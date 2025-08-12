import os
import requests
import json
import threading
import time
from queue import Queue, Empty
from flask import Flask, request, jsonify
import google.generativeai as genai

# --- Configuration ---
# Securely load environment variables
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
APPSHEET_API_KEY = os.environ.get('APPSHEET_API_KEY')
APPSHEET_APP_ID = os.environ.get('APPSHEET_APP_ID')
APPSHEET_TABLE_NAME = os.environ.get('APPSHEET_TABLE_NAME')
# The name of the key column in your AppSheet table (e.g., "ID", "AccountID")
APPSHEET_KEY_COLUMN = os.environ.get('APPSHEET_KEY_COLUMN', 'ID')

# --- Batching Configuration ---
MAX_BATCH_SIZE = 50  # Max number of rows to send to AppSheet in one go
BATCH_TIMEOUT = 30  # Max seconds to wait before sending a partial batch

# --- Prompts ---
PROMPT_START = 'Find the domain for the company: COMPANY_NAME. '
PROMPT_WEBSITE = 'The company\'s website might be: WEBSITE. But do not blindly believe it, double check if its appropriate or if there are better altnerative websites, for instance local websites for local subsidiaries.'
PROMPT_ADDRESS = 'If available, consider the address: ADDRESS. If it\'s an international company, but if the address points to another country than the international HQ country and there is a local domain, then use the local domain.'
PROMPT_END = ('Prioritize sources like company profiles on Linkedin, official company reports, or reputable financial news sites. Domains come in the format of basel.ch or facebook.com and generally do not have hosts as in website urls.'
              'If a value is not found, return \'N/A\' for that value.')

# --- Flask App & Worker Queues Initialization ---
app = Flask(__name__)
gemini_task_queue = Queue()
appsheet_update_queue = Queue()

# --- Logging Helper ---
def log(message):
    """Simple logger to print timestamped messages."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# --- Helper Functions ---

def construct_prompt(name, website, address):
    """Constructs the full prompt for the Gemini API."""
    prompt = PROMPT_START.replace("COMPANY_NAME", name)
    if website:
        prompt += PROMPT_WEBSITE.replace("WEBSITE", website)
    if address:
        prompt += PROMPT_ADDRESS.replace("ADDRESS", address)
    prompt += PROMPT_END
    log(f"📄 Constructed Prompt for {name}")
    return prompt

def call_gemini_api(prompt_text):
    """Calls the Gemini API, configured to return a JSON object directly."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config
        )
        schema = {
            "type": "object",
            "properties": {"domain": {"type": "string"}},
            "required": ["domain"]
        }
        response = model.generate_content([prompt_text, "output_schema:", str(schema)])
        return response
    except Exception as e:
        log(f"❌ Error calling Gemini API: {e}")
        return None

def parse_gemini_response(response):
    """Parses the Gemini response."""
    if not response or not response.text:
        log("❌ Gemini response is empty or invalid.")
        return None
    log(f"📝 Raw Gemini Response (JSON): {response.text}")
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        log(f"❌ Error parsing JSON from response: {e}")
        log(f"   L- Raw text that failed to parse: {response.text}")
        return None

# --- Background Workers ---

def gemini_worker():
    """Worker that gets tasks, calls Gemini, and puts results in the AppSheet queue."""
    while True:
        try:
            name, website, address, row_key_value = gemini_task_queue.get()
            log(f"🤖 Gemini Worker processing task for row key: {row_key_value}")
            prompt = construct_prompt(name, website, address)
            response = call_gemini_api(prompt)

            if not response:
                log(f"Aborting update for row {row_key_value} due to Gemini API error.")
                gemini_task_queue.task_done()
                continue

            ai_data = parse_gemini_response(response)
            if not ai_data:
                log(f"Aborting update for row {row_key_value} due to parsing error.")
                gemini_task_queue.task_done()
                continue

            # Add the result to the next queue for batching
            appsheet_update_queue.put({
                APPSHEET_KEY_COLUMN: row_key_value,
                "Domain": ai_data.get("domain", "Parse Error")
            })
            log(f"✅ Gemini processing complete for {row_key_value}. Pushed to AppSheet update queue.")
            gemini_task_queue.task_done()
        except Exception as e:
            log(f"🔥 An error occurred in the gemini_worker thread: {e}")


def appsheet_batch_worker():
    """Worker that collects results and sends them to AppSheet in batches."""
    batch = []
    last_send_time = time.time()

    while True:
        try:
            # Non-blocking get from the queue
            item = appsheet_update_queue.get_nowait()
            batch.append(item)
            log(f"📥 Added item for row key {item.get(APPSHEET_KEY_COLUMN)} to batch. Batch size: {len(batch)}")
            appsheet_update_queue.task_done()
        except Empty:
            # Queue is empty, do nothing
            pass

        # Check if we should send the batch
        time_since_last_send = time.time() - last_send_time
        if (len(batch) >= MAX_BATCH_SIZE) or (len(batch) > 0 and time_since_last_send >= BATCH_TIMEOUT):
            log(f"📦 Sending batch. Reason: {'Batch size limit' if len(batch) >= MAX_BATCH_SIZE else 'Timeout'}. Size: {len(batch)}")
            send_batch_to_appsheet(batch)
            batch = [] # Reset the batch
            last_send_time = time.time()

        # Sleep briefly to prevent this loop from pegging the CPU
        time.sleep(0.5)


def send_batch_to_appsheet(batch):
    """Sends a batch of rows to the AppSheet API with retry logic."""
    if not batch:
        return

    log(f"🚀 Preparing to send a batch of {len(batch)} rows to AppSheet.")
    appsheet_api_url = f"https://api.appsheet.com/api/v2/apps/{APPSHEET_APP_ID}/tables/{APPSHEET_TABLE_NAME}/Action"
    headers = {
        "Content-Type": "application/json",
        "ApplicationAccessKey": APPSHEET_API_KEY
    }
    payload = {
        "Action": "Edit",
        "Properties": {"Locale": "en-US"},
        "Rows": batch
    }

    log(f"📤 Sending this payload to AppSheet: {json.dumps(payload, indent=2)}")

    max_retries = 8
    base_delay = 5
    for attempt in range(max_retries):
        try:
            api_response = requests.post(appsheet_api_url, headers=headers, data=json.dumps(payload), timeout=60)
            api_response.raise_for_status()
            log(f"✅ Successfully updated AppSheet with batch. Status: {api_response.status_code}")
            return
        except requests.exceptions.RequestException as e:
            if e.response is not None and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (os.urandom(1)[0] / 255)
                    log(f"⚠️ Rate limited by AppSheet. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    log(f"❌ Max retries reached for batch. Giving up.")
                    break
            else:
                log(f"❌ An unrecoverable error occurred sending batch to AppSheet: {e}")
                if e.response:
                    log(f"   L- Server Response: {e.response.text}")
                break

# --- Start Worker Threads ---
# Start workers to process tasks from the Gemini queue
num_gemini_workers = 2
for i in range(num_gemini_workers):
    threading.Thread(target=gemini_worker, daemon=True).start()

# Start a single worker to handle batching and updating AppSheet
threading.Thread(target=appsheet_batch_worker, daemon=True).start()

# --- Main Flask Routes ---

@app.route('/start-ai-lookup', methods=['POST'])
def webhook():
    """Endpoint that AppSheet calls. Adds tasks to a queue for Gemini processing."""
    if not all([GEMINI_API_KEY, APPSHEET_API_KEY, APPSHEET_APP_ID, APPSHEET_TABLE_NAME]):
        log("🔥 Server-side configuration error: Missing environment variables.")
        return jsonify({"error": "Server is not configured correctly. Check logs."}), 500

    try:
        data = request.get_json(force=True)
        log(f"▶️ Received request data: {json.dumps(data, indent=2)}")
        
        name = data['Name']
        address = data.get('Address', '')
        website = data.get('Website', '')
        row_key_value = data[APPSHEET_KEY_COLUMN]
    except KeyError as e:
        log(f"🔥 Missing key in request body: {e}")
        return jsonify({"error": f"Request body is missing a required field: {e}"}), 400

    gemini_task_queue.put((name, website, address, row_key_value))
    log(f"✅ Task for row {row_key_value} added to Gemini queue. Queue size: {gemini_task_queue.qsize()}")
    
    return jsonify({"status": "AI processing task has been queued."}), 200

@app.route('/', methods=['GET'])
def index():
    """A simple root endpoint to check if the server is running."""
    return "<h1>Asynchronous AppSheet Batch Processor</h1><p>The server is running and ready to receive requests.</p>", 200
