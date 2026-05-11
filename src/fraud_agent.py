import anthropic 
from google.cloud import aiplatform
import json

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

ENDPOINT_ID = "755815288049500160"
aiplatform.init(project="end-to-end-llm", location="us-west1")
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

tools = [
    {
        "name": "check_fraud",
        "description": (
            "Score a credit card transaction for fraud probability. "
            "Returns the fraud probability, whether it exceeds the threshold, "
            "and the top contributing features with SHAP explanations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "V1":  {"type": "number"}, "V2":  {"type": "number"},
                "V3":  {"type": "number"}, "V4":  {"type": "number"},
                "V5":  {"type": "number"}, "V6":  {"type": "number"},
                "V7":  {"type": "number"}, "V8":  {"type": "number"},
                "V9":  {"type": "number"}, "V10": {"type": "number"},
                "V11": {"type": "number"}, "V12": {"type": "number"},
                "V13": {"type": "number"}, "V14": {"type": "number"},
                "V15": {"type": "number"}, "V16": {"type": "number"},
                "V17": {"type": "number"}, "V18": {"type": "number"},
                "V19": {"type": "number"}, "V20": {"type": "number"},
                "V21": {"type": "number"}, "V22": {"type": "number"},
                "V23": {"type": "number"}, "V24": {"type": "number"},
                "V25": {"type": "number"}, "V26": {"type": "number"},
                "V27": {"type": "number"}, "V28": {"type": "number"},
                "Amount": {"type": "number"},
                "top_k_features": {
                    "type": "integer",
                    "description": "Number of top SHAP features to return",
                    "default": 5,
                },
            },
            "required": [
                "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                "V21","V22","V23","V24","V25","V26","V27","V28","Amount",
            ],
        },
    }
]

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Call the fraud detection API and return the result as a JSON string."""
    if tool_name != "check_fraud":
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    top_k = tool_input.pop("top_k_features", 5)

    # Build the request in Vertex AI format
    payload = {
        "instances": [tool_input],
        "parameters": {"explain": True, "top_k_features": top_k},
    }

    #response = httpx.post(f"{API_URL}/predict", json=payload, timeout=30.0)
    response = endpoint.predict(
        instances=[tool_input],
        parameters={"explain": True, "top_k_features": 5}
    )
    #response.raise_for_status()
    return json.dumps(response.predictions)


user_message = ("Check this transaction for fraud: "
        "V1=-0.49, V2=-0.56, V3=0.59, V4=-1.66, V5=-0.26, "
        "V6=0.63, V7=0.27, V8=0.06, V9=-0.95, V10=0.22, "
        "V11=0.35, V12=0.02, V13=0.42, V14=-0.51, V15=-1.22, "
        "V16=1.03, V17=0.03, V18=-1.08, V19=1.69, V20=0.10, "
        "V21=-0.08, V22=-0.21, V23=0.19, V24=0.21, V25=-0.97, "
        "V26=-0.62, V27=0.20, V28=0.15, Amount=175.66")
messages = [{"role": "user", "content": user_message}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    system=(
        "You are a fraud analyst assistant. When the user provides "
        "transaction features, use the check_fraud tool to score them. "
        "Then explain the results in plain English, highlighting which "
        "features drove the prediction and what they might indicate."
    ),
    tools=tools,
    messages=messages,
)
print("Stop Reason: ", response.stop_reason)
print("Text: ", response.content[0].text)

def chat(user_message: str) -> str:
    """
    Send a message to Claude with the fraud detection tool available.
    Handles the tool-use loop: Claude may call the tool, we execute it
    and send the result back, repeating until Claude gives a final answer.
    """
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=(
            "You are a fraud analyst assistant. When the user provides "
            "transaction features, use the check_fraud tool to score them. "
            "Then explain the results in plain English, highlighting which "
            "features drove the prediction and what they might indicate."
        ),
        tools=tools,
        messages=messages,
    )

    # Loop while Claude wants to use tools
    while response.stop_reason == "tool_use":
        # Find the tool_use block(s) in Claude's response
        tool_use_block = next(
            block for block in response.content if block.type == "tool_use"
        )

        # Execute the tool
        result = execute_tool(tool_use_block.name, tool_use_block.input)

        # Append Claude's response and the tool result to the conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": result,
                }
            ],
        })

        # Send back to Claude with the tool result
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=(
                "You are a fraud analyst assistant. When the user provides "
                "transaction features, use the check_fraud tool to score them. "
                "Then explain the results in plain English, highlighting which "
                "features drove the prediction and what they might indicate."
            ),
            tools=tools,
            messages=messages,
        )

    # Extract the final text response
    final_text = next(
        (block.text for block in response.content if block.type == "text"),
        "No response generated.",
    )
    return final_text