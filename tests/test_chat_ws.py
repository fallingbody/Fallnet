import asyncio
import json
import httpx
import websockets

async def test_chat():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as ws:
        print("Connected to WebSocket!")
        
        async def listen():
            full_text = []
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("type") == "token":
                        t = data.get("token", "")
                        full_text.append(t)
                        print(t, end="", flush=True)
                    elif data.get("type") == "log" and "completed" in data.get("msg", ""):
                        break
                except Exception:
                    break
            print(f"\n\n--- Full Response Received ---\n{''.join(full_text)}")

        listener = asyncio.create_task(listen())
        
        # Trigger inference with 60s timeout
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post("http://127.0.0.1:8000/api/trigger", json={"prompt": "What is 2 + 2?"})
            print(f"\nTrigger Status Code: {resp.status_code}")
            
        await listener

if __name__ == "__main__":
    asyncio.run(test_chat())
