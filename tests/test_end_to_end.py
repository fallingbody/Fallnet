import asyncio
import httpx
import websockets
import json

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post("http://127.0.0.1:8000/api/trigger", json={"prompt": "What is 2 + 2?"})
            print("Trigger response status:", resp.status_code)
            
        full_tokens = []
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get("type") == "token":
                    t = data.get("token", "")
                    full_tokens.append(t)
                    print(t, end="", flush=True)
                elif data.get("type") == "log" and "completed" in data.get("msg", "").lower():
                    break
            except asyncio.TimeoutError:
                break
        print(f"\n\nFull Result: '{''.join(full_tokens)}'")

if __name__ == "__main__":
    asyncio.run(main())
