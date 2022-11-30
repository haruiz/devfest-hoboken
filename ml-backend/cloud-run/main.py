import uvicorn
import os

if __name__ == "__main__":
    uvicorn.run(
        "api:app", host="0.0.0.0", reload=True, port=int(os.environ.get("PORT", 8080))
    )  # run the fastapi app
