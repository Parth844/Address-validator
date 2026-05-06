from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict, predict_batch
import uvicorn

app = FastAPI(
    title="Address Validator API",
    description="Backend API for Indian Address Validation using XGBoost and Rule Engine",
    version="2.0.0"
)

class AddressRequest(BaseModel):
    address: str

class BatchRequest(BaseModel):
    addresses: list[str]

@app.get("/")
def read_root():
    return {"message": "Address Validator API is running", "version": "2.0.0"}

@app.post("/validate")
def validate_address(request: AddressRequest):
    try:
        result = predict(request.address)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate_batch")
def validate_batch(request: BatchRequest):
    try:
        results = predict_batch(request.addresses)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
