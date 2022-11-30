import typing

import numpy as np
from fastapi import Depends
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from starlette.requests import Request
from pydantic import BaseModel
from model_loader import ModelLoader

router = InferringRouter()



async def get_model(req: Request):
    return req.app.state.model


@cbv(router)
class IrisController:
    model: ModelLoader = Depends(get_model)

    @router.get("/")
    def welcome(self):
        return {"message": "Welcome to the Iris API"}

    @router.get("/info")
    def model_info(self):
        """Return model information, version, how to call"""
        return {"name": self.model.name, "version": self.model.version}

    @router.get("/health")
    def service_health(self):
        """Return service health"""
        return "ok"


    @router.post("/predict")
    async def predict(self, request: Request):
        request_body = await request.json()
        input_batch = list(
            map(
                lambda x: [
                    x["sepal_length"],
                    x["sepal_width"],
                    x["petal_length"],
                    x["petal_width"],
                ],
                request_body,
            )
        )
        predictions = self.model(input_batch)
        return {"predictions": predictions.tolist()}
