from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from iris_controller import router as iris_router
from users_controllers import router as users_router
from model_loader import ModelLoader


app = FastAPI()


@app.on_event("startup")
def load_model():
    # app.state.model = ModelLoader(
    #     path="models/tf/iris_model", name="iris", backend="tensorflow"
    # )
    app.state.model = ModelLoader(
        path="models/sklearn/iris_model.pk", name="iris", backend="sklearn"
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(
    iris_router,
    prefix="/iris",
    tags=["iris"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    users_router,
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


@app.get("/hi")
def root():
    return {"message": "Hello World from FastAPI and Docker Deployment course"}
