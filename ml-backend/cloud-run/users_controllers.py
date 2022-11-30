from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

router = InferringRouter()


@cbv(router)
class UsersController:
    @router.get("/")
    def welcome(self):
        return {"message": "Welcome to the Users API"}
