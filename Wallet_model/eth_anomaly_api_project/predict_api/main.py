from fastapi import FastAPI
from ethereum_eos import router as ethereum_eos_router
from base_eos import router as base_eos_router
from ethereum_token import router as ethereum_token_router
from base_token import router as base_token_router

app = FastAPI()

app.include_router(ethereum_eos_router)
app.include_router(base_eos_router)
app.include_router(ethereum_token_router)
app.include_router(base_token_router)