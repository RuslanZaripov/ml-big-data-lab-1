from fastapi import FastAPI
from logger import Logger

SHOW_LOG = True
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

app = FastAPI()

log.info('FastAPI app Initialized')


@app.get("/")
async def root():
    return {"message": "Hello World"}