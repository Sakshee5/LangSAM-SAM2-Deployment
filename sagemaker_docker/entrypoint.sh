#!/bin/bash

if [ "$1" = "serve" ]; then
    exec uvicorn app:app --host 0.0.0.0 --port 8080
else
    exec "$@"
fi
