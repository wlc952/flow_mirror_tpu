#!/bin/bash

set -e

MODEL_DIR="models"
MODEL_ARCHIVE="models.tar.gz"
MODEL_URL="https://github.com/wlc952/flow_mirror_tpu/releases/download/v0.1/models.tar.gz"

if [ ! -d "$MODEL_DIR" ]; then
    if [ ! -f "$MODEL_ARCHIVE" ]; then
        wget "$MODEL_URL"
    fi
    mkdir "$MODEL_DIR"
    tar xzf "$MODEL_ARCHIVE" -C "$MODEL_DIR"
    rm -rf "$MODEL_ARCHIVE"
fi