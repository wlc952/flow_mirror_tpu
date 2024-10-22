#!/bin/bash

set -e

MODEL_DIR="models"
MODEL_ARCHIVE="models.tar.gz"
MODEL_URL="https://modelscope.cn/models/wlc952/flow_mirror_aigchub/resolve/master/models.tar.gz"

if [ ! -d "$MODEL_DIR" ]; then
    if [ ! -f "$MODEL_ARCHIVE" ]; then
        wget "$MODEL_URL"
    fi
    mkdir "$MODEL_DIR"
    tar xzf "$MODEL_ARCHIVE" -C "$MODEL_DIR"
    rm -rf "$MODEL_ARCHIVE"
fi