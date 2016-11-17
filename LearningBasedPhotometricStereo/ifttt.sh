#!/bin/bash
CURL=/usr/bin/curl
IFTTT_EVENT="sh_finished"
IFTTT_KEY="dU5-yOWyff13qQV9VxHGLX"
IFTTT_URL="https://maker.ifttt.com/trigger/${IFTTT_EVENT}/with/key/${IFTTT_KEY}"

${CURL} -X POST \
    ${IFTTT_URL} \
    --header "Content-Type: application/json" \
    --data-binary "{\"value1\": \"$1\", \"value2\": \"$2\", \"value3\": \"$3\"}"

echo " "
