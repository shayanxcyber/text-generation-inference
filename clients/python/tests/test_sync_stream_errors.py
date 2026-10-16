import json

import pytest
import requests

from text_generation import Client
from text_generation.errors import (
    NotFoundError,
    OverloadedError,
    RateLimitExceededError,
)


def make_response(status_code, payload, event=False):
    response = requests.Response()
    response.status_code = status_code
    body = json.dumps(payload)
    response._content = (f"data: {body}\n\n" if event else body).encode("utf-8")
    response._content_consumed = True
    return response


def open_stream(client, endpoint):
    if endpoint == "completion":
        return client.completion("hello", stream=True)
    return client.chat([{"role": "user", "content": "hello"}], stream=True)


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
@pytest.mark.parametrize(
    "status_code,payload,error",
    [
        (404, {"error": "not found"}, NotFoundError),
        (429, {"error": "too many requests"}, RateLimitExceededError),
        (503, {"error": "busy", "error_type": "overloaded"}, OverloadedError),
    ],
)
def test_stream_reports_http_errors(monkeypatch, endpoint, status_code, payload, error):
    response = make_response(status_code, payload)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(error, match=payload["error"]):
        list(open_stream(Client("http://localhost:8080"), endpoint))


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
def test_stream_reports_error_events(monkeypatch, endpoint):
    response = make_response(
        200, {"error": "busy", "error_type": "overloaded"}, event=True
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    with pytest.raises(OverloadedError, match="busy"):
        list(open_stream(Client("http://localhost:8080"), endpoint))


@pytest.mark.parametrize("endpoint", ["completion", "chat"])
def test_stream_still_yields_valid_events(monkeypatch, endpoint):
    if endpoint == "completion":
        payload = {
            "index": 0,
            "text": "hello",
            "logprobs": None,
            "finish_reason": "length",
        }
    else:
        payload = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "tgi",
            "system_fingerprint": "test",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hello"}}
            ],
        }
    response = make_response(200, payload, event=True)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    chunks = list(open_stream(Client("http://localhost:8080"), endpoint))

    assert len(chunks) == 1
    if endpoint == "completion":
        assert chunks[0].text == "hello"
    else:
        assert chunks[0].choices[0].delta.content == "hello"
