from fastapi.testclient import TestClient
from apps.agentic_hr.api import app


client = TestClient(app)


def test_hr_chat_endpoint():
    response = client.post(
        "/hr/chat",
        json={"message": "What is the leave policy?"}
    )

    assert response.status_code == 200

    body = response.json()
    assert body["intent"] == "employment_policy"
    assert body["answer"] is not None
    assert "leave" in body["answer"].lower()
    assert isinstance(body["trace"], list)
