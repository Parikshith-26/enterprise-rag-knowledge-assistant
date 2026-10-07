import requests


BASE_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "ZX Bank Enterprise Knowledge Assistant"


def test_normal_question():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "What documents are required for an Agriculture Loan?"
        },
        timeout=120,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"]
    assert data["search_question"]
    assert data["answer"]
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0


def test_empty_question():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": ""
        },
        timeout=10,
    )

    assert response.status_code == 422


def test_whitespace_question():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "   "
        },
        timeout=10,
    )

    assert response.status_code == 422

    data = response.json()

    assert "Question cannot be empty." in str(data)


def test_oversized_question():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "A" * 2001
        },
        timeout=10,
    )

    assert response.status_code == 422


def test_conversation_history_limit():
    history = [
        {
            "role": "user",
            "content": "Previous question",
        }
        for _ in range(21)
    ]

    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "What is an Agriculture Loan?",
            "conversation_history": history,
        },
        timeout=10,
    )

    assert response.status_code == 422


def test_cors_preflight():
    response = requests.options(
        f"{BASE_URL}/ask",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=10,
    )

    assert response.status_code == 200

    assert (
        response.headers.get("access-control-allow-origin")
        == "http://localhost:8501"
    )

    allowed_methods = response.headers.get(
        "access-control-allow-methods",
        "",
    )

    assert "POST" in allowed_methods


if __name__ == "__main__":
    tests = [
        test_health,
        test_normal_question,
        test_empty_question,
        test_whitespace_question,
        test_oversized_question,
        test_conversation_history_limit,
        test_cors_preflight,
    ]

    passed = 0
    failed = 0

    for test in tests:

        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1

        except Exception as exc:
            print(f"FAIL: {test.__name__}")
            print(f"      {exc}")
            failed += 1

    print()
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed > 0:
        raise SystemExit(1)

    print("ALL API TESTS PASSED")