from src.generation.answer_generator import AnswerGenerator


def run_test(name, question, expected_phrases=None, expect_no_sources=False):
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)

    rag = AnswerGenerator(top_k=5)

    result = rag.generate_answer(question)

    answer = result["answer"]
    sources = result["sources"]

    print("Question:", question)
    print("\nAnswer:")
    print(answer)

    print("\nSearch Query:")
    print(result["search_question"])

    print("\nSources:", len(sources))

    passed = True

    # Check expected phrases
    if expected_phrases:
        answer_lower = answer.lower()

        for phrase in expected_phrases:
            if phrase.lower() in answer_lower:
                print(f"  PASS: '{phrase}' found")
            else:
                print(f"  FAIL: '{phrase}' missing")
                passed = False

    # Check that no sources are returned
    if expect_no_sources:
        if len(sources) == 0:
            print("  PASS: No sources returned")
        else:
            print("  FAIL: Sources were returned")
            passed = False

    print("\nRESULT:", "PASS" if passed else "FAIL")

    return passed


def main():
    tests = []

    # ---------------------------------------------------------
    # TEST 1: Empty question
    # ---------------------------------------------------------
    tests.append(
        run_test(
            "Empty Question",
            "",
            expect_no_sources=True,
        )
    )

    # ---------------------------------------------------------
    # TEST 2: Out-of-domain question
    # ---------------------------------------------------------
    tests.append(
        run_test(
            "Out-of-Domain Question",
            "What is the weather in Bangalore today?",
        )
    )

    # ---------------------------------------------------------
    # TEST 3: Missing information / hallucination prevention
    # ---------------------------------------------------------
    tests.append(
        run_test(
            "Unavailable Information",
            "What is the exact interest rate for a ZX Bank Agriculture Loan?",
        )
    )

    # ---------------------------------------------------------
    # TEST 4: Long / complex query
    # ---------------------------------------------------------
    tests.append(
        run_test(
            "Long Complex Query",
            (
                "I want to apply for an Agriculture Loan online. "
                "Can you explain exactly where I need to go, "
                "what I need to fill in, what documents I need to upload, "
                "how I can track the application, and whether there is "
                "any AI assistance available?"
            ),
            expected_phrases=[
                "Loans > Agriculture Loan",
                "application",
                "documents",
                "track",
                "Zia",
            ],
        )
    )

    # ---------------------------------------------------------
    # TEST 5: Normal factual question
    # ---------------------------------------------------------
    tests.append(
        run_test(
            "Normal Factual Question",
            "What documents are required for an Agriculture Loan?",
            expected_phrases=[
                "Identity Proof",
                "Address Proof",
                "Land Ownership",
            ],
        )
    )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 70)
    print("RELIABILITY TEST SUMMARY")
    print("=" * 70)

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\nALL RELIABILITY TESTS PASSED")
    else:
        print("\nSOME RELIABILITY TESTS FAILED")


if __name__ == "__main__":
    main()