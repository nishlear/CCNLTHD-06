from src.backend import db, services
from src.backend.outcomes import outcome_fields


def questions(cv_id="cv-a"):
    return [{"cv_id":cv_id,"category":category,"question":category+str(n)}
            for category in ("technical","experience","behavioral") for n in range(2)]


def test_completed_output_quality_and_replayed_save_are_separate(isolated):
    db.create_search("s","query",1)
    db.save_shortlist("s",[{"cv_id":"cv-a","score":.9,"matched_skills":[],"missing_skills":[],"rank":1}])
    db.save_review("s",["cv-a"],questions(),errors=[])
    assert outcome_fields("s")["outcome"] == "succeeded"
    db.save_review("s",["cv-a"],questions(),errors=[])
    assert outcome_fields("s")["question_count"] == 6
    db.save_review("s",["cv-a"],questions()[:3],errors=[])
    assert outcome_fields("s")["outcome"] == "completed_with_errors"
    db.save_review("s",["cv-a"],[],errors=[{"node":"generate_questions","code":"ConnectError"}])
    assert outcome_fields("s")["incomplete_cv_ids"] == ["cv-a"]
    db.save_review("s",[],[],errors=[])
    assert outcome_fields("s")["outcome"] == "succeeded"


def test_api_history_and_review_expose_same_failed_generation(client,candidates,monkeypatch):
    def broken(*args):
        raise ConnectionError("provider offline")
    monkeypatch.setattr(services,"generate_questions",broken)
    created=client.post("/searches",json={"query":"Python","top_k":1}).json()
    assert created["ranking_mode"] == "soft_priority" and not created["hard_filters_applied"]
    assert created["outcome"] == "pending"
    search_id=created["id"]
    result=client.post(f"/searches/{search_id}/review",json={"approved_cv_ids":["cv-a"]}).json()
    detail=client.get(f"/searches/{search_id}").json()
    history=client.get("/searches").json()[0]
    for row in (result,detail,history):
        assert row["status"] == "completed"
        assert row["outcome"] == "completed_with_errors"
        assert row["expected_question_count"] == 6 and row["question_count"] == 0
        assert row["errors"][0]["node"] == "generate_questions"
    assert client.post(f"/searches/{search_id}/review",json={"approved_cv_ids":["cv-a"]}).status_code == 409


def test_failed_and_empty_search_outcomes(client,monkeypatch):
    monkeypatch.setattr(services,"vector_search",lambda *args: [])
    result=client.post("/searches",json={"query":"Python"}).json()
    assert result["outcome"] == "succeeded" and result["approved_count"] == 0
    def broken(*args):
        raise OSError("network")
    monkeypatch.setattr(services,"vector_search",broken)
    assert client.post("/searches",json={"query":"Python"}).status_code == 500
    assert any(row["outcome"] == "failed" for row in client.get("/searches").json())
