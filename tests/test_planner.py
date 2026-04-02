from agents.planner import PlannerAgent


def test_planner_generates_questions_without_llm() -> None:
    agent = PlannerAgent()
    questions = agent.plan("RAG 系统设计", max_subquestions=4)

    assert len(questions) == 4
    assert all(item.question for item in questions)
