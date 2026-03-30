from app.models.input import CallTranscript

#Rules fpr each department to include in the prompt. These help guide the LLM to focus on relevant criteria for each type of call. If the department is unrecognized, it falls back to general guidelines.
DEPARTMENT_RULES: dict[str, str] = {
    "Scheduling": (
        "- Verify that appointment details (date, time, location, provider) were confirmed clearly.\n"
        "- Flag if the agent failed to offer a confirmation number or follow-up instructions.\n"
        "- Note if the caller's preferred time or location was acknowledged."
    ),
    "Onboarding": (
        "- Verify that lien agreement or financial responsibility was discussed.\n"
        "- Flag if consent or paperwork requirements were skipped.\n"
        "- Note if the agent explained next steps and what the patient should expect."
    ),
    "Helpdesk": (
        "- Verify that the caller's issue was understood and a resolution path was provided.\n"
        "- Flag if the agent left the caller without a clear next step.\n"
        "- Note warm transfer quality if the call was escalated."
    ),
    "Follow-Ups": (
        "- Verify that the purpose of the follow-up was clearly communicated.\n"
        "- Flag if required follow-up actions (referral status, test results) were left unaddressed.\n"
        "- Note if the agent set a clear callback or action timeline."
    ),
    "Records": (
        "- Pay extra attention to HIPAA: patient data should only be released after identity verification.\n"
        "- Flag if PHI (Protected Health Information) was discussed without confirming caller identity first.\n"
        "- Verify that the records request process was explained accurately."
    )
}

DEFAULT_DEPARTMENT_RULES = (
    "- Apply general healthcare call center standards.\n"
    "- Verify the caller's need was identified and addressed.\n"
    "- Note if the agent provided accurate information and a clear resolution."
)


def get_department_rules(department: str) -> str:
    return DEPARTMENT_RULES.get(department, DEFAULT_DEPARTMENT_RULES)


SYSTEM_PROMPT = """\
You are a call center QA analyst.
Your job is to analyze agent-caller transcripts and produce a structured quality report.

CORE PRINCIPLES — follow these without exception:
1. EVIDENCE-BASED: Only flag issues you can directly observe in the transcript.
   Do not infer, assume, or speculate about intent. If something is ambiguous, note the ambiguity.
2. PROPORTIONATE: Distinguish between critical violations and minor imperfections.
   "escalate" is reserved for HIPAA violations, clearly dangerous misinformation, or explicit rudeness.
   Minor mistakes, awkward phrasing, or incomplete scripts are "needs_review" at most.
3. NON-PUNITIVE: The purpose is to identify genuine issues and coach improvement — not to penalize.
   Always acknowledge what the agent did well alongside areas for improvement.
4. ACCURATE: The clinic serves real patients. Wrong medical or insurance information is a patient safety risk.
   Treat misinformation with appropriate severity.
5. PARSEABLE: Your output must be valid JSON matching the exact schema provided. No prose, no markdown fences.

ESCALATION THRESHOLDS — only escalate for:
- Confirmed HIPAA violation (PHI disclosed without identity verification)
- Dangerous misinformation (wrong medication, wrong dosage, false authorization status that could harm a patient)
- Explicit verbal rudeness, aggression, or threats toward a caller
- Any situation where a vulnerable patient may have been harmed or misled

SHORT OR UNCLEAR TRANSCRIPTS:
- If the transcript is very short (under ~100 words), note this in assessment_reasoning.
- Score conservatively: do not penalize for things you cannot observe.
- If quality cannot be assessed, set professionalism_score, accuracy_score, and resolution_score to 0.5 (neutral).

EDGE CASES:
- Disconnected calls: if the call ended abruptly, note it; do not penalize the agent.
- Calls with no issues: return overall_assessment "pass" and include at least one "positive_interaction" flag.
- Calls entirely in another language: note the language barrier; do not assess content accuracy.
"""

def build_user_prompt(call: CallTranscript) -> str:
    dept_rules = get_department_rules(call.department)

    return f"""\
Analyze the following call transcript and return a JSON quality report.

CALL METADATA:
- Call ID: {call.call_id}
- Agent: {call.agent_name}
- Date: {call.call_date}
- Duration: {call.call_duration_seconds} seconds total
- Department: {call.department}

DEPARTMENT-SPECIFIC CHECKLIST for {call.department}:
{dept_rules}

TRANSCRIPT:
{call.transcript}

---
Return ONLY valid JSON matching this exact schema (no markdown, no preamble):

{{
  "call_id": "{call.call_id}",
  "overall_assessment": "<pass | needs_review | escalate>",
  "assessment_reasoning": "<2-4 sentences>",
  "compliance_flags": [
    {{
      "type": "<hipaa_concern | misinformation | rudeness | protocol_violation | positive_interaction>",
      "severity": "<critical | moderate | minor | positive>",
      "description": "<1-2 sentences>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ],
  "agent_performance": {{
    "professionalism_score": <0.0–1.0>,
    "accuracy_score": <0.0–1.0>,
    "resolution_score": <0.0–1.0>,
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>", "<improvement 2>"]
  }},
  "escalation_required": <true | false>,
  "escalation_reason": "<reason string or null>"
}}
"""
