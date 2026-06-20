# Business Impact

## The problem

Companies that sell data or software to enterprise and institutional customers
are constantly asked to complete **Due Diligence Questionnaires (DDQs)** and
**security/vendor risk questionnaires**. These can run from dozens to several
hundred questions covering security, privacy, compliance, corporate structure,
data handling, and insurance.

The work has three painful characteristics:

1. **Repetitive** &mdash; the same questions recur across requesters with slightly
   different wording.
2. **High-stakes** &mdash; answers are contractual representations, so accuracy and
   consistency matter.
3. **Distributed knowledge** &mdash; the "right" answer often lives in a previous
   questionnaire, a policy PDF, or a colleague's memory.

The result is that skilled compliance, security, and operations staff spend hours
re-deriving answers they have already written before.

## The solution

The DDQ RAG Assistant centralizes every past answered questionnaire into one
searchable knowledge base and makes prior answers instantly reusable:

- Type a question and get a synthesized, **copy-ready** answer with its sources.
- Upload a brand-new blank questionnaire and have it **auto-answered** end to end.
- Keep answers **consistent** across submissions because they all draw from the
  same vetted source of truth.

## Where the value comes from

| Lever | Effect |
| --- | --- |
| Reuse of vetted answers | Less time re-writing; more consistent representations |
| Auto-extraction + auto-answer of full questionnaires | Turns a multi-hour task into a review task |
| Source + freshness on every answer | Reviewers trust and verify quickly |
| Template cache for repeat questions | Lower LLM cost and near-instant responses |
| Human-in-the-loop summary box | Speed without giving up final human sign-off |

## Design choices that protect the business

- **Grounded answers only.** The model is constrained to the company's own past
  responses and is instructed not to speculate, reducing the risk of inaccurate
  external representations.
- **Human review by design.** The tool produces a clean draft answer for a person
  to approve and submit; it is an accelerator, not an autopilot.
- **Confidence + freshness signals** direct reviewer attention to the answers most
  likely to need updating.

## Notes

- The figures and scenarios here are illustrative. Actual time and cost savings
  depend on questionnaire volume, length, and how complete the historical
  knowledge base is.
- All sample data in this repository is synthetic.
