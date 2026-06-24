# TraceFaith: Faithfulness Evaluation through Interventions for Cross-Modal Scientific Reasoning Traces

Research code for TraceFaith, a cross-modal faithfulness-evaluation project for multimodal scientific question answering.

## Project Aim

TraceFaith evaluates whether externally visible scientific reasoning traces are behaviorally load-bearing for a target vision-language model. The main experiment asks a target model to produce an original structured trace, asks an intervention model to generate compact natural-language component edits, constructs intervened traces deterministically, and reruns the target with each trace as the reasoning path.

The evaluation separates three claims:

- `grounding`: visual and textual evidence claims are supported by the item.
- `binding faithfulness`: evidence is assigned to the correct object, condition, relation, or scientific principle.
- `behavioral faithfulness`: changing a claimed support component changes target answer behavior under controlled interventions.
