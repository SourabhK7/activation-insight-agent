# Example Diagnosis — Activation Insight Agent

> This is an example output produced by running the agent against the included
> synthetic funnel (`data/sample_funnel.csv`). It demonstrates the format and
> calibration level the agent produces. Your own runs will look similar but
> will vary in phrasing (LLM outputs are not deterministic).
>
> Generated from: `python -m activation_agent run --data data/sample_funnel.csv`
>
> Input findings: `examples/findings.json`

---

## Headline

Of the 50,000 users who entered the funnel, 16.0% completed purchase. The single biggest drop-off is between product page view and add-to-cart, where 50.4% of users abandon — a step worth investigating regardless of segment. The most notable segment finding is that users acquired through paid social convert at 8.9%, roughly half the overall rate, with the gap concentrated at the product page step.

## Funnel overview

The funnel shows two structural drop-off points. The first and largest is between landing page view and product page view, where 29.7% of users leave — unsurprising for top-of-funnel but worth confirming is within expected bounds for this acquisition mix. The second and more actionable is between product page view and add-to-cart, where 50.4% of users drop. Below this point the funnel is reasonably healthy: 77.6% of add-to-cart users reach shipping, 65.7% of those reach payment, and 89.8% of those complete purchase. The shipping-to-payment transition (65.7% continuation) is the only mid-funnel step with meaningful drop-off, and it's masked in the overall number by segment-level heterogeneity (see below).

## Segment findings

**Paid-social acquisition underperforms substantially.** Users acquired through paid social (n=9,952, about 20% of the funnel) convert end-to-end at 8.9%, compared to 16.0% overall — a 7.1 percentage point gap. The divergence is almost entirely at the product-page-view-to-add-to-cart step: paid-social users continue to add-to-cart at only 26.7%, versus 49.6% overall. This pattern is consistent with an ad-to-landing mismatch, where the expectations set by the ad creative do not match what users see on the landing or product page. Downstream steps (add-to-cart onward) look similar for paid-social users vs. overall, which makes the mismatch hypothesis more specific: it's about initial intent, not about checkout friction.

**Desktop users convert noticeably better than overall.** Desktop users (n=22,554) convert at 20.7%, versus 16.0% overall — a 4.7 percentage point lift. The divergence is concentrated at the shipping-info-to-payment-info step, where desktop users continue at 84.9% vs. 65.7% overall. This is the mirror image of a mobile-specific problem: if desktop's rate is "normal" and mobile is dragging the overall rate down, the effective issue is mobile-side friction at the payment step, not desktop doing unusually well. The agent detected this from the desktop side because mobile's absolute delta fell just below the divergence threshold on end-to-end conversion, but the step-level delta at shipping→payment is large enough to act on.

## What this suggests (interpretation)

Two distinct issues, likely requiring different owners and fixes:

The paid-social underperformance is upstream — it's a marketing/creative problem, not a product problem. The affected users are reaching the product page but not forming purchase intent. This could be because the ad promised something the page doesn't deliver, because the landing page isn't tuned for the paid-social audience, or because the audience targeting is bringing in low-intent users. Investigation should start with the creative team comparing ad copy to landing page content, and with a look at which specific campaigns within paid social are contributing the drop (the analysis treats paid-social as a single bucket).

The desktop-vs-mobile gap at the payment step is downstream — it's a checkout UX problem, most likely mobile-specific. The fact that the gap is at payment entry specifically (not at shipping entry or purchase completion) suggests a technical or UX issue with the payment form on mobile: hard-to-tap fields, keyboard issues, validation failures, or a slow-loading payment widget. A session-replay review of mobile sessions that abandon at the shipping→payment transition would quickly confirm or rule out a specific cause.

Neither of these interpretations can be confirmed from funnel data alone — they're hypotheses to test.

## Caveats

- This is observational funnel data, not an experiment. Every pattern described above is a correlation. We cannot attribute causation without further investigation (session replays, user research, or an A/B test).
- The country-code attribute was analyzed but no country-level segment crossed the divergence threshold, which may mean no country-level issue exists, or may mean the effect is too small to detect at the available sample sizes. The largest country segment (US) had ~30,000 users; smaller segments (UK, Canada, CX) had 5-8,000 each.
- The paid-social segment is treated as a single bucket. In reality, paid-social likely contains multiple campaigns with very different performance. A per-campaign breakdown would be more actionable.
- The analysis uses a 4-percentage-point threshold for flagging divergent segments. Segments closer to the threshold (like device splits on end-to-end conversion) may contain real patterns that require looking at step-level behavior to see — which is why the desktop finding above is itself pointing at a mobile problem.

## Suggested next steps

1. **For paid social:** pull campaign-level breakdown of the paid_social bucket. If the drop is concentrated in 1-2 campaigns, pause or diagnose those; if it's uniform across campaigns, the issue is landing-page-side, and the fix is a landing page variant test tuned for paid-social traffic.
2. **For mobile payment:** run a session replay sampling of mobile users who reach shipping_info_entered but not payment_info_entered. Look specifically for payment-form load times, field validation behavior, and keyboard/autofill interaction. If a clear issue emerges, fix and re-measure on the mobile cohort.
3. **For the analysis itself:** re-run this diagnosis after 2 weeks of the above changes to confirm that targeted fixes moved the affected segments without regressing others. Comparing two diagnoses over time is a reasonable use of this tool.
