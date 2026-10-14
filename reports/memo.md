**To:** Sarah Okafor, Head of E-Commerce | **From:** Amiya Sekhar, Data & ML Team | **Date:** 13 October 2026
**Subject:** Product recommendations and customer groups for Q4 planning

## 1. Summary

We built a recommendation model from two years of purchase history and tested it on the last two months of sales it had never seen. The honest headline: a plain best-sellers list still beats the model at getting at least one suggestion right (26.8% vs 21.4% of customers), but the model catches more of each customer's actual new purchases (2.4% vs 1.7%), does much better for light buyers, and gives every customer a personal list. We recommend using the model's "customers who bought this also bought" suggestions on product pages now, and splitting personalised lists: best sellers for heavy buyers, model suggestions for light and newer buyers.

## 2. What we built and why this method

We took the table of which customer bought which product (5,471 customers by 3,687 products) and compressed it into 50 "taste patterns" using a method called Truncated SVD. We chose SVD because purchase data only records what people bought: SVD treats "not bought" as "no information", while the usual alternative, PCA, treats it as a mild dislike. In testing the two performed almost the same; SVD was slightly better on every measure and simpler to run.

## 3. Size and speed

- Full purchase table: 161 MB. Compressed patterns: 3.7 MB.
- Whole nightly job (build the patterns, product-page suggestions, lists for all customers): under 1 second on a laptop.

## 4. What we lose by compressing

The 50 patterns keep about 38% of the detail in the purchase table. The rest is mostly individual one-off buying that no shared pattern explains, and it matters little for ranking suggestions, which is what we need.

## 5. Results on purchases the model never saw

Test: the last two months of data (9 October to 9 December 2011), 1,931 returning customers who bought something new, 10 suggestions each.

| Measure | Best sellers | Our model |
|---|---|---|
| Hit rate: share of customers for whom at least one of the 10 suggestions was something they actually bought next | 26.8% | 21.4% |
| Share of each customer's new purchases caught by the 10 suggestions | 1.7% | 2.4% |

- Best sellers win on "at least one right" because Q4 buying is dominated by popular Christmas stock that most customers pick up anyway.
- The model is better for light buyers. For customers with 1 to 10 products in their history, it caught 10.9% of their new purchases against 2.4% for best sellers.
- The model's lists are personal: 97.6% of customers share no product between their model list and the best-seller list.
- Product-page suggestions are coherent: Regency cakestand leads to Regency teacups, tea plates and teapot; red retrospot lunch bag leads to the other lunch bag designs.

## 6. Recommendation

- Use the model's "customers who bought this also bought" table on product pages now.
- Personalised lists: show best sellers to heavy buyers and model suggestions to light and newer buyers.
- Next step: our check found that counting purchases as simply "bought / not bought" raised the hit rate by about 2.4 points. Test that, plus a blend with popularity, on a fresh time period before switching.

## 7. Customer groups for marketing

Five groups from the same patterns (map: `reports/figures/ml6_map_k5.png`). They describe buying tendencies, not hard boundaries.

- Occasional small buyers, 29.7%: few products, low spend, no distinctive items.
- Home décor & hearts, 23.9%: hanging hearts, zinc and ivory decorations, T-light holders.
- Kids' crafts & toys, 20.0%: paint sets, crayons, modelling clay, card games.
- Bulk bag buyers, 13.3%: jumbo and storage bags; highest spend, repeat trade-style orders.
- Baking & party, 13.2%: cake cases and paper doilies.

## 8. What could go wrong

- The test period was the Christmas peak; results may differ in other seasons.
- Rarely bought products can get odd matches because few customers bought them.
- 380 new customers had no history, so they can only get best sellers.
- We chose some settings after seeing test results, so the reported numbers are slightly optimistic.
- Very large one-off orders can skew a customer's suggestions; we dampen this, but it needs watching.
