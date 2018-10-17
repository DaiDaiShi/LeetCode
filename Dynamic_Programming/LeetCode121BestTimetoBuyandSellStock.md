# LeetCode 121. Best Time to Buy and Sell Stock

```java
Example 1:
                        +-----------------------+
prices(p): 	        | 7 | 1 | 5 | 3 | 6 | 4 |
                        +-----------------------+
minimum so far(m):    7   1   1   1   1   1
diff(p - m):          0   0   4   2   5   3
max diff:             5

// Brute Force Approach:
// find max(prices[j] - prices[i]), for every i and j such that j > i.

// But do we really need to calculate all the differences?
// Like for 6, do we have to calculate 6 - 7, 6 - 1, 6 - 5, 6 - 3?
// The answer is no, as long as we know the minimum number among 7, 1, 5, 3, which is 1, 
// we know the maximum difference, right?
// So how can we know the minimum number among all prices[i]s?
// That is easy, we can have a variable to keep track of the minimum number we encountered so far.
```
## code
```java
class Solution {
    public int maxProfit(int[] prices) {
        int N = prices.length;
        if(N <= 1) return 0;
        int max_gain_sofar = 0;
        int hold = prices[0];
        for(int i = 0; i < N; ++i){
            max_gain_sofar = Math.max(max_gain_sofar, prices[i] - hold);
            hold = Math.min(hold, prices[i]);
        }
        return max_gain_sofar;
    }
}
```
