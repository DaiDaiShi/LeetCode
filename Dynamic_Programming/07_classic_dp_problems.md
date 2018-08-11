[673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/)
```cc
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        //state: dp[i] represents the len of longest increasing subsequence[0, i]
        //recurrence equation: dp[i] = max(dp[j]) + 1, 0 < j < i
        //state: dp[0] = 0, dp[1] = 1
        int n = nums.size();
        if(n <= 1) return n;
        vector<int> dp(n);
        vector<int> counts(n);
        int LIS = 0;
        int res = 0;
        for(int i = 0; i < n; ++i){
            dp[i] = 1;
            for(int j = 0; j < i; ++j){
                if(nums[i] > nums[j]){
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            LIS = max(LIS, dp[i]);
            for(int j = 0; j < i; ++j){
                if(nums[i] > nums[j] && dp[i] == dp[j] + 1){
                    counts[i] += counts[j];
                }
            }
            counts[i] = counts[i] == 0 ? 1 : counts[i];
        }
        for(int i = 0; i < n; ++i){
            if(LIS == dp[i])
               res += counts[i];
        }
        return res;
    }
};
```
