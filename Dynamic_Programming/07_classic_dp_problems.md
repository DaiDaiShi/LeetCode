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
[790. Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/description/)
```cc
class Solution {
public:
    int numTilings(int M) {
        //state: dp[i][N], dp[i][L], dp[i][R]
        //recurrence equation: 
        // dp[i][N] = (dp[i - 1][N] + dp[i - 2][N] + dp[i - 1][L] + dp[i - 1][R]) % MOD;
        // dp[i][L] = (dp[i - 1][R] + dp[i - 2][N]) % MOD;
        // dp[i][R] = (dp[i - 1][L] + dp[i - 2][N]) % MOD;
        const int N = 0, L = 1, R = 2;
        const int MOD = 1000000007;
        vector<vector<long>> dp(M + 1, vector<long>(3));
        dp[0][N] = 1;
        
        dp[1][N] = 1;
        
        dp[2][N] = 2;
        dp[2][L] = 1;
        dp[2][R] = 1;
        
        for(int i = 3; i <= M; ++i){
            dp[i][N] = (dp[i - 1][N] + dp[i - 2][N] + dp[i - 1][L] + dp[i - 1][R]) % MOD;
            dp[i][L] = (dp[i - 1][R] + dp[i - 2][N]) % MOD;
            dp[i][R] = (dp[i - 1][L] + dp[i - 2][N]) % MOD;
        }
        
        return dp[M][N];
    }
};
```
[120. Triangle](https://leetcode.com/problems/triangle/description/)
```cc
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size();
        int m = n;
        for(int i = n - 2; i >= 0; --i){
            for(int j = 0; j < m - 1; ++j){
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1]);
            }
            --m;
        }
        return triangle[0][0];
    }
};
```
