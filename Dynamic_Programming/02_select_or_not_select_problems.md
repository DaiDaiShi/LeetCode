# Select or Not Select Problems
For some DP problems, you have two options at every step, updating state or remain the previous state.

For instance.

[198. House Robber](https://leetcode.com/problems/house-robber/description/)
```
The robber is not allowed to rob two adjacent houses, which means if the robber decides to rob house i, 
he has to give up robbing house i-1 and i+1. This also means this robber has to decide either to rob or not rob
when he is in front of a house, and his decisions will eventually determine the houses he robs.
```
## Recurrence Equation for these problems:
```cc
given a nums[]
state: dp[i], representing the optimum decision which can be made at step i
Recurrence Equation:   
if choosing nums[i], dp[i] = dp[j] + nums[i], where j < i
otherwise            dp[i] = dp[i - 1]
Thus                 dp[i] = min(dp[j] + nums[i], where j < i, dp[i - 1])
```
[198. House Robber](https://leetcode.com/problems/house-robber/description/)
```cc
class Solution {
public:
    int rob(vector<int>& nums) {
        //select or not select problem:
        //state: the maximum amount of money you can rob without alerting the police when getting house i
        //recurrence equation: 
        //select i,             dp[i - 2] + nums[i]
        //not select i,         dp[i - 1]
        //dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        //start state: dp[0] = 1, dp[1] = max(nums[0], nums[1])
        
        int n = nums.size();
        if(n == 0) return 0;
        if(n == 1) return nums[0];
        
        vector<int> opt(n);
        opt[0] = nums[0];
        opt[1] = max(nums[1], nums[0]);
        for(int i = 2; i < n; ++i){
            opt[i] = max(opt[i - 2] + nums[i], opt[i - 1]);
        }
        return opt[n - 1];
    }
};
```
