
# Game Problems
[877. Stone Game](https://leetcode.com/problems/stone-game/description/)
```cc
class Solution {
public:
    bool stoneGame(vector<int>& piles) {
        //state: 
        //  dp[i][j] represents the number with which one wins or lose the game. (+ means win, - means lose)
        //recurrence equation:
        //  dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
        //start state:
        //  dp[i][i] = piles[i];
        int n = piles.size();
        vector<vector<int>> dp(n, vector<int>(n));
        for(int l = 0; l < n; ++l){
            for(int i = 0; i + l < n; ++i){
                if(l == 0){
                    dp[i][i] = piles[i];
                    continue;
                }
                int j = i + l;
                dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
            }
        }
        return dp[0][n - 1];
    }
};
```
