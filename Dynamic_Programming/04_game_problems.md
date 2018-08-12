
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
[375. Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/description/)
```cc
class Solution {
public:
    int getMoneyAmount(int n) {
        //dp[i][j] represents the least money you need to have to guarantee a win for [i, j].
        //dp[i][j] = max(k + max(dp[i][k - 1], dp[k + 1][j]), dp[i][j])
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 1000000));
        for(int i = 0; i <= n; ++i) dp[i][i] = 0;
        for(int l = 0; l <= n; ++l) {
            for(int i = 1; i + l <= n; i++){
                int j = i + l;
                if(l == 1){
                    dp[i][j] = i;
                    continue;
                }
                for(int k = i + 1; k < j; k++){
                    dp[i][j] = min(dp[i][j], k + max(dp[i][k - 1], dp[k + 1][j]));
                }
            }
        }
        return dp[1][n];
    }
};
```
