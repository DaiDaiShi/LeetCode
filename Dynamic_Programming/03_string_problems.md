
# String Problems
[647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/description/)
```cc
class Solution {
public:
    int countSubstrings(string s) {
        //state: dp[i][j] represents if string[i, j] is a palindromic substring
        //recurrence equation:
        //if s[i] = s[j], dp[i][j] = dp[i - 1][j - 1]
        //start state: dp[i][i] = 1
        int n = s.size();
        vector<vector<bool>> dp(n, vector<bool>(n));
        
        for(int l = 0; l < n; ++l){
            for(int i = 0; i + l < n; i++){
                if(l == 0){
                    dp[i][i] = true;
                    continue;
                }
                int j = i + l;
                if(l == 1){
                    if(s[i] == s[j])
                        dp[i][j] = true;
                    continue;
                }
                dp[i][j] = dp[i + 1][j - 1] & (s[i] == s[j]);
            }
        }
        int count = 0;
        for(int i = 0; i < n; ++i){
            for(int j = i; j < n; ++j){
                count += dp[i][j];
            }
        }
        return count;
    }
};
```
