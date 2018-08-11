
# State
[576. Out of Boundary Paths](https://leetcode.com/problems/out-of-boundary-paths/description/)

Use one more dimension to represent state round.
e.g. dp\[i\]\[j\]\[u\] can be used to represent dp\[i\]\[j\] for round u.
```java
class Solution {
    public int findPaths(int m, int n, int N, int i_, int j_) {
        //state: dp[i][j][u] represents the number of paths to move the ball to cell[i][j] using u steps;
        if(N == 0) return 0;
        int[][][] dp = new int[m][n][N + 1];
        int[][] ds = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int MOD = 1000000007;
        
        dp[i_][j_][0] = 1;
        int count = 0;        
        for(int u = 1; u <= N; u++){
            for(int i = 0; i < m; ++i){
                for(int j = 0; j < n; ++j){
                    for(int[] d : ds){
                        int next_i = i + d[0];
                        int next_j = j + d[1];
                        if(next_i < 0 || next_i >= m || next_j < 0 || next_j >= n) {
                            count += dp[i][j][u - 1];
                            count %= MOD; 
                            continue;
                        }
                        dp[next_i][next_j][u] = (dp[next_i][next_j][u] + dp[i][j][u - 1]) % MOD;
                    }
                }
            }
        }
        return count;
    }
}
```
