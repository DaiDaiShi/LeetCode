
# [174. Dungeon Game](https://leetcode.com/problems/dungeon-game/description/)
```java
// given the example dungeon, lets label cells as follows: 
// +-+-+-+
// |1|2|3|
// +-+-+-+
// |4|5|6|
// +-+-+-+
// |7|8|9|
// +-+-+-+

The dungeon:                              Initial HP knight needed:       
+-------+-------+-------+                 +-------+-------+-------+
|       |       |       |                 |       |       |       |
|  -2   |  -3   |   3   |                 |   7   |   5   |   2   |
|       |       |       |                 |       |       |       |
+-------+-------+-------+                 +-------+-------+-------+
|       |       |       |                 |       |       |       |
|  -5   |  -10  |   1   |                 |   6   |   11  |   5   |
|       |       |       |                 |       |       |       |
+-------+-------+-------+                 +-------+-------+-------+
|       |       |       |                 |       |       |       |
|  10   |  30   |  -5(P)|                 |   1   |   1   |   6   |
|       |       |       |                 |       |       |       |
+-------+-------+-------+                 +-------+-------+-------+

Knight HP: 

// To solve this problem, we can start with the simpliest case.
// Let's say,

// If the knight starts from cell 9.
Initial HP: 6 (6 - 5 = 1), which means as long as knight has 6 HP when reaching cell 9, he would be fine.

// If the knight starts from cell 6.
Initial HP: 5 (5 + 1 == 6), which means as long as knight has 5 HP when reaching cell 6, he would be fine.

// If the knight starts from cell 8.
Initial HP: 1 (1 + 30 >= 6, HP needs to be at least 1, 0 means knight is already deat),
which means as long as knight has 1 HP when reaching cell 8, he would be fine.

// If knight starts from cell 5.
Emm..... knight now has two options, going right or going down.
If go right (5 --> 6 --> 9), Initial HP(R): 15 (15 - 10 = 5)
If go down  (5 --> 8 --> 9), Initial HP(D): 11 (11 - 10 = 1)
Hence, Initial HP = MIN(HP(R), HP(D)) = 11

// Sub-problems and state:
Let dp[i][j] denote Initial HP needed if the knight starts from dungeon[i][j].

// Recurrence relation:
dp[i][j] = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j];
if(dp[i][j] <= 0) dp[i][j] = 1;

```
## Code
```java
class Solution {
    public int calculateMinimumHP(int[][] dungeon) {
        if(dungeon == null || dungeon.length == 0 || dungeon[0].length == 0) return 1;
        
        int N = dungeon.length;
        int M = dungeon[0].length;
        int[][] dp = new int[N][M];
        dp[N - 1][M - 1] = 1 - dungeon[N - 1][M - 1];
        dp[N - 1][M - 1] = dp[N - 1][M - 1] <= 0 ? 1 : dp[N - 1][M - 1];
            
        for(int i = N - 1; i >= 0; --i){
            for(int j = M - 1; j >= 0; --j){
                if(i == N - 1 && j == M - 1) continue;
                int HP_D = i + 1 == N ? Integer.MAX_VALUE : dp[i + 1][j] - dungeon[i][j];
                int HP_R = j + 1 == M ? Integer.MAX_VALUE : dp[i][j + 1] - dungeon[i][j];
                int HP = Math.min(HP_D, HP_R);
                dp[i][j] = HP <= 0 ? 1 : HP;
            }    
        }
        
        return dp[0][0] ;
    }
}
```
