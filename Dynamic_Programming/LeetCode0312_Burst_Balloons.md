
# 312. Burst Balloons
link: https://leetcode.com/problems/burst-balloons/description/



## Code
```java
public int maxCoins(int[] nums) {
    int N = nums.length;
    if(N == 0) return 0;
    
    int[][] opt = new int[N][N];
    
    for(int len = 0; len < N; ++len){
        for(int i = 0; i + len < N; ++i){
            int j = i + len;
            for(int k = i; k <= j; ++k){
                // numbers on left ballon and right ballon.
                int left_num = i == 0 ? 1 : nums[i - 1];
                int right_num = j == N - 1 ? 1 : nums[j + 1];
                
                // left opt and right opt
                int left_opt = k == i ? 0 : opt[i][k - 1];
                int right_opt = k == j ? 0 : opt[k + 1][j];
                
                opt[i][j] = Math.max(
                	opt[i][j], 
                	left_num * nums[k] * right_num + left_opt + right_opt);
            }
        }
    }
    
    return opt[0][N - 1];
}
```
