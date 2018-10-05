# LeetCode 140. Word Break II
```java
// Hi This is Nick, Please make sure you've fully understood the problem before you continue.:)
Example 1:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]

// The basic idea for solving this problem is
The result of a string can be generated from the results of its substrings.

// Like, for string "catsanddog" (S) and one of its substrings "catsand" (T).
// Suppose we've known T can be broken into ["cat", "sand"] or ["cats", "and"],
// and "catsanddog"(S) = "catsand"(T) + "dog".
// We only need to check if "dog" is in the wordDict, which is true in this case.
// Let dp(S) denote the result of S.
dp(T) = 
["cat", "sand"], 
["apple", "pen"]

dp(S) <= dp(T) + "apple" = 
["cat", "sand", "apple"], 
["cats", "and", "apple"] 

// Thus, we can have
>> state: Let dp[i] denote all valid breaks for s(0, i)
E.g. in the above case,
s(0, 6) = "catsand"
dp[6] = ["cat", "sand"], ["cats", "and"]

// The way we solve this problem is like this:
wordDict = ["cat", "cats", "and", "sand", "dog"]
index:    0   1   2   3   4   5   6   7   8   9
        +---------------------------------------+
s:      | c | a | t | s | a | n | d | d | o | g |
        +---------------------------------------+
          i   
          j

dp[0] = []
dp[1] = []
dp[2] = ["cat"]
dp[3] = ["cats"]
dp[4] = []
dp[5] = []
dp[6] = ["cat sand","cats and"]
dp[7] = []
dp[8] = []
dp[9] = ["cat sand dog","cats and dog"]

>> recurrence relation:
dp[i] = 
for (int j = 0; j < i; j++) {
	substr = s[j, i];
  if (dp[j].size() > 0 && wordDict.contains(substr)) {
      for (String l : dp[j]) {
          dp[i].add(l + substr);
      }
  }
}

```
## Solutions:
https://leetcode.com/problems/word-break-ii/solution/
