
# Shortest Path Problems
[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/description/)
```cc
class Solution {
public:
    int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int K) {
        vector<int> dp(n, 1 << 16);
        unordered_map<int, unordered_map<int, int>> graph;
        buildGraph(graph, flights);
        
        queue<vector<int>> Q;
        Q.push({src, 0}); 
        while(!Q.empty() && K != -1){
            int l = Q.size();
            for(int i = 0; i < l; ++i){
                vector<int> top = Q.front();
                int cur = top[0];
                int all_cost = top[1];
                Q.pop();
                
                if(graph.find(cur) == graph.end()) continue;
                
                for(const auto& pair : graph[cur]){
                    int next = pair.first;
                    int cost = pair.second;
                    if(cost + all_cost < dp[next]){
                        Q.push({next, cost + all_cost});
                        dp[next] = cost + all_cost;                       
                    }
                }
            }
            K--;
        }
        return dp[dst] == 1 << 16 ? -1 : dp[dst];
    }
    
    void buildGraph(unordered_map<int, unordered_map<int, int>>& graph, const vector<vector<int>>& flights){
        int n = flights.size();
        for(int i = 0; i < n; ++i){
            int start = flights[i][0];
            int des = flights[i][1];
            int cost = flights[i][2];       
            graph[start][des] = cost;
        }
    }
};
```
