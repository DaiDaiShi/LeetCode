# Dynamic Programming (DP)
## 1. Definition
Dynamic programming is both a mathematical optimization method and a computer programming method. In both contexts it refers to simplifying a complicated problem by breaking it down into simpler sub-problems in a recursive manner. [(wiki)](https://en.wikipedia.org/wiki/Dynamic_programming)
## 2. When are complicated problem and sub-problems?
First Let's look at **Fibonacci sequence**.

A series of numbers in which the first 2 numbers are 0 and 1, and from the third number on, each number is the sum of the two preceding numbers.

The following is an example of fabinocci sequence

0, 1, 1, 2, 3, 5, 8, 13, 21, 34

From its definition, we can easily come up with its recurrence relation, which is 
```cc
f[0] = 0
f[1] = 1
f[i] = f[i - 1] + f[i - 2], i >= 2
```
Based on the above equation, intuitively, we can have a straightforward implementation:
```cc
int fib (int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}
```
The problem here is this implementation may calculate the same problems multiple times. Let's say if we are trying to calculate fib(4),
the above algorithm works like:
```sql
                   fib(4)
               /            \
             fib(3)      fib(2)
           /     \         /    \
        fib(2)  fib(1)   fib(1) fib(0)
       /     \  
     fib(1) fib(0)
```
You may be noticing here, fib(2) is calculated 2 times for calculating fib(4). And the total number of calculation is the number of (/) and (\\), which is 8 (2^3). 

Each node has a left and right child, we have 4 layers. We need O(2^N) time complexity to calculate fib(N) using the approach.

However we only need O(N) time complexity to get the result using DP approach. Let's think about this problem in a reverse way. We are given fib(0) and fib(1), so we can get fib(3) by adding them up. And we can deduce the rest from this just by adding the preceding two numbers.

The code is as followings

```cc
int fib (int n) {
    if (n <= 1) return n;
    vector<int> fib_seq(n + 1);
    fib_seq[0] = 0;
    fib_seq[1] = 1;
    for(int i = 2; i <= n; ++i) {
        fib_seq[i] = fib_seq[i - 1] + fib_seq[i - 2];
    }
    return fib_seq[n];
}
```
## 3. How do we know if a problem is a DP problem?

## 2. How do we tackle DP problems?

