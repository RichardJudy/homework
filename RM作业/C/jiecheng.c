#include <stdio.h>

// 定义递归函数 factorial
int factorial(int n) {
    if (n == 1)  // 递归终止条件
        return 1;
    else
        return n * factorial(n - 1);  // 递归公式 n! = n * (n-1)!
}

int main() {
    int n;
    scanf("%d", &n);
    printf("%d\n", factorial(n));
    return 0;
}
