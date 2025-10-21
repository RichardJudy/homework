#include <stdio.h>

void convert(unsigned long long x, int m) {
    if (x >= (unsigned long long)m)    // 先递归处理高位
        convert(x / m, m);
    int d = x % m;                     // 输出当前最低位
    putchar("0123456789ABCDEF"[d]);
}

int main(void) {
    unsigned long long X;
    int M;

    if (scanf("%llu %d", &X, &M) != 2) return 0;

    if (X == 0) {          // 虽然题目 X>=1，这里顺手兼容 0
        puts("0");
        return 0;
    }

    convert(X, M);
    putchar('\n');
    return 0;
}
