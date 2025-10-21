#include <stdio.h>
#include <string.h>

#define BASE 10000      // 每个“位”存 4 位十进制
#define WIDTH 4
#define MAXD 3000       // 足够存下 f(5000)（约 1050 位十进制 ≈ 263 个块）

// out = a + b，返回结果长度
int add(const int *a, int la, const int *b, int lb, int *out) {
    int n = (la > lb ? la : lb), carry = 0;
    for (int i = 0; i < n; ++i) {
        int x = (i < la ? a[i] : 0) + (i < lb ? b[i] : 0) + carry;
        out[i] = x % BASE;
        carry = x / BASE;
    }
    if (carry) out[n++] = carry;
    return n;
}

// 打印高精度数（小端存储）
void print_big(const int *a, int la) {
    if (la == 0) { puts("0"); return; }
    printf("%d", a[la-1]);                 // 最高块不补零
    for (int i = la-2; i >= 0; --i)
        printf("%0*d", WIDTH, a[i]);       // 其他块补零到 4 位
    putchar('\n');
}

int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 0;

    // f(1)=1, f(2)=2
    if (n == 1) { puts("1"); return 0; }
    if (n == 2) { puts("2"); return 0; }

    int a[MAXD] = {0}, b[MAXD] = {0}, c[MAXD] = {0};
    int la = 1, lb = 1, lc = 0;
    a[0] = 1;   // f(1)
    b[0] = 2;   // f(2)

    for (int i = 3; i <= n; ++i) {
        lc = add(a, la, b, lb, c);     // c = a + b
        memcpy(a, b, lb * sizeof(int)); la = lb;     // a = b
        memcpy(b, c, lc * sizeof(int)); lb = lc;     // b = c
    }

    print_big(b, lb);                   // f(n)
    return 0;
}
