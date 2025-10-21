#include <stdio.h>
#include <math.h>

// 判断是否为质数
int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i <= sqrt(n); i++)
        if (n % i == 0) return 0;
    return 1;
}

int main() {
    int N;
    scanf("%d", &N);

    for (int x = 4; x <= N; x += 2) {
        for (int p = 2; p <= x / 2; p++) {
            if (is_prime(p) && is_prime(x - p)) {
                printf("%d=%d+%d\n", x, p, x - p);
                break;
            }
        }
    }
    return 0;
}
