#include <stdio.h>
#include <math.h>

int isPrime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main() {
    int L;
    scanf("%d", &L);
    int sum = 0, count = 0, n = 2;
    while (1) {
        if (isPrime(n)) {
            if (sum + n > L) break;
            sum += n;
            printf("%d\n", n);
            count++;
        }
        n++;
    }
    printf("%d\n", count);
    return 0;
}