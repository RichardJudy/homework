#include <stdio.h>

unsigned int fib(unsigned int a) {
    if (a == 1 || a == 2) return 1;
    unsigned int p1 = 1, p2 = 1, p3 = 0;
    for (unsigned int i = 3; i <= a; ++i) {
        p3 = p1 + p2;
        p1 = p2;
        p2 = p3;
    }
    return p3;
}

int main() {
    int n;  
    scanf("%d", &n);
    for (int i = 1; i <= n; ++i) {
        unsigned int a;
        scanf("%u", &a);
        printf("%u\n", fib(a));
    }
    return 0;
}