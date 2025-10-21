#include <stdio.h>
int main() {
    double x;
    if (scanf("%lf", &x) != 1) return 0;
    printf("%.12f", x);
    return 0;
}
