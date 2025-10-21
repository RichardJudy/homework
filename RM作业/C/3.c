#include <stdio.h>
#include <string.h>
#include <ctype.h>

static const char D[] = "0123456789ABCDEF";

int is_pal(const char *s) {                 // 判断回文
    for (int i = 0, j = strlen(s)-1; i < j; i++, j--)
        if (s[i] != s[j]) return 0;
    return 1;
}

int val(char c) {                           // 字符 -> 数值
    c = toupper((unsigned char)c);
    return isdigit(c) ? c-'0' : c-'A'+10;
}

void revcpy(const char *s, char *r) {       // 反转复制
    int n = strlen(s);
    for (int i = 0; i < n; i++) r[i] = s[n-1-i];
    r[n] = '\0';
}

void add_baseN(const char *a, const char *b, int base, char *out) { // out=a+b
    int i = strlen(a)-1, j = strlen(b)-1, k = 0, carry = 0;
    char t[220];
    while (i >= 0 || j >= 0 || carry) {
        int x = (i >= 0) ? val(a[i--]) : 0;
        int y = (j >= 0) ? val(b[j--]) : 0;
        int s = x + y + carry;
        t[k++] = D[s % base];
        carry = s / base;
    }
    // 反转得到正序
    for (int u = 0; u < k; u++) out[u] = t[k-1-u];
    out[k] = '\0';
}

int main(void) {
    int N; char M[110], R[110], S[220];
    if (scanf("%d %s", &N, M) != 2) return 0;

    for (int step = 0; step <= 30; step++) {
        if (is_pal(M)) { printf("STEP=%d\n", step); return 0; }
        if (step == 30) break;
        revcpy(M, R);
        add_baseN(M, R, N, S);
        strcpy(M, S);
    }
    puts("Impossible!");
    return 0;
}

