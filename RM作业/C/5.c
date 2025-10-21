#include <stdio.h>

int main(void) {
    int ch, bal = 0;               // bal: 当前未配对的左括号数
    while ((ch = getchar()) != EOF && ch != '@') {
        if (ch == '(') bal++;
        else if (ch == ')') {
            if (bal == 0) {        // 提前出现多余的右括号
                puts("NO");
                return 0;
            }
            bal--;
        }
    }
    puts(bal == 0 ? "YES" : "NO"); // 全部配平才算 YES
    return 0;
}
