#include <stdio.h>

int ok(int k, int m) {
    int size = 2 * k;   // 当前圈人数
    int idx = 0;        // 下一次从谁开始数（0-based）
    for (int killed = 0; killed < k; ++killed) { // 必须先杀掉 k 个坏人
        idx = (idx + m - 1) % size;             // 本轮被杀的位置
        if (idx < k) return 0;                  // 杀到了好人，失败
        size--;                                  // 删除该坏人
        // 删除后，下一轮从 idx 开始（因为circle收缩，原 idx 的下一个顶到 idx）
    }
    return 1; // 成功：前 k 个出局的都是坏人
}

int main(void) {
    int k;
    if (scanf("%d", &k) != 1) return 0;

    int m = k + 1;                 // 最小从 k+1 开始试
    while (!ok(k, m)) ++m;         // 逐个尝试直到合法
    printf("%d\n", m);
    return 0;
}
