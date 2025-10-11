#include <stdio.h>
#include <stdlib.h>

int cmp(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}

int main() {
    int N;
    scanf("%d", &N);
    int *arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        scanf("%d", &arr[i]);
    }
    qsort(arr, N, sizeof(int), cmp);
    for (int i = 0; i < N; i++) {
        if (i > 0) printf(" ");
        printf("%d", arr[i]);
    }
    free(arr);
    return 0;
}