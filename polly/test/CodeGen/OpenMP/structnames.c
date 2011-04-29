#include <string.h>
#include <stdio.h>
#define N 5

float A[N];
float B[N];

void loop1_openmp() {
    for (int i = 0; i <= N; i++)
      A[i] = 0;

    for (int j = 0; j <= N; j++)
      for (int k = 0; k <= N; k++)
        B[k] += j;
}

int main () {
  int i;
  memset(A, 0, sizeof(float) * N);
  memset(B, 0, sizeof(float) * N);

  loop1_openmp();

  return 0;
}

