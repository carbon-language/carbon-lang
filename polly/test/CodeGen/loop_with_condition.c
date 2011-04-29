#include <string.h>
#define N 1024
int A[N];
int B[N];

void loop_with_condition() {
  int i;

  __sync_synchronize();
  for (i = 0; i < N; i++) {
    if (i <= N / 2)
      A[i] = 1;
    else
      A[i] = 2;
    B[i] = 3;
  }
  __sync_synchronize();
}

int main () {
  int i;

  memset(A, 0, sizeof(int) * N);
  memset(B, 0, sizeof(int) * N);

  loop_with_condition();

  for (i = 0; i < N; i++)
    if (B[i] != 3)
      return 1;

  for (i = 0; i < N; i++)
    if (i <= N / 2 && A[i] != 1)
      return 1;
    else if (i > N / 2 && A[i] != 2)
      return 1;
  return 0;
}

