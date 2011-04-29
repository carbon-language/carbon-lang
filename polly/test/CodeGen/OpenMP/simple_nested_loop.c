#include <string.h>
#define N 10

double A[N];
double B[N];

void loop_openmp() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[j] += j;
    }
  }
}

int main () {
  memset(A, 0, sizeof(float) * N);

  loop_openmp();

  return 0;
}

