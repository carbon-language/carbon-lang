#include <string.h>
#define N 10240000

float A[N];
float B[N];

void loop1_openmp() {
	for (int i = 0; i <= N; i++)
		A[i] = 0;
	for (int j = 0; j <= N; j++)
		B[j] = 0;
}


int main () {
  int i;
  memset(A, 0, sizeof(float) * N);
  memset(B, 1, sizeof(float) * N);

  loop1_openmp();
    
  return 0;
}

