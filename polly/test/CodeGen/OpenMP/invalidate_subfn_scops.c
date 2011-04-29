#define N 500000
float A[N];
int main() {
  int j, k;

  for(k = 0; k < N; k++)
    for (j = 0; j <= N; j++)
      A[j] = k;

  return 0;
}
