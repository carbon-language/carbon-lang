int bar1();
int bar2();
int bar3();
int k;
#define N 100
int A[N];

int main() {
  int i, j, z;

  __sync_synchronize();
  for (i = 0; i < N; i++) {
    if (i < 50)
      A[i] = 8;
    if (i < 4)
      A[i] = 9;
    if (i < 3)
      A[i] = 10;
  }
  __sync_synchronize();

  return A[z];
}
