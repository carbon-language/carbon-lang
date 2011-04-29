#define N 20

int main () {
  int i;
  int A[N];

  A[0] = 0;

  __sync_synchronize();

  for (i = 0; i < 0; i++)
    A[i] = 1;

  __sync_synchronize();

  if (A[0] == 0)
    return 0;
  else
    return 1;
}
