

void combinations(unsigned int n, unsigned *A) {
  unsigned int i, t = 1;
  A[0] = A[n] = 1;

  for (i = 1; i <= n/2; i++) {
    t = (t * (n+1-i)) / i;
    A[i] = A[n-i] = t;
  }
}
