int A[128][128];

int gpu_no_pure() {
  int i,j,k;

  for(i = 0; i < 128; i++)
    for(j = 0; j < 128; j++)
      for(k = 0; k < 256; k++)
        A[i][j] += i*123/(k+1)+5-j*k-123;

  return 0;
}

int main() {
  int b = gpu_no_pure();
  return 0;
}
