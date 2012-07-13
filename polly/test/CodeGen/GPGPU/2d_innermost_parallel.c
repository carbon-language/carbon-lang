int A[128][128];

int gpu_pure() {
  int i,j;

  for(i = 0; i < 128; i++)
    for(j = 0; j < 128; j++)
      A[i][j] = i*128 + j;

  return 0;
}

int main() {
  int b = gpu_pure();
  return 0;
}
