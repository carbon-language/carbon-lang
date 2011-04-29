#define M 1024
#define N 1024
#define K 1024
float A[K][M];
float B[N][K];
float C[M][N];
/*
void matmul_vec(void) {
  int i, j, k;


  /* With much unrolling
  for (i=0;i<=M;i++)
    for (j=0;j<=N;j+=4)
      for (k=0;k<=K;k+=8)
        for (kk=k;kk<=k+7;kk++)
          for (jj=j;jj<=j+3;jj++)
            C[i][jj] += A[kk][i] * B[jj][kk];
            vec_load    splat      scalar_load
   */
  /* Without unrolling
  for (i=0;i<=M;i++)
    for (j=0;j<=N;j+=4)
      for (k=0;k<=K;k++)
          for (jj=j;jj<=j+3;jj++)
            C[i][jj] += A[k][i] * B[jj][kk];
            vec_load    splat      scalar_load
   /

}
i*/
int main()
{
  int i, j, k;
  //matmul_vec();
  for(i=0; i<M/4; i++)
    for(k=0; k<K; k++) {
      for(j=0; j<N; j++)
        C[i+0][j] += A[k][i+0] * B[j][k];
        C[i+1][j] += A[k][i+1] * B[j][k];
        C[i+2][j] += A[k][i+2] * B[j][k];
        C[i+3][j] += A[k][i+3] * B[j][k];
      }

  return A[42][42];
}
