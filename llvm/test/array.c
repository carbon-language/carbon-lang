extern printf(const char *, double, double);

int
checkIdxCode(int N, int* A, float F[][30])
{
  int i, j;
  float sumA=0.0, sumF=0.0;
  for (i=0; i < 12; i++)
    {
      sumA = sumA + A[i];
      for (j=0; j < 10; j++)
	{
	  F[i][j] = 0.5 * (F[i][j-1] + F[i-1][j]);
	  sumF = sumF + F[i][j];
	}
    }
  printf("sumA = %lf, sumF = %lf\n", sumA, sumF);
}

#if 0
int
main(int argc, char** argv)
{
  int  N = argc+20;
  int* A = (int*) malloc(N * sizeof(int));
  float F[25][30];
  return checkIdxCode(N, A, F);
}

#endif
