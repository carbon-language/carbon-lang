#define N 1024
float A[N];
float B[N];

void simple_vec_stride_one(void) {
  int i;

  for (i = 0; i < 4; i++)
    B[i] = A[i];
}
int main()
{
  simple_vec_stride_one();
  return A[42];
}
