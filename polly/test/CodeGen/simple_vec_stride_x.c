#define N 1024
float A[N];
float B[N];

void simple_vec_stride_x(void) {
  int i;

  for (i = 0; i < 4; i++)
    B[2 * i] = A[2 * i];
}
int main()
{
  simple_vec_stride_x();
  return A[42];
}
