#define N 1024
float A[N];
float B[N];

void simple_vec_const(void) {
  int i;

  for (i = 0; i < 4; i++)
    B[i] = A[i] + 1;
}
int main()
{
  simple_vec_const();
  return A[42];
}
