#define N 10

void foo() {
  float A[N];
  int i = 0;

  for (i=0; i < N; i++)
    A[i] = 10;

  return;
}


int main()
{
	foo();
}
