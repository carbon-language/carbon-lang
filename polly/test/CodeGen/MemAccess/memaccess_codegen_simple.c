int A[100];

int memaccess_codegen_simple () {
  for (int i = 0; i < 12; i++)
    A[13] = A[i] + A[i-1];

  return 0;
}
