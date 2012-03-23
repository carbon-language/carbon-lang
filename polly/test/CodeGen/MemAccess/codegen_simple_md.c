int A[1040];

int codegen_simple_md() {
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j)
      A[32*i+j] = 100;

  return 0;
}
