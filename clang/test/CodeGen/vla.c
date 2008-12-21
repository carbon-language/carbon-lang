// RUN: clang %s -emit-llvm -o %t

int b(char* x);

// Extremely basic VLA test
void a(int x) {
  char arry[x];
  arry[0] = 10;
  b(arry);
}

void b(int n)
{
  sizeof(int[n]);
}
