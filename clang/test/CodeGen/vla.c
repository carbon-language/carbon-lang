// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

int b(char* x);

// Extremely basic VLA test
void a(int x) {
  char arry[x];
  arry[0] = 10;
  b(arry);
}

int c(int n)
{
  return sizeof(int[n]);
}

int f0(int x) {
  int vla[x];
  return vla[x-1];
}

void
f(int count)
{
 int a[count];

  do {  } while (0);

  if (a[0] != 3) {
  }
}

void g(int count) {
  // Make sure we emit sizes correctly in some obscure cases
  int (*a[5])[count];
  int (*b)[][count];
}

// rdar://8403108
// CHECK: define void @f_8403108
void f_8403108(unsigned x) {
  // CHECK: call i8* @llvm.stacksave()
  char s1[x];
  while (1) {
    // CHECK: call i8* @llvm.stacksave()
    char s2[x];
    if (1)
      break;
  // CHECK: call void @llvm.stackrestore(i8*
  }
  // CHECK: call void @llvm.stackrestore(i8*
}
