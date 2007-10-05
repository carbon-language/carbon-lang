// RUN: clang %s -emit-llvm | llvm-as | opt -std-compile-opts -disable-output

int foo(int i) {
  int j = 0;
  switch (i) {
  case 1 : 
    j = 2; break;
  case 2:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

    
int foo2(int i) {
  int j = 0;
  switch (i) {
  case 1 : 
    j = 2; break;
  case 2 ... 10:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

    
