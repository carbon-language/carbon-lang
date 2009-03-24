// RUN: clang-cc -emit-llvm %s -o %t &&
// RUN: grep "store i32 0, i32\* %cleanup" %t | count 2
void f(int n) {
  int a[n];
  
  {
    int b[n];
    
    if (n)
      return;
  }
  
  if (n)
    return;
}
