// RUN: %llvmgxx -S -emit-llvm %s -o - | FileCheck %s
// rdar://7309675
// PR4678

// test1 should be compmiled to be a varargs function in the IR even 
// though there is no way to do a va_begin.  Otherwise, the optimizer
// will warn about 'dropped arguments' at the call site.

// CHECK: define i32 @_Z5test1z(...)
int test1(...) {
  return -1;
}

// CHECK: call i32 (...)* @_Z5test1z(i32 0)
void test() {
  test1(0);
}


