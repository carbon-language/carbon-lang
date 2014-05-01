// RUN: %clang_cc1 %s -triple i686-pc-win32 -std=c++11 -fms-compatibility -emit-llvm -o - | FileCheck %s

template<class T>
void destroy(T *p) {
  p->~T();
}

extern "C" void f() {
  int a;
  destroy((void*)&a);
}

// CHECK-LABEL: define void @f()
// CHECK: call void @"\01??$destroy@X@@YAXPAX@Z"
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr void @"\01??$destroy@X@@YAXPAX@Z"(i8* %p)
//    The pseudo-dtor expr should not generate calls to anything.
// CHECK-NOT: call
// CHECK-NOT: invoke
// CHECK: ret void
