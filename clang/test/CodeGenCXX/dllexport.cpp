// RUN: %clang_cc1 -triple i686-pc-win32 -x c++ -emit-llvm < %s | FileCheck %s

#define DLLEXPORT __declspec(dllexport)

void DLLEXPORT a();
// CHECK-DAG: declare dllexport void @"\01?a@@YAXXZ"()

inline void DLLEXPORT b() {}
// CHECK-DAG: define weak_odr dllexport void @"\01?b@@YAXXZ"()

template <typename T> void c() {}
template void DLLEXPORT c<int>();
// CHECK-DAG: define weak_odr dllexport void @"\01??$c@H@@YAXXZ"()

struct S {
  void DLLEXPORT a() {}
  // CHECK-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?a@S@@QAEXXZ"

  struct T {
    void DLLEXPORT a() {}
    // CHECK-DAG: define weak_odr dllexport x86_thiscallcc void @"\01?a@T@S@@QAEXXZ"
  };
};


void user() {
  a();
}
