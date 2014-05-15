// RUN: %clang_cc1 -triple i686-pc-win32 -x c++ -O2 -disable-llvm-optzns -emit-llvm < %s | FileCheck %s

#define DLLIMPORT __declspec(dllimport)

void DLLIMPORT a();
// CHECK-DAG: declare dllimport void @"\01?a@@YAXXZ"()

inline void DLLIMPORT b() {}
// CHECK-DAG: define available_externally dllimport void @"\01?b@@YAXXZ"()

template <typename T> inline void c() {} // FIXME: MSVC accepts this without 'inline' too.
template void DLLIMPORT c<int>();
// CHECK-DAG: define available_externally dllimport void @"\01??$c@H@@YAXXZ"()

struct S {
  void DLLIMPORT a() {}
  // CHECK-DAG: define available_externally dllimport x86_thiscallcc void @"\01?a@S@@QAEXXZ"
};

void user(S* s) {
  a();
  b();
  c<int>();
  s->a();
}
