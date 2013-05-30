// RUN: %clang_cc1 -triple i686-pc-win32 -fms-compatibility %s -emit-llvm -o - | FileCheck %s

struct arbitrary_t {} arbitrary;
void *operator new(unsigned int size, arbitrary_t);

struct arbitrary2_t {} arbitrary2;
void *operator new[](unsigned int size, arbitrary2_t);

namespace PR13164 {
  void f() {
	// MSVC will fall back on the non-array operator new.
    void *a;
    int *p = new(arbitrary) int[4];
    // CHECK: call i8* @_Znwj11arbitrary_t(i32 16, %struct.arbitrary_t*
  }

  struct S {
    void *operator new[](unsigned int size, arbitrary_t);
  };

  void g() {
    S *s = new(arbitrary) S[2];
    // CHECK: call i8* @_ZN7PR131641SnaEj11arbitrary_t(i32 2, %struct.arbitrary_t*
    S *s1 = new(arbitrary) S;
    // CHECK: call i8* @_Znwj11arbitrary_t(i32 1, %struct.arbitrary_t*
  }

  struct T {
    void *operator new(unsigned int size, arbitrary2_t);
  };

  void h() {
    // This should still call the global operator new[].
    T *t = new(arbitrary2) T[2];
    // CHECK: call i8* @_Znaj12arbitrary2_t(i32 2, %struct.arbitrary2_t*
  }
}
