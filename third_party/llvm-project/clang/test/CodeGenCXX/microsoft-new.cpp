// RUN: %clang_cc1 -no-opaque-pointers -triple i686-pc-win32 -fms-compatibility %s -emit-llvm -o - | FileCheck %s

#include <stddef.h>

struct arbitrary_t {} arbitrary;
void *operator new(size_t size, arbitrary_t);

struct arbitrary2_t {} arbitrary2;
void *operator new[](size_t size, arbitrary2_t);

namespace PR13164 {
  void f() {
	// MSVC will fall back on the non-array operator new.
    void *a;
    int *p = new(arbitrary) int[4];
    // CHECK: call noundef i8* @"??2@YAPAXIUarbitrary_t@@@Z"(i32 noundef 16, %struct.arbitrary_t*
  }

  struct S {
    void *operator new[](size_t size, arbitrary_t);
  };

  void g() {
    S *s = new(arbitrary) S[2];
    // CHECK: call noundef i8* @"??_US@PR13164@@SAPAXIUarbitrary_t@@@Z"(i32 noundef 2, %struct.arbitrary_t*
    S *s1 = new(arbitrary) S;
    // CHECK: call noundef i8* @"??2@YAPAXIUarbitrary_t@@@Z"(i32 noundef 1, %struct.arbitrary_t*
  }

  struct T {
    void *operator new(size_t size, arbitrary2_t);
  };

  void h() {
    // This should still call the global operator new[].
    T *t = new(arbitrary2) T[2];
    // CHECK: call noundef i8* @"??_U@YAPAXIUarbitrary2_t@@@Z"(i32 noundef 2, %struct.arbitrary2_t*
  }
}
