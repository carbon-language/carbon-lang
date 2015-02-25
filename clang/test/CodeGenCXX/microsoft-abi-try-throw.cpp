// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -DTRY   | FileCheck %s -check-prefix=TRY
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -DTHROW | FileCheck %s -check-prefix=THROW

void external();

inline void not_emitted() {
  throw int(13); // no error
}

int main() {
  int rv = 0;
#ifdef TRY
  try {
    external(); // TRY: invoke void @"\01?external@@YAXXZ"
  } catch (int) {
    rv = 1;
    // TRY: call i8* @llvm.eh.begincatch
    // TRY: call void @llvm.eh.endcatch
  }
#endif
#ifdef THROW
  // THROW: call void @"\01?terminate@@YAXXZ"
  throw int(42);
#endif
  return rv;
}
