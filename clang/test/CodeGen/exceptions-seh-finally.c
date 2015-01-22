// RUN: %clang_cc1 %s -triple x86_64-pc-win32 -fexceptions -fms-extensions -emit-llvm -o - | FileCheck %s

void might_crash(void);
void cleanup(void);
int check_condition(void);
void basic_finally(void) {
  __try {
    might_crash();
  } __finally {
    cleanup();
  }
}

// CHECK-LABEL: define void @basic_finally()
// CHECK: invoke void @might_crash()
// CHECK: call void @cleanup()
//
// CHECK: landingpad
// CHECK-NEXT: cleanup
// CHECK: invoke void @cleanup()
//
// CHECK: landingpad
// CHECK-NEXT: catch i8* null
// CHECK: call void @abort()

// FIXME: This crashes.
#if 0
void basic_finally(void) {
  __try {
    might_crash();
  } __finally {
l:
    cleanup();
    if (check_condition())
      goto l;
  }
}
#endif
