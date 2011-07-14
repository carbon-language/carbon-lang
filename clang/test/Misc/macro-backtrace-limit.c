// RUN: %clang_cc1 -fsyntax-only -fmacro-backtrace-limit 5 %s > %t 2>&1 
// RUN: FileCheck %s < %t

#define M1(A, B) ((A) < (B))
#define M2(A, B) M1(A, B)
#define M3(A, B) M2(A, B)
#define M4(A, B) M3(A, B)
#define M5(A, B) M4(A, B)
#define M6(A, B) M5(A, B)
#define M7(A, B) M6(A, B)
#define M8(A, B) M7(A, B)
#define M9(A, B) M8(A, B)
#define M10(A, B) M9(A, B)
#define M11(A, B) M10(A, B)
#define M12(A, B) M11(A, B)

void f(int *ip, float *fp) {
  // CHECK: macro-backtrace-limit.c:31:7: warning: comparison of distinct pointer types ('int *' and 'float *')
  // CHECK: if (M12(ip, fp)) { }
  // CHECK: macro-backtrace-limit.c:15:19: note: expanded from:
  // CHECK: #define M12(A, B) M11(A, B)
  // CHECK: macro-backtrace-limit.c:14:19: note: expanded from:
  // CHECK: #define M11(A, B) M10(A, B)
  // CHECK: note: (skipping 7 expansions in backtrace; use -fmacro-backtrace-limit=0 to see all)
  // CHECK: macro-backtrace-limit.c:6:18: note: expanded from:
  // CHECK: #define M3(A, B) M2(A, B)
  // CHECK: macro-backtrace-limit.c:5:18: note: expanded from:
  // CHECK: #define M2(A, B) M1(A, B)
  // CHECK: macro-backtrace-limit.c:4:23: note: expanded from:
  // CHECK: #define M1(A, B) ((A) < (B))
  if (M12(ip, fp)) { }
}
