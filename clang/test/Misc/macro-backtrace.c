// Tests for macro expansion backtraces. The RUN and CHECK lines are grouped
// below the test code to reduce noise when updating them.

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
  if (M12(ip, fp)) { }
  // RUN: %clang_cc1 -fsyntax-only -fmacro-backtrace-limit 5 %s 2>&1 | FileCheck %s
  // CHECK: macro-backtrace.c:18:7: warning: comparison of distinct pointer types ('int *' and 'float *')
  // CHECK: if (M12(ip, fp)) { }
  // CHECK: macro-backtrace.c:15:19: note: expanded from:
  // CHECK: #define M12(A, B) M11(A, B)
  // CHECK: macro-backtrace.c:14:19: note: expanded from:
  // CHECK: #define M11(A, B) M10(A, B)
  // CHECK: note: (skipping 7 expansions in backtrace; use -fmacro-backtrace-limit=0 to see all)
  // CHECK: macro-backtrace.c:6:18: note: expanded from:
  // CHECK: #define M3(A, B) M2(A, B)
  // CHECK: macro-backtrace.c:5:18: note: expanded from:
  // CHECK: #define M2(A, B) M1(A, B)
  // CHECK: macro-backtrace.c:4:23: note: expanded from:
  // CHECK: #define M1(A, B) ((A) < (B))
}
