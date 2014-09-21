// RUN: $(dirname %s)/check_clang_tidy_fix.sh %s google-readability-casting %t -x c
// REQUIRES: shell

void f(const char *cpc) {
  const char *cpc2 = (const char*)cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: Redundant cast to the same type. [google-readability-casting]
  // CHECK-FIXES: const char *cpc2 = cpc;
  char *pc = (char*)cpc;
}
