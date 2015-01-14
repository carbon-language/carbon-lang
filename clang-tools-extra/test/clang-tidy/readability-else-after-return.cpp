// RUN: $(dirname %s)/check_clang_tidy.sh %s readability-else-after-return %t
// REQUIRES: shell

void f(int a) {
  if (a > 0)
    return;
  else // comment
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: don't use else after return
// CHECK-FIXES: {{^}}  // comment
    return;

  if (a > 0) {
    return;
  } else { // comment
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: don't use else after return
// CHECK-FIXES:  } // comment
    return;
  }

  if (a > 0) {
    f(0);
    if (a > 10)
      return;
  } else {
    return;
  }
}

