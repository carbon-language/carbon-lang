// RUN: %check_clang_tidy %s "llvm-namespace-comment,clang-diagnostic-*" %t -- --fix-notes
void foo(int a) {
  if (a = 1) {
    // CHECK-NOTES: [[@LINE-1]]:9: warning: using the result of an assignment as a condition without parentheses [clang-diagnostic-parentheses]
    // CHECK-NOTES: [[@LINE-2]]:9: note: place parentheses around the assignment to silence this warning
    // CHECK-NOTES: [[@LINE-3]]:9: note: use '==' to turn this assignment into an equality comparison
    // As we have 2 conflicting fixes in notes, don't apply any fix.
    // CHECK-FIXES: if (a = 1) {
  }
}
