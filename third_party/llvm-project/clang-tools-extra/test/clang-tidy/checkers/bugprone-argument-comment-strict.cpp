// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: [{key: StrictMode, value: true}]}" --

void f(int _with_underscores_);
void g(int x_);
void ignores_underscores() {
  f(/*With_Underscores=*/0);
// CHECK-NOTES: [[@LINE-1]]:5: warning: argument name 'With_Underscores' in comment does not match parameter name '_with_underscores_'
// CHECK-NOTES: [[@LINE-5]]:12: note: '_with_underscores_' declared here
// CHECK-FIXES: f(/*_with_underscores_=*/0);

  f(/*with_underscores=*/1);
// CHECK-NOTES: [[@LINE-1]]:5: warning: argument name 'with_underscores' in comment does not match parameter name '_with_underscores_'
// CHECK-NOTES: [[@LINE-10]]:12: note: '_with_underscores_' declared here
// CHECK-FIXES: f(/*_with_underscores_=*/1);
  f(/*_With_Underscores_=*/2);
// CHECK-NOTES: [[@LINE-1]]:5: warning: argument name '_With_Underscores_' in comment does not match parameter name '_with_underscores_'
// CHECK-NOTES: [[@LINE-14]]:12: note: '_with_underscores_' declared here
// CHECK-FIXES: f(/*_with_underscores_=*/2);
  g(/*X=*/3);
// CHECK-NOTES: [[@LINE-1]]:5: warning: argument name 'X' in comment does not match parameter name 'x_'
// CHECK-NOTES: [[@LINE-17]]:12: note: 'x_' declared here
// CHECK-FIXES: g(/*x_=*/3);
}
