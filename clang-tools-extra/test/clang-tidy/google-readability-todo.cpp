// RUN: $(dirname %s)/check_clang_tidy_fix.sh %s google-readability-todo %t -config="{User: 'some user'}" --
// REQUIRES: shell

//   TODOfix this1
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this1

//   TODO fix this2
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this2

// TODO fix this3
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this3

// TODO: fix this4
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this4

//   TODO(clang)fix this5

// TODO(foo):shave yaks
// TODO(bar):
// TODO(foo): paint bikeshed
// TODO(b/12345): find the holy grail
