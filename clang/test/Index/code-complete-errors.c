_Complex cd; // CHECK: code-complete-errors.c:1:1: warning: plain '_Complex' requires a type specifier; assuming '_Complex double'

struct s {
  int x, y;;
};

struct s s0 = { y: 5 }; // CHECK: code-complete-errors.c:7:20: warning: use of GNU old-style field designator extension

int f(int *ptr1, float *ptr2) {
  return ptr1 != ptr2; // CHECK: code-complete-errors.c:10:15: warning: comparison of distinct pointer types ('int *' and 'float *')
}

void g() {  }

// RUN: c-index-test -code-completion-at=%s:13:12 %s 2> %t
// RUN: FileCheck -check-prefix=CHECK %s < %t
