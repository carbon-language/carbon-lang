_Complex cd; // CHECK: code-complete-errors.c:1:1: warning: plain '_Complex' requires a type specifier; assuming '_Complex double'
// CHECK: FIX-IT: Insert " double" at 1:9
struct s {
  int x, y;; // CHECK: code-complete-errors.c:4:12: warning: extra ';' inside a struct
}; // CHECK: FIX-IT: Remove [4:12 - 4:13]

struct s s0 = { y: 5 }; // CHECK: code-complete-errors.c:7:20: warning: use of GNU old-style field designator extension
// CHECK: FIX-IT: Replace [7:17 - 7:19] with ".y = "
int f(int *ptr1, float *ptr2) {
  return ptr1 != ptr2; // CHECK: code-complete-errors.c:10:15:{10:10-10:14}{10:18-10:22}: warning: comparison of distinct pointer types ('int *' and 'float *')
}

#define expand_to_binary_function(ret, name, parm1, parm2, code) ret name(parm1, parm2) code

expand_to_binary_function(int, g, int *ip, float *fp, {
// CHECK: code-complete-errors.c:17:15:{17:12-17:14}{17:18-17:20}: warning: comparison of distinct pointer types ('int *' and 'float *')
    return ip == fp;
  })

void g() {  }

// RUN: c-index-test -code-completion-at=%s:19:12 -pedantic %s 2> %t
// RUN: FileCheck -check-prefix=CHECK %s < %t
