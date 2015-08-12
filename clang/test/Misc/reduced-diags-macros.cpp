// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s -strict-whitespace

#define NO_INITIATION(x) int a = x * 2
#define NO_DEFINITION(x) int c = x * 2

NO_INITIATION(a);
NO_DEFINITION(b);

// CHECK: {{.*}}:6:15: warning: variable 'a' is uninitialized when used within its own initialization
// CHECK-NEXT: NO_INITIATION(a);
// CHECK-NEXT: ~~~~~~~~~~~~~~^~
// CHECK-NEXT: {{.*}}:3:34: note: expanded from macro 'NO_INITIATION'
// CHECK-NEXT: #define NO_INITIATION(x) int a = x * 2
// CHECK-NEXT:                                  ^

// CHECK: {{.*}}:7:15: error: use of undeclared identifier 'b'
// CHECK-NEXT: NO_DEFINITION(b);
// CHECK-NEXT:               ^


#define F(x) x + 1
#define ADD(x,y) y + F(x)
#define SWAP_ARGU(x,y) ADD(y,x)

int  p = SWAP_ARGU(3, x);

// CHECK: {{.*}}:25:23: error: use of undeclared identifier 'x'
// CHECK-NEXT: int  p = SWAP_ARGU(3, x);
// CHECK-NEXT:                       ^
