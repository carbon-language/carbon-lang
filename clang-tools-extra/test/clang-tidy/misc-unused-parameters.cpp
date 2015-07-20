// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-unused-parameters %t
// REQUIRES: shell

// Basic removal
// =============
void a(int i) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void a(int  /*i*/) {}{{$}}

void b(int i = 1) {}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void b(int  /*i*/) {}{{$}}

void c(int *i) {}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void c(int * /*i*/) {}{{$}}

// Unchanged cases
// ===============
void g(int i);             // Don't remove stuff in declarations
void h(int i) { (void)i; } // Don't remove used parameters
