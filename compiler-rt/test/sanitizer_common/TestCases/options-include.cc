// RUN: %clangxx -O0 %s -o %t
// RUN: echo -e "symbolize=1\ninclude='%t.options2.txt'" >%t.options1.txt
// RUN: echo -e "help=1\n" >%t.options2.txt
// RUN: cat %t.options1.txt
// RUN: cat %t.options2.txt
// RUN: %tool_options="help=0:include='%t.options1.txt'" %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-VERBOSITY1 <%t.out
// RUN: %tool_options="include='%t.options1.txt',help=0" %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-VERBOSITY0 <%t.out
// RUN: %tool_options="include='%t.options-not-found.txt',help=1" not %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-NOT-FOUND < %t.out

#include <stdio.h>

int main() {
  fprintf(stderr, "done\n");
}

// CHECK-VERBOSITY1: Available flags for
// CHECK-VERBOSITY0-NOT: Available flags for
// CHECK-NOT-FOUND: Failed to read options from
