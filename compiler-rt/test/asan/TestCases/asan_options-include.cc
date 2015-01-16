// RUN: %clangxx_asan -O0 %s -o %t
// RUN: echo -e "symbolize=1\ninclude='%t.options2.txt'" >%t.options1.txt
// RUN: echo -e "verbosity=1\n" >%t.options2.txt
// RUN: cat %t.options1.txt
// RUN: cat %t.options2.txt
// RUN: ASAN_OPTIONS="verbosity=0:include='%t.options1.txt'" %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-VERBOSITY1 <%t.out
// RUN: ASAN_OPTIONS="include='%t.options1.txt',verbosity=0" %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-VERBOSITY0 <%t.out
// RUN: ASAN_OPTIONS="include='%t.options-not-found.txt',verbosity=0" not %run %t 2>&1 | tee %t.out
// RUN: FileCheck %s --check-prefix=CHECK-NOT-FOUND < %t.out

#include <stdio.h>

int main() {
  fprintf(stderr, "done\n");
}

// CHECK-VERBOSITY1: Parsed ASAN_OPTIONS:
// CHECK-VERBOSITY0-NOT: Parsed ASAN_OPTIONS:
// CHECK-NOT-FOUND: Failed to read options from
