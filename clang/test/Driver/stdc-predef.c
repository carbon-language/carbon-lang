// Test that clang preincludes stdc-predef.h, if the include file is available
//
// RUN: %clang %s -### -c 2>&1 \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck -check-prefix CHECK-PREDEF %s
// RUN: %clang %s -### -c -ffreestanding 2>&1 \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s
// RUN: %clang %s -c -E 2>&1 \
// RUN: --sysroot=%S/Inputs/basic_linux_tree \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s
// RUN: %clang -c %s -Xclang -verify -DCHECK_DUMMY=1 \
// RUN: --sysroot=%S/Inputs/stdc-predef
// expected-no-diagnostics
// RUN: %clang -x cpp-output %s -### -c 2>&1 \
// RUN: --sysroot=%S/Inputs/stdc-predef \
// RUN: | FileCheck --implicit-check-not "stdc-predef.h" %s

// CHECK-PREDEF: "-fsystem-include-if-exists" "stdc-predef.h"
int i;
#if CHECK_DUMMY
#if !DUMMY_STDC_PREDEF 
  #error "Expected macro symbol DUMMY_STDC_PREDEF is not defined."
#endif
#endif
