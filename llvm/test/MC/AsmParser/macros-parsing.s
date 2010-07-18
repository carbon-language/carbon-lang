// RUN: llvm-mc %s 2> %t.err
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err

.macros_on
.macros_off

// CHECK-ERRORS: .abort '"end"' detected
.abort "end"
