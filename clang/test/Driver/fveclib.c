// RUN: %clang -### -c -fveclib=none %s 2>&1 | FileCheck -check-prefix CHECK-NOLIB %s
// RUN: %clang -### -c -fveclib=Accelerate %s 2>&1 | FileCheck -check-prefix CHECK-ACCELERATE %s
// RUN: not %clang -c -fveclib=something %s 2>&1 | FileCheck -check-prefix CHECK-INVALID %s

// CHECK-NOLIB: "-fveclib=none"
// CHECK-ACCELERATE: "-fveclib=Accelerate"

// CHECK-INVALID: error: invalid value 'something' in '-fveclib=something'
