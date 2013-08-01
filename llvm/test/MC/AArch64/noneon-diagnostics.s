// RUN: not llvm-mc  -triple aarch64-none-linux-gnu -mattr=-neon < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

        fmla v3.4s, v12.4s, v17.4s
        fmla v1.2d, v30.2d, v20.2d
        fmla v9.2s, v9.2s, v0.2s
// CHECK-ERROR: error: instruction requires a CPU feature not currently enabled
// CHECK-ERROR-NEXT:    fmla v3.4s, v12.4s, v17.4s
// CHECK-ERROR-NEXT:    ^
// CHECK-ERROR-NEXT: error: instruction requires a CPU feature not currently enabled
// CHECK-ERROR-NEXT:    fmla v1.2d, v30.2d, v20.2d
// CHECK-ERROR-NEXT:    ^
// CHECK-ERROR-NEXT: error: instruction requires a CPU feature not currently enabled
// CHECK-ERROR-NEXT:    fmla v9.2s, v9.2s, v0.2s
// CHECK-ERROR-NEXT:    ^

        fmls v3.4s, v12.4s, v17.4s
        fmls v1.2d, v30.2d, v20.2d
        fmls v9.2s, v9.2s, v0.2s
// CHECK-ERROR: error: instruction requires a CPU feature not currently enabled
// CHECK-ERROR-NEXT:    fmls v3.4s, v12.4s, v17.4s
// CHECK-ERROR-NEXT:    ^
// CHECK-ERROR-NEXT: error: instruction requires a CPU feature not currently enabled
// CHECK-ERROR-NEXT:    fmls v1.2d, v30.2d, v20.2d
// CHECK-ERROR-NEXT:    ^
// CHECK-ERROR-NEXT: error: instruction requires a CPU feature not currently enabled
// CHECK-ERROR-NEXT:    fmls v9.2s, v9.2s, v0.2s
// CHECK-ERROR-NEXT:    ^
