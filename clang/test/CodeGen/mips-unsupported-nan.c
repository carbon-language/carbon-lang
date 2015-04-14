// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips2 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS2 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips3 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS3 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips4 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS4 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips32 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS32 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips32r2 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS32R2 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mipsel-unknown-linux -mnan=2008 -march=mips32r3 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-NOT-MIPS32R3 -check-prefix=CHECK-NAN2008 %s
// RUN: %clang -target mipsel-unknown-linux -mnan=legacy -march=mips32r6 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS32R6 -check-prefix=CHECK-NAN2008 %s
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips64 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS64 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mips64el-unknown-linux -mnan=2008 -march=mips64r2 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS64R2 -check-prefix=CHECK-NANLEGACY %s
// RUN: %clang -target mips64el-unknown-linux -mnan=legacy -march=mips64r6 -emit-llvm -S %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MIPS64R6 -check-prefix=CHECK-NAN2008 %s

// CHECK-MIPS2: warning: ignoring '-mnan=2008' option because the 'mips2' architecture does not support it
// CHECK-MIPS3: warning: ignoring '-mnan=2008' option because the 'mips3' architecture does not support it
// CHECK-MIPS4: warning: ignoring '-mnan=2008' option because the 'mips4' architecture does not support it
// CHECK-MIPS32: warning: ignoring '-mnan=2008' option because the 'mips32' architecture does not support it
// CHECK-MIPS32R2: warning: ignoring '-mnan=2008' option because the 'mips32r2' architecture does not support it
// CHECK-MIPS32R3: warning: ignoring '-mnan=2008' option because the 'mips32r3' architecture does not support it
// CHECK-MIPS32R6: warning: ignoring '-mnan=legacy' option because the 'mips32r6' architecture does not support it
// CHECK-MIPS64: warning: ignoring '-mnan=2008' option because the 'mips64' architecture does not support it
// CHECK-MIPS64R2: warning: ignoring '-mnan=2008' option because the 'mips64r2' architecture does not support it
// CHECK-MIPS64R6: warning: ignoring '-mnan=legacy' option because the 'mips64r6' architecture does not support it
// CHECK-NANLEGACY: float 0x7FF4000000000000
// CHECK-NAN2008: float 0x7FF8000000000000

float f =  __builtin_nan("");
