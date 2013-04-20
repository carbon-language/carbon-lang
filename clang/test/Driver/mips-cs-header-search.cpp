// REQUIRES: mips-registered-target
//
// Check frontend invocations on Mentor Graphics MIPS toolchain.
//
// = Big-endian, hard float
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-32 %s
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu"
// CHECK-BE-HF-32: "-internal-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/include"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/include-fixed"
// CHECK-BE-HF-32: "-internal-externc-isystem"
// CHECK-BE-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, hard float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mips16 \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-16 %s
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16"
// CHECK-BE-HF-16: "-internal-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/include"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/include-fixed/mips16"
// CHECK-BE-HF-16: "-internal-externc-isystem"
// CHECK-BE-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, hard float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -mmicromips \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-MICRO %s
// CHECK-BE-HF-MICRO: "-internal-isystem"
// CHECK-BE-HF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-MICRO: "-internal-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips"
// CHECK-BE-HF-MICRO: "-internal-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-MICRO: "-internal-externc-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/include"
// CHECK-BE-HF-MICRO: "-internal-externc-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/include-fixed/micromips"
// CHECK-BE-HF-MICRO: "-internal-externc-isystem"
// CHECK-BE-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, soft float
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-32 %s
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float"
// CHECK-BE-SF-32: "-internal-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/include"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/include-fixed/soft-float"
// CHECK-BE-SF-32: "-internal-externc-isystem"
// CHECK-BE-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, soft float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float -mips16 \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-16 %s
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16/soft-float"
// CHECK-BE-SF-16: "-internal-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/include"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/include-fixed/mips16/soft-float"
// CHECK-BE-SF-16: "-internal-externc-isystem"
// CHECK-BE-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, soft float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips-linux-gnu -msoft-float -mmicromips \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-MICRO %s
// CHECK-BE-SF-MICRO: "-internal-isystem"
// CHECK-BE-SF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-MICRO: "-internal-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-internal-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-MICRO: "-internal-externc-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/include"
// CHECK-BE-SF-MICRO: "-internal-externc-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/include-fixed/micromips/soft-float"
// CHECK-BE-SF-MICRO: "-internal-externc-isystem"
// CHECK-BE-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, hard float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips64-linux-gnu \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-HF-64 %s
// CHECK-BE-HF-64: "-internal-isystem"
// CHECK-BE-HF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-HF-64: "-internal-isystem"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/64"
// CHECK-BE-HF-64: "-internal-isystem"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-HF-64: "-internal-externc-isystem"
// CHECK-BE-HF-64: "[[TC]]/include"
// CHECK-BE-HF-64: "-internal-externc-isystem"
// CHECK-BE-HF-64: "[[TC]]/include-fixed/64"
// CHECK-BE-HF-64: "-internal-externc-isystem"
// CHECK-BE-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Big-endian, soft float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips64-linux-gnu -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BE-SF-64 %s
// CHECK-BE-SF-64: "-internal-isystem"
// CHECK-BE-SF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-BE-SF-64: "-internal-isystem"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float/64"
// CHECK-BE-SF-64: "-internal-isystem"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-BE-SF-64: "-internal-externc-isystem"
// CHECK-BE-SF-64: "[[TC]]/include"
// CHECK-BE-SF-64: "-internal-externc-isystem"
// CHECK-BE-SF-64: "[[TC]]/include-fixed/soft-float/64"
// CHECK-BE-SF-64: "-internal-externc-isystem"
// CHECK-BE-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, hard float
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mhard-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-32 %s
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/el"
// CHECK-EL-HF-32: "-internal-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/include"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/include-fixed/el"
// CHECK-EL-HF-32: "-internal-externc-isystem"
// CHECK-EL-HF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, hard float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mips16 \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-16 %s
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16/el"
// CHECK-EL-HF-16: "-internal-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/include"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/include-fixed/mips16/el"
// CHECK-EL-HF-16: "-internal-externc-isystem"
// CHECK-EL-HF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, hard float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mmicromips \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-MICRO %s
// CHECK-EL-HF-MICRO: "-internal-isystem"
// CHECK-EL-HF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-MICRO: "-internal-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips/el"
// CHECK-EL-HF-MICRO: "-internal-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-MICRO: "-internal-externc-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/include"
// CHECK-EL-HF-MICRO: "-internal-externc-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/include-fixed/micromips/el"
// CHECK-EL-HF-MICRO: "-internal-externc-isystem"
// CHECK-EL-HF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, soft float
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mfloat-abi=soft \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-32 %s
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float/el"
// CHECK-EL-SF-32: "-internal-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/include"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/include-fixed/soft-float/el"
// CHECK-EL-SF-32: "-internal-externc-isystem"
// CHECK-EL-SF-32: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, soft float, mips16
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mips16 -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-16 %s
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/mips16/soft-float/el"
// CHECK-EL-SF-16: "-internal-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/include"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/include-fixed/mips16/soft-float/el"
// CHECK-EL-SF-16: "-internal-externc-isystem"
// CHECK-EL-SF-16: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, soft float, micromips
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mipsel-linux-gnu -mmicromips -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-MICRO %s
// CHECK-EL-SF-MICRO: "-internal-isystem"
// CHECK-EL-SF-MICRO: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-MICRO: "-internal-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-internal-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-MICRO: "-internal-externc-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/include"
// CHECK-EL-SF-MICRO: "-internal-externc-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/include-fixed/micromips/soft-float/el"
// CHECK-EL-SF-MICRO: "-internal-externc-isystem"
// CHECK-EL-SF-MICRO: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, hard float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips64el-linux-gnu \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-HF-64 %s
// CHECK-EL-HF-64: "-internal-isystem"
// CHECK-EL-HF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-HF-64: "-internal-isystem"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/el/64"
// CHECK-EL-HF-64: "-internal-isystem"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-HF-64: "-internal-externc-isystem"
// CHECK-EL-HF-64: "[[TC]]/include"
// CHECK-EL-HF-64: "-internal-externc-isystem"
// CHECK-EL-HF-64: "[[TC]]/include-fixed/el/64"
// CHECK-EL-HF-64: "-internal-externc-isystem"
// CHECK-EL-HF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
//
// = Little-endian, soft float, 64-bit
// RUN: %clang -no-canonical-prefixes %s -### -c -o %t.o 2>&1 \
// RUN:     -target mips64el-linux-gnu -msoft-float \
// RUN:     -gcc-toolchain %S/Inputs/mips_cs_tree \
// RUN:   | FileCheck --check-prefix=CHECK-EL-SF-64 %s
// CHECK-EL-SF-64: "-internal-isystem"
// CHECK-EL-SF-64: "[[TC:[^"]+/lib/gcc/mips-linux-gnu/4.6.3]]/../../../../mips-linux-gnu/include/c++/4.6.3"
// CHECK-EL-SF-64: "-internal-isystem"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/mips-linux-gnu/soft-float/el/64"
// CHECK-EL-SF-64: "-internal-isystem"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/include/c++/4.6.3/backward"
// CHECK-EL-SF-64: "-internal-externc-isystem"
// CHECK-EL-SF-64: "[[TC]]/include"
// CHECK-EL-SF-64: "-internal-externc-isystem"
// CHECK-EL-SF-64: "[[TC]]/include-fixed/soft-float/el/64"
// CHECK-EL-SF-64: "-internal-externc-isystem"
// CHECK-EL-SF-64: "[[TC]]/../../../../mips-linux-gnu/libc/usr/include"
