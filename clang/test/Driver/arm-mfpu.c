// Test that different values of -mfpu pick correct ARM FPU target-feature(s).

// RUN: %clang -target arm-linux-eabi %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-target-feature" "+vfp2"
// CHECK-DEFAULT-NOT: "-target-feature" "+vfp3"
// CHECK-DEFAULT-NOT: "-target-feature" "+d16"
// CHECK-DEFAULT-NOT: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fpa %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpe2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpe3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -target arm-linux-eabi -mfpu=maverick %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// CHECK-FPA: "-target-feature" "-vfp2"
// CHECK-FPA: "-target-feature" "-vfp3"
// CHECK-FPA: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// CHECK-VFP3-D16: "-target-feature" "+vfp3"
// CHECK-VFP3-D16: "-target-feature" "+d16"
// CHECK-VFP3-D16: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP %s
// CHECK-VFP: "-target-feature" "+vfp2"
// CHECK-VFP: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// CHECK-VFP3: "-target-feature" "+vfp3"
// CHECK-VFP3: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4 %s
// CHECK-VFP4: "-target-feature" "+vfp4"
// CHECK-VFP4: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=vfp4-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=vfpv4-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP4-D16 %s
// CHECK-VFP4-D16: "-target-feature" "+vfp4"
// CHECK-VFP4-D16: "-target-feature" "+d16"
// CHECK-VFP4-D16: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fp4-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP4-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv4-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP4-SP-D16 %s
// CHECK-FP4-SP-D16: "-target-feature" "+vfp4"
// CHECK-FP4-SP-D16: "-target-feature" "+d16"
// CHECK-FP4-SP-D16: "-target-feature" "+fp-only-sp"
// CHECK-FP4-SP-D16: "-target-feature" "-neon"

// RUN: %clang -target arm-linux-eabi -mfpu=fp5-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-SP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-sp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-SP-D16 %s
// CHECK-FP5-SP-D16: "-target-feature" "+fp-armv8"
// CHECK-FP5-SP-D16: "-target-feature" "+fp-only-sp"
// CHECK-FP5-SP-D16: "-target-feature" "+d16"
// CHECK-FP5-SP-D16: "-target-feature" "-neon"
// CHECK-FP5-SP-D16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=fp5-dp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-DP-D16 %s
// RUN: %clang -target arm-linux-eabi -mfpu=fpv5-dp-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP5-DP-D16 %s
// CHECK-FP5-DP-D16: "-target-feature" "+fp-armv8"
// CHECK-FP5-DP-D16: "-target-feature" "+d16"
// CHECK-FP5-DP-D16: "-target-feature" "-neon"
// CHECK-FP5-DP-D16: "-target-feature" "-crypto"

// RUN: %clang -target arm-linux-eabi -mfpu=neon %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -msoft-float %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-FLOAT %s
// CHECK-SOFT-FLOAT: "-target-feature" "-neon"

// RUN: %clang -target armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV8-DEFAULT-SOFT-FP %s
// CHECK-ARMV8-DEFAULT-SOFT-FP: "-target-feature" "-neon"
// CHECK-ARMV8-DEFAULT-SOFT-FP: "-target-feature" "-crypto"

// RUN: %clang -target armv8 -mfpu=fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV8-SOFT-FLOAT %s
// CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "+fp-armv8"
// CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "-neon"
// CHECK-ARMV8-SOFT-FLOAT: "-target-feature" "-crypto"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FP-ARMV8 %s
// CHECK-FP-ARMV8-NOT: "-target-feature" "+neon"
// CHECK-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-FP-ARMV8: "-target-feature" "-neon"
// CHECK-FP-ARMV8: "-target-feature" "-crypto"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON-FP-ARMV8 %s
// CHECK-NEON-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-NEON-FP-ARMV8: "-target-feature" "+neon"
// CHECK-NEON-FP-ARMV8: "-target-feature" "-crypto"

// RUN: %clang -target armv8-linux-gnueabihf -mfpu=crypto-neon-fp-armv8 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRYPTO-NEON-FP-ARMV8 %s
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+fp-armv8"
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+neon"
// CHECK-CRYPTO-NEON-FP-ARMV8: "-target-feature" "+crypto"

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FP %s
// CHECK-NO-FP: "-target-feature" "-neon"
// CHECK-NO-FP: "-target-feature" "-crypto"
// CHECK-NO-FP: "-target-feature" "-vfp2"
// CHECK-NO-FP: "-target-feature" "-vfp3"
// CHECK-NO-FP: "-target-feature" "-vfp4"
// CHECK-NO-FP: "-target-feature" "-fp-armv8"

// RUN: %clang -target arm-linux-gnueabihf %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-HF %s
// CHECK-HF: "-target-cpu" "arm1136jf-s"

// RUN: %clang -target armv7-apple-darwin -x assembler %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=ASM %s
// ASM-NOT: -target-feature

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv7-linux-gnueabi -mfpu=none -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv6-linux-gnueabi -mfpu=none -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv5-linux-gnueabi -mfpu=none -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv4-linux-gnueabi -mfpu=none -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -mfpu=none -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv7-linux-gnueabi -mfpu=none -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv6-linux-gnueabi -mfpu=none -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv5-linux-gnueabi -mfpu=none -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv4-linux-gnueabi -mfpu=none -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv7-linux-gnueabi -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv6-linux-gnueabi -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv5-linux-gnueabi -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv4-linux-gnueabi -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv7-linux-gnueabi -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv6-linux-gnueabi -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv5-linux-gnueabi -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv4-linux-gnueabi -msoft-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv8-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv7-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv6-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv5-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// RUN: %clang -target armv4-linux-gnueabi -mfpu=none %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI-FP %s
// SOFT-ABI-FP-NOT: error:
// SOFT-ABI-FP-NOT: warning:
// SOFT-ABI-FP-NOT: "-target-feature" "+{{[^ ]*fp[^ ]*}}"
// SOFT-ABI-FP: "-target-feature" "-neon"
// SOFT-ABI-FP: "-target-feature" "-crypto"
// SOFT-ABI-FP: "-target-feature" "-vfp2"
// SOFT-ABI-FP: "-target-feature" "-vfp3"
// SOFT-ABI-FP: "-target-feature" "-vfp4"
// SOFT-ABI-FP: "-target-feature" "-fp-armv8"
// SOFT-ABI-FP-NOT: "-target-feature" "+{{[^ ]*fp[^ ]*}}"
// SOFT-ABI-FP: "-msoft-float"
// SOFT-ABI-FP: "-mfloat-abi" "soft"
// SOFT-ABI-FP-NOT: "-target-feature" "+{{[^ ]*fp[^ ]*}}"

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none -mfloat-abi=softfp %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=ABI-IGNORE %s
// ABI-IGNORE: warning: -mfpu=none implies soft-float, ignoring conflicting option '-mfloat-abi=softfp'

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none -mfloat-abi=hard %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=ABI-ERROR %s
// ABI-ERROR: error: -mfpu=none implies soft-float, which conflicts with option '-mfloat-abi=hard'

// RUN: %clang -target armv8-linux-gnueabi -mfpu=none -mhard-float %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=FP-ERROR %s
// FP-ERROR: error: -mfpu=none implies soft-float, which conflicts with option '-mhard-float'

// RUN: %clang -target armv8-linux-gnueabi -mfpu=vfp4 -mfloat-abi=soft %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI %s
// RUN: %clang -target armv8-linux-gnueabi -mfpu=vfp4 -mfloat-abi=softfp %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI %s
// RUN: %clang -target armv8-freebsd-gnueabi -mfpu=vfp4 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=SOFT-ABI %s
// SOFT-ABI-NOT: error:
// SOFT-ABI-NOT: warning:
// SOFT-ABI-NOT: "-msoft-float"
// SOFT-ABI: "-target-feature" "+vfp4"
// SOFT-ABI-NOT: "-msoft-float"
// SOFT-ABI: "-mfloat-abi" "soft"
// SOFT-ABI-NOT: "-msoft-float"

// Floating point features should be disabled only when explicitly requested,
// otherwise we must respect target inferred defaults.
//
// RUN: %clang -target armv8-freebsd-gnueabi %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=DEFAULT-FP %s
// DEAFULT-FP-NOT: error:
// DEFAULT-FP-NOT: warning:
// DEFAULT-FP-NOT: "-target-feature" "-{{[^ ]*fp[^ ]*}}"
// DEFAULT-FP: "-msoft-float" "-mfloat-abi" "soft"
// DEFAULT-FP-NOT: "-target-feature" "-{{[^ ]*fp[^ ]*}}"

// RUN: %clang -target armv8-linux-gnueabi -mfpu=vfp4 -mfloat-abi=hard %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=VFP-ABI %s
// VFP-ABI-NOT: error:
// VFP-ABI-NOT: warning:
// VFP-ABI-NOT: "-msoft-float"
// VFP-ABI: "-target-feature" "+vfp4"
// VFP-ABI-NOT: "-msoft-float"
// VFP-ABI: "-mfloat-abi" "hard"
// VFP-ABI-NOT: "-msoft-float"
