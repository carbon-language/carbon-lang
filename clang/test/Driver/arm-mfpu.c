// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=fpa %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=fpe2 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=fpe3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=maverick %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPA %s
// CHECK-FPA: "-target-feature" "-vfp2"
// CHECK-FPA: "-target-feature" "-vfp3"
// CHECK-FPA: "-target-feature" "-neon"

// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=vfp3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=vfpv3-d16 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3-D16 %s
// CHECK-VFP3-D16: "-target-feature" "+vfp3"
// CHECK-VFP3-D16: "-target-feature" "+d16"
// CHECK-VFP3-D16: "-target-feature" "-neon"

// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=vfp %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP %s
// CHECK-VFP: "-target-feature" "+vfp2"
// CHECK-VFP: "-target-feature" "-neon"

// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=vfp3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=vfpv3 %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VFP3 %s
// CHECK-VFP3: "-target-feature" "+vfp3"
// CHECK-VFP3: "-target-feature" "-neon"

// RUN: %clang -ccc-host-triple arm-linux-eabi -mfpu=neon %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: "-target-feature" "+neon"

