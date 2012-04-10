// Check passing options to the assembler for ARM targets.
//
// RUN: %clang -target arm-linux -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM %s
// CHECK-ARM: as{{(.exe)?}}"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MCPU %s
// CHECK-ARM-MCPU: as{{(.exe)?}}" "-mcpu=cortex-a8"
//
// RUN: %clang -target arm-linux -mfpu=neon -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MFPU %s
// CHECK-ARM-MFPU: as{{(.exe)?}}" "-mfpu=neon"
//
// RUN: %clang -target arm-linux -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-MARCH %s
// CHECK-ARM-MARCH: as{{(.exe)?}}" "-march=armv7-a"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -mfpu=neon -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-ALL %s
// CHECK-ARM-ALL: as{{(.exe)?}}" "-march=armv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target armv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ARM-TARGET %s
// CHECK-ARM-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mcpu=cortex-a8"
