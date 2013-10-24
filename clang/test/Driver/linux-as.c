// Check passing options to the assembler for various linux targets.
//
// RUN: %clang -target arm-linux -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM %s
// CHECK-ARM: as{{(.exe)?}}" "-mfloat-abi=soft"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MCPU %s
// CHECK-ARM-MCPU: as{{(.exe)?}}" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target arm-linux -mfpu=neon -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MFPU %s
// CHECK-ARM-MFPU: as{{(.exe)?}}" "-mfloat-abi=soft" "-mfpu=neon"
//
// RUN: %clang -target arm-linux -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MARCH %s
// CHECK-ARM-MARCH: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armv7-a"
//
// RUN: %clang -target arm-linux -mcpu=cortex-a8 -mfpu=neon -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-ALL %s
// CHECK-ARM-ALL: as{{(.exe)?}}" "-mfloat-abi=soft" "-march=armv7-a" "-mcpu=cortex-a8" "-mfpu=neon"
//
// RUN: %clang -target armv7-linux -mcpu=cortex-a8 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-TARGET %s
// CHECK-ARM-TARGET: as{{(.exe)?}}" "-mfpu=neon" "-mfloat-abi=soft" "-mcpu=cortex-a8"
//
// RUN: %clang -target armv8-linux -mcpu=cortex-a53 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-TARGET-V8 %s
// CHECK-ARM-TARGET-V8: as{{(.exe)?}}" "-mfpu=crypto-neon-fp-armv8" "-mfloat-abi=soft" "-mcpu=cortex-a53"
//
// RUN: %clang -target arm-linux -mfloat-abi=hard -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-MFLOAT-ABI %s
// CHECK-ARM-MFLOAT-ABI: as{{(.exe)?}}" "-mfloat-abi=hard"
//
// RUN: %clang -target arm-linux-androideabi -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-ANDROID %s
// CHECK-ARM-ANDROID: as{{(.exe)?}}" "-mfloat-abi=soft"
//
// RUN: %clang -target arm-linux-androideabi -march=armv7-a -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-ANDROID-SOFTFP %s
// CHECK-ARM-ANDROID-SOFTFP: as{{(.exe)?}}" "-mfloat-abi=softfp" "-march=armv7-a"
//
// RUN: %clang -target arm-linux-eabi -mhard-float -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ARM-HARDFP %s
// CHECK-ARM-HARDFP: as{{(.exe)?}}" "-mfloat-abi=hard"
//
// RUN: %clang -target ppc-linux -mcpu=invalid-cpu -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PPC-NO-MCPU %s
// CHECK-PPC-NO-MCPU-NOT: as{{.*}} "-mcpu=invalid-cpu"
//
// RUN: %clang -target s390x-linux -### -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-Z-DEFAULT-ARCH %s
// CHECK-Z-DEFAULT-ARCH: as{{.*}} "-march=z10"
//
// RUN: %clang -target s390x-linux -march=z196 -### \
// RUN:   -no-integrated-as -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-Z-ARCH-Z196 %s
// CHECK-Z-ARCH-Z196: as{{.*}} "-march=z196"
