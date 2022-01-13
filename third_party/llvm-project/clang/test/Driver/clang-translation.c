// RUN: %clang -target i386-unknown-unknown -### -S -O0 -Os %s -o %t.s -fverbose-asm -fvisibility=hidden 2>&1 | FileCheck -check-prefix=I386 %s
// I386: "-triple" "i386-unknown-unknown"
// I386: "-S"
// I386: "-disable-free"
// I386: "-mrelocation-model" "static"
// I386: "-mframe-pointer=all"
// I386: "-funwind-tables=2"
// I386: "-Os"
// I386: "-fvisibility"
// I386: "hidden"
// I386: "-o"
// I386: clang-translation

// RUN: %clang -target i386-unknown-unknown -### -S %s -fasynchronous-unwind-tables -fno-unwind-tables 2>&1 | FileCheck --check-prefix=UNWIND-TABLES %s
// UNWIND-TABLES: "-funwind-tables=2"

// RUN: %clang -target i386-apple-darwin9 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=YONAH %s
// RUN: %clang -target i386-apple-macosx10.11 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=YONAH %s
// YONAH: "-target-cpu"
// YONAH: "yonah"

// RUN: %clang -target x86_64-apple-darwin9 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=CORE2 %s
// RUN: %clang -target x86_64-apple-macosx10.11 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=CORE2 %s
// CORE2: "-target-cpu"
// CORE2: "core2"

// RUN: %clang -target x86_64h-apple-darwin -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=AVX2 %s
// RUN: %clang -target x86_64h-apple-macosx10.12 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=AVX2 %s
// AVX2: "-target-cpu"
// AVX2: "core-avx2"

// RUN: %clang -target x86_64h-apple-darwin -march=skx -### %s -o /dev/null 2>&1 | \
// RUN: FileCheck -check-prefix=X8664HSKX %s
// X8664HSKX: "-target-cpu"
// X8664HSKX: "skx"

// RUN: %clang -target i386-apple-macosx10.12 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=PENRYN %s
// RUN: %clang -target x86_64-apple-macosx10.12 -### -S %s -o %t.s 2>&1 | \
// RUN: FileCheck -check-prefix=PENRYN %s
// PENRYN: "-target-cpu"
// PENRYN: "penryn"


// RUN: %clang -target x86_64-apple-darwin10 -### -S %s -arch armv7 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV7_DEFAULT %s
// ARMV7_DEFAULT: clang
// ARMV7_DEFAULT: "-cc1"
// ARMV7_DEFAULT-NOT: "-msoft-float"
// ARMV7_DEFAULT: "-mfloat-abi" "soft"
// ARMV7_DEFAULT-NOT: "-msoft-float"
// ARMV7_DEFAULT: "-x" "c"

// RUN: %clang -target x86_64-apple-darwin10 -### -S %s -arch armv7 \
// RUN: -msoft-float 2>&1 | FileCheck -check-prefix=ARMV7_SOFTFLOAT %s
// ARMV7_SOFTFLOAT: clang
// ARMV7_SOFTFLOAT: "-cc1"
// ARMV7_SOFTFLOAT: "-target-feature"
// ARMV7_SOFTFLOAT: "-neon"
// ARMV7_SOFTFLOAT: "-msoft-float"
// ARMV7_SOFTFLOAT: "-mfloat-abi" "soft"
// ARMV7_SOFTFLOAT: "-x" "c"

// RUN: %clang -target x86_64-apple-darwin10 -### -S %s -arch armv7 \
// RUN: -mhard-float 2>&1 | FileCheck -check-prefix=ARMV7_HARDFLOAT %s
// ARMV7_HARDFLOAT: clang
// ARMV7_HARDFLOAT: "-cc1"
// ARMV7_HARDFLOAT-NOT: "-msoft-float"
// ARMV7_HARDFLOAT: "-mfloat-abi" "hard"
// ARMV7_HARDFLOAT-NOT: "-msoft-float"
// ARMV7_HARDFLOAT: "-x" "c"

// RUN: %clang -target arm64-apple-ios10 -### -S %s -arch arm64 2>&1 | \
// RUN: FileCheck -check-prefix=ARM64-APPLE %s
// ARM64-APPLE: -funwind-tables=2

// RUN: %clang -target arm64-apple-ios10 -### -ffreestanding -S %s -arch arm64 2>&1 | \
// RUN: FileCheck -check-prefix=ARM64-FREESTANDING-APPLE %s
//
// RUN: %clang -target arm64-apple-ios10 -### -fno-unwind-tables -ffreestanding -S %s -arch arm64 2>&1 | \
// RUN: FileCheck -check-prefix=ARM64-FREESTANDING-APPLE %s
//
// ARM64-FREESTANDING-APPLE-NOT: -funwind-tables

// RUN: %clang -target arm64-apple-ios10 -### -funwind-tables -S %s -arch arm64 2>&1 | \
// RUN: FileCheck -check-prefix=ARM64-EXPLICIT-UWTABLE-APPLE %s
//
// RUN: %clang -target arm64-apple-ios10 -### -ffreestanding -funwind-tables -S %s -arch arm64 2>&1 | \
// RUN: FileCheck -check-prefix=ARM64-EXPLICIT-UWTABLE-APPLE %s
//
// ARM64-EXPLICIT-UWTABLE-APPLE: -funwind-tables

// RUN: %clang -target arm64-apple-ios10 -fno-exceptions -### -S %s -arch arm64 2>&1 | \
// RUN: FileCheck -check-prefix=ARM64-APPLE-EXCEP %s
// ARM64-APPLE-EXCEP-NOT: -funwind-tables

// RUN: %clang -target armv7k-apple-watchos4.0 -### -S %s -arch armv7k 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV7K-APPLE %s
// ARMV7K-APPLE: -funwind-tables

// RUN: %clang -target arm-linux -### -S %s -march=armv5e 2>&1 | \
// RUN: FileCheck -check-prefix=ARMV5E %s
// ARMV5E: clang
// ARMV5E: "-cc1"
// ARMV5E: "-target-cpu" "arm1022e"

// RUN: %clang -target armv7-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER-HARD %s
// ARMv7_THREAD_POINTER-HARD: "-target-feature" "+read-tp-hard"

// RUN: %clang -target armv6t2-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang -target thumbv6t2-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang -target armv6k-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang -target armv6-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// RUN: %clang -target armv5t-linux -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARM_THREAD_POINTER-HARD %s
// ARM_THREAD_POINTER-HARD: "-target-feature" "+read-tp-hard"

// RUN: %clang -target armv5t-linux -mtp=cp15 -x assembler -### %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv5_THREAD_POINTER_ASSEMBLER %s
// ARMv5_THREAD_POINTER_ASSEMBLER-NOT: hardware TLS register is not supported for the armv5 sub-architecture

// RUN: %clang -target armv6-linux -mthumb -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=THUMBv6_THREAD_POINTER_UNSUPP %s
// RUN: %clang -target thumbv6-linux -mthumb -mtp=cp15 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=THUMBv6_THREAD_POINTER_UNSUPP %s
// THUMBv6_THREAD_POINTER_UNSUPP: hardware TLS register is not supported for the thumbv6 sub-architecture

// RUN: %clang -target armv7-linux -mtp=soft -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER_SOFT %s
// ARMv7_THREAD_POINTER_SOFT-NOT: "-target-feature" "+read-tp-hard"

// RUN: %clang -target armv7-linux -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv7_THREAD_POINTER_NON %s
// ARMv7_THREAD_POINTER_NON-NOT: "-target-feature" "+read-tp-hard"

// RUN: %clang -target aarch64-linux -### -S %s -arch armv8a 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_NON %s
// ARMv8_THREAD_POINTER_NON-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_NON-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_NON-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang -target aarch64-linux -### -S %s -arch armv8a -mtp=el0 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL0 %s
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL0-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang -target aarch64-linux -### -S %s -arch armv8a -mtp=el1 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL1 %s
// ARMv8_THREAD_POINTER_EL1: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL1-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL1-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang -target aarch64-linux -### -S %s -arch armv8a -mtp=el2 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL2 %s
// ARMv8_THREAD_POINTER_EL2-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL2: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL2-NOT: "-target-feature" "+tpidr-el3"

// RUN: %clang -target aarch64-linux -### -S %s -arch armv8a -mtp=el3 2>&1 | \
// RUN: FileCheck -check-prefix=ARMv8_THREAD_POINTER_EL3 %s
// ARMv8_THREAD_POINTER_EL3-NOT: "-target-feature" "+tpidr-el1"
// ARMv8_THREAD_POINTER_EL3-NOT: "-target-feature" "+tpidr-el2"
// ARMv8_THREAD_POINTER_EL3: "-target-feature" "+tpidr-el3"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=G5 2>&1 | FileCheck -check-prefix=PPCG5 %s
// PPCG5: clang
// PPCG5: "-cc1"
// PPCG5: "-target-cpu" "g5"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power7 2>&1 | FileCheck -check-prefix=PPCPWR7 %s
// PPCPWR7: clang
// PPCPWR7: "-cc1"
// PPCPWR7: "-target-cpu" "pwr7"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power8 2>&1 | FileCheck -check-prefix=PPCPWR8 %s
// PPCPWR8: clang
// PPCPWR8: "-cc1"
// PPCPWR8: "-target-cpu" "pwr8"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=630 2>&1 | FileCheck -check-prefix=PPC630 %s
// PPC630: clang
// PPC630: "-cc1"
// PPC630: "-target-cpu" "pwr3"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power3 2>&1 | FileCheck -check-prefix=PPCPOWER3 %s
// PPCPOWER3: clang
// PPCPOWER3: "-cc1"
// PPCPOWER3: "-target-cpu" "pwr3"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=pwr3 2>&1 | FileCheck -check-prefix=PPCPWR3 %s
// PPCPWR3: clang
// PPCPWR3: "-cc1"
// PPCPWR3: "-target-cpu" "pwr3"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power4 2>&1 | FileCheck -check-prefix=PPCPOWER4 %s
// PPCPOWER4: clang
// PPCPOWER4: "-cc1"
// PPCPOWER4: "-target-cpu" "pwr4"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=pwr4 2>&1 | FileCheck -check-prefix=PPCPWR4 %s
// PPCPWR4: clang
// PPCPWR4: "-cc1"
// PPCPWR4: "-target-cpu" "pwr4"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power5 2>&1 | FileCheck -check-prefix=PPCPOWER5 %s
// PPCPOWER5: clang
// PPCPOWER5: "-cc1"
// PPCPOWER5: "-target-cpu" "pwr5"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=pwr5 2>&1 | FileCheck -check-prefix=PPCPWR5 %s
// PPCPWR5: clang
// PPCPWR5: "-cc1"
// PPCPWR5: "-target-cpu" "pwr5"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power5x 2>&1 | FileCheck -check-prefix=PPCPOWER5X %s
// PPCPOWER5X: clang
// PPCPOWER5X: "-cc1"
// PPCPOWER5X: "-target-cpu" "pwr5x"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=pwr5x 2>&1 | FileCheck -check-prefix=PPCPWR5X %s
// PPCPWR5X: clang
// PPCPWR5X: "-cc1"
// PPCPWR5X: "-target-cpu" "pwr5x"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power6 2>&1 | FileCheck -check-prefix=PPCPOWER6 %s
// PPCPOWER6: clang
// PPCPOWER6: "-cc1"
// PPCPOWER6: "-target-cpu" "pwr6"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=pwr6 2>&1 | FileCheck -check-prefix=PPCPWR6 %s
// PPCPWR6: clang
// PPCPWR6: "-cc1"
// PPCPWR6: "-target-cpu" "pwr6"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power6x 2>&1 | FileCheck -check-prefix=PPCPOWER6X %s
// PPCPOWER6X: clang
// PPCPOWER6X: "-cc1"
// PPCPOWER6X: "-target-cpu" "pwr6x"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=pwr6x 2>&1 | FileCheck -check-prefix=PPCPWR6X %s
// PPCPWR6X: clang
// PPCPWR6X: "-cc1"
// PPCPWR6X: "-target-cpu" "pwr6x"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=power7 2>&1 | FileCheck -check-prefix=PPCPOWER7 %s
// PPCPOWER7: clang
// PPCPOWER7: "-cc1"
// PPCPOWER7: "-target-cpu" "pwr7"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=powerpc 2>&1 | FileCheck -check-prefix=PPCPOWERPC %s
// PPCPOWERPC: clang
// PPCPOWERPC: "-cc1"
// PPCPOWERPC: "-target-cpu" "ppc"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s -mcpu=powerpc64 2>&1 | FileCheck -check-prefix=PPCPOWERPC64 %s
// PPCPOWERPC64: clang
// PPCPOWERPC64: "-cc1"
// PPCPOWERPC64: "-target-cpu" "ppc64"

// RUN: %clang -target powerpc64-unknown-linux-gnu \
// RUN: -### -S %s 2>&1 | FileCheck -check-prefix=PPC64NS %s
// PPC64NS: clang
// PPC64NS: "-cc1"
// PPC64NS: "-target-cpu" "ppc64"

// RUN: %clang -target powerpc-fsl-linux -### -S %s \
// RUN: -mcpu=e500 2>&1 | FileCheck -check-prefix=PPCE500 %s
// PPCE500: clang
// PPCE500: "-cc1"
// PPCE500: "-target-cpu" "e500"

// RUN: %clang -target powerpc-fsl-linux -### -S %s \
// RUN: -mcpu=8548 2>&1 | FileCheck -check-prefix=PPC8548 %s
// PPC8548: clang
// PPC8548: "-cc1"
// PPC8548: "-target-cpu" "e500"

// RUN: %clang -target powerpc-fsl-linux -### -S %s \
// RUN: -mcpu=e500mc 2>&1 | FileCheck -check-prefix=PPCE500MC %s
// PPCE500MC: clang
// PPCE500MC: "-cc1"
// PPCE500MC: "-target-cpu" "e500mc"

// RUN: %clang -target powerpc64-fsl-linux -### -S \
// RUN: %s -mcpu=e5500 2>&1 | FileCheck -check-prefix=PPCE5500 %s
// PPCE5500: clang
// PPCE5500: "-cc1"
// PPCE5500: "-target-cpu" "e5500"

// RUN: %clang -target amd64-unknown-openbsd5.2 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=AMD64 %s
// AMD64: clang
// AMD64: "-cc1"
// AMD64: "-triple"
// AMD64: "amd64-unknown-openbsd5.2"
// AMD64: "-funwind-tables=2"

// RUN: %clang -target amd64--mingw32 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=AMD64-MINGW %s
// AMD64-MINGW: clang
// AMD64-MINGW: "-cc1"
// AMD64-MINGW: "-triple"
// AMD64-MINGW: "amd64-unknown-windows-gnu"
// AMD64-MINGW: "-funwind-tables=2"

// RUN: %clang -target i686-linux-android -### -S %s 2>&1 \
// RUN:        --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=ANDROID-X86 %s
// ANDROID-X86: clang
// ANDROID-X86: "-target-cpu" "i686"
// ANDROID-X86: "-target-feature" "+ssse3"

// RUN: %clang -target x86_64-linux-android -### -S %s 2>&1 \
// RUN:        --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=ANDROID-X86_64 %s
// ANDROID-X86_64: clang
// ANDROID-X86_64: "-target-cpu" "x86-64"
// ANDROID-X86_64: "-target-feature" "+sse4.2"
// ANDROID-X86_64: "-target-feature" "+popcnt"
// ANDROID-X86_64: "-target-feature" "+cx16"

// RUN: %clang -target mips-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS %s
// MIPS: clang
// MIPS: "-cc1"
// MIPS: "-target-cpu" "mips32r2"
// MIPS: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa32r6-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSR6 %s
// MIPSR6: clang
// MIPSR6: "-cc1"
// MIPSR6: "-target-cpu" "mips32r6"
// MIPSR6: "-mfloat-abi" "hard"

// RUN: %clang -target mipsel-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSEL %s
// MIPSEL: clang
// MIPSEL: "-cc1"
// MIPSEL: "-target-cpu" "mips32r2"
// MIPSEL: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa32r6el-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSR6EL %s
// MIPSR6EL: clang
// MIPSR6EL: "-cc1"
// MIPSR6EL: "-target-cpu" "mips32r6"
// MIPSR6EL: "-mfloat-abi" "hard"

// RUN: %clang -target mipsel-linux-android -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSEL-ANDROID %s
// MIPSEL-ANDROID: clang
// MIPSEL-ANDROID: "-cc1"
// MIPSEL-ANDROID: "-target-cpu" "mips32"
// MIPSEL-ANDROID: "-target-feature" "+fpxx"
// MIPSEL-ANDROID: "-target-feature" "+nooddspreg"
// MIPSEL-ANDROID: "-mfloat-abi" "hard"

// RUN: %clang -target mipsel-linux-android -### -S %s -mcpu=mips32r6 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSEL-ANDROID-R6 %s
// MIPSEL-ANDROID-R6: clang
// MIPSEL-ANDROID-R6: "-cc1"
// MIPSEL-ANDROID-R6: "-target-cpu" "mips32r6"
// MIPSEL-ANDROID-R6: "-target-feature" "+fp64"
// MIPSEL-ANDROID-R6: "-target-feature" "+nooddspreg"
// MIPSEL-ANDROID-R6: "-mfloat-abi" "hard"

// RUN: %clang -target mips64-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64 %s
// MIPS64: clang
// MIPS64: "-cc1"
// MIPS64: "-target-cpu" "mips64r2"
// MIPS64: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa64r6-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64R6 %s
// MIPS64R6: clang
// MIPS64R6: "-cc1"
// MIPS64R6: "-target-cpu" "mips64r6"
// MIPS64R6: "-mfloat-abi" "hard"

// RUN: %clang -target mips64el-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64EL %s
// MIPS64EL: clang
// MIPS64EL: "-cc1"
// MIPS64EL: "-target-cpu" "mips64r2"
// MIPS64EL: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa64r6el-linux-gnu -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64R6EL %s
// MIPS64R6EL: clang
// MIPS64R6EL: "-cc1"
// MIPS64R6EL: "-target-cpu" "mips64r6"
// MIPS64R6EL: "-mfloat-abi" "hard"

// RUN: %clang -target mips64-linux-gnuabi64 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64-GNUABI64 %s
// MIPS64-GNUABI64: clang
// MIPS64-GNUABI64: "-cc1"
// MIPS64-GNUABI64: "-target-cpu" "mips64r2"
// MIPS64-GNUABI64: "-target-abi" "n64"
// MIPS64-GNUABI64: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa64r6-linux-gnuabi64 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64R6-GNUABI64 %s
// MIPS64R6-GNUABI64: clang
// MIPS64R6-GNUABI64: "-cc1"
// MIPS64R6-GNUABI64: "-target-cpu" "mips64r6"
// MIPS64R6-GNUABI64: "-target-abi" "n64"
// MIPS64R6-GNUABI64: "-mfloat-abi" "hard"

// RUN: %clang -target mips64el-linux-gnuabi64 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64EL-GNUABI64 %s
// MIPS64EL-GNUABI64: clang
// MIPS64EL-GNUABI64: "-cc1"
// MIPS64EL-GNUABI64: "-target-cpu" "mips64r2"
// MIPS64EL-GNUABI64: "-target-abi" "n64"
// MIPS64EL-GNUABI64: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa64r6el-linux-gnuabi64 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64R6EL-GNUABI64 %s
// MIPS64R6EL-GNUABI64: clang
// MIPS64R6EL-GNUABI64: "-cc1"
// MIPS64R6EL-GNUABI64: "-target-cpu" "mips64r6"
// MIPS64R6EL-GNUABI64: "-target-abi" "n64"
// MIPS64R6EL-GNUABI64: "-mfloat-abi" "hard"

// RUN: %clang -target mips64-linux-gnuabin32 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSN32 %s
// MIPSN32: clang
// MIPSN32: "-cc1"
// MIPSN32: "-target-cpu" "mips64r2"
// MIPSN32: "-target-abi" "n32"
// MIPSN32: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa64r6-linux-gnuabin32 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSN32R6 %s
// MIPSN32R6: clang
// MIPSN32R6: "-cc1"
// MIPSN32R6: "-target-cpu" "mips64r6"
// MIPSN32R6: "-target-abi" "n32"
// MIPSN32R6: "-mfloat-abi" "hard"

// RUN: %clang -target mips64el-linux-gnuabin32 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSN32EL %s
// MIPSN32EL: clang
// MIPSN32EL: "-cc1"
// MIPSN32EL: "-target-cpu" "mips64r2"
// MIPSN32EL: "-target-abi" "n32"
// MIPSN32EL: "-mfloat-abi" "hard"

// RUN: %clang -target mipsisa64r6el-linux-gnuabin32 -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPSN32R6EL %s
// MIPSN32R6EL: clang
// MIPSN32R6EL: "-cc1"
// MIPSN32R6EL: "-target-cpu" "mips64r6"
// MIPSN32R6EL: "-target-abi" "n32"
// MIPSN32R6EL: "-mfloat-abi" "hard"

// RUN: %clang -target mips64el-linux-android -### -S %s 2>&1 | \
// RUN: FileCheck -check-prefix=MIPS64EL-ANDROID %s
// MIPS64EL-ANDROID: clang
// MIPS64EL-ANDROID: "-cc1"
// MIPS64EL-ANDROID: "-target-cpu" "mips64r6"
// MIPS64EL-ANDROID: "-mfloat-abi" "hard"
