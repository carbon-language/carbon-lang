// RUN: %clang -ccc-host-triple i386-unknown-unknown -### -S -O0 -Os %s -o %t.s -fverbose-asm -funwind-tables -fvisibility=hidden 2> %t.log
// RUN: grep '"-triple" "i386-unknown-unknown"' %t.log
// RUN: grep '"-S"' %t.log
// RUN: grep '"-disable-free"' %t.log
// RUN: grep '"-mrelocation-model" "static"' %t.log
// RUN: grep '"-mdisable-fp-elim"' %t.log
// RUN: grep '"-munwind-tables"' %t.log
// RUN: grep '"-Os"' %t.log
// RUN: grep '"-o" .*clang-translation.*' %t.log
// RUN: grep '"-masm-verbose"' %t.log
// RUN: grep '"-fvisibility" "hidden"' %t.log
// RUN: %clang -ccc-host-triple i386-apple-darwin9 -### -S %s -o %t.s 2> %t.log
// RUN: grep '"-target-cpu" "yonah"' %t.log
// RUN: %clang -ccc-host-triple x86_64-apple-darwin9 -### -S %s -o %t.s 2> %t.log
// RUN: grep '"-target-cpu" "core2"' %t.log

// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -### -S %s 2> %t.log \
// RUN:   -arch armv7
// RUN: FileCheck -check-prefix=ARMV7_DEFAULT %s < %t.log
// ARMV7_DEFAULT: clang
// ARMV7_DEFAULT: "-cc1"
// ARMV7_DEFAULT-NOT: "-msoft-float"
// ARMV7_DEFAULT: "-mfloat-abi" "soft"
// ARMV7_DEFAULT-NOT: "-msoft-float"
// ARMV7_DEFAULT: "-x" "c"

// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -### -S %s 2> %t.log \
// RUN:   -arch armv7 -msoft-float
// RUN: FileCheck -check-prefix=ARMV7_SOFTFLOAT %s < %t.log
// ARMV7_SOFTFLOAT: clang
// ARMV7_SOFTFLOAT: "-cc1"
// ARMV7_SOFTFLOAT: "-msoft-float"
// ARMV7_SOFTFLOAT: "-mfloat-abi" "soft"
// ARMV7_SOFTFLOAT: "-x" "c"

// RUN: %clang -ccc-host-triple x86_64-apple-darwin10 -### -S %s 2> %t.log \
// RUN:   -arch armv7 -mhard-float
// RUN: FileCheck -check-prefix=ARMV7_HARDFLOAT %s < %t.log
// ARMV7_HARDFLOAT: clang
// ARMV7_HARDFLOAT: "-cc1"
// ARMV7_HARDFLOAT-NOT: "-msoft-float"
// ARMV7_HARDFLOAT: "-mfloat-abi" "hard"
// ARMV7_HARDFLOAT-NOT: "-msoft-float"
// ARMV7_HARDFLOAT: "-x" "c"
