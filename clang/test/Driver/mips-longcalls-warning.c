// REQUIRES: mips-registered-target
// RUN: %clang -### -c -target mips-mti-elf -mlong-calls %s 2>&1 | FileCheck -check-prefix=IMPLICIT %s
// IMPLICIT: warning: ignoring '-mlong-calls' option as it is not currently supported with the implicit usage of -mabicalls

// RUN: %clang -### -c -target mips-mti-elf -mlong-calls -mabicalls %s 2>&1 | FileCheck -check-prefix=EXPLICIT %s
// EXPLICIT: warning: ignoring '-mlong-calls' option as it is not currently supported with -mabicalls
