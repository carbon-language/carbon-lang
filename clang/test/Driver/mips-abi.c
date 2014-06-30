// Check passing Mips ABI options to the backend.
//
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:        -mabi=32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-32 %s
// MIPS-ABI-32: "-target-cpu" "mips32r2"
// MIPS-ABI-32: "-target-abi" "o32"
//
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:        -mabi=o32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-O32 %s
// MIPS-ABI-O32: "-target-cpu" "mips32r2"
// MIPS-ABI-O32: "-target-abi" "o32"
//
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:        -mabi=n32 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-N32 %s
// MIPS-ABI-N32: "-target-cpu" "mips64r2"
// MIPS-ABI-N32: "-target-abi" "n32"
//
// RUN: %clang -target mips64-linux-gnu -### -c %s \
// RUN:        -mabi=64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-64 %s
// MIPS-ABI-64: "-target-cpu" "mips64r2"
// MIPS-ABI-64: "-target-abi" "n64"
//
// RUN: %clang -target mips64-linux-gnu -### -c %s \
// RUN:        -mabi=n64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-N64 %s
// MIPS-ABI-N64: "-target-cpu" "mips64r2"
// MIPS-ABI-N64: "-target-abi" "n64"
//
// RUN: not %clang -target mips64-linux-gnu -c %s \
// RUN:        -mabi=o64 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-O64 %s
// MIPS-ABI-O64: error: unknown target ABI 'o64'
//
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:        -mabi=eabi 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-EABI %s
// MIPS-ABI-EABI: "-target-cpu" "mips32r2"
// MIPS-ABI-EABI: "-target-abi" "eabi"
//
// RUN: not %clang -target mips-linux-gnu -c %s \
// RUN:        -mabi=unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=MIPS-ABI-UNKNOWN %s
// MIPS-ABI-UNKNOWN: error: unknown target ABI 'unknown'
