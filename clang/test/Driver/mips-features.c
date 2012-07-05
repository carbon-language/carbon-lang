// Check handling MIPS specific features options.
//
// -mips16
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-mips16 -mips16 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS16 %s
// CHECK-MIPS16: "-target-feature" "+mips16"
//
// -mno-mips16
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mips16 -mno-mips16 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMIPS16 %s
// CHECK-NOMIPS16: "-target-feature" "-mips16"
