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
//
// -mmicromips
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-micromips -mmicromips 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MICROMIPS %s
// CHECK-MICROMIPS: "-target-feature" "+micromips"
//
// -mno-micromips
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mmicromips -mno-micromips 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMICROMIPS %s
// CHECK-NOMICROMIPS: "-target-feature" "-micromips"
//
// -mdsp
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-dsp -mdsp 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MDSP %s
// CHECK-MDSP: "-target-feature" "+dsp"
//
// -mno-dsp
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mdsp -mno-dsp 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMDSP %s
// CHECK-NOMDSP: "-target-feature" "-dsp"
//
// -mdspr2
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-dspr2 -mdspr2 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MDSPR2 %s
// CHECK-MDSPR2: "-target-feature" "+dspr2"
//
// -mno-dspr2
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mdspr2 -mno-dspr2 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOMDSPR2 %s
// CHECK-NOMDSPR2: "-target-feature" "-dspr2"
//
// -mxgot
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mno-xgot -mxgot 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-XGOT %s
// CHECK-XGOT: "-mllvm" "-mxgot"
//
// -mno-xgot
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -mxgot -mno-xgot 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOXGOT %s
// CHECK-NOXGOT-NOT: "-mllvm" "-mxgot"
//
// -G
// RUN: %clang -target mips-linux-gnu -### -c %s \
// RUN:     -G 16 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS-G %s
// CHECK-MIPS-G: "-mllvm" "-mips-ssection-threshold=16"
