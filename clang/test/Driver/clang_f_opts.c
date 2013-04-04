// RUN: %clang -### -S -fasm -fblocks -fbuiltin -fno-math-errno -fcommon -fpascal-strings -fno-blocks -fno-builtin -fmath-errno -fno-common -fno-pascal-strings -fblocks -fbuiltin -fmath-errno -fcommon -fpascal-strings -fsplit-stack %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS1 %s
// RUN: %clang -### -S -fasm -fblocks -fbuiltin -fno-math-errno -fcommon -fpascal-strings -fno-asm -fno-blocks -fno-builtin -fmath-errno -fno-common -fno-pascal-strings -fno-show-source-location -fshort-enums -fshort-wchar %s 2>&1 | FileCheck -check-prefix=CHECK-OPTIONS2 %s

// CHECK-OPTIONS1: -split-stacks
// CHECK-OPTIONS1: -fgnu-keywords
// CHECK-OPTIONS1: -fblocks
// CHECK-OPTIONS1: -fpascal-strings

// CHECK_OPTIONS2: -fno-gnu-keywords
// CHECK-OPTIONS2: -fmath-errno
// CHECK-OPTIONS2: -fno-builtin
// CHECK-OPTIONS2: -fshort-enums
// CHECK-OPTIONS2: -fshort-wchar
// CHECK-OPTIONS2: -fno-common
// CHECK-OPTIONS2: -fno-show-source-location

// RUN: %clang -### -S -Wwrite-strings %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS1 %s
// WRITE-STRINGS1: -fconst-strings
// RUN: %clang -### -S -Wwrite-strings -Wno-write-strings %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS2 %s
// WRITE-STRINGS2-NOT: -fconst-strings
// RUN: %clang -### -S -Wwrite-strings -w %s 2>&1 | FileCheck -check-prefix=WRITE-STRINGS3 %s
// WRITE-STRINGS3: -fconst-strings

// RUN: %clang -### -x c++ -c %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -Wno-deprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// RUN: %clang -### -x c++ -c -Wno-deprecated -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -x c++ -c -w %s 2>&1 | FileCheck -check-prefix=DEPRECATED-ON-CHECK %s
// RUN: %clang -### -c %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// RUN: %clang -### -c -Wdeprecated %s 2>&1 | FileCheck -check-prefix=DEPRECATED-OFF-CHECK %s
// DEPRECATED-ON-CHECK: -fdeprecated-macro
// DEPRECATED-OFF-CHECK-NOT: -fdeprecated-macro

// RUN: %clang -### -S -ffp-contract=fast %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-FAST-CHECK %s
// RUN: %clang -### -S -ffast-math %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-FAST-CHECK %s
// RUN: %clang -### -S -ffp-contract=off %s 2>&1 | FileCheck -check-prefix=FP-CONTRACT-OFF-CHECK %s
// FP-CONTRACT-FAST-CHECK: -ffp-contract=fast
// FP-CONTRACT-OFF-CHECK: -ffp-contract=off

// RUN: %clang -### -S -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -fvectorize -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -ftree-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-vectorize -fvectorize %s 2>&1 | FileCheck -check-prefix=CHECK-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// RUN: %clang -### -S -ftree-vectorize -fno-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-VECTORIZE %s
// CHECK-VECTORIZE: "-vectorize-loops"
// CHECK-NO-VECTORIZE-NOT: "-vectorize-loops"

// RUN: %clang -### -S -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -fslp-vectorize -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -ftree-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-slp-vectorize -fslp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-SLP-VECTORIZE %s
// RUN: %clang -### -S -fno-tree-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// RUN: %clang -### -S -ftree-slp-vectorize -fno-slp-vectorize %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SLP-VECTORIZE %s
// CHECK-SLP-VECTORIZE: "-vectorize"
// CHECK-NO-SLP-VECTORIZE-NOT: "-vectorize"

// RUN: %clang -### -S -fextended-identifiers %s 2>&1 | FileCheck -check-prefix=CHECK-EXTENDED-IDENTIFIERS %s
// RUN: %clang -### -S -fno-extended-identifiers %s 2>&1 | FileCheck -check-prefix=CHECK-NO-EXTENDED-IDENTIFIERS %s
// CHECK-EXTENDED-IDENTIFIERS: "-cc1"
// CHECK-EXTENDED-IDENTIFIERS-NOT: "-fextended-identifiers"
// CHECK-NO-EXTENDED-IDENTIFIERS: error: unsupported option '-fno-extended-identifiers'
