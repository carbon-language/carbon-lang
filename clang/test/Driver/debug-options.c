// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// RUN: %clang -### -c -g %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g2 %s 2>&1 | FileCheck -check-prefix=G2 %s
// RUN: %clang -### -c -g3 %s 2>&1 | FileCheck -check-prefix=G3 %s
// RUN: %clang -### -c -ganything %s 2>&1 | FileCheck -check-prefix=GANY %s
// RUN: %clang -### -c -ggdb %s 2>&1 | FileCheck -check-prefix=GGDB %s
// RUN: %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=GFOO %s
// RUN: %clang -### -c -g -g0 %s 2>&1 | FileCheck -check-prefix=GG0 %s
// RUN: %clang -### -c -gline-tables-only %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO %s
// RUN: %clang -### -c -gline-tables-only -g %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO2 %s
// RUN: %clang -### -c -gline-tables-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO3 %s
//
// G: "-cc1"
// G: "-g"
//
// G2: "-cc1"
// G2: "-g"
//
// G3: "-cc1"
// G3: "-g"
//
// GANY: "-cc1"
// GANY-NOT: "-g"
//
// GGDB: "-cc1"
// GGDB: "-g"
//
// GFOO: "-cc1"
// GFOO-NOT: "-g"
//
// GG0: "-cc1"
// GG0-NOT: "-g"
//
// GLTO: "-cc1"
// GLTO-NOT: "-g"
// GLTO: "-gline-tables-only"
// GLTO-NOT: "-g"
//
// GLTO2: "-cc1"
// GLTO2-NOT: "-gline-tables-only"
// GLTO2: "-g"
// GLTO2-NOT: "-gline-tables-only"
//
// GLTO3: "-cc1"
// GLTO3-NOT: "-gline-tables-only"
