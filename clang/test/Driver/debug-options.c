// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// RUN: %clang -### -c -g %s 2>&1 | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g3 %s 2>&1 | FileCheck -check-prefix=G3 %s
// RUN: %clang -### -c -ganything %s 2>&1 | FileCheck -check-prefix=GANY %s
// RUN: %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=GFOO %s
// RUN: %clang -### -c -save-temps -g %s 2>&1 | FileCheck -check-prefix=SAVE %s
//
// G: "-cc1"
// G: "-g"
//
// G3: "-cc1"
// G3: "-g"
//
// GANY: "-cc1"
// GANY-NOT: "-g"
//
// GFOO: "-cc1"
// GFOO-NOT: "-g"
//
// SAVE: "-cc1as"
// SAVE-NOT: "-g"
