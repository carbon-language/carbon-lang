// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// RUN: %clang -### -c -g %s -target x86_64-linux-gnu 2>&1 \
                    | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g2 %s -target x86_64-linux-gnu 2>&1 \
                    | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g3 %s -target x86_64-linux-gnu 2>&1 \
                    | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb %s -target x86_64-linux-gnu 2>&1 \
                    | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-linux-gnu 2>&1 \
                    | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-linux-gnu 2>&1 \
                    | FileCheck -check-prefix=G %s

// RUN: %clang -### -c -g %s -target x86_64-apple-darwin 2>&1 \
                    | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -g2 %s -target x86_64-apple-darwin 2>&1 \
                    | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -g3 %s -target x86_64-apple-darwin 2>&1 \
                    | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb %s -target x86_64-apple-darwin 2>&1 \
                    | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-apple-darwin 2>&1 \
                    | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-apple-darwin 2>&1 \
                    | FileCheck -check-prefix=G_DARWIN %s

// RUN: %clang -### -c -gdwarf-2 %s 2>&1 | FileCheck -check-prefix=G_D2 %s
//
// RUN: %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -g -g0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -ggdb0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
//
// RUN: %clang -### -c -gline-tables-only %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_NO %s
//
// RUN: %clang -### -c -grecord-gcc-switches -gno-record-gcc-switches \
// RUN:        -gstrict-dwarf -gno-strict-dwarf %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GIGNORE %s
//
// RUN: %clang -### -c -ggnu-pubnames %s 2>&1 | FileCheck -check-prefix=GOPT %s
//
// RUN: %clang -### -fdebug-types-section %s 2>&1 \
// RUN:        | FileCheck -check-prefix=FDTS %s
//
// RUN: %clang -### -fdebug-types-section -fno-debug-types-section %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOFDTS %s
//
//
// G: "-cc1"
// G: "-g"
//
// G_DARWIN: "-cc1"
// G_DARWIN: "-gdwarf-2"
// 
// G_D2: "-cc1"
// G_D2: "-gdwarf-2"
//
// G_NO: "-cc1"
// G_NO-NOT: "-g"
//
// GLTO_ONLY: "-cc1"
// GLTO_ONLY-NOT: "-g"
// GLTO_ONLY: "-gline-tables-only"
// GLTO_ONLY-NOT: "-g"
//
// G_ONLY: "-cc1"
// G_ONLY-NOT: "-gline-tables-only"
// G_ONLY: "-g"
// G_ONLY-NOT: "-gline-tables-only"
//
// GLTO_NO: "-cc1"
// GLTO_NO-NOT: "-gline-tables-only"
//
// GIGNORE-NOT: "argument unused during compilation"
//
// GOPT: -generate-gnu-dwarf-pub-sections
//
// FDTS: "-backend-option" "-generate-type-units"
//
// NOFDTS-NOT: "-backend-option" "-generate-type-units"
