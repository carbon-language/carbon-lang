// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// RUN: %clang -### -c -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g2 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g3 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s

// RUN: %clang -### -c -g %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -g2 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -g3 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s

// On the PS4, -g defaults to -gno-column-info, and we always generate the
// arange section.
// RUN: %clang -### -c %s -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=G_PS4 %s
// RUN: %clang -### -c %s -g -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=G_PS4 %s
// RUN: %clang -### -c %s -g -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=NOCI %s
// RUN: %clang -### -c %s -g -gcolumn-info -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=CI %s

// RUN: %clang -### -c -gdwarf-2 %s 2>&1 | FileCheck -check-prefix=G_D2 %s
//
// RUN: %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -g -g0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -ggdb0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
//
// RUN: %clang -### -c -g1 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gmlt %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target i386-pc-solaris 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_NO %s
//
// RUN: %clang -### -c -grecord-gcc-switches -gno-record-gcc-switches \
// RUN:        -gstrict-dwarf -gno-strict-dwarf %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GIGNORE %s
//
// RUN: %clang -### -c -ggnu-pubnames %s 2>&1 | FileCheck -check-prefix=GOPT %s
//
// RUN: %clang -### -c -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=GARANGE %s
//
// RUN: %clang -### -fdebug-types-section %s 2>&1 \
// RUN:        | FileCheck -check-prefix=FDTS %s
//
// RUN: %clang -### -fdebug-types-section -fno-debug-types-section %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOFDTS %s
//
// RUN: %clang -### -g -gno-column-info %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOCI %s
//
// RUN: %clang -### -g %s 2>&1 | FileCheck -check-prefix=CI %s
//
// RUN: %clang -### -gmodules %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GEXTREFS %s
//
// G: "-cc1"
// G: "-g"
//
// G_DARWIN: "-cc1"
// G_DARWIN: "-gdwarf-2"
//
// G_PS4: "-cc1"
// G_PS4: "-generate-arange-section"
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
// GLTO_ONLY_DWARF2: "-cc1"
// GLTO_ONLY_DWARF2-NOT: "-g"
// GLTO_ONLY_DWARF2: "-gline-tables-only"
// GLTO_ONLY_DWARF2: "-gdwarf-2"
// GLTO_ONLY_DWARF2-NOT: "-g"
//
// G_ONLY: "-cc1"
// G_ONLY-NOT: "-gline-tables-only"
// G_ONLY: "-g"
// G_ONLY-NOT: "-gline-tables-only"
//
// G_ONLY_DWARF2: "-cc1"
// G_ONLY_DWARF2-NOT: "-gline-tables-only"
// G_ONLY_DWARF2: "-gdwarf-2"
// G_ONLY_DWARF2-NOT: "-gline-tables-only"
//
// GLTO_NO: "-cc1"
// GLTO_NO-NOT: "-gline-tables-only"
//
// GIGNORE-NOT: "argument unused during compilation"
//
// GOPT: -generate-gnu-dwarf-pub-sections
//
// GARANGE: -generate-arange-section
//
// FDTS: "-backend-option" "-generate-type-units"
//
// NOFDTS-NOT: "-backend-option" "-generate-type-units"
//
// CI: "-dwarf-column-info"
//
// NOCI-NOT: "-dwarf-column-info"
//
// GEXTREFS: "-g" "-dwarf-ext-refs" "-fmodule-format=obj"
