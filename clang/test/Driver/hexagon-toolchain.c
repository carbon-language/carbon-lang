// REQUIRES: hexagon-registered-target

// Tests disabled for now in non-Unix-like systems where we can't seem to find hexagon-as
// XFAIL: mingw32,win32

// Temporarily 
// XFAIL

// -----------------------------------------------------------------------------
// Test standard include paths
// -----------------------------------------------------------------------------

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK001 %s
// CHECK001: "-cc1" {{.*}} "-internal-externc-isystem" "[[INSTALL_DIR:.*]]/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include"
// CHECK001:   "-internal-externc-isystem" "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include-fixed"
// CHECK001:   "-internal-externc-isystem" "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include"
// CHECK001-NEXT: "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"

// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK002 %s
// CHECK002: "-cc1" {{.*}} "-internal-isystem" "[[INSTALL_DIR:.*]]/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include/c++/4.4.0"
// CHECK002:   "-internal-externc-isystem" "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include"
// CHECK002:   "-internal-externc-isystem" "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include-fixed"
// CHECK002:   "-internal-externc-isystem" "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include"
// CHECK002-NEXT: "[[INSTALL_DIR]]/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"

// -----------------------------------------------------------------------------
// Test -nostdinc, -nostdlibinc, -nostdinc++
// -----------------------------------------------------------------------------

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nostdinc \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK003 %s
// CHECK003: "-cc1"
// CHECK003-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include"
// CHECK003-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include-fixed"
// CHECK003-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include"
// CHECK003-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nostdlibinc \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK004 %s
// CHECK004: "-cc1"
// CHECK004-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include"
// CHECK004-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include-fixed"
// CHECK004-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include"
// CHECK004-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"

// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nostdlibinc \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK005 %s
// CHECK005: "-cc1"
// CHECK005-NOT: "-internal-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include/c++/4.4.0"
// CHECK005-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include"
// CHECK005-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/lib/gcc/hexagon/4.4.0/include-fixed"
// CHECK005-NOT: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include"
// CHECK005-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"

// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nostdinc++ \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK006 %s
// CHECK006: "-cc1"
// CHECK006-NOT: "-internal-isystem" "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/hexagon/include/c++/4.4.0"
// CHECK006-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"

// -----------------------------------------------------------------------------
// Test -march=<archname> -mcpu=<archname> -mv<number>
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -march=hexagonv3 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK007 %s
// CHECK007: "-cc1" {{.*}} "-target-cpu" "hexagonv3"
// CHECK007-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v3"
// CHECK007-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-ld"{{.*}} "-mv3"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -mcpu=hexagonv5 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK008 %s
// CHECK008: "-cc1" {{.*}} "-target-cpu" "hexagonv5"
// CHECK008-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v5"
// CHECK008-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-ld"{{.*}} "-mv5"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -mv2 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK009 %s
// CHECK009: "-cc1" {{.*}} "-target-cpu" "hexagonv2"
// CHECK009-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v2"
// CHECK009-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-ld"{{.*}} "-mv2"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK010 %s
// CHECK010: "-cc1" {{.*}} "-target-cpu" "hexagonv4"
// CHECK010-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v4"
// CHECK010-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-ld"{{.*}} "-mv4"

// -----------------------------------------------------------------------------
// Test Linker related args
// -----------------------------------------------------------------------------

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Defaults for C
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK011 %s
// CHECK011: "{{.*}}clang{{.*}}" "-cc1"
// CHECK011-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK011-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK011-NOT: "-static"
// CHECK011-NOT: "-shared"
// CHECK011: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK011: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK011: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK011: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK011: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK011: "-L[[GNU_DIR]]/lib/gcc"
// CHECK011: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK011: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK011: "{{[^"]+}}.o"
// CHECK011: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK011: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Defaults for C++
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK012 %s
// CHECK012: "{{.*}}clang{{.*}}" "-cc1"
// CHECK012-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK012-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK012-NOT: "-static"
// CHECK012-NOT: "-shared"
// CHECK012: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK012: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK012: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK012: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK012: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK012: "-L[[GNU_DIR]]/lib/gcc"
// CHECK012: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK012: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK012: "{{[^"]+}}.o"
// CHECK012: "-lstdc++" "-lm"
// CHECK012: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK012: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Additional Libraries (-L)
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -Lone -L two -L three \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK013 %s
// CHECK013: "{{.*}}clang{{.*}}" "-cc1"
// CHECK013-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK013-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK013: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK013: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK013: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK013: "-Lone" "-Ltwo" "-Lthree"
// CHECK013: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK013: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK013: "-L[[GNU_DIR]]/lib/gcc"
// CHECK013: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK013: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK013: "{{[^"]+}}.o"
// CHECK013: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK013: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -static, -shared
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -static \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK014 %s
// CHECK014: "{{.*}}clang{{.*}}" "-cc1"
// CHECK014-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK014-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK014: "-static"
// CHECK014: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK014: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK014: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK014: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK014: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK014: "-L[[GNU_DIR]]/lib/gcc"
// CHECK014: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK014: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK014: "{{[^"]+}}.o"
// CHECK014: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK014: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -shared \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK015 %s
// CHECK015: "{{.*}}clang{{.*}}" "-cc1"
// CHECK015-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK015-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK015: "-shared" "-call_shared"
// CHECK015-NOT: crt0_standalone.o
// CHECK015-NOT: crt0.o
// CHECK015: "[[GNU_DIR]]/hexagon/lib/v4/G0/initS.o"
// CHECK015: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4/G0"
// CHECK015: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/G0"
// CHECK015: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK015: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK015: "-L[[GNU_DIR]]/lib/gcc"
// CHECK015: "-L[[GNU_DIR]]/hexagon/lib/v4/G0"
// CHECK015: "-L[[GNU_DIR]]/hexagon/lib/G0"
// CHECK015: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK015: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK015: "{{[^"]+}}.o"
// CHECK015: "--start-group"
// CHECK015-NOT: "-lstandalone"
// CHECK015-NOT: "-lc"
// CHECK015: "-lgcc"
// CHECK015: "--end-group"
// CHECK015: "[[GNU_DIR]]/hexagon/lib/v4/G0/finiS.o"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -shared \
// RUN:   -static \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK016 %s
// CHECK016: "{{.*}}clang{{.*}}" "-cc1"
// CHECK016-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK016-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK016: "-shared" "-call_shared" "-static"
// CHECK016-NOT: crt0_standalone.o
// CHECK016-NOT: crt0.o
// CHECK016: "[[GNU_DIR]]/hexagon/lib/v4/G0/init.o"
// CHECK016: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4/G0"
// CHECK016: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/G0"
// CHECK016: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK016: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK016: "-L[[GNU_DIR]]/lib/gcc"
// CHECK016: "-L[[GNU_DIR]]/hexagon/lib/v4/G0"
// CHECK016: "-L[[GNU_DIR]]/hexagon/lib/G0"
// CHECK016: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK016: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK016: "{{[^"]+}}.o"
// CHECK016: "--start-group"
// CHECK016-NOT: "-lstandalone"
// CHECK016-NOT: "-lc"
// CHECK016: "-lgcc"
// CHECK016: "--end-group"
// CHECK016: "[[GNU_DIR]]/hexagon/lib/v4/G0/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -nostdlib, -nostartfiles, -nodefaultlibs
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nostdlib \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK017 %s
// CHECK017: "{{.*}}clang{{.*}}" "-cc1"
// CHECK017-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK017-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK017-NOT: crt0_standalone.o
// CHECK017-NOT: crt0.o
// CHECK017-NOT: init.o
// CHECK017: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK017: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK017: "-L[[GNU_DIR]]/lib/gcc"
// CHECK017: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK017: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK017: "{{[^"]+}}.o"
// CHECK017-NOT: "-lstdc++"
// CHECK017-NOT: "-lm"
// CHECK017-NOT: "--start-group"
// CHECK017-NOT: "-lstandalone"
// CHECK017-NOT: "-lc"
// CHECK017-NOT: "-lgcc"
// CHECK017-NOT: "--end-group"
// CHECK017-NOT: fini.o

// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nostartfiles \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK018 %s
// CHECK018: "{{.*}}clang{{.*}}" "-cc1"
// CHECK018-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK018-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK018-NOT: crt0_standalone.o
// CHECK018-NOT: crt0.o
// CHECK018-NOT: init.o
// CHECK018: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK018: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK018: "-L[[GNU_DIR]]/lib/gcc"
// CHECK018: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK018: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK018: "{{[^"]+}}.o"
// CHECK018: "-lstdc++"
// CHECK018: "-lm"
// CHECK018: "--start-group"
// CHECK018: "-lstandalone"
// CHECK018: "-lc"
// CHECK018: "-lgcc"
// CHECK018: "--end-group"
// CHECK018-NOT: fini.o

// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -nodefaultlibs \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK019 %s
// CHECK019: "{{.*}}clang{{.*}}" "-cc1"
// CHECK019-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK019-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK019: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK019: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK019: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK019: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK019: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK019: "-L[[GNU_DIR]]/lib/gcc"
// CHECK019: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK019: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK019: "{{[^"]+}}.o"
// CHECK019-NOT: "-lstdc++"
// CHECK019-NOT: "-lm"
// CHECK019-NOT: "--start-group"
// CHECK019-NOT: "-lstandalone"
// CHECK019-NOT: "-lc"
// CHECK019-NOT: "-lgcc"
// CHECK019-NOT: "--end-group"
// CHECK019: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -moslib
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -moslib=first -moslib=second \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK020 %s
// CHECK020: "{{.*}}clang{{.*}}" "-cc1"
// CHECK020-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK020-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK020-NOT: "-static"
// CHECK020-NOT: "-shared"
// CHECK020-NOT: crt0_standalone.o
// CHECK020: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK020: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK020: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK020: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK020: "-L[[GNU_DIR]]/lib/gcc"
// CHECK020: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK020: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK020: "{{[^"]+}}.o"
// CHECK020: "--start-group"
// CHECK020: "-lfirst" "-lsecond"
// CHECK020-NOT: "-lstandalone"
// CHECK020: "-lc" "-lgcc" "--end-group"
// CHECK020: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -moslib=first -moslib=second -moslib=standalone\
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK021 %s
// CHECK021: "{{.*}}clang{{.*}}" "-cc1"
// CHECK021-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK021-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK021-NOT: "-static"
// CHECK021-NOT: "-shared"
// CHECK021: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK021: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK021: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK021: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK021: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK021: "-L[[GNU_DIR]]/lib/gcc"
// CHECK021: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK021: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK021: "{{[^"]+}}.o"
// CHECK021: "--start-group"
// CHECK021: "-lfirst" "-lsecond"
// CHECK021: "-lstandalone"
// CHECK021: "-lc" "-lgcc" "--end-group"
// CHECK021: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Other args to pass to linker
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -ccc-cxx -x c++ -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -s \
// RUN:   -Tbss 0xdead -Tdata 0xbeef -Ttext 0xcafe \
// RUN:   -t \
// RUN:   -e start_here \
// RUN:   -uFoo -undefined Bar \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK022 %s
// CHECK022: "{{.*}}clang{{.*}}" "-cc1"
// CHECK022-NEXT: "[[GNU_DIR:.*]]/bin/hexagon-as"{{.*}}
// CHECK022-NEXT: "[[GNU_DIR]]/bin/hexagon-ld"
// CHECK022: "[[GNU_DIR]]/hexagon/lib/v4/crt0_standalone.o"
// CHECK022: "[[GNU_DIR]]/hexagon/lib/v4/crt0.o"
// CHECK022: "[[GNU_DIR]]/hexagon/lib/v4/init.o"
// CHECK022: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0/v4"
// CHECK022: "-L[[GNU_DIR]]/lib/gcc/hexagon/4.4.0"
// CHECK022: "-L[[GNU_DIR]]/lib/gcc"
// CHECK022: "-L[[GNU_DIR]]/hexagon/lib/v4"
// CHECK022: "-L[[GNU_DIR]]/hexagon/lib"
// CHECK022: "-Tbss" "0xdead" "-Tdata" "0xbeef" "-Ttext" "0xcafe"
// CHECK022: "-s"
// CHECK022: "-t"
// CHECK022: "-u" "Foo" "-undefined" "Bar"
// CHECK022: "{{[^"]+}}.o"
// CHECK022: "-lstdc++" "-lm"
// CHECK022: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK022: "[[GNU_DIR]]/hexagon/lib/v4/fini.o"
