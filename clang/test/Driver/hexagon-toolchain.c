// REQUIRES: hexagon-registered-target

// Tests disabled for now in non-Unix-like systems where we can't seem to find hexagon-as
// XFAIL: mingw32,win32

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
// CHECK007-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-gcc"{{.*}} "-mv3"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -mcpu=hexagonv5 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK008 %s
// CHECK008: "-cc1" {{.*}} "-target-cpu" "hexagonv5"
// CHECK008-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v5"
// CHECK008-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-gcc"{{.*}} "-mv5"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   -mv2 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK009 %s
// CHECK009: "-cc1" {{.*}} "-target-cpu" "hexagonv2"
// CHECK009-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v2"
// CHECK009-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-gcc"{{.*}} "-mv2"

// RUN: %clang -### -target hexagon-unknown-linux     \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK010 %s
// CHECK010: "-cc1" {{.*}} "-target-cpu" "hexagonv4"
// CHECK010-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-as"{{.*}} "-march=v4"
// CHECK010-NEXT: "{{.*}}/Inputs/hexagon_tree/qc/bin/../../gnu/bin/hexagon-gcc"{{.*}} "-mv4"
