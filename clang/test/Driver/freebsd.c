// REQUIRES: ppc32-registered-target,ppc64-registered-target
// RUN: %clang -ccc-clang-archs powerpc -no-canonical-prefixes \
// RUN:   -target powerpc-pc-freebsd8 %s    \
// RUN:   --sysroot=%S/Inputs/basic_freebsd_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC %s
// CHECK-PPC: clang{{.*}}" "-cc1" "-triple" "powerpc-pc-freebsd8"
// CHECK-PPC: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-PPC: "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "-L[[SYSROOT]]/usr/lib" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"
//
// RUN: %clang  -ccc-clang-archs powerpc64 -no-canonical-prefixes \
// RUN:   -target powerpc64-pc-freebsd8 %s                              \
// RUN:   --sysroot=%S/Inputs/basic_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64 %s
// CHECK-PPC64: clang{{.*}}" "-cc1" "-triple" "powerpc64-pc-freebsd8"
// CHECK-PPC64: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-PPC64: "--eh-frame-hdr" "-dynamic-linker" "{{.*}}ld-elf{{.*}}" "-o" "a.out" "{{.*}}crt1.o" "{{.*}}crti.o" "{{.*}}crtbegin.o" "-L[[SYSROOT]]/usr/lib" "{{.*}}.o" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "{{.*}}crtend.o" "{{.*}}crtn.o"
//
//
// Check that -m32 properly adjusts the toolchain flags.
//
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIB32 %s
// CHECK-LIB32: clang{{.*}}" "-cc1" "-triple" "i386-pc-freebsd8"
// CHECK-LIB32: ld{{.*}}" {{.*}} "-m" "elf_i386_fbsd"
//
// RUN: %clang -target x86_64-pc-freebsd8 -m32 %s 2>&1 \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -print-search-dirs 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIB32PATHS %s
// CHECK-LIB32PATHS: libraries: ={{.*:?}}/usr/lib32
//
// Check that the new linker flags are passed to FreeBSD
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDFLAGS8 %s
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd9 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDFLAGS9 %s
// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd10.0 -m32 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LDFLAGS9 %s
// CHECK-LDFLAGS8-NOT: --hash-style=both
// CHECK-LDFLAGS8: --enable-new-dtags
// CHECK-LDFLAGS9: --hash-style=both
// CHECK-LDFLAGS9: --enable-new-dtags

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -static %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC: crt1.o
// CHECK-STATIC: crtbeginT.o

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -shared %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED: crti.o
// CHECK-SHARED: crtbeginS.o

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 -pie %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-PIE %s
// CHECK-PIE: pie
// CHECK-PIE: Scrt1.o
// CHECK-PIE: crtbeginS.o

// RUN: %clang -no-canonical-prefixes -target x86_64-pc-freebsd8 %s \
// RUN:   --sysroot=%S/Inputs/multiarch_freebsd64_tree -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NORMAL %s
// CHECK-NORMAL: crt1.o
// CHECK-NORMAL: crtbegin.o
