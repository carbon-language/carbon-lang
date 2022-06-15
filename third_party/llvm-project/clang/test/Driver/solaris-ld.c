// General tests that ld invocations on Solaris targets sane. Note that we use
// sysroot to make these tests independent of the host system.

// Check sparc-sun-solaris2.11, 32bit
// RUN: %clang -### %s 2>&1 --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC32 %s
// CHECK-LD-SPARC32-NOT: warning:
// CHECK-LD-SPARC32: "-cc1" "-triple" "sparc-sun-solaris2.11"
// CHECK-LD-SPARC32-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-SPARC32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crt1.o"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtbegin.o"
// CHECK-LD-SPARC32-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2"
// CHECK-LD-SPARC32-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../.."
// CHECK-LD-SPARC32-SAME: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-SPARC32-SAME: "-zignore" "-latomic" "-zrecord"
// CHECK-LD-SPARC32-SAME: "-lgcc_s"
// CHECK-LD-SPARC32-SAME: "-lc"
// CHECK-LD-SPARC32-SAME: "-lgcc"
// CHECK-LD-SPARC32-SAME: "-lm"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtend.o"
// CHECK-LD-SPARC32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crtn.o"

// Check sparc-sun-solaris2.11, 64bit
// RUN: %clang -m64 -### %s 2>&1 --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-SPARC64 %s
// CHECK-LD-SPARC64-NOT: warning:
// CHECK-LD-SPARC64: "-cc1" "-triple" "sparcv9-sun-solaris2.11"
// CHECK-LD-SPARC64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-SPARC64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crt1.o"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/lib/sparcv9{{/|\\\\}}crti.o"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crtbegin.o"
// CHECK-LD-SPARC64-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9"
// CHECK-LD-SPARC64-SAME: "-L[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../../sparcv9"
// CHECK-LD-SPARC64-SAME: "-L[[SYSROOT]]/usr/lib/sparcv9"
// CHECK-LD-SPARC64-NOT:  "-latomic"
// CHECK-LD-SPARC64-SAME: "-lgcc_s"
// CHECK-LD-SPARC64-SAME: "-lc"
// CHECK-LD-SPARC64-SAME: "-lgcc"
// CHECK-LD-SPARC64-SAME: "-lm"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/sparcv9{{/|\\\\}}crtend.o"
// CHECK-LD-SPARC64-SAME: "[[SYSROOT]]/usr/lib/sparcv9{{/|\\\\}}crtn.o"

// Check i386-pc-solaris2.11, 32bit
// RUN: %clang -### %s 2>&1 --target=i386-pc-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32 %s
// CHECK-LD-X32-NOT: warning:
// CHECK-LD-X32: "-cc1" "-triple" "i386-pc-solaris2.11"
// CHECK-LD-X32-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-X32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crt1.o"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crti.o"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4{{/|\\\\}}crtbegin.o"
// CHECK-LD-X32-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4"
// CHECK-LD-X32-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../.."
// CHECK-LD-X32-SAME: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-X32-NOT:  "-latomic"
// CHECK-LD-X32-SAME: "-lgcc_s"
// CHECK-LD-X32-SAME: "-lc"
// CHECK-LD-X32-SAME: "-lgcc"
// CHECK-LD-X32-SAME: "-lm"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4{{/|\\\\}}crtend.o"
// CHECK-LD-X32-SAME: "[[SYSROOT]]/usr/lib{{/|\\\\}}crtn.o"

// Check i386-pc-solaris2.11, 64bit
// RUN: %clang -m64 -### %s 2>&1 \
// RUN:     --target=i386-pc-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X64 %s
// CHECK-LD-X64-NOT: warning:
// CHECK-LD-X64: "-cc1" "-triple" "x86_64-pc-solaris2.11"
// CHECK-LD-X64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-X64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/lib/amd64{{/|\\\\}}crt1.o"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/lib/amd64{{/|\\\\}}crti.o"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64{{/|\\\\}}crtbegin.o"
// CHECK-LD-X64-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64"
// CHECK-LD-X64-SAME: "-L[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../../amd64"
// CHECK-LD-X64-SAME: "-L[[SYSROOT]]/usr/lib/amd64"
// CHECK-LD-X64-NOT:  "-latomic"
// CHECK-LD-X64-SAME: "-lgcc_s"
// CHECK-LD-X64-SAME: "-lc"
// CHECK-LD-X64-SAME: "-lgcc"
// CHECK-LD-X64-SAME: "-lm"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/amd64{{/|\\\\}}crtend.o"
// CHECK-LD-X64-SAME: "[[SYSROOT]]/usr/lib/amd64{{/|\\\\}}crtn.o"

// Check the right -l flags are present with -shared
// RUN: %clang -### %s -shared 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SPARC32-SHARED %s
// CHECK-SPARC32-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARC32-SHARED-SAME: "-lgcc_s"
// CHECK-SPARC32-SHARED-SAME: "-lc"
// CHECK-SPARC32-SHARED-NOT: "-lgcc"
// CHECK-SPARC32-SHARED-NOT: "-lm"

// -r suppresses default -l and crt*.o, values-*.o like -nostdlib.
// RUN: %clang -### %s --target=sparc-sun-solaris2.11 -r 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-RELOCATABLE
// CHECK-RELOCATABLE:     "-L
// CHECK-RELOCATABLE:     "-r"
// CHECK-RELOCATABLE-NOT: "-l
// CHECK-RELOCATABLE-NOT: /crt{{[^.]+}}.o
// CHECK-RELOCATABLE-NOT: /values-{{[^.]+}}.o
