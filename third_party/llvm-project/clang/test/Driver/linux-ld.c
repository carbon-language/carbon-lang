// UNSUPPORTED: system-windows, default-pie-on-linux
// General tests that ld invocations on Linux targets sane. Note that we use
// sysroot to make these tests independent of the host system.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// CHECK-LD-32-NOT: warning:
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../i386-unknown-linux/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// CHECK-LD-64-NOT: warning:
// CHECK-LD-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64: "--eh-frame-hdr"
// CHECK-LD-64: "-m" "elf_x86_64"
// CHECK-LD-64: "-dynamic-linker"
// CHECK-LD-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64: "-L[[SYSROOT]]/lib"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-64: "-lc"
// CHECK-LD-64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X32 %s
// CHECK-LD-X32-NOT: warning:
// CHECK-LD-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-X32: "--eh-frame-hdr"
// CHECK-LD-X32: "-m" "elf32_x86_64"
// CHECK-LD-X32: "-dynamic-linker"
// CHECK-LD-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-X32: "-lc"
// CHECK-LD-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=compiler-rt \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RT %s
// CHECK-LD-RT-NOT: warning:
// CHECK-LD-RT: "-resource-dir" "[[RESDIR:[^"]*]]"
// CHECK-LD-RT: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RT: "--eh-frame-hdr"
// CHECK-LD-RT: "-m" "elf_x86_64"
// CHECK-LD-RT: "-dynamic-linker"
// CHECK-LD-RT: "[[RESDIR]]{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}clang_rt.crtbegin-x86_64.o"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-RT: "-L[[SYSROOT]]/lib"
// CHECK-LD-RT: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-RT: libclang_rt.builtins-x86_64.a"
// CHECK-LD-RT: "-lc"
// CHECK-LD-RT: libclang_rt.builtins-x86_64.a"
// CHECK-LD-RT: "[[RESDIR]]{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}clang_rt.crtend-x86_64.o"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-unknown-linux \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     --rtlib=compiler-rt \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RT-I686 %s
// CHECK-LD-RT-I686-NOT: warning:
// CHECK-LD-RT-I686: "-resource-dir" "[[RESDIR:[^"]*]]"
// CHECK-LD-RT-I686: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RT-I686: "--eh-frame-hdr"
// CHECK-LD-RT-I686: "-m" "elf_i386"
// CHECK-LD-RT-I686: "-dynamic-linker"
// CHECK-LD-RT-I686: "[[RESDIR]]{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}clang_rt.crtbegin-i386.o"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib/gcc/i686-unknown-linux/10.2.0"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib/gcc/i686-unknown-linux/10.2.0/../../../../i686-unknown-linux/lib"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/lib"
// CHECK-LD-RT-I686: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-RT-I686: libclang_rt.builtins-i386.a"
// CHECK-LD-RT-I686: "-lc"
// CHECK-LD-RT-I686: libclang_rt.builtins-i386.a"
// CHECK-LD-RT-I686: "[[RESDIR]]{{/|\\\\}}lib{{/|\\\\}}linux{{/|\\\\}}clang_rt.crtend-i386.o"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     --rtlib=compiler-rt \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RT-ANDROID %s
// CHECK-LD-RT-ANDROID-NOT: warning:
// CHECK-LD-RT-ANDROID: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RT-ANDROID: "--eh-frame-hdr"
// CHECK-LD-RT-ANDROID: "-m" "armelf_linux_eabi"
// CHECK-LD-RT-ANDROID: "-dynamic-linker"
// CHECK-LD-RT-ANDROID: libclang_rt.builtins-arm-android.a"
// CHECK-LD-RT-ANDROID: "-lc"
// CHECK-LD-RT-ANDROID: libclang_rt.builtins-arm-android.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GCC %s
// CHECK-LD-GCC-NOT: warning:
// CHECK-LD-GCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GCC: "--eh-frame-hdr"
// CHECK-LD-GCC: "-m" "elf_x86_64"
// CHECK-LD-GCC: "-dynamic-linker"
// CHECK-LD-GCC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-GCC: "-L[[SYSROOT]]/lib"
// CHECK-LD-GCC: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-GCC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GCC: "-lc"
// CHECK-LD-GCC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     -static-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64-STATIC-LIBGCC %s
// CHECK-LD-64-STATIC-LIBGCC-NOT: warning:
// CHECK-LD-64-STATIC-LIBGCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64-STATIC-LIBGCC: "--eh-frame-hdr"
// CHECK-LD-64-STATIC-LIBGCC: "-m" "elf_x86_64"
// CHECK-LD-64-STATIC-LIBGCC: "-dynamic-linker"
// CHECK-LD-64-STATIC-LIBGCC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/lib"
// CHECK-LD-64-STATIC-LIBGCC: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-64-STATIC-LIBGCC: "-lgcc" "-lgcc_eh"
// CHECK-LD-64-STATIC-LIBGCC: "-lc"
// CHECK-LD-64-STATIC-LIBGCC: "-lgcc" "-lgcc_eh"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-NO-LIBGCC %s
// CHECK-CLANG-NO-LIBGCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-NO-LIBGCC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-CLANG-NO-LIBGCC: "-lc"
// CHECK-CLANG-NO-LIBGCC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANGXX-NO-LIBGCC %s
// CHECK-CLANGXX-NO-LIBGCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANGXX-NO-LIBGCC: "-lgcc_s" "-lgcc"
// CHECK-CLANGXX-NO-LIBGCC: "-lc"
// CHECK-CLANGXX-NO-LIBGCC: "-lgcc_s" "-lgcc"
//
// RUN: %clang -static -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-NO-LIBGCC-STATIC %s
// CHECK-CLANG-NO-LIBGCC-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-NO-LIBGCC-STATIC: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "--end-group"
//
// RUN: %clang -static-pie -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-LD-STATIC-PIE %s
// CHECK-CLANG-LD-STATIC-PIE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-LD-STATIC-PIE: "-static"
// CHECK-CLANG-LD-STATIC-PIE: "-pie"
// CHECK-CLANG-LD-STATIC-PIE: "--no-dynamic-linker"
// CHECK-CLANG-LD-STATIC-PIE: "-z"
// CHECK-CLANG-LD-STATIC-PIE: "text"
// CHECK-CLANG-LD-STATIC-PIE: "-m" "elf_x86_64"
// CHECK-CLANG-LD-STATIC-PIE: "{{.*}}rcrt1.o"
// CHECK-CLANG-LD-STATIC-PIE: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "--end-group"
//
// RUN: %clang -static-pie -pie -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-LD-STATIC-PIE-PIE %s
// CHECK-CLANG-LD-STATIC-PIE-PIE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "-static"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "-pie"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "--no-dynamic-linker"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "-z"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "text"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "-m" "elf_x86_64"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "{{.*}}rcrt1.o"
// CHECK-CLANG-LD-STATIC-PIE-PIE: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "--end-group"
//
// RUN: %clang -static-pie -static -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-LD-STATIC-PIE-STATIC %s
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "-static"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "-pie"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "--no-dynamic-linker"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "-z"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "text"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "-m" "elf_x86_64"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "{{.*}}rcrt1.o"
// CHECK-CLANG-LD-STATIC-PIE-STATIC: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "--end-group"
//
// RUN: %clang -static-pie -nopie -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-LD-STATIC-PIE-NOPIE %s
// CHECK-CLANG-LD-STATIC-PIE-NOPIE: error: cannot specify 'nopie' along with 'static-pie'
//
// RUN: %clang -dynamic -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-NO-LIBGCC-DYNAMIC %s
// CHECK-CLANG-NO-LIBGCC-DYNAMIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-NO-LIBGCC-DYNAMIC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-CLANG-NO-LIBGCC-DYNAMIC: "-lc"
// CHECK-CLANG-NO-LIBGCC-DYNAMIC: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
//
// RUN: %clang -static-libgcc -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-STATIC-LIBGCC %s
// CHECK-CLANG-STATIC-LIBGCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-STATIC-LIBGCC: "-lgcc" "-lgcc_eh"
// CHECK-CLANG-STATIC-LIBGCC: "-lc"
// CHECK-CLANG-STATIC-LIBGCC: "-lgcc" "-lgcc_eh"
//
// RUN: %clang -static-libgcc -dynamic -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-STATIC-LIBGCC-DYNAMIC %s
// CHECK-CLANG-STATIC-LIBGCC-DYNAMIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-STATIC-LIBGCC-DYNAMIC: "-lgcc" "-lgcc_eh"
// CHECK-CLANG-STATIC-LIBGCC-DYNAMIC: "-lc"
// CHECK-CLANG-STATIC-LIBGCC-DYNAMIC: "-lgcc" "-lgcc_eh"
//
// RUN: %clang -shared-libgcc -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-SHARED-LIBGCC %s
// CHECK-CLANG-SHARED-LIBGCC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-SHARED-LIBGCC: "-lgcc_s" "-lgcc"
// CHECK-CLANG-SHARED-LIBGCC: "-lc"
// CHECK-CLANG-SHARED-LIBGCC: "-lgcc_s" "-lgcc"
//
// RUN: %clang -shared-libgcc -dynamic -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-SHARED-LIBGCC-DYNAMIC %s
// CHECK-CLANG-SHARED-LIBGCC-DYNAMIC: "-lgcc_s" "-lgcc"
// CHECK-CLANG-SHARED-LIBGCC-DYNAMIC: "-lc"
// CHECK-CLANG-SHARED-LIBGCC-DYNAMIC: "-lgcc_s" "-lgcc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-ANDROID-NONE %s
// CHECK-CLANG-ANDROID-NONE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-ANDROID-NONE: "-l:libunwind.a" "-ldl" "-lc"
//
// RUN: %clang -shared -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-ANDROID-SHARED %s
// CHECK-CLANG-ANDROID-SHARED: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-ANDROID-SHARED: "-l:libunwind.a" "-ldl" "-lc"
//
// RUN: %clang -static -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-CLANG-ANDROID-STATIC %s
// CHECK-CLANG-ANDROID-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-CLANG-ANDROID-STATIC: "--start-group" "{{[^"]*}}{{/|\\\\}}libclang_rt.builtins-aarch64-android.a" "-l:libunwind.a" "-lc" "--end-group"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1      \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     -static \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64-STATIC %s
// CHECK-LD-64-STATIC-NOT: warning:
// CHECK-LD-64-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64-STATIC: "--eh-frame-hdr"
// CHECK-LD-64-STATIC: "-m" "elf_x86_64"
// CHECK-LD-64-STATIC-NOT: "-dynamic-linker"
// CHECK-LD-64-STATIC: "-static"
// CHECK-LD-64-STATIC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbeginT.o"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/lib"
// CHECK-LD-64-STATIC: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-64-STATIC: "--start-group" "-lgcc" "-lgcc_eh" "-lc" "--end-group"

// RUN: %clang -### %s --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform -shared -static \
// RUN:   --gcc-toolchain= --sysroot=%S/Inputs/basic_linux_tree 2>&1 | FileCheck --check-prefix=CHECK-LD-SHARED-STATIC %s
// CHECK-LD-SHARED-STATIC: "-shared" "-static"
// CHECK-LD-SHARED-STATIC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbeginS.o"
// CHECK-LD-SHARED-STATIC: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtendS.o"

// Check that flags can be combined. The -static dominates.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform --unwindlib=platform \
// RUN:     -static-libgcc -static \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64-STATIC %s
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -rtlib=platform -m32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-32 %s
// CHECK-32-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../i386-unknown-linux/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../i386-unknown-linux/lib"
// CHECK-32-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -rtlib=platform -m64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-64 %s
// CHECK-32-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-64: "{{.*}}/usr/lib/gcc/i386-unknown-linux/10.2.0/64{{/|\\\\}}crtbegin.o"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../i386-unknown-linux/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/10.2.0/../../../../i386-unknown-linux/lib"
// CHECK-32-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform -m64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-64 %s
// CHECK-64-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=plaform -m32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-32 %s
// CHECK-64-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0/32{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X32 %s
// CHECK-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0/x32{{/|\\\\}}crtbegin.o"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/x32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../libx32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../libx32"
// CHECK-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-X32: "-L[[SYSROOT]]/lib"
// CHECK-X32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform -mx32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-X32 %s
// CHECK-64-TO-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-X32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0/x32{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/x32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-X32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -rtlib=platform -mx32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-X32 %s
// CHECK-32-TO-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-X32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0/x32{{/|\\\\}}crtbegin.o"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/x32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/lib/../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-X32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform -m64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X32-TO-64 %s
// CHECK-X32-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X32-TO-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-X32-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform -m32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X32-TO-32 %s
// CHECK-X32-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X32-TO-32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0/32{{/|\\\\}}crtbegin.o"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib/../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/10.2.0/../../../../x86_64-unknown-linux/lib"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-X32-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=platform -m32 \
// RUN:     --gcc-toolchain=%S/Inputs/multilib_64bit_linux_tree/usr \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-32-SYSROOT %s
// CHECK-64-TO-32-SYSROOT: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-32-SYSROOT: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/10.2.0/32{{/|\\\\}}crtbegin.o"
// CHECK-64-TO-32-SYSROOT: "-L{{[^"]*}}/Inputs/multilib_64bit_linux_tree/usr/lib/gcc/x86_64-unknown-linux/10.2.0/32"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-32-SYSROOT: "-L[[SYSROOT]]/usr/lib"
//
// Check that we support unusual patch version formats, including missing that
// component.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-unknown-linux -rtlib=platform -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing1/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION1 %s
// CHECK-GCC-VERSION1: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION1: "{{.*}}/Inputs/basic_linux_tree/usr/lib/gcc/i386-unknown-linux/10.2.0{{/|\\\\}}crtbegin.o"

// Test a simulated installation of libc++ on Linux, both through sysroot and
// the installation path of Clang.
// RUN: %clangxx -no-canonical-prefixes -x c++ %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-SYSROOT %s
// CHECK-BASIC-LIBCXX-SYSROOT: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
// CHECK-BASIC-LIBCXX-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BASIC-LIBCXX-SYSROOT: "--sysroot=[[SYSROOT]]"
// RUN: %clang -no-canonical-prefixes -x c++ %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-INSTALL %s
// CHECK-BASIC-LIBCXX-INSTALL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-INSTALL: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/bin/../include/c++/v1"
// CHECK-BASIC-LIBCXX-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BASIC-LIBCXX-INSTALL: "--sysroot=[[SYSROOT]]"
// CHECK-BASIC-LIBCXX-INSTALL: "-L[[SYSROOT]]/usr/bin/../lib"
//
// Test that we can use -stdlib=libc++ in a build system even when it
// occasionally links C code instead of C++ code.
// RUN: %clang -no-canonical-prefixes -x c %s -### -o %t.o 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-C-LINK %s
// CHECK-BASIC-LIBCXX-C-LINK-NOT: warning:
// CHECK-BASIC-LIBCXX-C-LINK: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-C-LINK: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-C-LINK-NOT: "-internal-isystem" "[[SYSROOT]]/usr/bin/../include/c++/v1"
// CHECK-BASIC-LIBCXX-C-LINK: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-BASIC-LIBCXX-C-LINK: "--sysroot=[[SYSROOT]]"
// CHECK-BASIC-LIBCXX-C-LINK: "-L[[SYSROOT]]/usr/bin/../lib"
//
// Check multi arch support on Ubuntu 12.04 LTS.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-unknown-linux-gnueabihf -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_12.04_LTS_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-12-04-ARM-HF %s
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/arm-linux-gnueabihf{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/arm-linux-gnueabihf{{/|\\\\}}crti.o"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabihf/4.6.3"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/lib/arm-linux-gnueabihf"
// CHECK-UBUNTU-12-04-ARM-HF: "-L[[SYSROOT]]/usr/lib/arm-linux-gnueabihf"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/gcc/arm-linux-gnueabihf/4.6.3{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-12-04-ARM-HF: "{{.*}}/usr/lib/arm-linux-gnueabihf{{/|\\\\}}crtn.o"
//
// Check Ubuntu 13.10 on x86-64 targeting arm-linux-gnueabihf.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabihf -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/x86-64_ubuntu_13.10 \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64-UBUNTU-13-10-ARM-HF %s
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-dynamic-linker" "{{(/usr/arm--linux-gnueabihf)?}}/lib/ld-linux-armhf.so.3"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib{{/|\\\\}}crti.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8{{/|\\\\}}crtbegin.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8{{/|\\\\}}crtend.o"
// CHECK-X86-64-UBUNTU-13-10-ARM-HF: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabihf/4.8/../../../../arm-linux-gnueabihf/lib/../lib{{/|\\\\}}crtn.o"
//
// Check Ubuntu 13.10 on x86-64 targeting arm-linux-gnueabi.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/x86-64_ubuntu_13.10 \
// RUN:   | FileCheck --check-prefix=CHECK-X86-64-UBUNTU-13-10-ARM %s
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-dynamic-linker" "{{(/usr/arm--linux-gnueabi)?}}/lib/ld-linux.so.3"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib{{/|\\\\}}crt1.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib{{/|\\\\}}crti.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7{{/|\\\\}}crtbegin.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabi/4.7"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/../lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "-L[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7{{/|\\\\}}crtend.o"
// CHECK-X86-64-UBUNTU-13-10-ARM: "{{.*}}/usr/lib/gcc-cross/arm-linux-gnueabi/4.7/../../../../arm-linux-gnueabi/lib/../lib{{/|\\\\}}crtn.o"
//
// Check Ubuntu 14.04 on powerpc64le.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-PPC64LE %s
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/powerpc64le-linux-gnu{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/powerpc64le-linux-gnu{{/|\\\\}}crti.o"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/lib/powerpc64le-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "-L[[SYSROOT]]/usr/lib/powerpc64le-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/gcc/powerpc64le-linux-gnu/4.8{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-14-04-PPC64LE: "{{.*}}/usr/lib/powerpc64le-linux-gnu{{/|\\\\}}crtn.o"
//
// Check Ubuntu 14.04 on x32.
// "/usr/lib/gcc/x86_64-linux-gnu/4.8/x32/crtend.o" "/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32/crtn.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-X32 %s
// CHECK-UBUNTU-14-04-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32{{/|\\\\}}crti.o"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/x32{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-14-04-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/x32"
// CHECK-UBUNTU-14-04-X32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32"
// CHECK-UBUNTU-14-04-X32-SAME: {{^}} "-L[[SYSROOT]]/lib/../libx32"
// CHECK-UBUNTU-14-04-X32-SAME: {{^}} "-L[[SYSROOT]]/usr/lib/../libx32"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/x32{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-14-04-X32: "{{.*}}/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../libx32{{/|\\\\}}crtn.o"
//
// Check fedora 18 on arm.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-unknown-linux-gnueabihf -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_18_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-18-ARM-HF %s
// CHECK-FEDORA-18-ARM-HF: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib{{/|\\\\}}crti.o"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2{{/|\\\\}}crtbegin.o"
// CHECK-FEDORA-18-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2"
// CHECK-FEDORA-18-ARM-HF: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2{{/|\\\\}}crtend.o"
// CHECK-FEDORA-18-ARM-HF: "{{.*}}/usr/lib/gcc/armv7hl-redhat-linux-gnueabi/4.7.2/../../../../lib{{/|\\\\}}crtn.o"
//
// Check Fedora 21 on AArch64.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_21_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-21-AARCH64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_21_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-21-AARCH64 %s
// CHECK-FEDORA-21-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0{{/|\\\\}}crtbegin.o"
// CHECK-FEDORA-21-AARCH64: "-L[[SYSROOT]]/usr/lib/gcc/aarch64-redhat-linux/4.9.0"
// CHECK-FEDORA-21-AARCH64: "-L[[SYSROOT]]/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0{{/|\\\\}}crtend.o"
// CHECK-FEDORA-21-AARCH64: "{{.*}}/usr/lib/gcc/aarch64-redhat-linux/4.9.0/../../../../lib64{{/|\\\\}}crtn.o"
//
// Check Fedora 31 on riscv64.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=riscv64-redhat-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/fedora_31_riscv64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FEDORA-31-RISCV64 %s
// CHECK-FEDORA-31-RISCV64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FEDORA-31-RISCV64: "{{.*}}/usr/lib/gcc/riscv64-redhat-linux/9/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-FEDORA-31-RISCV64: "{{.*}}/usr/lib/gcc/riscv64-redhat-linux/9{{/|\\\\}}crti.o"
// CHECK-FEDORA-31-RISCV64: "{{.*}}/usr/lib/gcc/riscv64-redhat-linux/9{{/|\\\\}}crtbegin.o"
// CHECK-FEDORA-31-RISCV64: "-L[[SYSROOT]]/usr/lib/gcc/riscv64-redhat-linux/9"
// CHECK-FEDORA-31-RISCV64: "-L[[SYSROOT]]/usr/lib/gcc/riscv64-redhat-linux/9/../../../../lib64"
// CHECK-FEDORA-31-RISCV64: "{{.*}}/usr/lib/gcc/riscv64-redhat-linux/9{{/|\\\\}}crtend.o"
// CHECK-FEDORA-31-RISCV64: "{{.*}}/usr/lib/gcc/riscv64-redhat-linux/9{{/|\\\\}}crtn.o"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-unknown-linux-gnueabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ubuntu_12.04_LTS_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-12-04-ARM %s
// CHECK-UBUNTU-12-04-ARM: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/arm-linux-gnueabi{{/|\\\\}}crt1.o"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/arm-linux-gnueabi{{/|\\\\}}crti.o"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1{{/|\\\\}}crtbegin.o"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/usr/lib/gcc/arm-linux-gnueabi/4.6.1"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/lib/arm-linux-gnueabi"
// CHECK-UBUNTU-12-04-ARM: "-L[[SYSROOT]]/usr/lib/arm-linux-gnueabi"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/gcc/arm-linux-gnueabi/4.6.1{{/|\\\\}}crtend.o"
// CHECK-UBUNTU-12-04-ARM: "{{.*}}/usr/lib/arm-linux-gnueabi{{/|\\\\}}crtn.o"
//
// Test the setup that shipped in SUSE 10.3 on ppc64.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-suse-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/suse_10.3_ppc64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-SUSE-10-3-PPC64 %s
// CHECK-SUSE-10-3-PPC64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-SUSE-10-3-PPC64: "{{.*}}/usr/lib/gcc/powerpc64-suse-linux/4.1.2/64{{/|\\\\}}crtbegin.o"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-suse-linux/4.1.2/64"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-suse-linux/4.1.2/../../../../lib64"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-SUSE-10-3-PPC64: "-L[[SYSROOT]]/usr/lib/../lib64"
//
// Check openSuse Leap 42.2 on AArch64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_42.2_aarch64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-42-2-AARCH64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_42.2_aarch64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-42-2-AARCH64 %s
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-42-2-AARCH64: "-L[[SYSROOT]]/usr/lib64/gcc/aarch64-suse-linux/4.8"
// CHECK-OPENSUSE-42-2-AARCH64: "-L[[SYSROOT]]/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-42-2-AARCH64: "{{.*}}/usr/lib64/gcc/aarch64-suse-linux/4.8/../../../../lib64{{/|\\\\}}crtn.o"
//
// Check openSUSE Tumbleweed on armv6hl
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv6hl-suse-linux-gnueabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv6hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV6HL %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv6hl-suse-linux-gnueabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv6hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV6HL %s
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "-L[[SYSROOT]]/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5"
// CHECK-OPENSUSE-TW-ARMV6HL: "-L[[SYSROOT]]/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-TW-ARMV6HL: "{{.*}}/usr/lib/gcc/armv6hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crtn.o"
//
// Check openSUSE Tumbleweed on armv7hl
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7hl-suse-linux-gnueabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv7hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV7HL %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7hl-suse-linux-gnueabi -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_armv7hl_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-ARMV7HL %s
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5"
// CHECK-OPENSUSE-TW-ARMV7HL: "-L[[SYSROOT]]/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-TW-ARMV7HL: "{{.*}}/usr/lib/gcc/armv7hl-suse-linux-gnueabi/5/../../../../lib{{/|\\\\}}crtn.o"
//
// Check openSUSE Tumbleweed on riscv64
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=riscv64-suse-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_riscv64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-RISCV64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=riscv64-suse-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_riscv64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-RISCV64 %s
// CHECK-OPENSUSE-TW-RISCV64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-TW-RISCV64: "{{.*}}/usr/lib64/gcc/riscv64-suse-linux/9/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-TW-RISCV64: "{{.*}}/usr/lib64/gcc/riscv64-suse-linux/9/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-TW-RISCV64: "{{.*}}/usr/lib64/gcc/riscv64-suse-linux/9{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-TW-RISCV64: "-L[[SYSROOT]]/usr/lib64/gcc/riscv64-suse-linux/9"
// CHECK-OPENSUSE-TW-RISCV64: "-L[[SYSROOT]]/usr/lib64/gcc/riscv64-suse-linux/9/../../../../lib64"
// CHECK-OPENSUSE-TW-RISCV64: "{{.*}}/usr/lib64/gcc/riscv64-suse-linux/9{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-TW-RISCV64: "{{.*}}/usr/lib64/gcc/riscv64-suse-linux/9/../../../../lib64{{/|\\\\}}crtn.o"
//
// Check openSUSE Tumbleweed on ppc
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/opensuse_tumbleweed_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OPENSUSE-TW-PPC %s
// CHECK-OPENSUSE-TW-PPC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OPENSUSE-TW-PPC: "{{.*}}/usr/lib{{/|\\\\}}crt1.o"
// CHECK-OPENSUSE-TW-PPC: "{{.*}}/usr/lib{{/|\\\\}}crti.o"
// CHECK-OPENSUSE-TW-PPC: "{{.*}}/usr/lib/gcc/powerpc64-suse-linux/9{{/|\\\\}}crtbegin.o"
// CHECK-OPENSUSE-TW-PPC: "-L[[SYSROOT]]/usr/lib/gcc/powerpc64-suse-linux/9"
// CHECK-OPENSUSE-TW-PPC: "{{.*}}/usr/lib/gcc/powerpc64-suse-linux/9{{/|\\\\}}crtend.o"
// CHECK-OPENSUSE-TW-PPC: "{{.*}}/usr/lib/crtn.o"
//
// Check dynamic-linker for different archs
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabi \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: "{{.*}}ld{{(.exe)?}}"
// CHECK-ARM: "-m" "armelf_linux_eabi"
// CHECK-ARM: "-dynamic-linker" "{{.*}}/lib/ld-linux.so.3"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabi -mfloat-abi=hard \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-ABIHF %s
// CHECK-ARM-ABIHF: "{{.*}}ld{{(.exe)?}}"
// CHECK-ARM-ABIHF: "-m" "armelf_linux_eabi"
// CHECK-ARM-ABIHF: "-dynamic-linker" "{{.*}}/lib/ld-linux-armhf.so.3"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-gnueabihf \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-HF %s
// CHECK-ARM-HF: "{{.*}}ld{{(.exe)?}}"
// CHECK-ARM-HF: "-m" "armelf_linux_eabi"
// CHECK-ARM-HF: "-dynamic-linker" "{{.*}}/lib/ld-linux-armhf.so.3"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64 %s
// CHECK-PPC64: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64: "-m" "elf64ppc"
// CHECK-PPC64: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.1"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu -mabi=elfv1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64-ELFv1 %s
// CHECK-PPC64-ELFv1: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64-ELFv1: "-m" "elf64ppc"
// CHECK-PPC64-ELFv1: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.1"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-linux-gnu -mabi=elfv2 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64-ELFv2 %s
// CHECK-PPC64-ELFv2: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64-ELFv2: "-m" "elf64ppc"
// CHECK-PPC64-ELFv2: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE %s
// CHECK-PPC64LE: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64LE: "-m" "elf64lppc"
// CHECK-PPC64LE: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu -mabi=elfv1 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE-ELFv1 %s
// CHECK-PPC64LE-ELFv1: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64LE-ELFv1: "-m" "elf64lppc"
// CHECK-PPC64LE-ELFv1: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.1"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64le-linux-gnu -mabi=elfv2 \
// RUN:   | FileCheck --check-prefix=CHECK-PPC64LE-ELFv2 %s
// CHECK-PPC64LE-ELFv2: "{{.*}}ld{{(.exe)?}}"
// CHECK-PPC64LE-ELFv2: "-m" "elf64lppc"
// CHECK-PPC64LE-ELFv2: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld64.so.2"
//
// Check that we do not pass --hash-style=gnu or --hash-style=both to
// hexagon linux linker
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=hexagon-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-HEXAGON %s
// CHECK-HEXAGON: "{{.*}}{{hexagon-link|ld}}{{(.exe)?}}"
// CHECK-HEXAGON-NOT: "--hash-style={{gnu|both}}"
//
// Check that we do not pass --hash-style=gnu and --hash-style=both to linker
// and provide correct path to the dynamic linker and emulation mode when build
// for MIPS platforms.
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS %s
// CHECK-MIPS: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS: "-m" "elf32btsmip"
// CHECK-MIPS: "-dynamic-linker" "{{.*}}/lib/ld.so.1"
// CHECK-MIPS-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSEL %s
// CHECK-MIPSEL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPSEL: "-m" "elf32ltsmip"
// CHECK-MIPSEL: "-dynamic-linker" "{{.*}}/lib/ld.so.1"
// CHECK-MIPSEL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mipsel-linux-gnu -mnan=2008 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPSEL-NAN2008 %s
// CHECK-MIPSEL-NAN2008: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPSEL-NAN2008: "-m" "elf32ltsmip"
// CHECK-MIPSEL-NAN2008: "-dynamic-linker" "{{.*}}/lib/ld-linux-mipsn8.so.1"
// CHECK-MIPSEL-NAN2008-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mipsel-linux-gnu -mcpu=mips32r6 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS32R6EL %s
// CHECK-MIPS32R6EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS32R6EL: "-m" "elf32ltsmip"
// CHECK-MIPS32R6EL: "-dynamic-linker" "{{.*}}/lib/ld-linux-mipsn8.so.1"
// CHECK-MIPS32R6EL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64 %s
// CHECK-MIPS64: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64: "-m" "elf64btsmip"
// CHECK-MIPS64: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL %s
// CHECK-MIPS64EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL: "-m" "elf64ltsmip"
// CHECK-MIPS64EL: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64EL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-gnu -mnan=2008 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-NAN2008 %s
// CHECK-MIPS64EL-NAN2008: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-NAN2008: "-m" "elf64ltsmip"
// CHECK-MIPS64EL-NAN2008: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld-linux-mipsn8.so.1"
// CHECK-MIPS64EL-NAN2008-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-gnu -mcpu=mips64r6 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64R6EL %s
// CHECK-MIPS64R6EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64R6EL: "-m" "elf64ltsmip"
// CHECK-MIPS64R6EL: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld-linux-mipsn8.so.1"
// CHECK-MIPS64R6EL-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnu -mabi=n32 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-N32 %s
// CHECK-MIPS64-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64-N32: "-m" "elf32btsmipn32"
// CHECK-MIPS64-N32: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld.so.1"
// CHECK-MIPS64-N32-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -mabi=n32 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-N32 %s
// CHECK-MIPS64EL-N32: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-N32: "-m" "elf32ltsmipn32"
// CHECK-MIPS64EL-N32: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld.so.1"
// CHECK-MIPS64EL-N32-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-linux-gnu -mabi=n32 \
// RUN:   -mnan=2008 | FileCheck --check-prefix=CHECK-MIPS64EL-N32-NAN2008 %s
// CHECK-MIPS64EL-N32-NAN2008: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-N32-NAN2008: "-m" "elf32ltsmipn32"
// CHECK-MIPS64EL-N32-NAN2008: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld-linux-mipsn8.so.1"
// CHECK-MIPS64EL-N32-NAN2008-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64el-redhat-linux \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-REDHAT %s
// CHECK-MIPS64EL-REDHAT: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-REDHAT: "-m" "elf64ltsmip"
// CHECK-MIPS64EL-REDHAT: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64EL-REDHAT-NOT: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld-musl-mipsel.so.1"
// CHECK-MIPS64EL-REDHAT-NOT: "--hash-style={{gnu|both}}"

// Check that we pass --hash-style=both for pre-M Android versions and
// --hash-style=gnu for newer Android versions.
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-linux-android21 \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-HASH-STYLE-L %s
// CHECK-ANDROID-HASH-STYLE-L: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-HASH-STYLE-L: "--hash-style=both"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-linux-android23 \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-HASH-STYLE-M %s
// CHECK-ANDROID-HASH-STYLE-M: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-HASH-STYLE-M: "--hash-style=gnu"

// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-linux-android21 \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-WARN-SHARED-TEXTREL %s
// CHECK-ANDROID-WARN-SHARED-TEXTREL: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-WARN-SHARED-TEXTREL: "--warn-shared-textrel"

// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64-linux-gnuabin32 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-GNUABIN32 %s
// CHECK-MIPS64EL-GNUABIN32: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-GNUABIN32: "-m" "elf32btsmipn32"
// CHECK-MIPS64EL-GNUABIN32: "-dynamic-linker" "{{.*}}/lib{{(32)?}}/ld.so.1"
// CHECK-MIPS64EL-GNUABIN32-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 --target=mips64-linux-gnuabi64 \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-GNUABI64 %s
// CHECK-MIPS64EL-GNUABI64: "{{.*}}ld{{(.exe)?}}"
// CHECK-MIPS64EL-GNUABI64: "-m" "elf64btsmip"
// CHECK-MIPS64EL-GNUABI64: "-dynamic-linker" "{{.*}}/lib{{(64)?}}/ld.so.1"
// CHECK-MIPS64EL-GNUABI64-NOT: "--hash-style={{gnu|both}}"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=sparc-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-SPARCV8 %s
// CHECK-SPARCV8: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARCV8: "-m" "elf32_sparc"
// CHECK-SPARCV8: "-dynamic-linker" "{{(/usr/sparc-unknown-linux-gnu)?}}/lib/ld-linux.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=sparcel-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-SPARCV8EL %s
// CHECK-SPARCV8EL: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARCV8EL: "-m" "elf32_sparc"
// CHECK-SPARCV8EL: "-dynamic-linker" "{{(/usr/sparcel-unknown-linux-gnu)?}}/lib/ld-linux.so.2"
//
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=sparcv9-unknown-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-SPARCV9 %s
// CHECK-SPARCV9: "{{.*}}ld{{(.exe)?}}"
// CHECK-SPARCV9: "-m" "elf64_sparc"
// CHECK-SPARCV9: "-dynamic-linker" "{{(/usr/sparcv9-unknown-linux-gnu)?}}/lib{{(64)?}}/ld-linux.so.2"

// Test linker invocation on Android.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// CHECK-ANDROID: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID: "-z" "now"
// CHECK-ANDROID: "-z" "relro"
// CHECK-ANDROID: "--enable-new-dtags"
// CHECK-ANDROID: "{{.*}}{{/|\\\\}}crtbegin_dynamic.o"
// CHECK-ANDROID: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-NOT: "-lgcc_s"
// CHECK-ANDROID-NOT: "-lgcc"
// CHECK-ANDROID: "-l:libunwind.a"
// CHECK-ANDROID: "-ldl"
// CHECK-ANDROID-NOT: "-lgcc_s"
// CHECK-ANDROID-NOT: "-lgcc"
// CHECK-ANDROID: "{{.*}}{{/|\\\\}}crtend_android.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SO %s
// CHECK-ANDROID-SO: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID-SO-NOT: "-Bsymbolic"
// CHECK-ANDROID-SO: "{{.*}}{{/|\\\\}}crtbegin_so.o"
// CHECK-ANDROID-SO: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-SO-NOT: "-lgcc_s"
// CHECK-ANDROID-SO-NOT: "-lgcc"
// CHECK-ANDROID-SO: "-l:libunwind.a"
// CHECK-ANDROID-SO: "-ldl"
// CHECK-ANDROID-SO-NOT: "-lgcc_s"
// CHECK-ANDROID-SO-NOT: "-lgcc"
// CHECK-ANDROID-SO: "{{.*}}{{/|\\\\}}crtend_so.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -static \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-STATIC %s
// CHECK-ANDROID-STATIC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID-STATIC: "{{.*}}{{/|\\\\}}crtbegin_static.o"
// CHECK-ANDROID-STATIC: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-STATIC-NOT: "-lgcc_eh"
// CHECK-ANDROID-STATIC-NOT: "-lgcc"
// CHECK-ANDROID-STATIC: "-l:libunwind.a"
// CHECK-ANDROID-STATIC-NOT: "-ldl"
// CHECK-ANDROID-STATIC-NOT: "-lgcc_eh"
// CHECK-ANDROID-STATIC-NOT: "-lgcc"
// CHECK-ANDROID-STATIC: "{{.*}}{{/|\\\\}}crtend_android.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot  \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot  \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot  \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -rtlib=platform --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -pie \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PIE %s
// CHECK-ANDROID-PIE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ANDROID-PIE: "{{.*}}{{/|\\\\}}crtbegin_dynamic.o"
// CHECK-ANDROID-PIE: "-L[[SYSROOT]]/usr/lib"
// CHECK-ANDROID-PIE-NOT: "-lgcc_s"
// CHECK-ANDROID-PIE-NOT: "-lgcc"
// CHECK-ANDROID-PIE: "-l:libunwind.a"
// CHECK-ANDROID-PIE-NOT: "-lgcc_s"
// CHECK-ANDROID-PIE-NOT: "-lgcc"
// CHECK-ANDROID-PIE: "{{.*}}{{/|\\\\}}crtend_android.o"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-32 %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-64 %s
// CHECK-ANDROID-32: "-dynamic-linker" "/system/bin/linker"
// CHECK-ANDROID-64: "-dynamic-linker" "/system/bin/linker64"
//
// Test that -pthread does not add -lpthread on Android.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -pthread \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-androideabi -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-linux-android -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD %s
// CHECK-ANDROID-PTHREAD-NOT: -lpthread
//
// RUN: %clang -no-canonical-prefixes %t.o -### -o %t 2>&1 \
// RUN:     --target=arm-linux-androideabi -pthread \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-PTHREAD-LINK %s
// CHECK-ANDROID-PTHREAD-LINK-NOT: argument unused during compilation: '-pthread'
//
// Check linker invocation on Debian 6 MIPS 32/64-bit.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPSEL %s
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPSEL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/lib/../lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib/../lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPSEL: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL %s
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/64{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnu -rtlib=platform -mabi=n32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL-N32 %s
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib32{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib32{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "{{.*}}/usr/lib/gcc/mipsel-linux-gnu/4.4/n32{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/n32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/gcc/mipsel-linux-gnu/4.4/../../../../lib32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL-N32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnuabi64 -rtlib=platform -mabi=32 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL-O32 %s
// CHECK-DEBIAN-ML-MIPS64EL-O32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../../libo32{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../../libo32{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/32{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/32"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../../libo32"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "-L[[SYSROOT]]/libo32"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "-L[[SYSROOT]]/usr/libo32"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL-O32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-unknown-linux-gnu --rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64-GNUABI %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-linux-gnuabi64 -rtlib=platform -mabi=n64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64-GNUABI %s
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/mips64-linux-gnuabi64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/mips64-linux-gnuabi64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/lib/mips64-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib/mips64-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/gcc/mips64-linux-gnuabi64/4.9{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-ML-MIPS64-GNUABI: "{{.*}}/usr/lib/mips64-linux-gnuabi64{{/|\\\\}}crtn.o"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-unknown-linux-gnu -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL-GNUABI %s
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-linux-gnuabi64 -rtlib=platform -mabi=n64 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-ML-MIPS64EL-GNUABI %s
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/mips64el-linux-gnuabi64{{/|\\\\}}crt1.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/mips64el-linux-gnuabi64{{/|\\\\}}crti.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9{{/|\\\\}}crtbegin.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/lib/mips64el-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib/mips64el-linux-gnuabi64"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/lib"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "-L[[SYSROOT]]/usr/lib"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/gcc/mips64el-linux-gnuabi64/4.9{{/|\\\\}}crtend.o"
// CHECK-DEBIAN-ML-MIPS64EL-GNUABI: "{{.*}}/usr/lib/mips64el-linux-gnuabi64{{/|\\\\}}crtn.o"
//
// Test linker invocation for Freescale SDK (OpenEmbedded).
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc-fsl-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/freescale_ppc_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FSL-PPC %s
// CHECK-FSL-PPC: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FSL-PPC: "-m" "elf32ppclinux"
// CHECK-FSL-PPC: "{{.*}}{{/|\\\\}}crt1.o"
// CHECK-FSL-PPC: "{{.*}}{{/|\\\\}}crtbegin.o"
// CHECK-FSL-PPC: "-L[[SYSROOT]]/usr/lib"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-fsl-linux -rtlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/freescale_ppc64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-FSL-PPC64 %s
// CHECK-FSL-PPC64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-FSL-PPC64: "-m" "elf64ppc"
// CHECK-FSL-PPC64: "{{.*}}{{/|\\\\}}crt1.o"
// CHECK-FSL-PPC64: "{{.*}}{{/|\\\\}}crtbegin.o"
//
// Check that crtfastmath.o is linked with -ffast-math and with -Ofast.
// RUN: %clang --target=x86_64-unknown-linux -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -funsafe-math-optimizations\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast -O3\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -O3 -Ofast\
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -ffast-math -fno-fast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast -fno-fast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -Ofast -fno-unsafe-math-optimizations \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -fno-fast-math -Ofast  \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// RUN: %clang --target=x86_64-unknown-linux -### %s -fno-unsafe-math-optimizations -Ofast \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-CRTFASTMATH %s
// We don't have crtfastmath.o in the i386 tree, use it to check that file
// detection works.
// RUN: %clang --target=i386-unknown-linux -### %s -ffast-math \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOCRTFASTMATH %s
// CHECK-CRTFASTMATH: usr/lib/gcc/x86_64-unknown-linux/10.2.0{{/|\\\\}}crtfastmath.o
// CHECK-NOCRTFASTMATH-NOT: crtfastmath.o

// Check that we link in gcrt1.o when compiling with -pg
// RUN: %clang -pg --target=x86_64-unknown-linux -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>& 1 \
// RUN:   | FileCheck --check-prefix=CHECK-PG %s
// CHECK-PG: gcrt1.o

// GCC forwards -u to the linker.
// RUN: %clang -u asdf --target=x86_64-unknown-linux -### %s \
// RUN:        --gcc-toolchain="" \
// RUN:        --sysroot=%S/Inputs/basic_linux_tree 2>& 1 \
// RUN:   | FileCheck --check-prefix=CHECK-u %s
// CHECK-u: "-u" "asdf"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armeb-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMEB %s
// CHECK-ARMEB: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMEB-NOT: "--be8"
// CHECK-ARMEB: "-EB"
// CHECK-ARMEB: "-m" "armelfb_linux_eabi"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armebv7-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EB %s
// CHECK-ARMV7EB: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMV7EB: "--be8"
// CHECK-ARMV7EB: "-EB"
// CHECK-ARMV7EB: "-m" "armelfb_linux_eabi"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-unknown-linux \
// RUN:     -mbig-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EB %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armebv7-unknown-linux \
// RUN:     -mbig-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EB %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EL %s
// CHECK-ARMV7EL: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-ARMV7EL-NOT: "--be8"
// CHECK-ARMV7EL: "-EL"
// CHECK-ARMV7EL: "-m" "armelf_linux_eabi"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armebv7-unknown-linux \
// RUN:     -mlittle-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EL %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-unknown-linux \
// RUN:     -mlittle-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ARMV7EL %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64_be-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64BE %s
// CHECK-AARCH64BE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-AARCH64BE-NOT: "--be8"
// CHECK-AARCH64BE: "-EB"
// CHECK-AARCH64BE: "-m" "aarch64linuxb"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-unknown-linux \
// RUN:     -mbig-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64BE %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64_be-unknown-linux \
// RUN:     -mbig-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64BE %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-unknown-linux \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64LE %s
// CHECK-AARCH64LE: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-AARCH64LE-NOT: "--be8"
// CHECK-AARCH64LE: "-EL"
// CHECK-AARCH64LE: "-m" "aarch64linux"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64_be-unknown-linux \
// RUN:     -mlittle-endian \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-AARCH64LE %s

// Check dynamic-linker for musl-libc
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=i386-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-X86 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-X86_64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPS %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mipsel-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPSEL %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPS64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=mips64el-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-MIPS64EL %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-PPC %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpc64-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-PPC64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=powerpcspe-pc-linux-musl \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-PPCSPE %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumb-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARM %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumb-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbv7-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbeb-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEB %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbeb-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=thumbv7eb-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARM %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armeb-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEB %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armeb-pc-linux-musleabihf \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=armv7eb-pc-linux-musleabi -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-ARMEBHF %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-AARCH64 %s
// RUN: %clang %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64_be-pc-linux-musleabi \
// RUN:   | FileCheck --check-prefix=CHECK-MUSL-AARCH64_BE %s
// CHECK-MUSL-X86:        "-dynamic-linker" "/lib/ld-musl-i386.so.1"
// CHECK-MUSL-X86_64:     "-dynamic-linker" "/lib/ld-musl-x86_64.so.1"
// CHECK-MUSL-MIPS:       "-dynamic-linker" "/lib/ld-musl-mips.so.1"
// CHECK-MUSL-MIPSEL:     "-dynamic-linker" "/lib/ld-musl-mipsel.so.1"
// CHECK-MUSL-MIPS64:     "-dynamic-linker" "/lib/ld-musl-mips64.so.1"
// CHECK-MUSL-MIPS64EL:   "-dynamic-linker" "/lib/ld-musl-mips64el.so.1"
// CHECK-MUSL-PPC:        "-dynamic-linker" "/lib/ld-musl-powerpc.so.1"
// CHECK-MUSL-PPC64:      "-dynamic-linker" "/lib/ld-musl-powerpc64.so.1"
// CHECK-MUSL-PPCSPE:     "-dynamic-linker" "/lib/ld-musl-powerpc-sf.so.1"
// CHECK-MUSL-ARM:        "-dynamic-linker" "/lib/ld-musl-arm.so.1"
// CHECK-MUSL-ARMHF:      "-dynamic-linker" "/lib/ld-musl-armhf.so.1"
// CHECK-MUSL-ARMEB:      "-dynamic-linker" "/lib/ld-musl-armeb.so.1"
// CHECK-MUSL-ARMEBHF:    "-dynamic-linker" "/lib/ld-musl-armebhf.so.1"
// CHECK-MUSL-AARCH64:    "-dynamic-linker" "/lib/ld-musl-aarch64.so.1"
// CHECK-MUSL-AARCH64_BE: "-dynamic-linker" "/lib/ld-musl-aarch64_be.so.1"

// Check whether multilib gcc install works fine on Gentoo with gcc-config
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu -rtlib=platform --unwindlib=platform \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GENTOO %s
// CHECK-LD-GENTOO-NOT: warning:
// CHECK-LD-GENTOO: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GENTOO: "--eh-frame-hdr"
// CHECK-LD-GENTOO: "-m" "elf_x86_64"
// CHECK-LD-GENTOO: "-dynamic-linker"
// CHECK-LD-GENTOO: "{{.*}}/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3{{/|\\\\}}crtbegin.o"
// CHECK-LD-GENTOO: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3"
// CHECK-LD-GENTOO: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../../../x86_64-pc-linux-gnu/lib"
// CHECK-LD-GENTOO: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GENTOO: "-lc"
// CHECK-LD-GENTOO: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=i686-unknown-linux-gnu -rtlib=platform --unwindlib=platform \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GENTOO-32 %s
// CHECK-LD-GENTOO-32-NOT: warning:
// CHECK-LD-GENTOO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GENTOO-32: "--eh-frame-hdr"
// CHECK-LD-GENTOO-32: "-m" "elf_i386"
// CHECK-LD-GENTOO-32: "-dynamic-linker"
// CHECK-LD-GENTOO-32: "{{.*}}/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/32{{/|\\\\}}crtbegin.o"
// CHECK-LD-GENTOO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/32"
// CHECK-LD-GENTOO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../../../x86_64-pc-linux-gnu/lib"
// CHECK-LD-GENTOO-32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GENTOO-32: "-lc"
// CHECK-LD-GENTOO-32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnux32 -rtlib=platform --unwindlib=platform \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-LD-GENTOO-X32 %s
// CHECK-LD-GENTOO-X32-NOT: warning:
// CHECK-LD-GENTOO-X32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-GENTOO-X32: "--eh-frame-hdr"
// CHECK-LD-GENTOO-X32: "-m" "elf32_x86_64"
// CHECK-LD-GENTOO-X32: "-dynamic-linker"
// CHECK-LD-GENTOO-X32: "{{.*}}/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/x32{{/|\\\\}}crtbegin.o"
// CHECK-LD-GENTOO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/x32"
// CHECK-LD-GENTOO-X32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/../../../../x86_64-pc-linux-gnu/lib"
// CHECK-LD-GENTOO-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-GENTOO-X32: "-lc"
// CHECK-LD-GENTOO-X32: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux-gnu \
// RUN:     --gcc-toolchain="%S/Inputs/rhel_7_tree/opt/rh/devtoolset-7/root/usr" \
// RUN:     --sysroot=%S/Inputs/rhel_7_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-RHEL7-DTS %s
// CHECK-LD-RHEL7-DTS: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-RHLE7-DTS: Selected GCC installation: [[GCC_INSTALL:[[SYSROOT]]/lib/gcc/x86_64-redhat-linux/7]]
// CHECK-LD-RHEL7-DTS-NOT: /usr/bin/ld
// CHECK-LD-RHLE7-DTS: [[GCC_INSTALL]/../../../bin/ld

// Check whether gcc7 install works fine on Amazon Linux AMI
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-amazon-linux -rtlib=libgcc --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/ami_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-AMI %s
// CHECK-LD-AMI-NOT: warning:
// CHECK-LD-AMI: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-AMI: "--eh-frame-hdr"
// CHECK-LD-AMI: "-m" "elf_x86_64"
// CHECK-LD-AMI: "-dynamic-linker"
// CHECK-LD-AMI: "{{.*}}/usr/lib/gcc/x86_64-amazon-linux/7{{/|\\\\}}crtbegin.o"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-amazon-linux/7"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-amazon-linux/7/../../../../lib64"
// CHECK-LD-AMI: "-L[[SYSROOT]]/lib"
// CHECK-LD-AMI: "-L[[SYSROOT]]/usr/lib"
// CHECK-LD-AMI: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-LD-AMI: "-lc"
// CHECK-LD-AMI: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"

// Check whether the OpenEmbedded ARM libs are added correctly.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=arm-oe-linux-gnueabi -rtlib=libgcc --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/openembedded_arm_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OE-ARM %s

// CHECK-OE-ARM: "-cc1" "-triple" "armv4t-oe-linux-gnueabi"
// CHECK-OE-ARM: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OE-ARM: "-m" "armelf_linux_eabi" "-dynamic-linker"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../lib{{/|\\\\}}crt1.o"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../lib{{/|\\\\}}crti.o"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0{{/|\\\\}}crtbegin.o"
// CHECK-OE-ARM: "-L[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0"
// CHECK-OE-ARM: "-L[[SYSROOT]]/usr/lib"
// CHECK-OE-ARM: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0{{/|\\\\}}crtend.o"
// CHECK-OE-ARM: "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../lib{{/|\\\\}}crtn.o"

// Check whether the OpenEmbedded AArch64 libs are added correctly.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=aarch64-oe-linux -rtlib=libgcc --unwindlib=platform \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/openembedded_aarch64_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OE-AARCH64 %s

// CHECK-OE-AARCH64: "-cc1" "-triple" "aarch64-oe-linux"
// CHECK-OE-AARCH64: ld{{.*}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-OE-AARCH64: "-m" "aarch64linux" "-dynamic-linker"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../lib64{{/|\\\\}}crt1.o"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../lib64{{/|\\\\}}crti.o"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0{{/|\\\\}}crtbegin.o"
// CHECK-OE-AARCH64: "-L[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0"
// CHECK-OE-AARCH64: "-L[[SYSROOT]]/usr/lib64"
// CHECK-OE-AARCH64: "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0{{/|\\\\}}crtend.o"
// CHECK-OE-AARCH64: "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../lib64{{/|\\\\}}crtn.o"
