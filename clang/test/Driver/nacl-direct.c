// Test clang changes for NaCl Support including:
//    include paths, library paths, emulation, default static
//
// RUN: %clang -### -o %t.o %s 2>&1 \
// RUN:     -target i686-unknown-nacl \
// RUN:   | FileCheck --check-prefix=CHECK-I686 %s
// CHECK-I686: {{.*}}clang{{.*}}" "-cc1"
// CHECK-I686: "-fuse-init-array"
// CHECK-I686: "-target-cpu" "pentium4"
// CHECK-I686: "-resource-dir" "{{.*}}/lib/clang/[[VER:[0-9.]+]]"
// CHECK-I686: "-internal-isystem" "{{.*}}/../lib/clang/[[VER]]/include"
// CHECK-I686: "-internal-isystem" "{{.*}}/../x86_64-nacl/usr/include"
// CHECK-I686: "-internal-isystem" "{{.*}}/../x86_64-nacl/include"
// CHECK-I686: /as" "--32"
// CHECK-I686: /ld"
// CHECK-I686: "--build-id"
// CHECK-I686: "-m" "elf_i386_nacl"
// CHECK-I686: "-static"
// CHECK-I686: "-L{{.*}}/../x86_64-nacl/lib32"
// CHECK-I686: "-L{{.*}}/../x86_64-nacl/usr/lib32"
// CHECK-I686: "-L{{.*}}/../lib/clang/[[VER]]/lib/i686-nacl"
// CHECK-I686-NOT: -lpthread
//
// RUN: %clang -### -o %t.o %s 2>&1 \
// RUN:     -target x86_64-unknown-nacl \
// RUN:   | FileCheck --check-prefix=CHECK-x86_64 %s
// CHECK-x86_64: {{.*}}clang{{.*}}" "-cc1"
// CHECK-x86_64: "-fuse-init-array"
// CHECK-x86_64: "-target-cpu" "x86-64"
// CHECK-x86_64: "-resource-dir" "{{.*}}/lib/clang/[[VER:[0-9.]+]]"
// CHECK-x86_64: "-internal-isystem" "{{.*}}/../lib/clang/[[VER]]/include"
// CHECK-x86_64: "-internal-isystem" "{{.*}}/../x86_64-nacl/usr/include"
// CHECK-x86_64: "-internal-isystem" "{{.*}}/../x86_64-nacl/include"
// CHECK-x86_64: /as" "--64"
// CHECK-x86_64: /ld"
// CHECK-x86_64: "--build-id"
// CHECK-x86_64: "-m" "elf_x86_64_nacl"
// CHECK-x86_64: "-static"
// CHECK-x86_64: "-L{{.*}}/../x86_64-nacl/lib"
// CHECK-x86_64: "-L{{.*}}/../x86_64-nacl/usr/lib"
// CHECK-x86_64: "-L{{.*}}/../lib/clang/[[VER]]/lib/x86_64-nacl"
// CHECK-X86_64-NOT: -lpthread
//
// RUN: %clang -### -o %t.o %s 2>&1 \
// RUN:     -target armv7a-unknown-nacl-gnueabihf \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARM: "-fuse-init-array"
// CHECK-ARM: "-target-cpu" "cortex-a8"
// CHECK-ARM: "-target-abi" "aapcs-linux"
// CHECK-ARM: "-mfloat-abi" "hard"
// CHECK-ARM: "-resource-dir" "{{.*}}/lib/clang/[[VER:[0-9.]+]]"
// CHECK-ARM: "-internal-isystem" "{{.*}}/../lib/clang/[[VER]]/include"
// CHECK-ARM: "-internal-isystem" "{{.*}}/../arm-nacl/usr/include"
// CHECK-ARM: "-internal-isystem" "{{.*}}/../arm-nacl/include"
// CHECK-ARM: /as"
// CHECK-ARM: /ld"
// CHECK-ARM: "--build-id"
// CHECK-ARM: "-m" "armelf_nacl"
// CHECK-ARM: "-static"
// CHECK-ARM: "-L{{.*}}/../arm-nacl/lib"
// CHECK-ARM: "-L{{.*}}/../arm-nacl/usr/lib"
// CHECK-ARM: "-L{{.*}}/../lib/clang/[[VER]]/lib/arm-nacl"
// CHECK-ARM-NOT: -lpthread

// Check that even when the target arch is just "arm" (as will be the case when
// it is inferred from the binary name) that we get the right ABI flags
// RUN: %clang -### -o %t.o %s 2>&1 \
// RUN:     -target arm-nacl \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-NOV7 %s
// CHECK-ARM-NOV7: "-triple" "armv7--nacl-gnueabihf"
// CHECK-ARM-NOV7: "-target-abi" "aapcs-linux"
// CHECK-ARM-NOV7: "-mfloat-abi" "hard"

// Test clang c++ include dirs and link line when using clang++

// RUN: %clangxx -### -o %t.o %s 2>&1 \
// RUN:     -target armv7a-unknown-nacl-gnueabihf \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-CXX %s
// CHECK-ARM-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-ARM-CXX: "-resource-dir" "{{.*}}/lib/clang/[[VER:[0-9.]+]]"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}/../arm-nacl/include/c++/v1"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}/../lib/clang/[[VER]]/include"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}/../arm-nacl/usr/include"
// CHECK-ARM-CXX: "-internal-isystem" "{{.*}}/../arm-nacl/include"
// CHECK-ARM-CXX: "-lpthread"

// RUN: %clangxx -### -o %t.o %s 2>&1 \
// RUN:     -target i686-unknown-nacl \
// RUN:   | FileCheck --check-prefix=CHECK-I686-CXX %s
// CHECK-I686-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-I686-CXX: "-resource-dir" "{{.*}}/lib/clang/[[VER:[0-9.]+]]"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}/../x86_64-nacl/include/c++/v1"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}/../lib/clang/[[VER]]/include"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}/../x86_64-nacl/usr/include"
// CHECK-I686-CXX: "-internal-isystem" "{{.*}}/../x86_64-nacl/include"
// CHECK-I686-CXX: "-lpthread"

// RUN: %clangxx -### -o %t.o %s 2>&1 \
// RUN:     -target x86_64-unknown-nacl \
// RUN:   | FileCheck --check-prefix=CHECK-x86_64-CXX %s
// CHECK-x86_64-CXX: {{.*}}clang{{.*}}" "-cc1"
// CHECK-x86_64-CXX: "-resource-dir" "{{.*}}/lib/clang/[[VER:[0-9.]+]]"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}/../x86_64-nacl/include/c++/v1"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}/../lib/clang/[[VER]]/include"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}/../x86_64-nacl/usr/include"
// CHECK-x86_64-CXX: "-internal-isystem" "{{.*}}/../x86_64-nacl/include"
// CHECK-x86_64-CXX: "-lpthread"
