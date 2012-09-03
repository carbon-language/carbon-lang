// Test header and library paths when Clang is used with Android standalone
// toolchain.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi \
// RUN:     -B%S/Inputs/basic_android_tree \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck  %s
// CHECK: {{.*}}clang{{.*}}" "-cc1"
// CHECK: "-internal-isystem" "{{.*}}/arm-linux-androideabi/include/c++/4.4.3"
// CHECK: "-internal-isystem" "{{.*}}/arm-linux-androideabi/include/c++/4.4.3/arm-linux-androideabi"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/include"
// CHECK: "-internal-externc-isystem" "{{.*}}/sysroot/usr/include"
// CHECK: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.4.3"
// CHECK: "-L{{.*}}/lib/gcc/arm-linux-androideabi/4.4.3/../../../../arm-linux-androideabi/lib"
// CHECK: "-L{{.*}}/sysroot/usr/lib"
