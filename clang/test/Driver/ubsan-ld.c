// Test UndefinedBehaviorSanitizer ld flags.

// RUN: %clang -fsanitize=undefined %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX %s
// CHECK-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX-NOT: "-lc"
// CHECK-LINUX: libclang_rt.ubsan-i386.a"
// CHECK-LINUX: "-lpthread"

// RUN: %clang -fsanitize=bounds %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX1 %s
// CHECK-LINUX1: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX1-NOT: libclang_rt.ubsan-i386.a"
// CHECK-LINUX1-NOT: "-lpthread"
