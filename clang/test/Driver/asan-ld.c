// Test AddressSanitizer ld flags.

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX %s
//
// CHECK-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX-NOT: "-lc"
// CHECK-LINUX: libclang_rt.asan-i386.a"
// CHECK-LINUX: "-lpthread"
// CHECK-LINUX: "-ldl"
// CHECK-LINUX: "-export-dynamic"

// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-CXX %s
//
// CHECK-LINUX-CXX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX-CXX-NOT: "-lc"
// CHECK-LINUX-CXX: libclang_rt.asan-i386.a"
// CHECK-LINUX-CXX: "-lpthread"
// CHECK-LINUX-CXX: "-ldl"
// CHECK-LINUX-CXX: "-export-dynamic"
// CHECK-LINUX-CXX: stdc++

// RUN: %clang -no-canonical-prefixes %s -### -o /dev/null -fsanitize=address \
// RUN:     -target i386-unknown-linux --sysroot=%S/Inputs/basic_linux_tree \
// RUN:     -lstdc++ -static 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX-CXX-STATIC %s
//
// CHECK-LINUX-CXX-STATIC: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX-CXX-STATIC-NOT: stdc++
// CHECK-LINUX-CXX-STATIC: libclang_rt.asan-i386.a"
// CHECK-LINUX-CXX-STATIC: stdc++

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
//
// CHECK-ANDROID: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-NOT: "-lc"
// CHECK-ANDROID: libclang_rt.asan-arm-android.so"
// CHECK-ANDROID-NOT: "-lpthread"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -fsanitize=address \
// RUN:     --sysroot=%S/Inputs/basic_android_tree/sysroot \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SHARED %s
//
// CHECK-ANDROID-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-SHARED-NOT: "-lc"
// CHECK-ANDROID-SHARED: libclang_rt.asan-arm-android.so"
// CHECK-ANDROID-SHARED-NOT: "-lpthread"
