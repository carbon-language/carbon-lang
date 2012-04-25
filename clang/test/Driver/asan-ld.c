// Test AddressSanitizer ld flags.

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target i386-unknown-linux -faddress-sanitizer \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LINUX %s
// CHECK-LINUX: "{{.*}}ld{{(.exe)?}}"
// CHECK-LINUX-NOT: "-lc"
// CHECK-LINUX: lib/linux/libclang_rt.asan-i386.a"
// CHECK-LINUX: "-lpthread"
// CHECK-LINUX: "-ldl"
// CHECK-LINUX: "-export-dynamic"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -faddress-sanitizer \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID %s
// CHECK-ANDROID: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-NOT: "-lc"
// CHECK-ANDROID: "-u" "__asan_preinit" "-lasan"
// CHECK-ANDROID: "-lasan_preload" "-ldl"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -target arm-linux-androideabi -faddress-sanitizer \
// RUN:     --sysroot=%S/Inputs/basic_android_tree \
// RUN:     -shared \
// RUN:   | FileCheck --check-prefix=CHECK-ANDROID-SHARED %s
// CHECK-ANDROID-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-ANDROID-SHARED-NOT: "-lc"
// CHECK-ANDROID-SHARED-NOT: "-lasan"
// CHECK-ANDROID-SHARED: "-lasan_preload" "-ldl"
