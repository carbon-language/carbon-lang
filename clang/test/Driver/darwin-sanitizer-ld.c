// Test sanitizer link flags on Darwin.

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fsanitize=address %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ASAN %s

// CHECK-ASAN: "{{.*}}ld" 
// CHECK-ASAN: libclang_rt.asan_osx.a"
// CHECK-ASAN: "-lstdc++"
// CHECK-ASAN: "-framework" "CoreFoundation"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fPIC -shared -fsanitize=address %s -o %t.so 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DYN-ASAN %s

// CHECK-DYN-ASAN: "{{.*}}ld"
// CHECK-DYN-ASAN: "-dylib"
// CHECK-DYN-ASAN-NOT: libclang_rt.asan_osx.a
// CHECK-DYN-ASAN: "-undefined"
// CHECK-DYN-ASAN: "dynamic_lookup"
// CHECK-DYN-ASAN-NOT: libclang_rt.asan_osx.a

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fsanitize=undefined %s -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UBSAN %s

// CHECK-UBSAN: "{{.*}}ld" 
// CHECK-UBSAN: libclang_rt.ubsan_osx.a"
// CHECK-UBSAN: "-lstdc++"

// RUN: %clang -no-canonical-prefixes -### -target x86_64-darwin \
// RUN:   -fPIC -shared -fsanitize=undefined %s -o %t.so 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DYN-UBSAN %s

// CHECK-DYN-UBSAN: "{{.*}}ld"
// CHECK-DYN-UBSAN: "-dylib"
// CHECK-DYN-UBSAN-NOT: libclang_rt.ubsan_osx.a
// CHECK-DYN-UBSAN: "-undefined"
// CHECK-DYN-UBSAN: "dynamic_lookup"
// CHECK-DYN-UBSAN-NOT: libclang_rt.ubsan_osx.a
