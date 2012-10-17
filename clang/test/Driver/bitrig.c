// RUN: %clang -no-canonical-prefixes -target amd64-pc-bitrig %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-C %s
// CHECK-LD-C: clang{{.*}}" "-cc1" "-triple" "amd64-pc-bitrig"
// CHECK-LD-C: ld{{.*}}" {{.*}} "-lc" "-lclang_rt.amd64"

// RUN: %clangxx -no-canonical-prefixes -target amd64-pc-bitrig %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-CXX %s
// CHECK-LD-CXX: clang{{.*}}" "-cc1" "-triple" "amd64-pc-bitrig" 
// CHECK-LD-CXX: ld{{.*}}" {{.*}} "-lstdc++" "-lm" "-lc" "-lclang_rt.amd64"

// RUN: %clangxx -stdlib=libc++ -no-canonical-prefixes -target amd64-pc-bitrig %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-CXX-STDLIB %s
// CHECK-LD-CXX-STDLIB: clang{{.*}}" "-cc1" "-triple" "amd64-pc-bitrig"
// CHECK-LD-CXX-STDLIB: ld{{.*}}" {{.*}} "-lc++" "-lcxxrt" "-lgcc" "-lm" "-lc" "-lclang_rt.amd64"
