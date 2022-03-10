// RUN: %clangxx -target x86_64-linux-gnu -fmemory-profile %s -### 2>&1 | FileCheck %s
// RUN: %clangxx -target x86_64-linux-gnu -fmemory-profile=foo %s -### 2>&1 | FileCheck %s --check-prefix=DIR
// RUN: %clangxx -target x86_64-linux-gnu -fmemory-profile -fno-memory-profile %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: %clangxx -target x86_64-linux-gnu -fmemory-profile=foo -fno-memory-profile %s -### 2>&1 | FileCheck %s --check-prefix=OFF
// CHECK: "-cc1" {{.*}} "-fmemory-profile"
// CHECK: ld{{.*}}libclang_rt.memprof{{.*}}libclang_rt.memprof_cxx
// DIR: "-cc1" {{.*}} "-fmemory-profile=foo"
// DIR: ld{{.*}}libclang_rt.memprof{{.*}}libclang_rt.memprof_cxx
// OFF-NOT: "-fmemory-profile"
// OFF-NOT: libclang_rt.memprof
