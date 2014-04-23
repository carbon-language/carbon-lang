// RUN: env INCLUDE=my_system_headers %clang -c %s -### 2>&1 | FileCheck %s
// RUN: env INCLUDE=my_system_headers %clang_cl -c %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK: "-internal-isystem" "my_system_headers"
