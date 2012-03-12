// RUN: env LIBRARY_PATH=%T/test1 %clang -x c %s -### 2>&1 | FileCheck %s
// CHECK: "-L" "{{.*}}/test1"
