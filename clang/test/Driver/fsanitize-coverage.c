// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=0 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-0
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address                       %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-0
// CHECK-SANITIZE-COVERAGE-0-NOT: fsanitize-coverage-type
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=memory -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=leak -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=undefined -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=bool -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=dataflow -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-1
// CHECK-SANITIZE-COVERAGE-1: fsanitize-coverage-type=1
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=4 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-4
// CHECK-SANITIZE-COVERAGE-4: fsanitize-coverage-type=3
// CHECK-SANITIZE-COVERAGE-4: fsanitize-coverage-indirect-calls
// RUN: %clang -target x86_64-linux-gnu -fsanitize=address -fsanitize-coverage=5 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-5
// CHECK-SANITIZE-COVERAGE-5: error: invalid value '5' in '-fsanitize-coverage=5'
// RUN: %clang -target x86_64-linux-gnu -fsanitize=thread   -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-UNUSED
// RUN: %clang -target x86_64-linux-gnu                     -fsanitize-coverage=1 %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-SANITIZE-COVERAGE-UNUSED
// CHECK-SANITIZE-COVERAGE-UNUSED: argument unused during compilation: '-fsanitize-coverage=1'

// RUN: %clang_cl -fsanitize=address -fsanitize-coverage=1 -c -### -- %s 2>&1 | FileCheck %s -check-prefix=CLANG-CL-COVERAGE
// CLANG-CL-COVERAGE-NOT: error:
// CLANG-CL-COVERAGE-NOT: warning:
// CLANG-CL-COVERAGE-NOT: argument unused
// CLANG-CL-COVERAGE-NOT: unknown argument
// CLANG-CL-COVERAGE: -fsanitize=address
// CLANG-CL-COVERAGE: -fsanitize-coverage-type=1
