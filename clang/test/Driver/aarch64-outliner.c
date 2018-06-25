// REQUIRES: aarch64-registered-target
// RUN: %clang -target aarch64 -moutline -S %s -### 2>&1 | FileCheck %s -check-prefix=ON
// ON: "-mllvm" "-enable-machine-outliner"
// RUN: %clang -target aarch64 -moutline -mno-outline -S %s -### 2>&1 | FileCheck %s -check-prefix=OFF
// OFF-NOT: "-mllvm" "-enable-machine-outliner"
