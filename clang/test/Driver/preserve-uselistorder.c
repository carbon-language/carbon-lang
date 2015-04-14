// RUN: %clang -target x86_64-apple-darwin -emit-llvm -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-emit-llvm-bc"
// CHECK: "-preserve-bc-uselistorder"

// RUN: %clang -target x86_64-apple-darwin -flto -arch x86_64 %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=LTO %s
// LTO:      "-emit-llvm-bc"
// LTO-NOT:  "-preserve-bc-uselistorder"
