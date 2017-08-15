// Check that non-Windows, non-Darwin OSs use LLVMgold.so.
// REQUIRES: !system-darwin && !system-windows
// RUN: %clang -target x86_64-unknown-linux -### %s -flto 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LTO-PLUGIN %s
//
// CHECK-LTO-PLUGIN: "-plugin" "{{.*}}/LLVMgold.so"
