// Check that Windows uses LLVMgold.dll.
// REQUIRES: system-windows
// RUN: %clang -target x86_64-unknown-linux -### %s -flto 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-LTO-PLUGIN %s
//
// CHECK-LTO-PLUGIN: "-plugin" "{{.*}}/LLVMgold.dll"
