// Check that we split debug output properly
//
// REQUIRES: asserts
// RUN: %clang -target x86_64-unknown-linux-gnu -gsplit-dwarf -c -### %s 2> %t
// RUN: FileCheck -check-prefix=CHECK-ACTIONS < %t %s
//
// CHECK-ACTIONS: objcopy{{.*}}--extract-dwo{{.*}}"split-debug.dwo"
// CHECK-ACTIONS: objcopy{{.*}}--strip-dwo{{.*}}"split-debug.o"


