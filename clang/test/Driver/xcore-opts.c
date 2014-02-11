// RUN: %clang -target xcore %s -g -Wl,L1Arg,L2Arg -Wa,A1Arg,A2Arg -fverbose-asm -### -o %t.o 2>&1 | FileCheck %s

// CHECK: "-nostdsysteminc"
// CHECK: "-momit-leaf-frame-pointer"
// CHECK-NOT: "-mdisable-fp-elim"
// CHECK: "-fno-signed-char"
// CHECK: "-fno-use-cxa-atexit"
// CHECK: "-fno-common"
// CHECH: xcc" "-o"
// CHECK: "-c" "-g" "-fverbose-asm" "A1Arg" "A2Arg"
// CHECK: xcc" "-o"
// CHECK: "L1Arg" "L2Arg"

