// RUN: %clang -target xcore -O1 -o - -emit-llvm -S %s | FileCheck %s

// CHECK: @g1 = global
int g1;
// CHECK: @g2 = common global i32 0, align 4
int g2 __attribute__((common));

// CHECK: define zeroext i8 @testchar()
// CHECK: ret i8 -1
char testchar (void) {
  return (char)-1;
}

// CHECK: "no-frame-pointer-elim"="false"
// CHECK: "no-frame-pointer-elim-non-leaf"="false"
