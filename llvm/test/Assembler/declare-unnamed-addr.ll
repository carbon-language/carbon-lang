; RUN: not llvm-as  %s -o /dev/null 2>%t
; RUN: FileCheck -input-file=%t %s

declare unnamed_addr i32 @zed()

// CHECK: error: only definitions can have unnamed_addr
// CHECK: declare unnamed_addr i32 @zed()
// CHECK:         ^
