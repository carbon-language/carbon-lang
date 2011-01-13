; RUN: not llvm-as  %s -o /dev/null 2>%t
; RUN: FileCheck -input-file=%t %s

@foo = external unnamed_addr global i8*

// CHECK: error: only definitions can have unnamed_addr
// CHECK: @foo = external unnamed_addr global i8*
// CHECK:                 ^
