; RUN:  llvm-dis < %s.bc| FileCheck %s

; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
