; RUN:  llvm-dis < %s.bc| FileCheck %s
; RUN:  llvm-uselistorder < %s.bc -preserve-bc-use-list-order -num-shuffles=5

; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
