; RUN:  llvm-dis < %s.bc| FileCheck %s
; RUN:  verify-uselistorder < %s.bc -preserve-bc-use-list-order

; CHECK: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer
