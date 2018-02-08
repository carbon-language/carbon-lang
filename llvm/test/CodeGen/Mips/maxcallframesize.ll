; RUN: llc < %s -mtriple=mips-unknown-linux -stop-before=prologepilog | FileCheck %s

; Test that maxCallFrameSize is being computed early on.

@glob = external global i32*

declare void @bar(i32*, [20000 x i8]* byval)

define void @foo([20000 x i8]* %addr) {
  %tmp = alloca [4 x i32], align 32
  %tmp0 = getelementptr [4 x i32], [4 x i32]* %tmp, i32 0, i32 0
  call void @bar(i32* %tmp0, [20000 x i8]* byval %addr)
  ret void
}

; CHECK: adjustsStack:    true
; CHECK: maxCallFrameSize: 20008
