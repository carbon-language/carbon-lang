; PR2094
; RUN: llvm-as < %s | llc -march=x86-64 | grep movslq
; RUN: llvm-as < %s | llc -march=x86-64 | not grep movq

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

define i32 @sad16_sse2(i8* %v, i8* %blk2, i8* %blk1, i32 %stride, i32 %h) nounwind  {
entry:
	%tmp12 = sext i32 %stride to i64		; <i64> [#uses=1]
	%mrv = call {i32, i8*, i8*} asm sideeffect "$0 $1 $2 $3 $4 $5 $6",
         "=r,=r,=r,r,r,r,r"( i64 %tmp12, i32 %h, i8* %blk1, i8* %blk2 ) nounwind
        %tmp6 = getresult {i32, i8*, i8*} %mrv, 0
	%tmp7 = call i32 asm sideeffect "set $0",
             "=r,~{dirflag},~{fpsr},~{flags}"( ) nounwind
	ret i32 %tmp7
}
