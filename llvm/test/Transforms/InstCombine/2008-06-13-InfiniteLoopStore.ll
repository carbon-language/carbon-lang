; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {store i32} | count 2
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

@g_139 = global i32 0           ; <i32*> [#uses=2]

define void @func_56(i32 %p_60) nounwind  {
entry:
        store i32 1, i32* @g_139, align 4
        %tmp1 = icmp ne i32 %p_60, 0            ; <i1> [#uses=1]
        %tmp12 = zext i1 %tmp1 to i8            ; <i8> [#uses=1]
        %toBool = icmp ne i8 %tmp12, 0          ; <i1> [#uses=1]
        br i1 %toBool, label %bb, label %return

bb:             ; preds = %bb, %entry
        store i32 1, i32* @g_139, align 4
        br label %bb

return:         ; preds = %entry
        ret void
}

