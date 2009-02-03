; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep extractelement

define void @get_image() nounwind {
entry:
        %0 = call i32 @fgetc(i8* null) nounwind               ; <i32> [#uses=1]
        %1 = trunc i32 %0 to i8         ; <i8> [#uses=1]
        %tmp2 = insertelement <100 x i8> zeroinitializer, i8 %1, i32 1          ; <<100 x i8>> [#uses=1]
        %tmp1 = extractelement <100 x i8> %tmp2, i32 0          ; <i8> [#uses=1]
        %2 = icmp eq i8 %tmp1, 80               ; <i1> [#uses=1]
        br i1 %2, label %bb2, label %bb3

bb2:            ; preds = %entry
        br label %bb3

bb3:            ; preds = %bb2, %entry
        unreachable
}

declare i32 @fgetc(i8*)
