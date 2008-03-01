; This entire chain of computation should be optimized away, but
; wasn't because the two multiplies were not detected as being identical.
;
; RUN: llvm-as < %s | opt -gcse -instcombine -dce | \
; RUN:    llvm-dis | not grep sub

define i32 @vnum_test4(i32* %data) {
        %idx1 = getelementptr i32* %data, i64 1         ; <i32*> [#uses=1]
        %idx2 = getelementptr i32* %data, i64 3         ; <i32*> [#uses=1]
        %reg1101 = load i32* %idx1              ; <i32> [#uses=2]
        %reg1111 = load i32* %idx2              ; <i32> [#uses=2]
        %reg109 = mul i32 %reg1101, %reg1111            ; <i32> [#uses=1]
        %reg108 = mul i32 %reg1111, %reg1101            ; <i32> [#uses=1]
        %reg121 = sub i32 %reg108, %reg109              ; <i32> [#uses=1]
        ret i32 %reg121
}

