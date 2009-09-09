; RUN: llc < %s -march=thumb -mattr=+thumb2 | \
; RUN:   grep smulbt | count 1
; RUN: llc < %s -march=thumb -mattr=+thumb2 | \
; RUN:   grep smultt | count 1

@x = weak global i16 0          ; <i16*> [#uses=1]
@y = weak global i16 0          ; <i16*> [#uses=0]

define i32 @f1(i32 %y) {
        %tmp = load i16* @x             ; <i16> [#uses=1]
        %tmp1 = add i16 %tmp, 2         ; <i16> [#uses=1]
        %tmp2 = sext i16 %tmp1 to i32           ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp2, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f2(i32 %x, i32 %y) {
        %tmp1 = ashr i32 %x, 16         ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp3, %tmp1            ; <i32> [#uses=1]
        ret i32 %tmp4
}
