; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | grep srwi
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep rlwimi

define i32 @baz(i64 %a) {
        %tmp29 = lshr i64 %a, 24                ; <i64> [#uses=1]
        %tmp23 = trunc i64 %tmp29 to i32                ; <i32> [#uses=1]
        %tmp410 = lshr i32 %tmp23, 9            ; <i32> [#uses=1]
        %tmp45 = trunc i32 %tmp410 to i16               ; <i16> [#uses=1]
        %tmp456 = sext i16 %tmp45 to i32                ; <i32> [#uses=1]
        ret i32 %tmp456
}

