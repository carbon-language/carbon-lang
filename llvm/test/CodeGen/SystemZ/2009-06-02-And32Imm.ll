; RUN: llc < %s -march=systemz | grep nilf | count 1
; RUN: llc < %s -march=systemz | grep nill | count 1

define i32 @gnu_dev_major(i64 %__dev) nounwind readnone {
entry:
        %shr = lshr i64 %__dev, 8               ; <i64> [#uses=1]
        %shr8 = trunc i64 %shr to i32           ; <i32> [#uses=1]
        %shr2 = lshr i64 %__dev, 32             ; <i64> [#uses=1]
        %conv = trunc i64 %shr2 to i32          ; <i32> [#uses=1]
        %and3 = and i32 %conv, -4096            ; <i32> [#uses=1]
        %and6 = and i32 %shr8, 4095             ; <i32> [#uses=1]
        %conv5 = or i32 %and6, %and3            ; <i32> [#uses=1]
        ret i32 %conv5
}
