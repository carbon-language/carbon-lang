; RUN: llc -enable-misched -enable-aa-sched-mi < %s

; This generates a decw instruction, which has two MMOs, and an alias SU edge
; query involving that instruction. Make sure this does not crash.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%s1 = type { i16, i16, i32 }
%c1 = type { %s1*, %u1, i16, i8 }
%u1 = type { i64 }

declare zeroext i1 @bar(i64*, i32) #5

define i32 @foo() #0 align 2 {
entry:
  %temp_rhs = alloca %c1, align 8
  br i1 undef, label %if.else56, label %cond.end.i

cond.end.i:
  %significand.i18.i = getelementptr inbounds %c1, %c1* %temp_rhs, i64 0, i32 1
  %exponent.i = getelementptr inbounds %c1, %c1* %temp_rhs, i64 0, i32 2
  %0 = load i16, i16* %exponent.i, align 8
  %sub.i = add i16 %0, -1
  store i16 %sub.i, i16* %exponent.i, align 8
  %parts.i.i = bitcast %u1* %significand.i18.i to i64**
  %1 = load i64*, i64** %parts.i.i, align 8
  %call5.i = call zeroext i1 @bar(i64* %1, i32 undef) #1
  unreachable

if.else56:
  unreachable
}

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

