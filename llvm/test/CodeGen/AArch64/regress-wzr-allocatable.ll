; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -O0

; When WZR wasn't marked as reserved, this function tried to allocate
; it at O0 and then generated an internal fault (mostly incidentally)
; when it discovered that it was already in use for a multiplication.

; I'm not really convinced this is a good test since it could easily
; stop testing what it does now with no-one any the wiser. However, I
; can't think of a better way to force the allocator to use WZR
; specifically.

define void @test() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond6

for.cond6:                                        ; preds = %for.body9, %for.end
  br i1 undef, label %for.body9, label %while.cond30

for.body9:                                        ; preds = %for.cond6
  store i16 0, i16* undef, align 2
  %0 = load i32* undef, align 4
  %1 = load i32* undef, align 4
  %mul15 = mul i32 %0, %1
  %add16 = add i32 %mul15, 32768
  %div = udiv i32 %add16, 65535
  %add17 = add i32 %div, 1
  store i32 %add17, i32* undef, align 4
  br label %for.cond6

while.cond30:                                     ; preds = %for.cond6
  ret void
}
