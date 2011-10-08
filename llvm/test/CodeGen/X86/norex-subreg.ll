; RUN: llc -O0 < %s
target triple = "x86_64-apple-macosx10.7"

; This test case extracts a sub_8bit_hi sub-register:
;
;	%R8B<def> = COPY %BH, %EBX<imp-use,kill>
;	%ESI<def> = MOVZX32_NOREXrr8 %R8B<kill>
;
; The register allocation above is invalid, %BH can only be encoded without an
; REX prefix, so the destination register must be GR8_NOREX.  The code above
; triggers an assertion in copyPhysReg.
;
; <rdar://problem/10248099>

define void @f() nounwind uwtable ssp {
entry:
  %0 = load i32* undef, align 4
  %add = add i32 0, %0
  %conv1 = trunc i32 %add to i16
  %bf.value = and i16 %conv1, 255
  %1 = and i16 %bf.value, 255
  %2 = shl i16 %1, 8
  %3 = load i16* undef, align 1
  %4 = and i16 %3, 255
  %5 = or i16 %4, %2
  store i16 %5, i16* undef, align 1
  %6 = load i16* undef, align 1
  %7 = lshr i16 %6, 8
  %bf.clear2 = and i16 %7, 255
  %conv3 = zext i16 %bf.clear2 to i32
  %rem = srem i32 %conv3, 15
  %conv4 = trunc i32 %rem to i16
  %bf.value5 = and i16 %conv4, 255
  %8 = and i16 %bf.value5, 255
  %9 = shl i16 %8, 8
  %10 = or i16 undef, %9
  store i16 %10, i16* undef, align 1
  ret void
}
