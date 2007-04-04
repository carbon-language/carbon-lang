; Make sure this testcase is supported by all code generators. Either the
; intrinsic is supported natively or IntrinsicLowering provides it.
; RUN: llvm-as < %s | llc


declare i32 @llvm.bit.part.select.i32.i32(i32 %x, i32 %hi, i32 %lo)
declare i16 @llvm.bit.part.select.i16.i16(i16 %x, i32 %hi, i32 %lo)
define i32 @bit_part_select(i32 %A, i16 %B) {
  %a = call i32 @llvm.bit.part.select.i32.i32(i32 %A, i32 8, i32 0)
  %b = call i16 @llvm.bit.part.select.i16.i16(i16 %B, i32 8, i32 0)
  %c = zext i16 %b to i32
  %d = add i32 %a, %c
  ret i32 %d
}
