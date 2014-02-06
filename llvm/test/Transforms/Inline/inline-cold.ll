; RUN: opt < %s -inline -S -inlinecold-threshold=75 | FileCheck %s

; Test that functions with attribute Cold are not inlined while the 
; same function without attribute Cold will be inlined.

@a = global i32 4

; This function should be larger than the cold threshold (75), but smaller
; than the regular threshold.
; Function Attrs: nounwind readnone uwtable
define i32 @simpleFunction(i32 %a) #0 {
entry:
  %a1 = load volatile i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32* @a
  %x7 = add i32 %x6, %a6
  %a8 = load volatile i32* @a
  %x8 = add i32 %x7, %a8
  %a9 = load volatile i32* @a
  %x9 = add i32 %x8, %a9
  %a10 = load volatile i32* @a
  %x10 = add i32 %x9, %a10
  %a11 = load volatile i32* @a
  %x11 = add i32 %x10, %a11
  %a12 = load volatile i32* @a
  %x12 = add i32 %x11, %a12
  %add = add i32 %x12, %a
  ret i32 %add
}

; Function Attrs: nounwind cold readnone uwtable
define i32 @ColdFunction(i32 %a) #1 {
; CHECK-LABEL: @ColdFunction
; CHECK: ret
entry:
  %a1 = load volatile i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32* @a
  %x7 = add i32 %x6, %a6
  %a8 = load volatile i32* @a
  %x8 = add i32 %x7, %a8
  %a9 = load volatile i32* @a
  %x9 = add i32 %x8, %a9
  %a10 = load volatile i32* @a
  %x10 = add i32 %x9, %a10
  %a11 = load volatile i32* @a
  %x11 = add i32 %x10, %a11
  %a12 = load volatile i32* @a
  %x12 = add i32 %x11, %a12
  %add = add i32 %x12, %a
  ret i32 %add
}

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #0 {
; CHECK-LABEL: @bar
; CHECK: call i32 @ColdFunction(i32 5)
; CHECK-NOT: call i32 @simpleFunction(i32 6)
; CHECK: ret
entry:
  %0 = tail call i32 @ColdFunction(i32 5)
  %1 = tail call i32 @simpleFunction(i32 6)
  %add = add i32 %0, %1
  ret i32 %add
}

attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind cold readnone uwtable }
