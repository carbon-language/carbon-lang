; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

; Test load/store of global value from global offset table.
@seed = common global i64 0, align 8

define void @Initrand() nounwind {
entry:
; CHECK: @Initrand
; CHECK: adrp x[[REG:[0-9]+]], _seed@GOTPAGE
; CHECK: ldr x[[REG2:[0-9]+]], [x[[REG]], _seed@GOTPAGEOFF]
; CHECK: str x{{[0-9]+}}, [x[[REG2]]]
  store i64 74755, i64* @seed, align 8
  ret void
}

define i32 @Rand() nounwind {
entry:
; CHECK: @Rand
; CHECK: adrp x[[REG:[0-9]+]], _seed@GOTPAGE
; CHECK: ldr x[[REG2:[0-9]+]], [x[[REG]], _seed@GOTPAGEOFF]
; CHECK: movz x[[REG3:[0-9]+]], #0x51d
; CHECK: ldr x[[REG4:[0-9]+]], [x[[REG2]]]
; CHECK: mul x[[REG5:[0-9]+]], x[[REG4]], x[[REG3]]
; CHECK: movz x[[REG6:[0-9]+]], #0x3619
; CHECK: add x[[REG7:[0-9]+]], x[[REG5]], x[[REG6]]
; CHECK: orr x[[REG8:[0-9]+]], xzr, #0xffff
; CHECK: and x[[REG9:[0-9]+]], x[[REG7]], x[[REG8]]
; CHECK: str x[[REG9]], [x[[REG]]]
; CHECK: ldr x{{[0-9]+}}, [x[[REG]]]
  %0 = load i64* @seed, align 8
  %mul = mul nsw i64 %0, 1309
  %add = add nsw i64 %mul, 13849
  %and = and i64 %add, 65535
  store i64 %and, i64* @seed, align 8
  %1 = load i64* @seed, align 8
  %conv = trunc i64 %1 to i32
  ret i32 %conv
}
