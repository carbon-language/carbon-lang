; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @sub1
; CHECK: %y = sub i32 0, %x
; CHECK: %z = sdiv i32 %y, 337
; CHECK: ret i32 %z
define i32 @sub1(i32 %x) {
  %y = sub i32 0, %x
  %z = sdiv i32 %y, 337
  ret i32 %z
}

; CHECK: @sub2
; CHECK: %z = sdiv i32 %x, -337
; CHECK: ret i32 %z
define i32 @sub2(i32 %x) {
  %y = sub nsw i32 0, %x
  %z = sdiv i32 %y, 337
  ret i32 %z
}

; CHECK: @shl_icmp
; CHECK: %B = icmp eq i64 %X, 0
; CHECK: ret i1 %B
define i1 @shl_icmp(i64 %X) nounwind {
  %A = shl nuw i64 %X, 2   ; X/4
  %B = icmp eq i64 %A, 0
  ret i1 %B
}

; CHECK: @shl1
; CHECK: %B = shl nuw nsw i64 %A, 8
; CHECK: ret i64 %B
define i64 @shl1(i64 %X, i64* %P) nounwind {
  %A = and i64 %X, 312
  store i64 %A, i64* %P  ; multiple uses of A.
  %B = shl i64 %A, 8
  ret i64 %B
}

; CHECK: @preserve1
; CHECK: add nsw i32 %x, 5
define i32 @preserve1(i32 %x) nounwind {
  %add = add nsw i32 %x, 2
  %add3 = add nsw i32 %add, 3
  ret i32 %add3
}

; CHECK: @nopreserve1
; CHECK: add i8 %x, -126
define i8 @nopreserve1(i8 %x) nounwind {
  %add = add nsw i8 %x, 127
  %add3 = add nsw i8 %add, 3
  ret i8 %add3
}