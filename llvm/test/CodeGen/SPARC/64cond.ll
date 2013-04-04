; RUN: llc < %s -march=sparcv9 | FileCheck %s
; Testing 64-bit conditionals.

; CHECK: cmpri
; CHECK: subcc %i1, 1
; CHECK: bpe %xcc,
define void @cmpri(i64* %p, i64 %x) {
entry:
  %tobool = icmp eq i64 %x, 1
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i64 %x, i64* %p, align 8
  br label %if.end

if.end:
  ret void
}

; CHECK: cmprr
; CHECK: subcc %i1, %i2
; CHECK: bpgu %xcc,
define void @cmprr(i64* %p, i64 %x, i64 %y) {
entry:
  %tobool = icmp ugt i64 %x, %y
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i64 %x, i64* %p, align 8
  br label %if.end

if.end:
  ret void
}

; CHECK: selecti32_xcc
; CHECK: subcc %i0, %i1
; CHECK: movg %xcc, %i2, %i3
; CHECK: or %g0, %i3, %i0
define i32 @selecti32_xcc(i64 %x, i64 %y, i32 %a, i32 %b) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, i32 %a, i32 %b
  ret i32 %rv
}

; CHECK: selecti64_xcc
; CHECK: subcc %i0, %i1
; CHECK: movg %xcc, %i2, %i3
; CHECK: or %g0, %i3, %i0
define i64 @selecti64_xcc(i64 %x, i64 %y, i64 %a, i64 %b) {
entry:
  %tobool = icmp sgt i64 %x, %y
  %rv = select i1 %tobool, i64 %a, i64 %b
  ret i64 %rv
}
