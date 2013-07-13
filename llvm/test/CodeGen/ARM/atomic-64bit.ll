; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabihf | FileCheck %s --check-prefix=CHECK-THUMB

define i64 @test1(i64* %ptr, i64 %val) {
; CHECK-LABEL: test1:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: adds [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: adc [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test1:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: adds.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: adc.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw add i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test2(i64* %ptr, i64 %val) {
; CHECK-LABEL: test2:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: sbc [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test2:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: subs.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: sbc.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw sub i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test3(i64* %ptr, i64 %val) {
; CHECK-LABEL: test3:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: and [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: and [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test3:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: and.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: and.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw and i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test4(i64* %ptr, i64 %val) {
; CHECK-LABEL: test4:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: orr [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: orr [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test4:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: orr.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: orr.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw or i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test5(i64* %ptr, i64 %val) {
; CHECK-LABEL: test5:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: eor [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: eor [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test5:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: eor.w [[REG3:[a-z0-9]+]], [[REG1]]
; CHECK-THUMB: eor.w [[REG4:[a-z0-9]+]], [[REG2]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw xor i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test6(i64* %ptr, i64 %val) {
; CHECK-LABEL: test6:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test6:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw xchg i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test7(i64* %ptr, i64 %val1, i64 %val2) {
; CHECK-LABEL: test7:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: cmp [[REG1]]
; CHECK: cmpeq [[REG2]]
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test7:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: cmp [[REG1]]
; CHECK-THUMB: it eq
; CHECK-THUMB: cmpeq [[REG2]]
; CHECK-THUMB: bne
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = cmpxchg i64* %ptr, i64 %val1, i64 %val2 seq_cst
  ret i64 %r
}

; Compiles down to cmpxchg
; FIXME: Should compile to a single ldrexd
define i64 @test8(i64* %ptr) {
; CHECK-LABEL: test8:
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: cmp [[REG1]]
; CHECK: cmpeq [[REG2]]
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test8:
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: cmp [[REG1]]
; CHECK-THUMB: it eq
; CHECK-THUMB: cmpeq [[REG2]]
; CHECK-THUMB: bne
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = load atomic i64* %ptr seq_cst, align 8
  ret i64 %r
}

; Compiles down to atomicrmw xchg; there really isn't any more efficient
; way to write it.
define void @test9(i64* %ptr, i64 %val) {
; CHECK-LABEL: test9:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test9:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: strexd {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  store atomic i64 %val, i64* %ptr seq_cst, align 8
  ret void
}

define i64 @test10(i64* %ptr, i64 %val) {
; CHECK-LABEL: test10:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: blt
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test10:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: subs.w {{[a-z0-9]+}}, [[REG1]], [[REG3:[a-z0-9]+]]
; CHECK-THUMB: sbcs.w {{[a-z0-9]+}}, [[REG2]], [[REG4:[a-z0-9]+]]
; CHECK-THUMB: blt
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw min i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test11(i64* %ptr, i64 %val) {
; CHECK-LABEL: test11:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: blo
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}


; CHECK-THUMB: test11:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: subs.w {{[a-z0-9]+}}, [[REG1]], [[REG3:[a-z0-9]+]]
; CHECK-THUMB: sbcs.w {{[a-z0-9]+}}, [[REG2]], [[REG4:[a-z0-9]+]]
; CHECK-THUMB: blo
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw umin i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test12(i64* %ptr, i64 %val) {
; CHECK-LABEL: test12:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: bge
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test12:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: subs.w {{[a-z0-9]+}}, [[REG1]], [[REG3:[a-z0-9]+]]
; CHECK-THUMB: sbcs.w {{[a-z0-9]+}}, [[REG2]], [[REG4:[a-z0-9]+]]
; CHECK-THUMB: bge
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}

  %r = atomicrmw max i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test13(i64* %ptr, i64 %val) {
; CHECK-LABEL: test13:
; CHECK: dmb {{ish$}}
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: bhs
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb {{ish$}}

; CHECK-THUMB: test13:
; CHECK-THUMB: dmb {{ish$}}
; CHECK-THUMB: ldrexd [[REG1:[a-z0-9]+]], [[REG2:[a-z0-9]+]]
; CHECK-THUMB: subs.w {{[a-z0-9]+}}, [[REG1]], [[REG3:[a-z0-9]+]]
; CHECK-THUMB: sbcs.w {{[a-z0-9]+}}, [[REG2]], [[REG4:[a-z0-9]+]]
; CHECK-THUMB: bhs
; CHECK-THUMB: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK-THUMB: cmp
; CHECK-THUMB: bne
; CHECK-THUMB: dmb {{ish$}}
  %r = atomicrmw umax i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

