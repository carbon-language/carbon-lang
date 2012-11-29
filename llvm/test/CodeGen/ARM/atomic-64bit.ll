; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s

define i64 @test1(i64* %ptr, i64 %val) {
; CHECK: test1:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: adds [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: adc [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw add i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test2(i64* %ptr, i64 %val) {
; CHECK: test2:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: sbc [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw sub i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test3(i64* %ptr, i64 %val) {
; CHECK: test3:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: and [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: and [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw and i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test4(i64* %ptr, i64 %val) {
; CHECK: test4:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: orr [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: orr [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw or i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test5(i64* %ptr, i64 %val) {
; CHECK: test5:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: eor [[REG3:(r[0-9]?[02468])]], [[REG1]]
; CHECK: eor [[REG4:(r[0-9]?[13579])]], [[REG2]]
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw xor i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test6(i64* %ptr, i64 %val) {
; CHECK: test6:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw xchg i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test7(i64* %ptr, i64 %val1, i64 %val2) {
; CHECK: test7:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: cmp [[REG1]]
; CHECK: cmpeq [[REG2]]
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = cmpxchg i64* %ptr, i64 %val1, i64 %val2 seq_cst
  ret i64 %r
}

; Compiles down to cmpxchg
; FIXME: Should compile to a single ldrexd
define i64 @test8(i64* %ptr) {
; CHECK: test8:
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: cmp [[REG1]]
; CHECK: cmpeq [[REG2]]
; CHECK: bne
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = load atomic i64* %ptr seq_cst, align 8
  ret i64 %r
}

; Compiles down to atomicrmw xchg; there really isn't any more efficient
; way to write it.
define void @test9(i64* %ptr, i64 %val) {
; CHECK: test9:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: strexd {{[a-z0-9]+}}, {{r[0-9]?[02468]}}, {{r[0-9]?[13579]}}
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  store atomic i64 %val, i64* %ptr seq_cst, align 8
  ret void
}

define i64 @test10(i64* %ptr, i64 %val) {
; CHECK: test10:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: ble
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw min i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test11(i64* %ptr, i64 %val) {
; CHECK: test11:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: bls
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw umin i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test12(i64* %ptr, i64 %val) {
; CHECK: test12:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: bge
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw max i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

define i64 @test13(i64* %ptr, i64 %val) {
; CHECK: test13:
; CHECK: dmb ish
; CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], [[REG2:(r[0-9]?[13579])]]
; CHECK: subs {{[a-z0-9]+}}, [[REG1]], [[REG3:(r[0-9]?[02468])]]
; CHECK: sbcs {{[a-z0-9]+}}, [[REG2]], [[REG4:(r[0-9]?[13579])]]
; CHECK: bhs
; CHECK: strexd {{[a-z0-9]+}}, [[REG3]], [[REG4]]
; CHECK: cmp
; CHECK: bne
; CHECK: dmb ish
  %r = atomicrmw umax i64* %ptr, i64 %val seq_cst
  ret i64 %r
}

