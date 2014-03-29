; RUN: llc -march=arm64 < %s | FileCheck %s
; rdar://10232252

@object = external hidden global i64, section "__DATA, __objc_ivar", align 8

; base + offset (imm9)
; CHECK: @t1
; CHECK: ldr xzr, [x{{[0-9]+}}, #8]
; CHECK: ret
define void @t1() {
  %incdec.ptr = getelementptr inbounds i64* @object, i64 1
  %tmp = load volatile i64* %incdec.ptr, align 8
  ret void
}

; base + offset (> imm9)
; CHECK: @t2
; CHECK: sub [[ADDREG:x[0-9]+]], x{{[0-9]+}}, #264
; CHECK: ldr xzr, [
; CHECK: [[ADDREG]]]
; CHECK: ret
define void @t2() {
  %incdec.ptr = getelementptr inbounds i64* @object, i64 -33
  %tmp = load volatile i64* %incdec.ptr, align 8
  ret void
}

; base + unsigned offset (> imm9 and <= imm12 * size of type in bytes)
; CHECK: @t3
; CHECK: ldr xzr, [x{{[0-9]+}}, #32760]
; CHECK: ret
define void @t3() {
  %incdec.ptr = getelementptr inbounds i64* @object, i64 4095
  %tmp = load volatile i64* %incdec.ptr, align 8
  ret void
}

; base + unsigned offset (> imm12 * size of type in bytes)
; CHECK: @t4
; CHECK: add [[ADDREG:x[0-9]+]], x{{[0-9]+}}, #32768
; CHECK: ldr xzr, [
; CHECK: [[ADDREG]]]
; CHECK: ret
define void @t4() {
  %incdec.ptr = getelementptr inbounds i64* @object, i64 4096
  %tmp = load volatile i64* %incdec.ptr, align 8
  ret void
}

; base + reg
; CHECK: @t5
; CHECK: ldr xzr, [x{{[0-9]+}}, x{{[0-9]+}}, lsl #3]
; CHECK: ret
define void @t5(i64 %a) {
  %incdec.ptr = getelementptr inbounds i64* @object, i64 %a
  %tmp = load volatile i64* %incdec.ptr, align 8
  ret void
}

; base + reg + imm
; CHECK: @t6
; CHECK: add [[ADDREG:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #3
; CHECK-NEXT: add [[ADDREG]], [[ADDREG]], #32768
; CHECK: ldr xzr, [
; CHECK: [[ADDREG]]]
; CHECK: ret
define void @t6(i64 %a) {
  %tmp1 = getelementptr inbounds i64* @object, i64 %a
  %incdec.ptr = getelementptr inbounds i64* %tmp1, i64 4096
  %tmp = load volatile i64* %incdec.ptr, align 8
  ret void
}
