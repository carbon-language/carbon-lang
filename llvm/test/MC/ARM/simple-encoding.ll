;RUN: llc -mtriple=armv7-apple-darwin -show-mc-encoding < %s | FileCheck %s


;FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;       should run on .s source files rather than using llc to generate the
;       assembly.

define i32 @foo(i32 %a, i32 %b) {
entry:
; CHECK: foo
; CHECK: trap                         @ encoding: [0xf0,0x00,0xf0,0x07]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]

  tail call void @llvm.trap()
  ret i32 undef
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK: f2
; CHECK: add  r0, r1, r0              @ encoding: [0x00,0x00,0x81,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %b, %a
  ret i32 %add
}


define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK: f3
; CHECK: add  r0, r0, r1, lsl #3      @ encoding: [0x81,0x01,0x80,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %mul = shl i32 %b, 3
  %add = add nsw i32 %mul, %a
  ret i32 %add
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK: f4
; CHECK: add r0, r0, #254, 28         @ encoding: [0xfe,0x0e,0x80,0xe2]
; CHECK:                              @ 4064
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %a, 4064
  ret i32 %add
}

define i32 @f5(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK: f5
; CHECK: cmp r0, r1                   @ encoding: [0x01,0x00,0x50,0xe1]
; CHECK: mov r0, r2                   @ encoding: [0x02,0x00,0xa0,0xe1]
; CHECK: movgt r0, r1                 @ encoding: [0x01,0x00,0xa0,0xc1]
  %cmp = icmp sgt i32 %a, %b
  %retval.0 = select i1 %cmp, i32 %b, i32 %c
  ret i32 %retval.0
}

define i64 @f6(i64 %a, i64 %b, i64 %c) {
entry:
; CHECK: f6
; CHECK: adds r0, r2, r0              @ encoding: [0x00,0x00,0x92,0xe0]
; CHECK: adc r1, r3, r1               @ encoding: [0x01,0x10,0xa3,0xe0]
  %add = add nsw i64 %b, %a
  ret i64 %add
}

define i32 @f7(i32 %a, i32 %b) {
entry:
; CHECK: f7
; CHECK: uxtab  r0, r0, r1            @ encoding: [0x71,0x00,0xe0,0xe6]
  %and = and i32 %b, 255
  %add = add i32 %and, %a
  ret i32 %add
}

define i32 @f8(i32 %a) {
entry:
; CHECK: f8
; CHECK: movt r0, #42405              @ encoding: [0xa5,0x05,0x4a,0xe3]
  %and = and i32 %a, 65535
  %or = or i32 %and, -1515913216
  ret i32 %or
}

define i32 @f9() {
entry:
; CHECK: f9
; CHECK: movw r0, #42405              @ encoding: [0xa5,0x05,0x0a,0xe3]
  ret i32 42405
}

define i64 @f10(i64 %a) {
entry:
; CHECK: f10
; CHECK: asrs  r1, r1, #1             @ encoding: [0xc1,0x10,0xb0,0xe1]
; CHECK: rrx r0, r0                   @ encoding: [0x60,0x00,0xa0,0xe1]
  %shr = ashr i64 %a, 1
  ret i64 %shr
}

define i32 @f11([1 x i32] %A.coerce0, [1 x i32] %B.coerce0) {
entry:
; CHECK: f11
; CHECK: ubfx  r1, r1, #8, #5         @ encoding: [0x51,0x14,0xe4,0xe7]
; CHECK: sbfx  r0, r0, #13, #7        @ encoding: [0xd0,0x06,0xa6,0xe7]
  %tmp11 = extractvalue [1 x i32] %A.coerce0, 0
  %tmp4 = extractvalue [1 x i32] %B.coerce0, 0
  %0 = shl i32 %tmp11, 12
  %bf.val.sext = ashr i32 %0, 25
  %1 = lshr i32 %tmp4, 8
  %bf.clear2 = and i32 %1, 31
  %mul = mul nsw i32 %bf.val.sext, %bf.clear2
  ret i32 %mul
}

define i32 @f12(i32 %a) {
; CHECK: f12:
; CHECK: bfc  r0, #4, #20             @ encoding: [0x1f,0x02,0xd7,0xe7]
    %tmp = and i32 %a, 4278190095
    ret i32 %tmp
}

define i64 @f13() {
; CHECK: f13:
; CHECK: mvn r0, #0                   @ encoding: [0x00,0x00,0xe0,0xe3]
; CHECK: mvn r1, #2, 2                @ encoding: [0x02,0x11,0xe0,0xe3]
entry:
        ret i64 9223372036854775807
}

define i32 @f14(i32 %x, i32 %y) {
; CHECK: f14:
; CHECK: smmul  r0, r1, r0            @ encoding: [0x11,0xf0,0x50,0xe7]
        %tmp = sext i32 %x to i64
        %tmp1 = sext i32 %y to i64
        %tmp2 = mul i64 %tmp1, %tmp
        %tmp3 = lshr i64 %tmp2, 32
        %tmp3.upgrd.1 = trunc i64 %tmp3 to i32
        ret i32 %tmp3.upgrd.1
}

define i32 @f15(i32 %x, i32 %y) {
; CHECK: f15:
; CHECK: umull  r1, r0, r1, r0        @ encoding: [0x91,0x10,0x80,0xe0]
        %tmp = zext i32 %x to i64
        %tmp1 = zext i32 %y to i64
        %tmp2 = mul i64 %tmp1, %tmp
        %tmp3 = lshr i64 %tmp2, 32
        %tmp3.upgrd.2 = trunc i64 %tmp3 to i32
        ret i32 %tmp3.upgrd.2
}
declare void @llvm.trap() nounwind
