;RUN: llc -mtriple=armv7-apple-darwin -show-mc-encoding -disable-cgp-branch-opts -join-physregs < %s | FileCheck %s


;FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;       should run on .s source files rather than using llc to generate the
;       assembly. There's also a large number of instruction encodings the
;       compiler never generates, so we need the integrated assembler to be
;       able to test those at all.

declare void @llvm.trap() nounwind
declare i32 @llvm.ctlz.i32(i32)

define i32 @foo(i32 %a, i32 %b) {
; CHECK: foo
; CHECK: trap                         @ encoding: [0xfe,0xde,0xff,0xe7]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]

  tail call void @llvm.trap()
  ret i32 undef
}

define i32 @f2(i32 %a, i32 %b) {
; CHECK: f2
; CHECK: add  r0, r1, r0              @ encoding: [0x00,0x00,0x81,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %b, %a
  ret i32 %add
}


define i32 @f3(i32 %a, i32 %b) {
; CHECK: f3
; CHECK: add  r0, r0, r1, lsl #3      @ encoding: [0x81,0x01,0x80,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %mul = shl i32 %b, 3
  %add = add nsw i32 %mul, %a
  ret i32 %add
}

define i32 @f4(i32 %a, i32 %b) {
; CHECK: f4
; CHECK: add r0, r0, #4064            @ encoding: [0xfe,0x0e,0x80,0xe2]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %a, 4064
  ret i32 %add
}

define i32 @f5(i32 %a, i32 %b, i32 %c) {
; CHECK: f5
; CHECK: cmp r0, r1                   @ encoding: [0x01,0x00,0x50,0xe1]
; CHECK: mov r0, r2                   @ encoding: [0x02,0x00,0xa0,0xe1]
; CHECK: movgt r0, r1                 @ encoding: [0x01,0x00,0xa0,0xc1]
  %cmp = icmp sgt i32 %a, %b
  %retval.0 = select i1 %cmp, i32 %b, i32 %c
  ret i32 %retval.0
}

define i64 @f6(i64 %a, i64 %b, i64 %c) {
; CHECK: f6
; CHECK: adds r0, r2, r0              @ encoding: [0x00,0x00,0x92,0xe0]
; CHECK: adc r1, r3, r1               @ encoding: [0x01,0x10,0xa3,0xe0]
  %add = add nsw i64 %b, %a
  ret i64 %add
}

define i32 @f7(i32 %a, i32 %b) {
; CHECK: f7
; CHECK: uxtab  r0, r0, r1            @ encoding: [0x71,0x00,0xe0,0xe6]
  %and = and i32 %b, 255
  %add = add i32 %and, %a
  ret i32 %add
}

define i32 @f8(i32 %a) {
; CHECK: f8
; CHECK: movt r0, #42405              @ encoding: [0xa5,0x05,0x4a,0xe3]
  %and = and i32 %a, 65535
  %or = or i32 %and, -1515913216
  ret i32 %or
}

define i32 @f9() {
; CHECK: f9
; CHECK: movw r0, #42405              @ encoding: [0xa5,0x05,0x0a,0xe3]
  ret i32 42405
}

define i64 @f10(i64 %a) {
; CHECK: f10
; CHECK: asrs  r1, r1, #1             @ encoding: [0xc1,0x10,0xb0,0xe1]
; CHECK: rrx r0, r0                   @ encoding: [0x60,0x00,0xa0,0xe1]
  %shr = ashr i64 %a, 1
  ret i64 %shr
}

define i32 @f11([1 x i32] %A.coerce0, [1 x i32] %B.coerce0) {
; CHECK: f11
; CHECK: ubfx  r1, r1, #8, #5         @ encoding: [0x51,0x14,0xe4,0xe7]
; CHECK: sbfx  r0, r0, #13, #7        @ encoding: [0xd0,0x06,0xa6,0xe7]
  %tmp1 = extractvalue [1 x i32] %A.coerce0, 0
  %tmp2 = extractvalue [1 x i32] %B.coerce0, 0
  %tmp3 = shl i32 %tmp1, 12
  %bf.val.sext = ashr i32 %tmp3, 25
  %tmp4 = lshr i32 %tmp2, 8
  %bf.clear2 = and i32 %tmp4, 31
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
; CHECK: mvn r1, #-2147483648         @ encoding: [0x02,0x11,0xe0,0xe3]
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

define i32 @f16(i16 %x, i32 %y) {
; CHECK: f16:
; CHECK: smulbt r0, r0, r1            @ encoding: [0xc0,0x01,0x60,0xe1]
        %tmp1 = add i16 %x, 2
        %tmp2 = sext i16 %tmp1 to i32
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp2, %tmp3
        ret i32 %tmp4
}

define i32 @f17(i32 %x, i32 %y) {
; CHECK: f17:
; CHECK: smultt r0, r1, r0            @ encoding: [0xe1,0x00,0x60,0xe1]
        %tmp1 = ashr i32 %x, 16
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp3, %tmp1
        ret i32 %tmp4
}

define i32 @f18(i32 %a, i16 %x, i32 %y) {
; CHECK: f18:
; CHECK: smlabt r0, r1, r2, r0        @ encoding: [0xc1,0x02,0x00,0xe1]
        %tmp = sext i16 %x to i32
        %tmp2 = ashr i32 %y, 16
        %tmp3 = mul i32 %tmp2, %tmp
        %tmp5 = add i32 %tmp3, %a
        ret i32 %tmp5
}

define i32 @f19(i32 %x) {
; CHECK: f19
; CHECK: clz r0, r0                   @ encoding: [0x10,0x0f,0x6f,0xe1]
        %tmp.1 = call i32 @llvm.ctlz.i32( i32 %x )
        ret i32 %tmp.1
}

define i32 @f20(i32 %X) {
; CHECK: f20
; CHECK: rev16 r0, r0                 @ encoding: [0xb0,0x0f,0xbf,0xe6]
        %tmp1 = lshr i32 %X, 8
        %X15 = bitcast i32 %X to i32
        %tmp4 = shl i32 %X15, 8
        %tmp2 = and i32 %tmp1, 16711680
        %tmp5 = and i32 %tmp4, -16777216
        %tmp9 = and i32 %tmp1, 255
        %tmp13 = and i32 %tmp4, 65280
        %tmp6 = or i32 %tmp5, %tmp2
        %tmp10 = or i32 %tmp6, %tmp13
        %tmp14 = or i32 %tmp10, %tmp9
        ret i32 %tmp14
}

define i32 @f21(i32 %X) {
; CHECK: f21
; CHECK: revsh r0, r0                 @ encoding: [0xb0,0x0f,0xff,0xe6]
        %tmp1 = lshr i32 %X, 8
        %tmp1.upgrd.1 = trunc i32 %tmp1 to i16
        %tmp3 = trunc i32 %X to i16
        %tmp2 = and i16 %tmp1.upgrd.1, 255
        %tmp4 = shl i16 %tmp3, 8
        %tmp5 = or i16 %tmp2, %tmp4
        %tmp5.upgrd.2 = sext i16 %tmp5 to i32
        ret i32 %tmp5.upgrd.2
}

define i32 @f22(i32 %X, i32 %Y) {
; CHECK: f22
; CHECK: pkhtb   r0, r0, r1, asr #22  @ encoding: [0x51,0x0b,0x80,0xe6]
	%tmp1 = and i32 %X, -65536
	%tmp2 = lshr i32 %Y, 22
	%tmp3 = or i32 %tmp2, %tmp1
	ret i32 %tmp3
}

define i32 @f23(i32 %X, i32 %Y) {
; CHECK: f23
; CHECK: pkhbt   r0, r0, r1, lsl #18  @ encoding: [0x11,0x09,0x80,0xe6]
	%tmp1 = and i32 %X, 65535
	%tmp2 = shl i32 %Y, 18
	%tmp3 = or i32 %tmp1, %tmp2
	ret i32 %tmp3
}

define void @f24(i32 %a) {
; CHECK: f24
; CHECK: cmp r0, #65536               @ encoding: [0x01,0x08,0x50,0xe3]
        %b = icmp ugt i32 %a, 65536
        br i1 %b, label %r, label %r
r:
        ret void
}
