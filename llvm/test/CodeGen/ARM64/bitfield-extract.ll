; RUN: llc < %s -march=arm64 | FileCheck %s
%struct.X = type { i8, i8, [2 x i8] }
%struct.Y = type { i32, i8 }
%struct.Z = type { i8, i8, [2 x i8], i16 }
%struct.A = type { i64, i8 }

define void @foo(%struct.X* nocapture %x, %struct.Y* nocapture %y) nounwind optsize ssp {
; CHECK-LABEL: foo:
; CHECK: ubfm
; CHECK-NOT: and
; CHECK: ret

  %tmp = bitcast %struct.X* %x to i32*
  %tmp1 = load i32* %tmp, align 4
  %b = getelementptr inbounds %struct.Y* %y, i64 0, i32 1
  %bf.clear = lshr i32 %tmp1, 3
  %bf.clear.lobit = and i32 %bf.clear, 1
  %frombool = trunc i32 %bf.clear.lobit to i8
  store i8 %frombool, i8* %b, align 1
  ret void
}

define i32 @baz(i64 %cav1.coerce) nounwind {
; CHECK-LABEL: baz:
; CHECK: sbfm  w0, w0, #0, #3
  %tmp = trunc i64 %cav1.coerce to i32
  %tmp1 = shl i32 %tmp, 28
  %bf.val.sext = ashr exact i32 %tmp1, 28
  ret i32 %bf.val.sext
}

define i32 @bar(i64 %cav1.coerce) nounwind {
; CHECK-LABEL: bar:
; CHECK: sbfm  w0, w0, #4, #9
  %tmp = trunc i64 %cav1.coerce to i32
  %cav1.sroa.0.1.insert = shl i32 %tmp, 22
  %tmp1 = ashr i32 %cav1.sroa.0.1.insert, 26
  ret i32 %tmp1
}

define void @fct1(%struct.Z* nocapture %x, %struct.A* nocapture %y) nounwind optsize ssp {
; CHECK-LABEL: fct1:
; CHECK: ubfm
; CHECK-NOT: and
; CHECK: ret

  %tmp = bitcast %struct.Z* %x to i64*
  %tmp1 = load i64* %tmp, align 4
  %b = getelementptr inbounds %struct.A* %y, i64 0, i32 0
  %bf.clear = lshr i64 %tmp1, 3
  %bf.clear.lobit = and i64 %bf.clear, 1
  store i64 %bf.clear.lobit, i64* %b, align 8
  ret void
}

define i64 @fct2(i64 %cav1.coerce) nounwind {
; CHECK-LABEL: fct2:
; CHECK: sbfm  x0, x0, #0, #35
  %tmp = shl i64 %cav1.coerce, 28
  %bf.val.sext = ashr exact i64 %tmp, 28
  ret i64 %bf.val.sext
}

define i64 @fct3(i64 %cav1.coerce) nounwind {
; CHECK-LABEL: fct3:
; CHECK: sbfm  x0, x0, #4, #41
  %cav1.sroa.0.1.insert = shl i64 %cav1.coerce, 22
  %tmp1 = ashr i64 %cav1.sroa.0.1.insert, 26
  ret i64 %tmp1
}

define void @fct4(i64* nocapture %y, i64 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct4:
; CHECK: ldr [[REG1:x[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], x1, #16, #39
; CHECK-NEXT: str [[REG1]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, -16777216
  %shr = lshr i64 %x, 16
  %and1 = and i64 %shr, 16777215
  %or = or i64 %and, %and1
  store i64 %or, i64* %y, align 8
  ret void
}

define void @fct5(i32* nocapture %y, i32 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct5:
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], w1, #16, #18
; CHECK-NEXT: str [[REG1]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, -8
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  store i32 %or, i32* %y, align 8
  ret void
}

; Check if we can still catch bfm instruction when we drop some low bits
define void @fct6(i32* nocapture %y, i32 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct6:
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], w1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: lsr [[REG2:w[0-9]+]], [[REG1]], #2
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, -8
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  %shr1 = lshr i32 %or, 2
  store i32 %shr1, i32* %y, align 8
  ret void
}


; Check if we can still catch bfm instruction when we drop some high bits
define void @fct7(i32* nocapture %y, i32 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct7:
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], w1, #16, #18
; lsl is an alias of ubfm
; CHECK-NEXT: lsl [[REG2:w[0-9]+]], [[REG1]], #2
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, -8
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  %shl = shl i32 %or, 2
  store i32 %shl, i32* %y, align 8
  ret void
}


; Check if we can still catch bfm instruction when we drop some low bits
; (i64 version)
define void @fct8(i64* nocapture %y, i64 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct8:
; CHECK: ldr [[REG1:x[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], x1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: lsr [[REG2:x[0-9]+]], [[REG1]], #2
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, -8
  %shr = lshr i64 %x, 16
  %and1 = and i64 %shr, 7
  %or = or i64 %and, %and1
  %shr1 = lshr i64 %or, 2
  store i64 %shr1, i64* %y, align 8
  ret void
}


; Check if we can still catch bfm instruction when we drop some high bits
; (i64 version)
define void @fct9(i64* nocapture %y, i64 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct9:
; CHECK: ldr [[REG1:x[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], x1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: lsl [[REG2:x[0-9]+]], [[REG1]], #2
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, -8
  %shr = lshr i64 %x, 16
  %and1 = and i64 %shr, 7
  %or = or i64 %and, %and1
  %shl = shl i64 %or, 2
  store i64 %shl, i64* %y, align 8
  ret void
}

; Check if we can catch bfm instruction when lsb is 0 (i.e., no lshr)
; (i32 version)
define void @fct10(i32* nocapture %y, i32 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct10:
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], w1, #0, #2
; lsl is an alias of ubfm
; CHECK-NEXT: lsl [[REG2:w[0-9]+]], [[REG1]], #2
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, -8
  %and1 = and i32 %x, 7
  %or = or i32 %and, %and1
  %shl = shl i32 %or, 2
  store i32 %shl, i32* %y, align 8
  ret void
}

; Check if we can catch bfm instruction when lsb is 0 (i.e., no lshr)
; (i64 version)
define void @fct11(i64* nocapture %y, i64 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct11:
; CHECK: ldr [[REG1:x[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], x1, #0, #2
; lsl is an alias of ubfm
; CHECK-NEXT: lsl [[REG2:x[0-9]+]], [[REG1]], #2
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, -8
  %and1 = and i64 %x, 7
  %or = or i64 %and, %and1
  %shl = shl i64 %or, 2
  store i64 %shl, i64* %y, align 8
  ret void
}

define zeroext i1 @fct12bis(i32 %tmp2) unnamed_addr nounwind ssp align 2 {
; CHECK-LABEL: fct12bis:
; CHECK-NOT: and
; CHECK: ubfm w0, w0, #11, #11
  %and.i.i = and i32 %tmp2, 2048
  %tobool.i.i = icmp ne i32 %and.i.i, 0
  ret i1 %tobool.i.i
}

; Check if we can still catch bfm instruction when we drop some high bits
; and some low bits
define void @fct12(i32* nocapture %y, i32 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct12:
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], w1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: ubfm [[REG2:w[0-9]+]], [[REG1]], #2, #29
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, -8
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  %shl = shl i32 %or, 2
  %shr2 = lshr i32 %shl, 4
  store i32 %shr2, i32* %y, align 8
  ret void
}

; Check if we can still catch bfm instruction when we drop some high bits
; and some low bits
; (i64 version)
define void @fct13(i64* nocapture %y, i64 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct13:
; CHECK: ldr [[REG1:x[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], x1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: ubfm [[REG2:x[0-9]+]], [[REG1]], #2, #61
; CHECK-NEXT: str [[REG2]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, -8
  %shr = lshr i64 %x, 16
  %and1 = and i64 %shr, 7
  %or = or i64 %and, %and1
  %shl = shl i64 %or, 2
  %shr2 = lshr i64 %shl, 4
  store i64 %shr2, i64* %y, align 8
  ret void
}


; Check if we can still catch bfm instruction when we drop some high bits
; and some low bits
define void @fct14(i32* nocapture %y, i32 %x, i32 %x1) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct14:
; CHECK: ldr [[REG1:w[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], w1, #16, #23
; lsr is an alias of ubfm
; CHECK-NEXT: lsr [[REG2:w[0-9]+]], [[REG1]], #4
; CHECK-NEXT: bfm [[REG2]], w2, #5, #7
; lsl is an alias of ubfm
; CHECK-NEXT: lsl [[REG3:w[0-9]+]], [[REG2]], #2
; CHECK-NEXT: str [[REG3]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, -256
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 255
  %or = or i32 %and, %and1
  %shl = lshr i32 %or, 4
  %and2 = and i32 %shl, -8
  %shr1 = lshr i32 %x1, 5
  %and3 = and i32 %shr1, 7
  %or1 = or i32 %and2, %and3
  %shl1 = shl i32 %or1, 2
  store i32 %shl1, i32* %y, align 8
  ret void
}

; Check if we can still catch bfm instruction when we drop some high bits
; and some low bits
; (i64 version)
define void @fct15(i64* nocapture %y, i64 %x, i64 %x1) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct15:
; CHECK: ldr [[REG1:x[0-9]+]],
; CHECK-NEXT: bfm [[REG1]], x1, #16, #23
; lsr is an alias of ubfm
; CHECK-NEXT: lsr [[REG2:x[0-9]+]], [[REG1]], #4
; CHECK-NEXT: bfm [[REG2]], x2, #5, #7
; lsl is an alias of ubfm
; CHECK-NEXT: lsl [[REG3:x[0-9]+]], [[REG2]], #2
; CHECK-NEXT: str [[REG3]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, -256
  %shr = lshr i64 %x, 16
  %and1 = and i64 %shr, 255
  %or = or i64 %and, %and1
  %shl = lshr i64 %or, 4
  %and2 = and i64 %shl, -8
  %shr1 = lshr i64 %x1, 5
  %and3 = and i64 %shr1, 7
  %or1 = or i64 %and2, %and3
  %shl1 = shl i64 %or1, 2
  store i64 %shl1, i64* %y, align 8
  ret void
}

; Check if we can still catch bfm instruction when we drop some high bits
; and some low bits and a masking operation has to be kept
define void @fct16(i32* nocapture %y, i32 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct16:
; CHECK: ldr [[REG1:w[0-9]+]],
; Create the constant
; CHECK: movz [[REGCST:w[0-9]+]], #26, lsl #16
; CHECK: movk [[REGCST]], #33120
; Do the masking
; CHECK: and [[REG2:w[0-9]+]], [[REG1]], [[REGCST]]
; CHECK-NEXT: bfm [[REG2]], w1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: ubfm [[REG3:w[0-9]+]], [[REG2]], #2, #29
; CHECK-NEXT: str [[REG3]],
; CHECK-NEXT: ret
  %0 = load i32* %y, align 8
  %and = and i32 %0, 1737056
  %shr = lshr i32 %x, 16
  %and1 = and i32 %shr, 7
  %or = or i32 %and, %and1
  %shl = shl i32 %or, 2
  %shr2 = lshr i32 %shl, 4
  store i32 %shr2, i32* %y, align 8
  ret void
}


; Check if we can still catch bfm instruction when we drop some high bits
; and some low bits and a masking operation has to be kept
; (i64 version)
define void @fct17(i64* nocapture %y, i64 %x) nounwind optsize inlinehint ssp {
entry:
; CHECK-LABEL: fct17:
; CHECK: ldr [[REG1:x[0-9]+]],
; Create the constant
; CHECK: movz [[REGCST:x[0-9]+]], #26, lsl #16
; CHECK: movk [[REGCST]], #33120
; Do the masking
; CHECK: and [[REG2:x[0-9]+]], [[REG1]], [[REGCST]]
; CHECK-NEXT: bfm [[REG2]], x1, #16, #18
; lsr is an alias of ubfm
; CHECK-NEXT: ubfm [[REG3:x[0-9]+]], [[REG2]], #2, #61
; CHECK-NEXT: str [[REG3]],
; CHECK-NEXT: ret
  %0 = load i64* %y, align 8
  %and = and i64 %0, 1737056
  %shr = lshr i64 %x, 16
  %and1 = and i64 %shr, 7
  %or = or i64 %and, %and1
  %shl = shl i64 %or, 2
  %shr2 = lshr i64 %shl, 4
  store i64 %shr2, i64* %y, align 8
  ret void
}

define i64 @fct18(i32 %xor72) nounwind ssp {
; CHECK-LABEL: fct18:
; CHECK: ubfm x0, x0, #9, #16
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %result = and i64 %conv82, 255
  ret i64 %result
}
