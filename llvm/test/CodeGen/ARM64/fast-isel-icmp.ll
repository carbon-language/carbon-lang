; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

define i32 @icmp_eq_imm(i32 %a) nounwind ssp {
entry:
; CHECK: icmp_eq_imm
; CHECK: cmp  w0, #31
; CHECK: csinc w0, wzr, wzr, ne
  %cmp = icmp eq i32 %a, 31
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_eq_neg_imm(i32 %a) nounwind ssp {
entry:
; CHECK: icmp_eq_neg_imm
; CHECK: cmn  w0, #7
; CHECK: csinc w0, wzr, wzr, ne
  %cmp = icmp eq i32 %a, -7
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_eq(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_eq
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, ne
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ne(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_ne
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, eq
  %cmp = icmp ne i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ugt(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_ugt
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, ls
  %cmp = icmp ugt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_uge(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_uge
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, cc
  %cmp = icmp uge i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ult(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_ult
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, cs
  %cmp = icmp ult i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_ule(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_ule
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, hi
  %cmp = icmp ule i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_sgt(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_sgt
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, le
  %cmp = icmp sgt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_sge(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_sge
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, lt
  %cmp = icmp sge i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_slt(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_slt
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, ge
  %cmp = icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_sle(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: icmp_sle
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, gt
  %cmp = icmp sle i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @icmp_i64(i64 %a, i64 %b) nounwind ssp {
entry:
; CHECK: icmp_i64
; CHECK: cmp  x0, x1
; CHECK: csinc w{{[0-9]+}}, wzr, wzr, gt
  %cmp = icmp sle i64 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define zeroext i1 @icmp_eq_i16(i16 %a, i16 %b) nounwind ssp {
entry:
; CHECK: icmp_eq_i16
; CHECK: sxth w0, w0
; CHECK: sxth w1, w1
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, ne
  %cmp = icmp eq i16 %a, %b
  ret i1 %cmp
}

define zeroext i1 @icmp_eq_i8(i8 %a, i8 %b) nounwind ssp {
entry:
; CHECK: icmp_eq_i8
; CHECK: sxtb w0, w0
; CHECK: sxtb w1, w1
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, ne
  %cmp = icmp eq i8 %a, %b
  ret i1 %cmp
}

define i32 @icmp_i16_unsigned(i16 %a, i16 %b) nounwind {
entry:
; CHECK: icmp_i16_unsigned
; CHECK: uxth w0, w0
; CHECK: uxth w1, w1
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, cs
  %cmp = icmp ult i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i8_signed(i8 %a, i8 %b) nounwind {
entry:
; CHECK: @icmp_i8_signed
; CHECK: sxtb w0, w0
; CHECK: sxtb w1, w1
; CHECK: cmp  w0, w1
; CHECK: csinc w0, wzr, wzr, le
  %cmp = icmp sgt i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define i32 @icmp_i16_signed_const(i16 %a) nounwind {
entry:
; CHECK: icmp_i16_signed_const
; CHECK: sxth w0, w0
; CHECK: cmn  w0, #233
; CHECK: csinc w0, wzr, wzr, ge
; CHECK: and w0, w0, #0x1
  %cmp = icmp slt i16 %a, -233
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i8_signed_const(i8 %a) nounwind {
entry:
; CHECK: icmp_i8_signed_const
; CHECK: sxtb w0, w0
; CHECK: cmp  w0, #124
; CHECK: csinc w0, wzr, wzr, le
; CHECK: and w0, w0, #0x1
  %cmp = icmp sgt i8 %a, 124
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i1_unsigned_const(i1 %a) nounwind {
entry:
; CHECK: icmp_i1_unsigned_const
; CHECK: and w0, w0, #0x1
; CHECK: cmp  w0, #0
; CHECK: csinc w0, wzr, wzr, cs
; CHECK: and w0, w0, #0x1
  %cmp = icmp ult i1 %a, 0
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}
