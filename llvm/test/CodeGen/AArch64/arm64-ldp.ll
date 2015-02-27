; RUN: llc < %s -march=arm64 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=arm64 -aarch64-unscaled-mem-op=true\
; RUN:   -verify-machineinstrs | FileCheck -check-prefix=LDUR_CHK %s

; CHECK: ldp_int
; CHECK: ldp
define i32 @ldp_int(i32* %p) nounwind {
  %tmp = load i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32* %add.ptr, align 4
  %add = add nsw i32 %tmp1, %tmp
  ret i32 %add
}

; CHECK: ldp_sext_int
; CHECK: ldpsw
define i64 @ldp_sext_int(i32* %p) nounwind {
  %tmp = load i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; CHECK: ldp_long
; CHECK: ldp
define i64 @ldp_long(i64* %p) nounwind {
  %tmp = load i64* %p, align 8
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 1
  %tmp1 = load i64* %add.ptr, align 8
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK: ldp_float
; CHECK: ldp
define float @ldp_float(float* %p) nounwind {
  %tmp = load float* %p, align 4
  %add.ptr = getelementptr inbounds float, float* %p, i64 1
  %tmp1 = load float* %add.ptr, align 4
  %add = fadd float %tmp, %tmp1
  ret float %add
}

; CHECK: ldp_double
; CHECK: ldp
define double @ldp_double(double* %p) nounwind {
  %tmp = load double* %p, align 8
  %add.ptr = getelementptr inbounds double, double* %p, i64 1
  %tmp1 = load double* %add.ptr, align 8
  %add = fadd double %tmp, %tmp1
  ret double %add
}

; Test the load/store optimizer---combine ldurs into a ldp, if appropriate
define i32 @ldur_int(i32* %a) nounwind {
; LDUR_CHK: ldur_int
; LDUR_CHK: ldp     [[DST1:w[0-9]+]], [[DST2:w[0-9]+]], [x0, #-8]
; LDUR_CHK-NEXT: add     w{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32* %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

define i64 @ldur_sext_int(i32* %a) nounwind {
; LDUR_CHK: ldur_sext_int
; LDUR_CHK: ldpsw     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-8]
; LDUR_CHK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @ldur_long(i64* %a) nounwind ssp {
; LDUR_CHK: ldur_long
; LDUR_CHK: ldp     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-16]
; LDUR_CHK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -1
  %tmp1 = load i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %a, i64 -2
  %tmp2 = load i64* %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define float @ldur_float(float* %a) {
; LDUR_CHK: ldur_float
; LDUR_CHK: ldp     [[DST1:s[0-9]+]], [[DST2:s[0-9]+]], [x0, #-8]
; LDUR_CHK-NEXT: add     s{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds float, float* %a, i64 -1
  %tmp1 = load float* %p1, align 2
  %p2 = getelementptr inbounds float, float* %a, i64 -2
  %tmp2 = load float* %p2, align 2
  %tmp3 = fadd float %tmp1, %tmp2
  ret float %tmp3
}

define double @ldur_double(double* %a) {
; LDUR_CHK: ldur_double
; LDUR_CHK: ldp     [[DST1:d[0-9]+]], [[DST2:d[0-9]+]], [x0, #-16]
; LDUR_CHK-NEXT: add     d{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds double, double* %a, i64 -1
  %tmp1 = load double* %p1, align 2
  %p2 = getelementptr inbounds double, double* %a, i64 -2
  %tmp2 = load double* %p2, align 2
  %tmp3 = fadd double %tmp1, %tmp2
  ret double %tmp3
}

; Now check some boundary conditions
define i64 @pairUpBarelyIn(i64* %a) nounwind ssp {
; LDUR_CHK: pairUpBarelyIn
; LDUR_CHK-NOT: ldur
; LDUR_CHK: ldp     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-256]
; LDUR_CHK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -31
  %tmp1 = load i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %a, i64 -32
  %tmp2 = load i64* %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyInSext(i32* %a) nounwind ssp {
; LDUR_CHK: pairUpBarelyInSext
; LDUR_CHK-NOT: ldur
; LDUR_CHK: ldpsw     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-256]
; LDUR_CHK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -63
  %tmp1 = load i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i64 -64
  %tmp2 = load i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyOut(i64* %a) nounwind ssp {
; LDUR_CHK: pairUpBarelyOut
; LDUR_CHK-NOT: ldp
; Don't be fragile about which loads or manipulations of the base register
; are used---just check that there isn't an ldp before the add
; LDUR_CHK: add
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -32
  %tmp1 = load i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %a, i64 -33
  %tmp2 = load i64* %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyOutSext(i32* %a) nounwind ssp {
; LDUR_CHK: pairUpBarelyOutSext
; LDUR_CHK-NOT: ldp
; Don't be fragile about which loads or manipulations of the base register
; are used---just check that there isn't an ldp before the add
; LDUR_CHK: add
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -64
  %tmp1 = load i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i64 -65
  %tmp2 = load i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @pairUpNotAligned(i64* %a) nounwind ssp {
; LDUR_CHK: pairUpNotAligned
; LDUR_CHK-NOT: ldp
; LDUR_CHK: ldur
; LDUR_CHK-NEXT: ldur
; LDUR_CHK-NEXT: add
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -18
  %bp1 = bitcast i64* %p1 to i8*
  %bp1p1 = getelementptr inbounds i8, i8* %bp1, i64 1
  %dp1 = bitcast i8* %bp1p1 to i64*
  %tmp1 = load i64* %dp1, align 1

  %p2 = getelementptr inbounds i64, i64* %a, i64 -17
  %bp2 = bitcast i64* %p2 to i8*
  %bp2p1 = getelementptr inbounds i8, i8* %bp2, i64 1
  %dp2 = bitcast i8* %bp2p1 to i64*
  %tmp2 = load i64* %dp2, align 1

  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @pairUpNotAlignedSext(i32* %a) nounwind ssp {
; LDUR_CHK: pairUpNotAlignedSext
; LDUR_CHK-NOT: ldp
; LDUR_CHK: ldursw
; LDUR_CHK-NEXT: ldursw
; LDUR_CHK-NEXT: add
; LDUR_CHK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -18
  %bp1 = bitcast i32* %p1 to i8*
  %bp1p1 = getelementptr inbounds i8, i8* %bp1, i64 1
  %dp1 = bitcast i8* %bp1p1 to i32*
  %tmp1 = load i32* %dp1, align 1

  %p2 = getelementptr inbounds i32, i32* %a, i64 -17
  %bp2 = bitcast i32* %p2 to i8*
  %bp2p1 = getelementptr inbounds i8, i8* %bp2, i64 1
  %dp2 = bitcast i8* %bp2p1 to i32*
  %tmp2 = load i32* %dp2, align 1

  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
 ret i64 %tmp3
}
