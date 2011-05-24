; RUN: llc -march=mipsel -mcpu=4ke < %s | FileCheck %s

%0 = type { i8, i16, i32, i64, double, i32, [4 x i8] }
%struct.S1 = type { i8, i16, i32, i64, double, i32 }
%struct.S2 = type { [4 x i32] }
%struct.S3 = type { i8 }

@f1.s1 = internal unnamed_addr constant %0 { i8 1, i16 2, i32 3, i64 4, double 5.000000e+00, i32 6, [4 x i8] undef }, align 8
@f1.s2 = internal unnamed_addr constant %struct.S2 { [4 x i32] [i32 7, i32 8, i32 9, i32 10] }, align 4

define void @f1() nounwind {
entry:
; CHECK: lw  $[[R0:[0-9]+]], %got(f1.s1)($gp)
; CHECK: addiu $[[R1:[0-9]+]], $sp, 16
; CHECK: addiu $[[R0:[0-9]+]], $[[R0]], %lo(f1.s1)
; CHECK: lw  $[[R2:[0-9]+]], 8($[[R0]])
; CHECK: lw  $[[R3:[0-9]+]], 16($[[R0]])
; CHECK: lw  $[[R4:[0-9]+]], 20($[[R0]])
; CHECK: lw  $[[R5:[0-9]+]], 24($[[R0]])
; CHECK: lw  $[[R6:[0-9]+]], 28($[[R0]])
; CHECK: lw  $[[R7:[0-9]+]], 12($[[R0]])
; CHECK: ori $[[R8:[0-9]+]], $[[R1]], 4
; CHECK: sw  $[[R2]], 16($sp)
; CHECK: sw  $[[R7]], 0($[[R8]])
; CHECK: sw  $[[R3]], 24($sp)
; CHECK: sw  $[[R4]], 28($sp)
; CHECK: sw  $[[R5]], 32($sp)
; CHECK: sw  $[[R6]], 36($sp)
; CHECK: lw  $6, 0($[[R0]])
; CHECK: lw  $7, 4($[[R0]])
  %agg.tmp10 = alloca %struct.S3, align 4
  call void @callee1(float 2.000000e+01, %struct.S1* byval bitcast (%0* @f1.s1 to %struct.S1*)) nounwind
  call void @callee2(%struct.S2* byval @f1.s2) nounwind
  %tmp11 = getelementptr inbounds %struct.S3* %agg.tmp10, i32 0, i32 0
  store i8 11, i8* %tmp11, align 4
  call void @callee3(float 2.100000e+01, %struct.S3* byval %agg.tmp10, %struct.S1* byval bitcast (%0* @f1.s1 to %struct.S1*)) nounwind
  ret void
}

declare void @callee1(float, %struct.S1* byval)

declare void @callee2(%struct.S2* byval)

declare void @callee3(float, %struct.S3* byval, %struct.S1* byval)

define void @f2(float %f, %struct.S1* nocapture byval %s1) nounwind {
entry:
; CHECK: addiu $sp, $sp, -56
; CHECK: addiu $[[R0:[0-9]+]], $sp, 64
; CHECK: ori $[[R1:[0-9]+]], $[[R0]], 4
; CHECK: ori $[[R0:[0-9]+]], $[[R0]], 2
; CHECK: sw  $6, 64($sp)
; CHECK: sw  $7, 0($[[R1]])
; CHECK: ldc1 $f[[F0:[0-9]+]], 80($sp)
; CHECK: lw  $[[R2:[0-9]+]], 0($[[R1]])
; CHECK: lh  $[[R1:[0-9]+]], 0($[[R0]])
; CHECK: lb  $[[R0:[0-9]+]], 64($sp)
; CHECK: lw  $[[R3:[0-9]+]], 72($sp)
; CHECK: lw  $[[R4:[0-9]+]], 76($sp)
; CHECK: lw  $4, 88($sp)
; CHECK: sw  $[[R3]], 16($sp)
; CHECK: sw  $[[R4]], 20($sp)
; CHECK: sw  $[[R2]], 24($sp)
; CHECK: sw  $[[R1]], 28($sp)
; CHECK: sw  $[[R0]], 32($sp)
; CHECK: mfc1 $6, $f[[F0]]

  %i2 = getelementptr inbounds %struct.S1* %s1, i32 0, i32 5
  %tmp = load i32* %i2, align 4, !tbaa !0
  %d = getelementptr inbounds %struct.S1* %s1, i32 0, i32 4
  %tmp1 = load double* %d, align 8, !tbaa !3
  %ll = getelementptr inbounds %struct.S1* %s1, i32 0, i32 3
  %tmp2 = load i64* %ll, align 8, !tbaa !4
  %i = getelementptr inbounds %struct.S1* %s1, i32 0, i32 2
  %tmp3 = load i32* %i, align 4, !tbaa !0
  %s = getelementptr inbounds %struct.S1* %s1, i32 0, i32 1
  %tmp4 = load i16* %s, align 2, !tbaa !5
  %c = getelementptr inbounds %struct.S1* %s1, i32 0, i32 0
  %tmp5 = load i8* %c, align 1, !tbaa !1
  tail call void @callee4(i32 %tmp, double %tmp1, i64 %tmp2, i32 %tmp3, i16 signext %tmp4, i8 signext %tmp5, float %f) nounwind
  ret void
}

declare void @callee4(i32, double, i64, i32, i16 signext, i8 signext, float)

define void @f3(%struct.S2* nocapture byval %s2) nounwind {
entry:
; CHECK: addiu $sp, $sp, -56
; CHECK: addiu $[[R0:[0-9]+]], $sp, 56
; CHECK: ori $[[R0:[0-9]+]], $[[R0:[0-9]+]], 4
; CHECK: sw  $4, 56($sp)
; CHECK: sw  $5, 0($[[R0:[0-9]+]])
; CHECK: sw  $6, 64($sp)
; CHECK: sw  $7, 68($sp)
; CHECK: lw  $[[R0:[0-9]+]], 68($sp)
; CHECK: lw  $4, 56($sp)
; CHECK: sw  $[[R0:[0-9]+]], 24($sp)

  %arrayidx = getelementptr inbounds %struct.S2* %s2, i32 0, i32 0, i32 0
  %tmp = load i32* %arrayidx, align 4, !tbaa !0
  %arrayidx2 = getelementptr inbounds %struct.S2* %s2, i32 0, i32 0, i32 3
  %tmp3 = load i32* %arrayidx2, align 4, !tbaa !0
  tail call void @callee4(i32 %tmp, double 2.000000e+00, i64 3, i32 %tmp3, i16 signext 4, i8 signext 5, float 6.000000e+00) nounwind
  ret void
}

define void @f4(float %f, %struct.S3* nocapture byval %s3, %struct.S1* nocapture byval %s1) nounwind {
entry:
; CHECK: addiu $sp, $sp, -56
; CHECK: addiu $[[R0:[0-9]+]], $sp, 64
; CHECK: ori $[[R2:[0-9]+]], $[[R0]], 4
; CHECK: sw  $5, 60($sp)
; CHECK: sw  $6, 64($sp)
; CHECK: sw  $7, 0($[[R2]])
; CHECK: lw  $[[R1:[0-9]+]], 88($sp)
; CHECK: lb  $[[R0:[0-9]+]], 60($sp)
; CHECK: lw  $4, 0($[[R2]])
; CHECK: sw  $[[R1]], 24($sp)
; CHECK: sw  $[[R0]], 32($sp)

  %i = getelementptr inbounds %struct.S1* %s1, i32 0, i32 2
  %tmp = load i32* %i, align 4, !tbaa !0
  %i2 = getelementptr inbounds %struct.S1* %s1, i32 0, i32 5
  %tmp1 = load i32* %i2, align 4, !tbaa !0
  %c = getelementptr inbounds %struct.S3* %s3, i32 0, i32 0
  %tmp2 = load i8* %c, align 1, !tbaa !1
  tail call void @callee4(i32 %tmp, double 2.000000e+00, i64 3, i32 %tmp1, i16 signext 4, i8 signext %tmp2, float 6.000000e+00) nounwind
  ret void
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"double", metadata !1}
!4 = metadata !{metadata !"long long", metadata !1}
!5 = metadata !{metadata !"short", metadata !1}
