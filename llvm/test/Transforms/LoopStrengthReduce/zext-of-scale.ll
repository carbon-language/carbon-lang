; RUN: opt  < %s -S -loop-reduce | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct = type { [8 x i8] }

declare void @use_32(i32)
declare void @use_64(i64)

define void @f(i32 %tmp156, i32* %length_buf_1, i32* %length_buf_0, %struct* %b,
                %struct* %c, %struct* %d, %struct* %e, i32* %length_buf_2,
                i32 %tmp160) {
; CHECK-LABEL: @f(
entry:
  %begin151 = getelementptr inbounds %struct, %struct* %b, i64 0, i32 0, i64 12
  %tmp21 = bitcast i8* %begin151 to i32*
  %begin157 = getelementptr inbounds %struct, %struct* %c, i64 0, i32 0, i64 16
  %tmp23 = bitcast i8* %begin157 to double*
  %begin163 = getelementptr inbounds %struct, %struct* %d, i64 0, i32 0, i64 16
  %tmp25 = bitcast i8* %begin163 to double*
  %length.i820 = load i32, i32* %length_buf_1, align 4, !range !0
  %enter = icmp ne i32 %tmp156, -1
  br i1 %enter, label %ok_146, label %block_81_2

ok_146:
  %var_13 = phi double [ %tmp186, %ok_161 ], [ 0.000000e+00, %entry ]
  %var_17 = phi i32 [ %tmp187, %ok_161 ], [ %tmp156, %entry ]
  %tmp174 = zext i32 %var_17 to i64
  %tmp175 = icmp ult i32 %var_17, %length.i820
  br i1 %tmp175, label %ok_152, label %block_81_2

ok_152:
  %tmp176 = getelementptr inbounds i32, i32* %tmp21, i64 %tmp174
  %tmp177 = load i32, i32* %tmp176, align 4
  %tmp178 = zext i32 %tmp177 to i64
  %length.i836 = load i32, i32* %length_buf_2, align 4, !range !0
  %tmp179 = icmp ult i32 %tmp177, %length.i836
  br i1 %tmp179, label %ok_158, label %block_81_2

ok_158:
  %tmp180 = getelementptr inbounds double, double* %tmp23, i64 %tmp178
  %tmp181 = load double, double* %tmp180, align 8
  %length.i = load i32, i32* %length_buf_0, align 4, !range !0
  %tmp182 = icmp slt i32 %var_17, %length.i
  br i1 %tmp182, label %ok_161, label %block_81_2

ok_161:
; CHECK-LABEL: ok_161:
; CHECK: add
; CHECK-NOT: add
  %tmp183 = getelementptr inbounds double, double* %tmp25, i64 %tmp174
  %tmp184 = load double, double* %tmp183, align 8
  %tmp185 = fmul double %tmp181, %tmp184
  %tmp186 = fadd double %var_13, %tmp185
  %tmp187 = add nsw i32 %var_17, 1
  %tmp188 = icmp slt i32 %tmp187, %tmp160
; CHECK: br
  br i1 %tmp188, label %ok_146, label %block_81

block_81:
  call void @use_64(i64 %tmp174)  ;; pre-inc use
  call void @use_32(i32 %tmp187)  ;; post-inc use
  ret void

block_81_2:
  ret void
}

!0 = !{i32 0, i32 2147483647}
