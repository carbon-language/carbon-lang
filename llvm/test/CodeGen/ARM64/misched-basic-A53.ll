; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a53 -pre-RA-sched=source -enable-misched -verify-misched -debug-only=misched -o - 2>&1 > /dev/null | FileCheck %s
;
; The Cortex-A53 machine model will cause the MADD instruction to be scheduled
; much higher than the ADD instructions in order to hide latency. When not
; specifying a subtarget, the MADD will remain near the end of the block.
;
; CHECK: ********** MI Scheduling **********
; CHECK: main
; CHECK: *** Final schedule for BB#2 ***
; CHECK: SU(13)
; CHECK: MADDWrrr
; CHECK: SU(4)
; CHECK: ADDWri
; CHECK: ********** INTERVALS **********
@main.x = private unnamed_addr constant [8 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 4
@main.y = private unnamed_addr constant [8 x i32] [i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2], align 4

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca [8 x i32], align 4
  %y = alloca [8 x i32], align 4
  %i = alloca i32, align 4
  %xx = alloca i32, align 4
  %yy = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = bitcast [8 x i32]* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([8 x i32]* @main.x to i8*), i64 32, i32 4, i1 false)
  %1 = bitcast [8 x i32]* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ([8 x i32]* @main.y to i8*), i64 32, i32 4, i1 false)
  store i32 0, i32* %xx, align 4
  store i32 0, i32* %yy, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32* %i, align 4
  %cmp = icmp slt i32 %2, 8
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [8 x i32]* %x, i32 0, i64 %idxprom
  %4 = load i32* %arrayidx, align 4
  %add = add nsw i32 %4, 1
  store i32 %add, i32* %xx, align 4
  %5 = load i32* %xx, align 4
  %add1 = add nsw i32 %5, 12
  store i32 %add1, i32* %xx, align 4
  %6 = load i32* %xx, align 4
  %add2 = add nsw i32 %6, 23
  store i32 %add2, i32* %xx, align 4
  %7 = load i32* %xx, align 4
  %add3 = add nsw i32 %7, 34
  store i32 %add3, i32* %xx, align 4
  %8 = load i32* %i, align 4
  %idxprom4 = sext i32 %8 to i64
  %arrayidx5 = getelementptr inbounds [8 x i32]* %y, i32 0, i64 %idxprom4
  %9 = load i32* %arrayidx5, align 4
  %10 = load i32* %yy, align 4
  %mul = mul nsw i32 %10, %9
  store i32 %mul, i32* %yy, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %11 = load i32* %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %12 = load i32* %xx, align 4
  %13 = load i32* %yy, align 4
  %add6 = add nsw i32 %12, %13
  ret i32 %add6
}


; The Cortex-A53 machine model will cause the FDIVvvv_42 to be raised to
; hide latency. Whereas normally there would only be a single FADDvvv_4s
; after it, this test checks to make sure there are more than one.
;
; CHECK: ********** MI Scheduling **********
; CHECK: neon4xfloat:BB#0
; CHECK: *** Final schedule for BB#0 ***
; CHECK: FDIVv4f32
; CHECK: FADDv4f32
; CHECK: FADDv4f32
; CHECK: ********** INTERVALS **********
define <4 x float> @neon4xfloat(<4 x float> %A, <4 x float> %B) {
        %tmp1 = fadd <4 x float> %A, %B;
        %tmp2 = fadd <4 x float> %A, %tmp1;
        %tmp3 = fadd <4 x float> %A, %tmp2;
        %tmp4 = fadd <4 x float> %A, %tmp3;
        %tmp5 = fadd <4 x float> %A, %tmp4;
        %tmp6 = fadd <4 x float> %A, %tmp5;
        %tmp7 = fadd <4 x float> %A, %tmp6;
        %tmp8 = fadd <4 x float> %A, %tmp7;
        %tmp9 = fdiv <4 x float> %A, %B;
        %tmp10 = fadd <4 x float> %tmp8, %tmp9;

        ret <4 x float> %tmp10
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
