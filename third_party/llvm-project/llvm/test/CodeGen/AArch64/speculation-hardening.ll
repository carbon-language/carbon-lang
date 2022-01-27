; RUN: sed -e 's/SLHATTR/speculative_load_hardening/' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu | FileCheck %s --check-prefixes=CHECK,SLH
; RUN: sed -e 's/SLHATTR//' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu | FileCheck %s --check-prefixes=CHECK,NOSLH
; RUN: sed -e 's/SLHATTR/speculative_load_hardening/' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -global-isel | FileCheck %s --check-prefixes=CHECK,SLH
; RUN: sed -e 's/SLHATTR//' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -global-isel | FileCheck %s --check-prefixes=CHECK,NOSLH
; RUN: sed -e 's/SLHATTR/speculative_load_hardening/' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -fast-isel | FileCheck %s --check-prefixes=CHECK,SLH
; RUN: sed -e 's/SLHATTR//' %s | llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -fast-isel | FileCheck %s --check-prefixes=CHECK,NOSLH

define i32 @f(i8* nocapture readonly %p, i32 %i, i32 %N) local_unnamed_addr SLHATTR {
; CHECK-LABEL: f
entry:
; SLH:  cmp sp, #0
; SLH:  csetm x16, ne
; NOSLH-NOT:  cmp sp, #0
; NOSLH-NOT:  csetm x16, ne

; SLH:  mov [[TMPREG:x[0-9]+]], sp
; SLH:  and [[TMPREG]], [[TMPREG]], x16
; SLH:  mov sp, [[TMPREG]]
; NOSLH-NOT:  mov {{x[0-9]+}}, sp
; NOSLH-NOT:  and [[TMPREG:x[0-9]+]], [[TMPREG]], x16
; NOSLH-NOT:  mov sp, {{x[0-9]+}}
  %call = tail call i32 @tail_callee(i32 %i)
; SLH:  cmp sp, #0
; SLH:  csetm x16, ne
; NOSLH-NOT:  cmp sp, #0
; NOSLH-NOT:  csetm x16, ne
  %cmp = icmp slt i32 %call, %N
  br i1 %cmp, label %if.then, label %return
; CHECK: b.[[COND:(ge)|(lt)|(ne)|(eq)]]

if.then:                                          ; preds = %entry
; NOSLH-NOT: csel x16, x16, xzr, {{(lt)|(ge)|(eq)|(ne)}}
; SLH-DAG: csel x16, x16, xzr, {{(lt)|(ge)|(eq)|(ne)}}
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i8, i8* %p, i64 %idxprom
  %0 = load i8, i8* %arrayidx, align 1
; CHECK-DAG:      ldrb [[LOADED:w[0-9]+]],
  %conv = zext i8 %0 to i32
  br label %return

; SLH-DAG: csel x16, x16, xzr, [[COND]]
; NOSLH-NOT: csel x16, x16, xzr, [[COND]]
return:                                           ; preds = %entry, %if.then
  %retval.0 = phi i32 [ %conv, %if.then ], [ 0, %entry ]
; SLH:  mov [[TMPREG:x[0-9]+]], sp
; SLH:  and [[TMPREG]], [[TMPREG]], x16
; SLH:  mov sp, [[TMPREG]]
; NOSLH-NOT:  mov {{x[0-9]+}}, sp
; NOSLH-NOT:  and [[TMPREG:x[0-9]+]], [[TMPREG]], x16
; NOSLH-NOT:  mov sp, {{x[0-9]+}}
  ret i32 %retval.0
}

; Make sure that for a tail call, taint doesn't get put into SP twice.
define i32 @tail_caller(i32 %a) local_unnamed_addr SLHATTR {
; CHECK-LABEL: tail_caller:
; SLH:     mov [[TMPREG:x[0-9]+]], sp
; SLH:     and [[TMPREG]], [[TMPREG]], x16
; SLH:     mov sp, [[TMPREG]]
; NOSLH-NOT:     mov {{x[0-9]+}}, sp
; NOSLH-NOT:     and [[TMPREG:x[0-9]+]], [[TMPREG]], x16
; NOSLH-NOT:     mov sp, {{x[0-9]+}}
; SLH:     b tail_callee
; SLH-NOT:        cmp sp, #0
  %call = tail call i32 @tail_callee(i32 %a)
  ret i32 %call
}

declare i32 @tail_callee(i32) local_unnamed_addr

; Verify that no cb(n)z/tb(n)z instructions are produced when implementing
; SLH
define i32 @compare_branch_zero(i32, i32) SLHATTR {
; CHECK-LABEL: compare_branch_zero
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %then, label %else
;SLH-NOT:   cb{{n?}}z
;NOSLH:     cb{{n?}}z
then:
  %4 = sdiv i32 5, %1
  ret i32 %4
else:
  %5 = sdiv i32 %1, %0
  ret i32 %5
}

define i32 @test_branch_zero(i32, i32) SLHATTR {
; CHECK-LABEL: test_branch_zero
  %3 = and i32 %0, 16
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %then, label %else
;SLH-NOT:   tb{{n?}}z
;NOSLH:     tb{{n?}}z
then:
  %5 = sdiv i32 5, %1
  ret i32 %5
else:
  %6 = sdiv i32 %1, %0
  ret i32 %6
}

define i32 @landingpad(i32 %l0, i32 %l1) SLHATTR personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: landingpad
entry:
; SLH:  cmp sp, #0
; SLH:  csetm x16, ne
; NOSLH-NOT:  cmp sp, #0
; NOSLH-NOT:  csetm x16, ne
; CHECK: bl _Z10throwing_fv
  invoke void @_Z10throwing_fv()
          to label %exit unwind label %lpad
; SLH:  cmp sp, #0
; SLH:  csetm x16, ne

lpad:
  %l4 = landingpad { i8*, i32 }
          catch i8* null
; SLH:  cmp sp, #0
; SLH:  csetm x16, ne
; NOSLH-NOT:  cmp sp, #0
; NOSLH-NOT:  csetm x16, ne
  %l5 = extractvalue { i8*, i32 } %l4, 0
  %l6 = tail call i8* @__cxa_begin_catch(i8* %l5)
  %l7 = icmp sgt i32 %l0, %l1
  br i1 %l7, label %then, label %else
; GlobalISel lowers the branch to a b.ne sometimes instead of b.ge as expected..
; CHECK: b.[[COND:(le)|(gt)|(ne)|(eq)]]

then:
; SLH-DAG: csel x16, x16, xzr, [[COND]]
  %l9 = sdiv i32 %l0, %l1
  br label %postif

else:
; SLH-DAG: csel x16, x16, xzr, {{(gt)|(le)|(eq)|(ne)}}
  %l11 = sdiv i32 %l1, %l0
  br label %postif

postif:
  %l13 = phi i32 [ %l9, %then ], [ %l11, %else ]
  tail call void @__cxa_end_catch()
  br label %exit

exit:
  %l15 = phi i32 [ %l13, %postif ], [ 0, %entry ]
  ret i32 %l15
}

declare i32 @__gxx_personality_v0(...)
declare void @_Z10throwing_fv() local_unnamed_addr
declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr
declare void @__cxa_end_catch() local_unnamed_addr
