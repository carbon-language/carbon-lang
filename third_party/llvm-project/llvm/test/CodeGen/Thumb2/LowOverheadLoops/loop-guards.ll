; RUN: llc -mtriple=thumbv8.1m.main -disable-arm-loloops=false -mattr=+lob -stop-after=arm-low-overhead-loops --verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv8.1m.main -disable-arm-loloops=false -mattr=+lob -stop-after=arm-low-overhead-loops --verify-machineinstrs %s -o - | FileCheck %s --check-prefix=CHECK-GLOBAL

; Not implemented as a mir test so that changes the generic HardwareLoop can
; also be tested. These functions have been taken from
; Transforms/HardwareLoops/loop-guards.ll in which can be seen the generation
; of a few test.set intrinsics, but only one (ne_trip_count) gets generated
; here. Simplifications result in icmps changing and maybe also the CFG. So,
; TODO: Teach the HardwareLoops some better pattern recognition.

; CHECK-GLOBAL-NOT: DoLoopStart
; CHECK-GLOBAL-NOT: WhileLoopStart
; CHECK-GLOBAL-NOT: LoopEnd

; CHECK: ne_and_guard
; CHECK: body:
; CHECK: bb.0.entry:
; CHECK:   t2CMPri renamable $lr, 0
; CHECK:   tBcc %bb.4
; CHECK: bb.2.while.body.preheader:
; CHECK-NOT:   $lr = t2DLS killed renamable $lr
; CHECK: bb.3.while.body:
; CHECK:   $lr = t2LEUpdate killed renamable $lr, %bb.3
define void @ne_and_guard(i1 zeroext %t1, i1 zeroext %t2, i32* nocapture %a, i32* nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  %cmp6 = icmp ne i32 %N, 0
  %or.cond = and i1 %brmerge.demorgan, %cmp6
  br i1 %or.cond, label %while.body, label %if.end

while.body:                                       ; preds = %while.body, %entry
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.08 = phi i32* [ %incdec.ptr3, %while.body ], [ %a, %entry ]
  %b.addr.07 = phi i32* [ %incdec.ptr, %while.body ], [ %b, %entry ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.07, i32 1
  %tmp = load i32, i32* %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %a.addr.08, i32 1
  store i32 %tmp, i32* %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %entry
  ret void
}

; TODO: This could generate WLS
; CHECK: ne_preheader
; CHECK: body:
; CHECK: bb.0.entry:
; CHECK:   t2CMPri renamable $lr, 0
; CHECK:   tBcc %bb.4
; CHECK: bb.2.while.body.preheader:
; CHECK-NOT:   $lr = t2DLS killed renamable $lr
; CHECK: bb.3.while.body:
; CHECK:   $lr = t2LEUpdate killed renamable $lr, %bb.3
define void @ne_preheader(i1 zeroext %t1, i1 zeroext %t2, i32* nocapture %a, i32* nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.preheader, label %if.end

while.preheader:                                  ; preds = %entry
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %while.body, label %if.end

while.body:                                       ; preds = %while.body, %while.preheader
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %while.preheader ]
  %a.addr.08 = phi i32* [ %incdec.ptr3, %while.body ], [ %a, %while.preheader ]
  %b.addr.07 = phi i32* [ %incdec.ptr, %while.body ], [ %b, %while.preheader ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.07, i32 1
  %tmp = load i32, i32* %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %a.addr.08, i32 1
  store i32 %tmp, i32* %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %while.preheader, %entry
  ret void
}

; TODO: This could generate WLS
; CHECK: eq_preheader
; CHECK: body:
; CHECK: bb.0.entry:
; CHECK:   t2CMPri renamable $lr, 0
; CHECK:   tBcc %bb.4
; CHECK: bb.2.while.body.preheader:
; CHECK-NOT:   $lr = t2DLS killed renamable $lr
; CHECK: bb.3.while.body:
; CHECK:   $lr = t2LEUpdate killed renamable $lr, %bb.3
define void @eq_preheader(i1 zeroext %t1, i1 zeroext %t2, i32* nocapture %a, i32* nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.preheader, label %if.end

while.preheader:                                  ; preds = %entry
  %cmp = icmp eq i32 %N, 0
  br i1 %cmp, label %if.end, label %while.body

while.body:                                       ; preds = %while.body, %while.preheader
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %while.preheader ]
  %a.addr.08 = phi i32* [ %incdec.ptr3, %while.body ], [ %a, %while.preheader ]
  %b.addr.07 = phi i32* [ %incdec.ptr, %while.body ], [ %b, %while.preheader ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.07, i32 1
  %tmp = load i32, i32* %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %a.addr.08, i32 1
  store i32 %tmp, i32* %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %while.preheader, %entry
  ret void
}

; TODO: This could generate WLS
; CHECK: ne_prepreheader
; CHECK: body:
; CHECK: bb.0.entry:
; CHECK:   t2CMPri renamable $lr, 0
; CHECK:   tBcc %bb.4
; CHECK: bb.2.while.body.preheader:
; CHECK-NOT:   $lr = t2DLS killed renamable $lr
; CHECK: bb.3.while.body:
; CHECK:   $lr = t2LEUpdate killed renamable $lr, %bb.3
define void @ne_prepreheader(i1 zeroext %t1, i1 zeroext %t2, i32* nocapture %a, i32* nocapture readonly %b, i32 %N) {
entry:
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %while.preheader, label %if.end

while.preheader:                                  ; preds = %entry
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.body, label %if.end

while.body:                                       ; preds = %while.body, %while.preheader
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %while.preheader ]
  %a.addr.08 = phi i32* [ %incdec.ptr3, %while.body ], [ %a, %while.preheader ]
  %b.addr.07 = phi i32* [ %incdec.ptr, %while.body ], [ %b, %while.preheader ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.07, i32 1
  %tmp = load i32, i32* %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %a.addr.08, i32 1
  store i32 %tmp, i32* %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %while.preheader, %entry
  ret void
}

; CHECK: be_ne
; CHECK: body:
; CHECK: bb.0.entry:
; CHECK:   $lr =
; CHECK: bb.2.do.body:
; CHECK:   $lr = t2LEUpdate killed renamable $lr, %bb.2
define void @be_ne(i32* nocapture %a, i32* nocapture readonly %b, i32 %N) {
entry:
  %cmp = icmp ne i32 %N, 0
  %sub = sub i32 %N, 1
  %be = select i1 %cmp, i32 0, i32 %sub
  %cmp.1 = icmp ne i32 %be, 0
  br i1 %cmp.1, label %do.body, label %if.end

do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi i32* [ %incdec.ptr, %do.body ], [ %b, %entry ]
  %a.addr.0 = phi i32* [ %incdec.ptr3, %do.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.0, i32 1
  %tmp = load i32, i32* %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %a.addr.0, i32 1
  store i32 %tmp, i32* %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp.2 = icmp ult i32 %inc, %N
  br i1 %cmp.2, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK: ne_trip_count
; CHECK: body:
; CHECK: bb.0.entry:
; CHECK:   $lr = t2WLS killed renamable $r3, %bb.3
; CHECK: bb.1.do.body.preheader:
; CHECK: bb.2.do.body:
; CHECK:   $lr = t2LEUpdate killed renamable $lr, %bb.2
define void @ne_trip_count(i1 zeroext %t1, i32* nocapture %a, i32* nocapture readonly %b, i32 %N) {
entry:
  br label %do.body.preheader

do.body.preheader:
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %do.body, label %if.end

do.body:
  %b.addr.0 = phi i32* [ %incdec.ptr, %do.body ], [ %b, %do.body.preheader ]
  %a.addr.0 = phi i32* [ %incdec.ptr3, %do.body ], [ %a, %do.body.preheader ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %do.body.preheader ]
  %incdec.ptr = getelementptr inbounds i32, i32* %b.addr.0, i32 1
  %tmp = load i32, i32* %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, i32* %a.addr.0, i32 1
  store i32 %tmp, i32* %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp.1 = icmp ult i32 %inc, %N
  br i1 %cmp.1, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}
