; RUN: opt -passes='require<scalar-evolution>,require<aa>,loop(print-access-info)' -disable-output  < %s 2>&1 | FileCheck %s

%s1 = type { [32000 x double], [32000 x double], [32000 x double] }

define i32 @load_with_pointer_phi_no_runtime_checks(%s1* %data) {
; CHECK-LABEL: load_with_pointer_phi_no_runtime_checks
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Memory dependences are safe
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp5 = icmp ult i64 %iv, 15999
  %arrayidx = getelementptr inbounds %s1, %s1 * %data, i64 0, i32 0, i64 %iv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %loop.header
  %gep.1 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 1, i64 %iv
  br label %loop.latch

if.else:                                          ; preds = %loop.header
  %gep.2 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 2, i64 %iv
  br label %loop.latch

loop.latch:                                          ; preds = %if.else, %if.then
  %gep.2.sink = phi double* [ %gep.2, %if.else ], [ %gep.1, %if.then ]
  %v8 = load double, double* %gep.2.sink, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %arrayidx, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_no_runtime_checks(%s1* %data) {
; CHECK-LABEL: 'store_with_pointer_phi_no_runtime_checks'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Memory dependences are safe
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp5 = icmp ult i64 %iv, 15999
  %arrayidx = getelementptr inbounds %s1, %s1 * %data, i64 0, i32 0, i64 %iv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %loop.header
  %gep.1 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 1, i64 %iv
  br label %loop.latch

if.else:                                          ; preds = %loop.header
  %gep.2 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 2, i64 %iv
  br label %loop.latch

loop.latch:                                          ; preds = %if.else, %if.then
  %gep.2.sink = phi double* [ %gep.2, %if.else ], [ %gep.1, %if.then ]
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %gep.2.sink, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_runtime_checks(double* %A, double* %B, double* %C) {
; CHECK-LABEL: 'store_with_pointer_phi_runtime_checks'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Memory dependences are safe with run-time checks
; CHECK:         Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[GROUP_B:.+]]):
; CHECK-NEXT:        %gep.1 = getelementptr inbounds double, double* %B, i64 %iv
; CHECK-NEXT:      Against group ([[GROUP_C:.+]]):
; CHECK-NEXT:        %gep.2 = getelementptr inbounds double, double* %C, i64 %iv
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[GROUP_B]]):
; CHECK-NEXT:        %gep.1 = getelementptr inbounds double, double* %B, i64 %iv
; CHECK-NEXT:      Against group ([[GROUP_A:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[GROUP_C]]):
; CHECK-NEXT:        %gep.2 = getelementptr inbounds double, double* %C, i64 %iv
; CHECK-NEXT:      Against group ([[GROUP_A]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp5 = icmp ult i64 %iv, 15999
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %loop.header
  %gep.1 = getelementptr inbounds double, double* %B, i64 %iv
  br label %loop.latch

if.else:                                          ; preds = %loop.header
  %gep.2 = getelementptr inbounds double, double* %C, i64 %iv
  br label %loop.latch

loop.latch:                                          ; preds = %if.else, %if.then
  %gep.2.sink = phi double* [ %gep.2, %if.else ], [ %gep.1, %if.then ]
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %gep.2.sink, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @load_with_pointer_phi_outside_loop(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: 'load_with_pointer_phi_outside_loop'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %v8 = load double, double* %ptr, align 8 ->
; CHECK-NEXT:          store double %mul16, double* %arrayidx, align 8
;
entry:
  br i1 %c.0, label %if.then, label %if.else

if.then:
  br label %loop.ph

if.else:
  %ptr.select = select i1 %c.1, double* %C, double* %B
  br label %loop.ph

loop.ph:
  %ptr = phi double* [ %A, %if.then ], [ %ptr.select, %if.else ]
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop.header ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %ptr, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %arrayidx, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_outside_loop(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: 'store_with_pointer_phi_outside_loop'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %v8 = load double, double* %arrayidx, align 8 ->
; CHECK-NEXT:          store double %mul16, double* %ptr, align 8
;
entry:
  br i1 %c.0, label %if.then, label %if.else

if.then:
  br label %loop.ph

if.else:
  %ptr.select = select i1 %c.1, double* %C, double* %B
  br label %loop.ph

loop.ph:
  %ptr = phi double* [ %A, %if.then ], [ %ptr.select, %if.else ]
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop.header ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %ptr, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_incoming_phi(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: 'store_with_pointer_phi_incoming_phi'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %v8 = load double, double* %arrayidx, align 8 ->
; CHECK-NEXT:          store double %mul16, double* %ptr.2, align 8
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[GROUP_C:.+]]):
; CHECK-NEXT:      double* %C
; CHECK-NEXT:      Against group ([[GROUP_B:.+]]):
; CHECK-NEXT:      double* %B
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[GROUP_C]]):
; CHECK-NEXT:      double* %C
; CHECK-NEXT:      Against group ([[GROUP_A:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:      double* %A
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[GROUP_B]]):
; CHECK-NEXT:      double* %B
; CHECK-NEXT:      Against group ([[GROUP_A]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:      double* %A
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[GROUP_C]]:
; CHECK-NEXT:        (Low: %C High: (8 + %C))
; CHECK-NEXT:          Member: %C
; CHECK-NEXT:      Group [[GROUP_B]]:
; CHECK-NEXT:        (Low: %B High: (8 + %B))
; CHECK-NEXT:          Member: %B
; CHECK-NEXT:      Group [[GROUP_A]]:
; CHECK-NEXT:        (Low: %A High: (256000 + %A))
; CHECK-NEXT:          Member: {%A,+,8}<nuw><%loop.header>
; CHECK-NEXT:          Member: %A
; CHECK-EMPTY
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  br i1 %c.0, label %loop.then, label %loop.latch

loop.then:
  br i1 %c.0, label %loop.then.2, label %loop.else.2

loop.then.2:
  br label %merge.2

loop.else.2:
  br label %merge.2


merge.2:
  %ptr = phi double* [ %A, %loop.then.2 ], [ %B, %loop.else.2 ]
  br label %loop.latch


loop.latch:
  %ptr.2 = phi double* [ %ptr, %merge.2], [ %C, %loop.header ]
  store double %mul16, double* %ptr.2, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

; Test cases with pointer phis forming a cycle.
define i32 @store_with_pointer_phi_incoming_phi_irreducible_cycle(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: 'store_with_pointer_phi_incoming_phi_irreducible_cycle'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %v8 = load double, double* %arrayidx, align 8 ->
; CHECK-NEXT:          store double %mul16, double* %ptr.3, align 8
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[GROUP_C:.+]]):
; CHECK-NEXT:      double* %C
; CHECK-NEXT:      Against group ([[GROUP_B:.+]]):
; CHECK-NEXT:      double* %B
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[GROUP_C]]):
; CHECK-NEXT:      double* %C
; CHECK-NEXT:      Against group ([[GROUP_A:.+]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:      double* %A
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[GROUP_B]]):
; CHECK-NEXT:      double* %B
; CHECK-NEXT:      Against group ([[GROUP_A]]):
; CHECK-NEXT:        %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
; CHECK-NEXT:      double* %A
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[GROUP_C]]
; CHECK-NEXT:        (Low: %C High: (8 + %C))
; CHECK-NEXT:          Member: %C
; CHECK-NEXT:      Group [[GROUP_B]]
; CHECK-NEXT:        (Low: %B High: (8 + %B))
; CHECK-NEXT:          Member: %B
; CHECK-NEXT:      Group [[GROUP_A]]
; CHECK-NEXT:        (Low: %A High: (256000 + %A))
; CHECK-NEXT:          Member: {%A,+,8}<nuw><%loop.header>
; CHECK-NEXT:          Member: %A
; CHECK-EMPTY
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  br i1 %c.0, label %loop.then, label %loop.latch

loop.then:
  br i1 %c.0, label %BB.A, label %BB.B

BB.A:
  %ptr = phi double* [ %A, %loop.then ], [ %ptr.2, %BB.B ]
  br label %BB.B

BB.B:
  %ptr.2 = phi double* [ %ptr, %BB.A ], [ %B, %loop.then ]
  br i1 %c.1, label %loop.latch, label %BB.A

loop.latch:
  %ptr.3 = phi double* [ %ptr.2, %BB.B ], [ %C, %loop.header ]
  store double %mul16, double* %ptr.3, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_outside_loop_select(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: 'store_with_pointer_phi_outside_loop_select'
; CHECK-NEXT:  loop.header:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop.
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %v8 = load double, double* %arrayidx, align 8 ->
; CHECK-NEXT:          store double %mul16, double* %ptr, align 8
;
entry:
  br i1 %c.0, label %if.then, label %if.else

if.then:
  br label %loop.ph

if.else:
  %ptr.select = select i1 %c.1, double* %C, double* %B
  br label %loop.ph

loop.ph:
  %ptr = phi double* [ %A, %if.then ], [ %ptr.select, %if.else ]
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop.header ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %ptr, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_in_same_bb_use_other_phi(double* %A, double* %B, double* %C, double* %D, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: Loop access info in function 'store_with_pointer_phi_in_same_bb_use_other_phi':
; CHECK-NEXT:   loop.header:
; CHECK-NEXT:     Report: cannot identify array bounds
; CHECK-NEXT:     Dependences:
; CHECK-NEXT:     Run-time memory checks:
; CHECK-NEXT:     Grouped accesses:
; CHECK-EMPTY:
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %ptr.0 = phi double* [ %C, %entry ], [ %D, %loop.header ]
  %ptr.1 = phi double* [ %B, %entry ], [ %ptr.0, %loop.header ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.header ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %ptr.1, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define void @phi_load_store_memdep_check(i1 %c, i16* %A, i16* %B, i16* %C) {
; CHECK-LABEL: Loop access info in function 'phi_load_store_memdep_check':
; CHECK-NEXT:   for.body:
; CHECK-NEXT:    Report: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %lv3 = load i16, i16* %c.sink, align 2 ->
; CHECK-NEXT:          store i16 %add, i16* %c.sink, align 1
; CHECK-EMPTY:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %lv3 = load i16, i16* %c.sink, align 2 ->
; CHECK-NEXT:          store i16 %add, i16* %c.sink, align 1
; CHECK-EMPTY:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          %lv = load i16, i16* %A, align 1 ->
; CHECK-NEXT:          store i16 %lv, i16* %A, align 1
; CHECK-EMPTY:
; CHECK-NEXT:      Unknown:
; CHECK-NEXT:          store i16 %lv, i16* %A, align 1 ->
; CHECK-NEXT:          %lv2 = load i16, i16* %A, align 1
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group ([[GROUP_A:.+]]):
; CHECK-NEXT:      i16* %A
; CHECK-NEXT:      i16* %A
; CHECK-NEXT:      Against group ([[GROUP_C:.+]]):
; CHECK-NEXT:      i16* %C
; CHECK-NEXT:      i16* %C
; CHECK-NEXT:    Check 1:
; CHECK-NEXT:      Comparing group ([[GROUP_A]]):
; CHECK-NEXT:      i16* %A
; CHECK-NEXT:      i16* %A
; CHECK-NEXT:      Against group ([[GROUP_B:.+]]):
; CHECK-NEXT:      i16* %B
; CHECK-NEXT:      i16* %B
; CHECK-NEXT:    Check 2:
; CHECK-NEXT:      Comparing group ([[GROUP_C]]):
; CHECK-NEXT:      i16* %C
; CHECK-NEXT:      i16* %C
; CHECK-NEXT:      Against group ([[GROUP_B]]):
; CHECK-NEXT:      i16* %B
; CHECK-NEXT:      i16* %B
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group [[GROUP_A]]
; CHECK-NEXT:        (Low: %A High: (2 + %A))
; CHECK-NEXT:          Member: %A
; CHECK-NEXT:          Member: %A
; CHECK-NEXT:      Group [[GROUP_C]]
; CHECK-NEXT:        (Low: %C High: (2 + %C))
; CHECK-NEXT:          Member: %C
; CHECK-NEXT:          Member: %C
; CHECK-NEXT:      Group [[GROUP_B]]
; CHECK-NEXT:        (Low: %B High: (2 + %B))
; CHECK-NEXT:          Member: %B
; CHECK-NEXT:          Member: %B
; CHECK-EMPTY:
;
entry:
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %iv = phi i16 [ 0, %entry ], [ %iv.next, %if.end ]
  %lv = load i16, i16* %A, align 1
  store i16 %lv, i16* %A, align 1
  br i1 %c, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %lv2 = load i16, i16* %A, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %c.sink = phi i16* [ %B, %if.then ], [ %C, %for.body ]
  %lv3 = load i16, i16* %c.sink
  %add = add i16 %lv3, 10
  store i16 %add, i16* %c.sink, align 1
  %iv.next = add nuw nsw i16 %iv, 1
  %tobool.not = icmp eq i16 %iv.next, 1000
  br i1 %tobool.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %if.end
  ret void
}
