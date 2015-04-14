; RUN: opt < %s -analyze -branch-prob | FileCheck %s

define i32 @test1(i32 %i, i32* %a) {
; CHECK: Printing analysis {{.*}} for function 'test1'
entry:
  br label %body
; CHECK: edge entry -> body probability is 16 / 16 = 100%

body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body
; CHECK: edge body -> exit probability is 4 / 128
; CHECK: edge body -> body probability is 124 / 128

exit:
  ret i32 %sum
}

define i32 @test2(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test2'
entry:
  %cond = icmp ult i32 %i, 42
  br i1 %cond, label %then, label %else, !prof !0
; CHECK: edge entry -> then probability is 64 / 68
; CHECK: edge entry -> else probability is 4 / 68

then:
  br label %exit
; CHECK: edge then -> exit probability is 16 / 16 = 100%

else:
  br label %exit
; CHECK: edge else -> exit probability is 16 / 16 = 100%

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

!0 = !{!"branch_weights", i32 64, i32 4}

define i32 @test3(i32 %i, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK: Printing analysis {{.*}} for function 'test3'
entry:
  switch i32 %i, label %case_a [ i32 1, label %case_b
                                 i32 2, label %case_c
                                 i32 3, label %case_d
                                 i32 4, label %case_e ], !prof !1
; CHECK: edge entry -> case_a probability is 4 / 80
; CHECK: edge entry -> case_b probability is 4 / 80
; CHECK: edge entry -> case_c probability is 64 / 80
; CHECK: edge entry -> case_d probability is 4 / 80
; CHECK: edge entry -> case_e probability is 4 / 80

case_a:
  br label %exit
; CHECK: edge case_a -> exit probability is 16 / 16 = 100%

case_b:
  br label %exit
; CHECK: edge case_b -> exit probability is 16 / 16 = 100%

case_c:
  br label %exit
; CHECK: edge case_c -> exit probability is 16 / 16 = 100%

case_d:
  br label %exit
; CHECK: edge case_d -> exit probability is 16 / 16 = 100%

case_e:
  br label %exit
; CHECK: edge case_e -> exit probability is 16 / 16 = 100%

exit:
  %result = phi i32 [ %a, %case_a ],
                    [ %b, %case_b ],
                    [ %c, %case_c ],
                    [ %d, %case_d ],
                    [ %e, %case_e ]
  ret i32 %result
}

!1 = !{!"branch_weights", i32 4, i32 4, i32 64, i32 4, i32 4}

define i32 @test4(i32 %x) nounwind uwtable readnone ssp {
; CHECK: Printing analysis {{.*}} for function 'test4'
entry:
  %conv = sext i32 %x to i64
  switch i64 %conv, label %return [
    i64 0, label %sw.bb
    i64 1, label %sw.bb
    i64 2, label %sw.bb
    i64 5, label %sw.bb1
  ], !prof !2
; CHECK: edge entry -> return probability is 7 / 85
; CHECK: edge entry -> sw.bb probability is 14 / 85
; CHECK: edge entry -> sw.bb1 probability is 64 / 85

sw.bb:
  br label %return

sw.bb1:
  br label %return

return:
  %retval.0 = phi i32 [ 5, %sw.bb1 ], [ 1, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}

!2 = !{!"branch_weights", i32 7, i32 6, i32 4, i32 4, i32 64}

declare void @coldfunc() cold

define i32 @test5(i32 %a, i32 %b, i1 %flag) {
; CHECK: Printing analysis {{.*}} for function 'test5'
entry:
  br i1 %flag, label %then, label %else
; CHECK: edge entry -> then probability is 4 / 68
; CHECK: edge entry -> else probability is 64 / 68

then:
  call void @coldfunc()
  br label %exit
; CHECK: edge then -> exit probability is 16 / 16 = 100%

else:
  br label %exit
; CHECK: edge else -> exit probability is 16 / 16 = 100%

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

declare i32 @regular_function(i32 %i)

define i32 @test_cold_call_sites(i32* %a) {
; Test that edges to blocks post-dominated by cold call sites
; are marked as not expected to be taken.
; TODO(dnovillo) The calls to regular_function should not be merged, but
; they are currently being merged. Convert this into a code generation test
; after that is fixed.

; CHECK: Printing analysis {{.*}} for function 'test_cold_call_sites'
; CHECK: edge entry -> then probability is 4 / 68 = 5.88235%
; CHECK: edge entry -> else probability is 64 / 68 = 94.1176% [HOT edge]

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then, label %else

then:
  ; This function is not declared cold, but this call site is.
  %val4 = call i32 @regular_function(i32 %val1) cold
  br label %exit

else:
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  %val3 = call i32 @regular_function(i32 %val2)
  br label %exit

exit:
  %ret = phi i32 [ %val4, %then ], [ %val3, %else ]
  ret i32 %ret
}

define i32 @zero1(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'zero1'
entry:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %then, label %else
; CHECK: edge entry -> then probability is 12 / 32
; CHECK: edge entry -> else probability is 20 / 32

then:
  br label %exit

else:
  br label %exit

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

define i32 @zero2(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'zero2'
entry:
  %cond = icmp ne i32 %i, -1
  br i1 %cond, label %then, label %else
; CHECK: edge entry -> then probability is 20 / 32
; CHECK: edge entry -> else probability is 12 / 32

then:
  br label %exit

else:
  br label %exit

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

