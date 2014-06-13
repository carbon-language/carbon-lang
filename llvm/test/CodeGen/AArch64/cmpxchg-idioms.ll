; RUN: llc -mtriple=aarch64-apple-ios7.0 -o - %s | FileCheck %s

define i32 @test_return(i32* %p, i32 %oldval, i32 %newval) {
; CHECK-LABEL: test_return:

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr [[LOADED:w[0-9]+]], [x0]
; CHECK: cmp [[LOADED]], w1
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxr [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x0]
; CHECK: cbnz [[STATUS]], [[LOOP]]

; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: orr w0, wzr, #0x1
; CHECK: ret

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: mov w0, wzr
; CHECK: ret

  %pair = cmpxchg i32* %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = extractvalue { i32, i1 } %pair, 1
  %conv = zext i1 %success to i32
  ret i32 %conv
}

define i1 @test_return_bool(i8* %value, i8 %oldValue, i8 %newValue) {
; CHECK-LABEL: test_return_bool:

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxrb [[LOADED:w[0-9]+]], [x0]
; CHECK: cmp [[LOADED]], w1, uxtb
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxrb [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x0]
; CHECK: cbnz [[STATUS]], [[LOOP]]

; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
  ; FIXME: DAG combine should be able to deal with this.
; CHECK: orr [[TMP:w[0-9]+]], wzr, #0x1
; CHECK: eor w0, [[TMP]], #0x1
; CHECK: ret

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: mov [[TMP:w[0-9]+]], wzr
; CHECK: eor w0, [[TMP]], #0x1
; CHECK: ret

  %pair = cmpxchg i8* %value, i8 %oldValue, i8 %newValue acq_rel monotonic
  %success = extractvalue { i8, i1 } %pair, 1
  %failure = xor i1 %success, 1
  ret i1 %failure
}

define void @test_conditional(i32* %p, i32 %oldval, i32 %newval) {
; CHECK-LABEL: test_conditional:

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxr [[LOADED:w[0-9]+]], [x0]
; CHECK: cmp [[LOADED]], w1
; CHECK: b.ne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: stlxr [[STATUS:w[0-9]+]], w2, [x0]
; CHECK: cbnz [[STATUS]], [[LOOP]]

; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: b _bar

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: b _baz

  %pair = cmpxchg i32* %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = extractvalue { i32, i1 } %pair, 1
  br i1 %success, label %true, label %false

true:
  tail call void @bar() #2
  br label %end

false:
  tail call void @baz() #2
  br label %end

end:
  ret void
}

declare void @bar()
declare void @baz()
