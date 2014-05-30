; RUN: llc -mtriple=thumbv7s-apple-ios7.0 -o - %s | FileCheck %s

define i32 @test_return(i32* %p, i32 %oldval, i32 %newval) {
; CHECK-LABEL: test_return:

; CHECK: dmb ishst

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldrex [[LOADED:r[0-9]+]], [r0]
; CHECK: cmp [[LOADED]], r1
; CHECK: bne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: strex [[STATUS:r[0-9]+]], {{r[0-9]+}}, [r0]
; CHECK: cmp [[STATUS]], #0
; CHECK: bne [[LOOP]]

; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: movs r0, #1
; CHECK: dmb ish
; CHECK: bx lr

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: movs r0, #0
; CHECK: dmb ish
; CHECK: bx lr

  %loaded = cmpxchg i32* %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = icmp eq i32 %loaded, %oldval
  %conv = zext i1 %success to i32
  ret i32 %conv
}

define i1 @test_return_bool(i8* %value, i8 %oldValue, i8 %newValue) {
; CHECK-LABEL: test_return_bool:

; CHECK: uxtb [[OLDBYTE:r[0-9]+]], r1
; CHECK: dmb ishst

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldrexb [[LOADED:r[0-9]+]], [r0]
; CHECK: cmp [[LOADED]], [[OLDBYTE]]

; CHECK: itt ne
; CHECK: movne r0, #1
; CHECK: bxne lr

; CHECK: strexb [[STATUS:r[0-9]+]], {{r[0-9]+}}, [r0]
; CHECK: cmp [[STATUS]], #0
; CHECK: bne [[LOOP]]

; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: movs r0, #0
; CHECK: bx lr

  %loaded = cmpxchg i8* %value, i8 %oldValue, i8 %newValue acq_rel monotonic
  %failure = icmp ne i8 %loaded, %oldValue
  ret i1 %failure
}

define void @test_conditional(i32* %p, i32 %oldval, i32 %newval) {
; CHECK-LABEL: test_conditional:

; CHECK: dmb ishst

; CHECK: [[LOOP:LBB[0-9]+_[0-9]+]]:
; CHECK: ldrex [[LOADED:r[0-9]+]], [r0]
; CHECK: cmp [[LOADED]], r1
; CHECK: bne [[FAILED:LBB[0-9]+_[0-9]+]]

; CHECK: strex [[STATUS:r[0-9]+]], r2, [r0]
; CHECK: cmp [[STATUS]], #0
; CHECK: bne [[LOOP]]

; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: dmb ish
; CHECK: b.w _bar

; CHECK: [[FAILED]]:
; CHECK-NOT: cmp {{r[0-9]+}}, {{r[0-9]+}}
; CHECK: dmb ish
; CHECK: b.w _baz

  %loaded = cmpxchg i32* %p, i32 %oldval, i32 %newval seq_cst seq_cst
  %success = icmp eq i32 %loaded, %oldval
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
