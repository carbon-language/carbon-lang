; RUN: llc -mtriple arm-unknown -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

define void @test_void_return() {
; CHECK-LABEL: name: test_void_return
; CHECK: BX_RET 14, _
entry:
  ret void
}

define i32 @test_add(i32 %x, i32 %y) {
; CHECK-LABEL: name: test_add
; CHECK: liveins: %r0, %r1
; CHECK: [[VREGX:%[0-9]+]]{{.*}} = COPY %r0
; CHECK: [[VREGY:%[0-9]+]]{{.*}} = COPY %r1
; CHECK: [[SUM:%[0-9]+]]{{.*}} = G_ADD [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}

define i32 @test_many_args(i32 %p0, i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) {
; CHECK-LABEL: name: test_many_args
; CHECK: fixedStack:
; CHECK-DAG: id: [[P4:[0-9]]]{{.*}}offset: 0{{.*}}size: 4
; CHECK-DAG: id: [[P5:[0-9]]]{{.*}}offset: 4{{.*}}size: 4
; CHECK: liveins: %r0, %r1, %r2, %r3
; CHECK: [[VREGP2:%[0-9]+]]{{.*}} = COPY %r2
; CHECK: [[FIP5:%[0-9]+]]{{.*}} = G_FRAME_INDEX %fixed-stack.[[P5]]
; CHECK: [[VREGP5:%[0-9]+]]{{.*}} = G_LOAD [[FIP5]]
; CHECK: [[SUM:%[0-9]+]]{{.*}} = G_ADD [[VREGP2]], [[VREGP5]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i32 %p2, %p5
  ret i32 %sum
}
