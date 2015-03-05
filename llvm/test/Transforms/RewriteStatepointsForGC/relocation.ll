; RUN: opt %s -rewrite-statepoints-for-gc -S 2>&1 | FileCheck %s


declare void @foo()

define i64 addrspace(1)* @test1(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj2, i1 %condition) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test1
; CHECK-DAG: %obj.relocated
; CHECK-DAG: %obj2.relocated
  %safepoint_token = call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @foo, i32 0, i32 0, i32 0)
  br label %joint

joint:
; CHECK-LABEL: joint:
; CHECK: %phi1 = phi i64 addrspace(1)* [ %obj.relocated, %entry ], [ %obj3, %joint2 ]
  %phi1 = phi i64 addrspace(1)* [ %obj, %entry ], [ %obj3, %joint2 ]
  br i1 %condition, label %use, label %joint2

use:
  br label %joint2

joint2:
; CHECK-LABEL: joint2:
; CHECK: %phi2 = phi i64 addrspace(1)* [ %obj.relocated, %use ], [ %obj2.relocated, %joint ]
; CHECK: %obj3 = getelementptr i64, i64 addrspace(1)* %obj2.relocated, i32 1
  %phi2 = phi i64 addrspace(1)* [ %obj, %use ], [ %obj2, %joint ]
  %obj3 = getelementptr i64, i64 addrspace(1)* %obj2, i32 1
  br label %joint
}

declare i64 addrspace(1)* @generate_obj()

declare void @consume_obj(i64 addrspace(1)*)

declare i1 @rt()

define void @test2() gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
  %obj_init = call i64 addrspace(1)* @generate_obj()
  %obj = getelementptr i64, i64 addrspace(1)* %obj_init, i32 42
  br label %loop

loop:
; CHECK: loop:
; CHECK-DAG: [ %obj_init.relocated, %loop.backedge ]
; CHECK-DAG: [ %obj_init, %entry ]
; CHECK-DAG: [ %obj.relocated, %loop.backedge ]
; CHECK-DAG: [ %obj, %entry ]
  %index = phi i32 [ 0, %entry ], [ %index.inc, %loop.backedge ]
; CHECK-NOT: %location = getelementptr i64, i64 addrspace(1)* %obj, i32 %index
  %location = getelementptr i64, i64 addrspace(1)* %obj, i32 %index
  call void @consume_obj(i64 addrspace(1)* %location)
  %index.inc = add i32 %index, 1
  %condition = call i1 @rt()
  br i1 %condition, label %loop_x, label %loop_y

loop_x:
  br label %loop.backedge

loop.backedge:
  %safepoint_token = call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @do_safepoint, i32 0, i32 0, i32 0)
  br label %loop

loop_y:
  br label %loop.backedge
}

declare void @some_call(i8 addrspace(1)*)

define void @relocate_merge(i1 %cnd, i8 addrspace(1)* %arg) gc "statepoint-example" {
; CHECK-LABEL: @relocate_merge
bci_0:
  br i1 %cnd, label %if_branch, label %else_branch

if_branch:
; CHECK-LABEL: if_branch:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  %safepoint_token = call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @foo, i32 0, i32 0, i32 0)
  br label %join

else_branch:
; CHECK-LABEL: else_branch:
; CHECK: gc.statepoint
; CHECK: gc.relocate
  %safepoint_token1 = call i32 (void ()*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()* @foo, i32 0, i32 0, i32 0)
  br label %join

join:
; We need to end up with a single relocation phi updated from both paths 
; CHECK-LABEL: join:
; CHECK: phi i8 addrspace(1)*
; CHECK-DAG: [ %arg.relocated, %if_branch ]
; CHECK-DAG: [ %arg.relocated4, %else_branch ]
; CHECK-NOT: phi
  call void (i8 addrspace(1)*)* @some_call(i8 addrspace(1)* %arg)
  ret void
}

; Make sure a use in a statepoint gets properly relocated at a previous one.  
; This is basically just making sure that statepoints aren't accidentally 
; treated specially.
define void @test3(i64 addrspace(1)* %obj) gc "statepoint-example" {
entry:
; CHECK-LABEL: @test3
; CHECK: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: gc.statepoint
  %safepoint_token = call i32 (void (i64)*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_isVoidi64f(void (i64)* undef, i32 1, i32 0, i64 undef, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  %safepoint_token1 = call i32 (i32 (i64 addrspace(1)*)*, i32, i32, ...)* @llvm.experimental.gc.statepoint.p0f_i32p1i64f(i32 (i64 addrspace(1)*)* undef, i32 1, i32 0, i64 addrspace(1)* %obj, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret void
}

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidi64f(void (i64)*, i32, i32, ...)

declare i32 @llvm.experimental.gc.statepoint.p0f_i32p1i64f(i32 (i64 addrspace(1)*)*, i32, i32, ...)


declare void @do_safepoint()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(void ()*, i32, i32, ...)
