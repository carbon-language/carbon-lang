; RUN: opt %s -rewrite-statepoints-for-gc -S | FileCheck  %s

define i64 addrspace(1)* @test(<2 x i64 addrspace(1)*> %vec, i32 %idx) gc "statepoint-example" {
; CHECK-LABEL: @test
; CHECK: extractelement
; CHECK: extractelement
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%base_ee, %base_ee)
; CHECK: gc.relocate
; CHECK-DAG: ; (%base_ee, %obj)
; Note that the second extractelement is actually redundant here.  A correct output would
; be to reuse the existing obj as a base since it is actually a base pointer.
entry:
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)

  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test2(<2 x i64 addrspace(1)*>* %ptr, i1 %cnd, i32 %idx1, i32 %idx2) 
    gc "statepoint-example" {
; CHECK-LABEL: test2
entry:
  br i1 %cnd, label %taken, label %untaken
taken:
  %obja = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge
untaken:
  %objb = load <2 x i64 addrspace(1)*>, <2 x i64 addrspace(1)*>* %ptr
  br label %merge
merge:
  %vec = phi <2 x i64 addrspace(1)*> [%obja, %taken], [%objb, %untaken]
  br i1 %cnd, label %taken2, label %untaken2
taken2:
  %obj0 = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx1
  br label %merge2
untaken2:
  %obj1 = extractelement <2 x i64 addrspace(1)*> %vec, i32 %idx2
  br label %merge2
merge2:
; CHECK-LABEL: merge2:
; CHECK: %obj.base = phi i64 addrspace(1)*
; CHECK: %obj = phi i64 addrspace(1)*
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%obj.base, %obj)
; CHECK: gc.relocate
; CHECK-DAG: ; (%obj.base, %obj.base)
  %obj = phi i64 addrspace(1)* [%obj0, %taken2], [%obj1, %untaken2]
  %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}

define i64 addrspace(1)* @test3(i64 addrspace(1)* %ptr) 
    gc "statepoint-example" {
; CHECK-LABEL: test3
entry:
  %vec = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %ptr, i32 0
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
; CHECK: insertelement
; CHECK: extractelement
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%ptr, %obj)
   %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}
define i64 addrspace(1)* @test4(i64 addrspace(1)* %ptr) 
    gc "statepoint-example" {
; CHECK-LABEL: test4
entry:
  %derived = getelementptr i64, i64 addrspace(1)* %ptr, i64 16
  %veca = insertelement <2 x i64 addrspace(1)*> undef, i64 addrspace(1)* %derived, i32 0
  %vec = insertelement <2 x i64 addrspace(1)*> %veca, i64 addrspace(1)* %ptr, i32 1
  %obj = extractelement <2 x i64 addrspace(1)*> %vec, i32 0
; CHECK: statepoint
; CHECK: gc.relocate
; CHECK-DAG: ; (%ptr, %obj)
; CHECK: gc.relocate
; CHECK-DAG: ; (%ptr, %ptr)
   %safepoint_token = call i32 (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0)
  ret i64 addrspace(1)* %obj
}

declare void @do_safepoint()

declare i32 @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
