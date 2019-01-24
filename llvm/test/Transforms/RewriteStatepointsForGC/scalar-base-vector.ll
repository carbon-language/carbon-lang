; RUN: opt < %s -rewrite-statepoints-for-gc -S | FileCheck  %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -S | FileCheck  %s

declare void @do_safepoint()
declare i8 addrspace(1)* @def_ptr()

define i32 addrspace(1)* @test1(i8 addrspace(1)* %base1, <2 x i64> %offsets) gc "statepoint-example" {
entry:
  br i1 undef, label %first, label %second

first:
  %base2 = call i8 addrspace(1)* @def_ptr() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %second

second:
; CHECK-LABEL: @test1(
; CHECK: gc.statepoint
; CHECK-DAG: (%ptr.base, %ptr)
; CHECK-DAG: (%ptr.base, %ptr.base)
  %phi = phi i8 addrspace(1)* [ %base1, %entry ], [ %base2, %first ]
  %base.i32 = bitcast i8 addrspace(1)* %phi to i32 addrspace(1)*
  %vec = getelementptr i32, i32 addrspace(1)* %base.i32, <2 x i64> %offsets
  %ptr = extractelement <2 x i32 addrspace(1)*> %vec, i32 1
  call void @do_safepoint() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i32 addrspace(1)* %ptr
}
