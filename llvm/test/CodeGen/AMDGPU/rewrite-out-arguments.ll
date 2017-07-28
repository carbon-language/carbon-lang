; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-rewrite-out-arguments < %s | FileCheck %s

; CHECK: %void_one_out_arg_i32_1_use = type { i32 }
; CHECK: %void_one_out_arg_i32_1_use_align = type { i32 }
; CHECK: %void_one_out_arg_i32_2_use = type { i32 }
; CHECK: %void_one_out_arg_i32_2_stores = type { i32 }
; CHECK: %void_one_out_arg_i32_2_stores_clobber = type { i32 }
; CHECK: %void_one_out_arg_i32_pre_call_may_clobber = type { i32 }
; CHECK: %void_one_out_arg_v2i32_1_use = type { <2 x i32> }
; CHECK: %void_one_out_arg_struct_1_use = type { %struct }
; CHECK: %struct = type { i32, i8, float }
; CHECK: %i32_one_out_arg_i32_1_use = type { i32, i32 }
; CHECK: %unused_different_type = type { float }
; CHECK: %multiple_same_return_noalias = type { i32, i32 }
; CHECK: %multiple_same_return_mayalias = type { i32, i32 }
; CHECK: %multiple_same_return_mayalias_order = type { i32, i32 }
; CHECK: %i1_one_out_arg_i32_1_use = type { i1, i32 }
; CHECK: %i1_zeroext_one_out_arg_i32_1_use = type { i1, i32 }
; CHECK: %i1_signext_one_out_arg_i32_1_use = type { i1, i32 }
; CHECK: %p1i32_noalias_one_out_arg_i32_1_use = type { i32 addrspace(1)*, i32 }
; CHECK: %func_ptr_type = type { void ()* }
; CHECK: %out_arg_small_array = type { [4 x i32] }
; CHECK: %num_regs_reach_limit = type { [15 x i32], i32 }
; CHECK: %num_regs_reach_limit_leftover = type { [15 x i32], i32, i32 }
; CHECK: %preserve_debug_info = type { i32 }
; CHECK: %preserve_metadata = type { i32 }

; CHECK-LABEL: define void @no_ret_blocks() #0 {
; CHECK-NEXT: unreachable
define void @no_ret_blocks() #0 {
  unreachable
}

; CHECK-LABEL: @void_one_out_arg_i32_no_use(
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_no_use(i32* %val) #0 {
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @skip_byval_arg(
; CHECK-NEXT: store i32 0, i32* %val
; CHECK-NEXT: ret void
define void @skip_byval_arg(i32* byval %val) #0 {
  store i32 0, i32* %val
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @skip_optnone(
; CHECK-NEXT: store i32 0, i32* %val
; CHECK-NEXT: ret void
define void @skip_optnone(i32* byval %val) #1 {
  store i32 0, i32* %val
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @skip_volatile(
; CHECK-NEXT: store volatile i32 0, i32* %val
; CHECK-NEXT: ret void
define void @skip_volatile(i32* byval %val) #0 {
  store volatile i32 0, i32* %val
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @skip_atomic(
; CHECK-NEXT: store atomic i32 0, i32* %val
; CHECK-NEXT: ret void
define void @skip_atomic(i32* byval %val) #0 {
  store atomic i32 0, i32* %val seq_cst, align 4
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @skip_store_pointer_val(
; CHECK-NEXT: store i32* %val, i32** undef
; CHECK-NEXT: ret void
define void @skip_store_pointer_val(i32* %val) #0 {
  store i32* %val, i32** undef
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @skip_store_gep(
; CHECK-NEXT: %gep = getelementptr inbounds i32, i32* %val, i32 1
; CHECK-NEXT: store i32 0, i32* %gep
; CHECK-NEXT: ret void
define void @skip_store_gep(i32* %val) #0 {
  %gep = getelementptr inbounds i32, i32* %val, i32 1
  store i32 0, i32* %gep
  ret void
}

; CHECK-LABEL: define void @skip_sret(i32* sret %sret, i32* %out) #0 {
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret void
define void @skip_sret(i32* sret %sret, i32* %out) #0 {
  store i32 1, i32* %sret
  store i32 0, i32* %out
  ret void
}

; CHECK-LABEL: define private %void_one_out_arg_i32_1_use @void_one_out_arg_i32_1_use.body(i32* %val) #0 {
; CHECK-NEXT: ret %void_one_out_arg_i32_1_use zeroinitializer

; CHECK-LABEL: @void_one_out_arg_i32_1_use(
; CHECK-NEXT: %2 = call %void_one_out_arg_i32_1_use @void_one_out_arg_i32_1_use.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_i32_1_use %2, 0
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_1_use(i32* %val) #0 {
  store i32 0, i32* %val
  ret void
}

; CHECK-LABEL: define private %void_one_out_arg_i32_1_use_align @void_one_out_arg_i32_1_use_align.body(i32* align 8 %val) #0 {
; CHECK-NEXT: ret %void_one_out_arg_i32_1_use_align zeroinitializer

; CHECK-LABEL: @void_one_out_arg_i32_1_use_align(
; CHECK-NEXT: %2 = call %void_one_out_arg_i32_1_use_align @void_one_out_arg_i32_1_use_align.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_i32_1_use_align %2, 0
; CHECK-NEXT: store i32 %3, i32* %0, align 8
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_1_use_align(i32* align 8 %val) #0 {
  store i32 0, i32* %val, align 8
  ret void
}

; CHECK-LABEL: define private %void_one_out_arg_i32_2_use @void_one_out_arg_i32_2_use.body(i1 %arg0, i32* %val) #0 {
; CHECK: br i1 %arg0, label %ret0, label %ret1

; CHECK: ret0:
; CHECK-NEXT: ret %void_one_out_arg_i32_2_use zeroinitializer

; CHECK: ret1:
; CHECK-NEXT: ret %void_one_out_arg_i32_2_use { i32 9 }

; CHECK-LABEL: define void @void_one_out_arg_i32_2_use(i1, i32*) #2 {
; CHECK-NEXT: %3 = call %void_one_out_arg_i32_2_use @void_one_out_arg_i32_2_use.body(i1 %0, i32* undef)
; CHECK-NEXT: %4 = extractvalue %void_one_out_arg_i32_2_use %3, 0
; CHECK-NEXT: store i32 %4, i32* %1, align 4
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_2_use(i1 %arg0, i32* %val) #0 {
  br i1 %arg0, label %ret0, label %ret1

ret0:
  store i32 0, i32* %val
  ret void

ret1:
  store i32 9, i32* %val
  ret void
}

declare void @may.clobber()

; CHECK-LABEL: define private %void_one_out_arg_i32_2_stores @void_one_out_arg_i32_2_stores.body(i32* %val) #0 {
; CHECK-NEXT: store i32 0, i32* %val
; CHECK-NEXT: ret %void_one_out_arg_i32_2_stores { i32 1 }

; CHECK-LABEL: define void @void_one_out_arg_i32_2_stores(i32*) #2 {
; CHECK-NEXT: %2 = call %void_one_out_arg_i32_2_stores @void_one_out_arg_i32_2_stores.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_i32_2_stores %2, 0
; CHECK-NEXT: store i32 %3, i32* %0, align 4
define void @void_one_out_arg_i32_2_stores(i32* %val) #0 {
  store i32 0, i32* %val
  store i32 1, i32* %val
  ret void
}

; CHECK-LABEL: define private %void_one_out_arg_i32_2_stores_clobber @void_one_out_arg_i32_2_stores_clobber.body(i32* %val) #0 {
; CHECK-NEXT: store i32 0, i32* %val
; CHECK-NEXT: call void @may.clobber()
; CHECK-NEXT: ret %void_one_out_arg_i32_2_stores_clobber { i32 1 }

; CHECK-LABEL: define void @void_one_out_arg_i32_2_stores_clobber(i32*) #2 {
; CHECK-NEXT: %2 = call %void_one_out_arg_i32_2_stores_clobber @void_one_out_arg_i32_2_stores_clobber.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_i32_2_stores_clobber %2, 0
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_2_stores_clobber(i32* %val) #0 {
  store i32 0, i32* %val
  call void @may.clobber()
  store i32 1, i32* %val
  ret void
}

; CHECK-NOT: define

; CHECK-LABEL: define void @void_one_out_arg_i32_call_may_clobber(i32* %val) #0 {
; CHECK-NEXT: store i32 0, i32* %val
; CHECK-NEXT: call void @may.clobber()
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_call_may_clobber(i32* %val) #0 {
  store i32 0, i32* %val
  call void @may.clobber()
  ret void
}

; CHECK-LABEL: define private %void_one_out_arg_i32_pre_call_may_clobber @void_one_out_arg_i32_pre_call_may_clobber.body(i32* %val) #0 {
; CHECK-NEXT: call void @may.clobber()
; CHECK-NEXT: ret %void_one_out_arg_i32_pre_call_may_clobber zeroinitializer

; CHECK-LABEL: @void_one_out_arg_i32_pre_call_may_clobber(i32*) #2 {
; CHECK-NEXT: %2 = call %void_one_out_arg_i32_pre_call_may_clobber @void_one_out_arg_i32_pre_call_may_clobber.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_i32_pre_call_may_clobber %2, 0
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_pre_call_may_clobber(i32* %val) #0 {
  call void @may.clobber()
  store i32 0, i32* %val
  ret void
}

; CHECK-LABEL: define void @void_one_out_arg_i32_reload(i32* %val) #0 {
; CHECK: store i32 0, i32* %val
; CHECK: %load = load i32, i32* %val, align 4
; CHECK: ret void
define void @void_one_out_arg_i32_reload(i32* %val) #0 {
  store i32 0, i32* %val
  %load = load i32, i32* %val, align 4
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @void_one_out_arg_i32_store_in_different_block(
; CHECK-NEXT: %load = load i32, i32 addrspace(1)* undef
; CHECK-NEXT: store i32 0, i32* %out
; CHECK-NEXT: br label %ret
; CHECK: ret:
; CHECK-NEXT: ret void
define void @void_one_out_arg_i32_store_in_different_block(i32* %out) #0 {
  %load = load i32, i32 addrspace(1)* undef
  store i32 0, i32* %out
  br label %ret

ret:
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @unused_out_arg_one_branch(
; CHECK: ret0:
; CHECK-NEXT: ret void

; CHECK: ret1:
; CHECK-NEXT: store i32 9, i32* %val
; CHECK-NEXT: ret void
define void @unused_out_arg_one_branch(i1 %arg0, i32* %val) #0 {
  br i1 %arg0, label %ret0, label %ret1

ret0:
  ret void

ret1:
  store i32 9, i32* %val
  ret void
}

; CHECK-LABEL: define private %void_one_out_arg_v2i32_1_use @void_one_out_arg_v2i32_1_use.body(<2 x i32>* %val) #0 {
; CHECK-NEXT: ret %void_one_out_arg_v2i32_1_use { <2 x i32> <i32 17, i32 9> }

; CHECK-LABEL: define void @void_one_out_arg_v2i32_1_use(<2 x i32>*) #2 {
; CHECK-NEXT: %2 = call %void_one_out_arg_v2i32_1_use @void_one_out_arg_v2i32_1_use.body(<2 x i32>* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_v2i32_1_use %2, 0
; CHECK-NEXT: store <2 x i32> %3, <2 x i32>* %0, align 8
; CHCEK-NEXT: ret void
define void @void_one_out_arg_v2i32_1_use(<2 x i32>* %val) #0 {
  store <2 x i32> <i32 17, i32 9>, <2 x i32>* %val
  ret void
}

%struct = type { i32, i8, float }

; CHECK-LABEL: define private %void_one_out_arg_struct_1_use @void_one_out_arg_struct_1_use.body(%struct* %out) #0 {
; CHECK-NEXT: ret %void_one_out_arg_struct_1_use { %struct { i32 9, i8 99, float 4.000000e+00 } }

; Normally this is split into element accesses which we don't handle.
; CHECK-LABEL: define void @void_one_out_arg_struct_1_use(%struct*) #2 {
; CHECK-NEXT: %2 = call %void_one_out_arg_struct_1_use @void_one_out_arg_struct_1_use.body(%struct* undef)
; CHECK-NEXT: %3 = extractvalue %void_one_out_arg_struct_1_use %2, 0
; CHECK-NEXT: store %struct %3, %struct* %0, align 4
; CHECK-NEXT: ret void
define void @void_one_out_arg_struct_1_use(%struct* %out) #0 {
  store %struct { i32 9, i8 99, float 4.0 }, %struct* %out
  ret void
}

; CHECK-LABEL: define private %i32_one_out_arg_i32_1_use @i32_one_out_arg_i32_1_use.body(i32* %val) #0 {
; CHECK-NEXT: ret %i32_one_out_arg_i32_1_use { i32 9, i32 24 }

; CHECK-LABEL: define i32 @i32_one_out_arg_i32_1_use(i32*) #2 {
; CHECK-NEXT: %2 = call %i32_one_out_arg_i32_1_use @i32_one_out_arg_i32_1_use.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %i32_one_out_arg_i32_1_use %2, 1
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: %4 = extractvalue %i32_one_out_arg_i32_1_use %2, 0
; CHECK-NEXT: ret i32 %4
define i32 @i32_one_out_arg_i32_1_use(i32* %val) #0 {
  store i32 24, i32* %val
  ret i32 9
}

; CHECK-LABEL: define private %unused_different_type @unused_different_type.body(i32* %arg0, float* nocapture %arg1) #0 {
; CHECK-NEXT: ret %unused_different_type { float 4.000000e+00 }

; CHECK-LABEL: define void @unused_different_type(i32*, float* nocapture) #2 {
; CHECK-NEXT: %3 = call %unused_different_type @unused_different_type.body(i32* %0, float* undef)
; CHECK-NEXT: %4 = extractvalue %unused_different_type %3, 0
; CHECK-NEXT: store float %4, float* %1, align 4
; CHECK-NEXT  ret void
define void @unused_different_type(i32* %arg0, float* nocapture %arg1) #0 {
  store float 4.0, float* %arg1, align 4
  ret void
}

; CHECK-LABEL: define private %multiple_same_return_noalias @multiple_same_return_noalias.body(i32* noalias %out0, i32* noalias %out1) #0 {
; CHECK-NEXT: ret %multiple_same_return_noalias { i32 1, i32 2 }

; CHECK-LABEL: define void @multiple_same_return_noalias(
; CHECK-NEXT: %3 = call %multiple_same_return_noalias @multiple_same_return_noalias.body(i32* undef, i32* undef)
; CHECK-NEXT: %4 = extractvalue %multiple_same_return_noalias %3, 0
; CHECK-NEXT: store i32 %4, i32* %0, align 4
; CHECK-NEXT: %5 = extractvalue %multiple_same_return_noalias %3, 1
; CHECK-NEXT: store i32 %5, i32* %1, align 4
; CHECK-NEXT: ret void
define void @multiple_same_return_noalias(i32* noalias %out0, i32* noalias %out1) #0 {
  store i32 1, i32* %out0, align 4
  store i32 2, i32* %out1, align 4
  ret void
}

; CHECK-LABEL: define private %multiple_same_return_mayalias @multiple_same_return_mayalias.body(i32* %out0, i32* %out1) #0 {
; CHECK-NEXT: ret %multiple_same_return_mayalias { i32 2, i32 1 }

; CHECK-LABEL: define void @multiple_same_return_mayalias(i32*, i32*) #2 {
; CHECK-NEXT: %3 = call %multiple_same_return_mayalias @multiple_same_return_mayalias.body(i32* undef, i32* undef)
; CHECK-NEXT: %4 = extractvalue %multiple_same_return_mayalias %3, 0
; CHECK-NEXT: store i32 %4, i32* %0, align 4
; CHECK-NEXT: %5 = extractvalue %multiple_same_return_mayalias %3, 1
; CHECK-NEXT: store i32 %5, i32* %1, align 4
; CHECK-NEXT: ret void
define void @multiple_same_return_mayalias(i32* %out0, i32* %out1) #0 {
 store i32 1, i32* %out0, align 4
 store i32 2, i32* %out1, align 4
 ret void
}

; CHECK-LABEL: define private %multiple_same_return_mayalias_order @multiple_same_return_mayalias_order.body(i32* %out0, i32* %out1) #0 {
; CHECK-NEXT: ret %multiple_same_return_mayalias_order { i32 1, i32 2 }

; CHECK-LABEL: define void @multiple_same_return_mayalias_order(i32*, i32*) #2 {
; CHECK-NEXT: %3 = call %multiple_same_return_mayalias_order @multiple_same_return_mayalias_order.body(i32* undef, i32* undef)
; CHECK-NEXT: %4 = extractvalue %multiple_same_return_mayalias_order %3, 0
; CHECK-NEXT: store i32 %4, i32* %0, align 4
; CHECK-NEXT: %5 = extractvalue %multiple_same_return_mayalias_order %3, 1
; CHECK-NEXT: store i32 %5, i32* %1, align 4
; CHECK-NEXT: ret void
define void @multiple_same_return_mayalias_order(i32* %out0, i32* %out1) #0 {
 store i32 2, i32* %out1, align 4
 store i32 1, i32* %out0, align 4
 ret void
}

; Currently this fails to convert because the store won't be found if
; it isn't in the same block as the return.
; CHECK-LABEL: define i32 @store_in_entry_block(i1 %arg0, i32* %out) #0 {
; CHECK-NOT: call
define i32 @store_in_entry_block(i1 %arg0, i32* %out) #0 {
entry:
  %val0 = load i32, i32 addrspace(1)* undef
  store i32 %val0, i32* %out
  br i1 %arg0, label %if, label %endif

if:
  %val1 = load i32, i32 addrspace(1)* undef
  br label %endif

endif:
  %phi = phi i32 [ 0, %entry ], [ %val1, %if ]
  ret i32 %phi
}

; CHECK-LABEL: define private %i1_one_out_arg_i32_1_use @i1_one_out_arg_i32_1_use.body(i32* %val) #0 {
; CHECK-NEXT: ret %i1_one_out_arg_i32_1_use { i1 true, i32 24 }

; CHECK-LABEL: define i1 @i1_one_out_arg_i32_1_use(i32*) #2 {
; CHECK: %2 = call %i1_one_out_arg_i32_1_use @i1_one_out_arg_i32_1_use.body(i32* undef)
; CHECK: %3 = extractvalue %i1_one_out_arg_i32_1_use %2, 1
; CHECK: store i32 %3, i32* %0, align 4
; CHECK: %4 = extractvalue %i1_one_out_arg_i32_1_use %2, 0
; CHECK: ret i1 %4
define i1 @i1_one_out_arg_i32_1_use(i32* %val) #0 {
  store i32 24, i32* %val
  ret i1 true
}

; Make sure we don't leave around return attributes that are
; incompatible with struct return types.

; CHECK-LABEL: define private %i1_zeroext_one_out_arg_i32_1_use @i1_zeroext_one_out_arg_i32_1_use.body(i32* %val) #0 {
; CHECK-NEXT: ret %i1_zeroext_one_out_arg_i32_1_use { i1 true, i32 24 }

; CHECK-LABEL: define zeroext i1 @i1_zeroext_one_out_arg_i32_1_use(i32*) #2 {
; CHECK-NEXT: %2 = call %i1_zeroext_one_out_arg_i32_1_use @i1_zeroext_one_out_arg_i32_1_use.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %i1_zeroext_one_out_arg_i32_1_use %2, 1
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: %4 = extractvalue %i1_zeroext_one_out_arg_i32_1_use %2, 0
; CHECK-NEXT: ret i1 %4
define zeroext i1 @i1_zeroext_one_out_arg_i32_1_use(i32* %val) #0 {
  store i32 24, i32* %val
  ret i1 true
}

; CHECK-LABEL: define private %i1_signext_one_out_arg_i32_1_use @i1_signext_one_out_arg_i32_1_use.body(i32* %val) #0 {
; CHECK-NEXT: ret %i1_signext_one_out_arg_i32_1_use { i1 true, i32 24 }

; CHECK-LABEL: define signext i1 @i1_signext_one_out_arg_i32_1_use(i32*) #2 {
; CHECK-NEXT: %2 = call %i1_signext_one_out_arg_i32_1_use @i1_signext_one_out_arg_i32_1_use.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %i1_signext_one_out_arg_i32_1_use %2, 1
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: %4 = extractvalue %i1_signext_one_out_arg_i32_1_use %2, 0
; CHECK-NEXT: ret i1 %4
define signext i1 @i1_signext_one_out_arg_i32_1_use(i32* %val) #0 {
  store i32 24, i32* %val
  ret i1 true
}

; CHECK-LABEL: define private %p1i32_noalias_one_out_arg_i32_1_use @p1i32_noalias_one_out_arg_i32_1_use.body(i32* %val) #0 {
; CHECK-NEXT: ret %p1i32_noalias_one_out_arg_i32_1_use { i32 addrspace(1)* null, i32 24 }

; CHECK-LABEL: define noalias i32 addrspace(1)* @p1i32_noalias_one_out_arg_i32_1_use(i32*) #2 {
; CHECK-NEXT: %2 = call %p1i32_noalias_one_out_arg_i32_1_use @p1i32_noalias_one_out_arg_i32_1_use.body(i32* undef)
; CHECK-NEXT: %3 = extractvalue %p1i32_noalias_one_out_arg_i32_1_use %2, 1
; CHECK-NEXT: store i32 %3, i32* %0, align 4
; CHECK-NEXT: %4 = extractvalue %p1i32_noalias_one_out_arg_i32_1_use %2, 0
; CHECK-NEXT: ret i32 addrspace(1)* %4
define noalias i32 addrspace(1)* @p1i32_noalias_one_out_arg_i32_1_use(i32* %val) #0 {
  store i32 24, i32* %val
  ret i32 addrspace(1)* null
}

; CHECK-LABEL: define void @void_one_out_non_private_arg_i32_1_use(i32 addrspace(1)* %val) #0 {
; CHECK-NEXT: store i32 0, i32 addrspace(1)* %val
; CHECK-NEXT: ret void
define void @void_one_out_non_private_arg_i32_1_use(i32 addrspace(1)* %val) #0 {
  store i32 0, i32 addrspace(1)* %val
  ret void
}

; CHECK-LABEL: define private %func_ptr_type @func_ptr_type.body(void ()** %out) #0 {
; CHECK-LABEL: define void @func_ptr_type(void ()**) #2 {
; CHECK: %2 = call %func_ptr_type @func_ptr_type.body(void ()** undef)
define void @func_ptr_type(void()** %out) #0 {
  %func = load void()*, void()** undef
  store void()* %func, void()** %out
  ret void
}

; CHECK-LABEL: define private %out_arg_small_array @out_arg_small_array.body([4 x i32]* %val) #0 {
; CHECK-NEXT: ret %out_arg_small_array { [4 x i32] [i32 0, i32 1, i32 2, i32 3] }

; CHECK-LABEL: define void @out_arg_small_array([4 x i32]*) #2 {
define void @out_arg_small_array([4 x i32]* %val) #0 {
  store [4 x i32] [i32 0, i32 1, i32 2, i32 3], [4 x i32]* %val
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define void @out_arg_large_array([17 x i32]* %val) #0 {
; CHECK-NEXT: store [17 x i32] zeroinitializer, [17 x i32]* %val
; CHECK-NEXT: ret void
define void @out_arg_large_array([17 x i32]* %val) #0 {
  store [17 x i32] zeroinitializer, [17 x i32]* %val
  ret void
}

; CHECK-NOT: define
; CHECK-LABEL: define <16 x i32> @num_regs_return_limit(i32* %out, i32 %val) #0 {
define <16 x i32> @num_regs_return_limit(i32* %out, i32 %val) #0 {
  %load = load volatile <16 x i32>, <16 x i32> addrspace(1)* undef
  store i32 %val, i32* %out
  ret <16 x i32> %load
}

; CHECK-LABEL: define private %num_regs_reach_limit @num_regs_reach_limit.body(i32* %out, i32 %val) #0 {
; CHECK: define [15 x i32] @num_regs_reach_limit(i32*, i32) #2 {
; CHECK-NEXT: call %num_regs_reach_limit @num_regs_reach_limit.body(i32* undef, i32 %1)
define [15 x i32] @num_regs_reach_limit(i32* %out, i32 %val) #0 {
  %load = load volatile [15 x i32], [15 x i32] addrspace(1)* undef
  store i32 %val, i32* %out
  ret [15 x i32] %load
}

; CHECK-LABEL: define private %num_regs_reach_limit_leftover @num_regs_reach_limit_leftover.body(i32* %out0, i32* %out1, i32 %val0) #0 {
; CHECK-NEXT: %load0 = load volatile [15 x i32], [15 x i32] addrspace(1)* undef
; CHECK-NEXT: %load1 = load volatile i32, i32 addrspace(1)* undef
; CHECK-NEXT: %1 = insertvalue %num_regs_reach_limit_leftover undef, [15 x i32] %load0, 0
; CHECK-NEXT: %2 = insertvalue %num_regs_reach_limit_leftover %1, i32 %load1, 1
; CHECK-NEXT: %3 = insertvalue %num_regs_reach_limit_leftover %2, i32 %val0, 2
; CHECK-NEXT: ret %num_regs_reach_limit_leftover %3

; CHECK-LABEL: define [15 x i32] @num_regs_reach_limit_leftover(i32*, i32*, i32) #2 {
; CHECK-NEXT: %4 = call %num_regs_reach_limit_leftover @num_regs_reach_limit_leftover.body(i32* undef, i32* undef, i32 %2)
; CHECK-NEXT: %5 = extractvalue %num_regs_reach_limit_leftover %4, 1
; CHECK-NEXT: store i32 %5, i32* %0, align 4
; CHECK-NEXT: %6 = extractvalue %num_regs_reach_limit_leftover %4, 2
; CHECK-NEXT: store i32 %6, i32* %1, align 4
; CHECK-NEXT: %7 = extractvalue %num_regs_reach_limit_leftover %4, 0
; CHECK-NEXT: ret [15 x i32] %7
define [15 x i32] @num_regs_reach_limit_leftover(i32* %out0, i32* %out1, i32 %val0) #0 {
  %load0 = load volatile [15 x i32], [15 x i32] addrspace(1)* undef
  %load1 = load volatile i32, i32 addrspace(1)* undef
  store i32 %val0, i32* %out0
  store i32 %load1, i32* %out1
  ret [15 x i32] %load0
}

; CHECK-LABEL: define private %preserve_debug_info @preserve_debug_info.body(i32 %arg0, i32* %val) #0 {
; CHECK-NEXT: call void @may.clobber(), !dbg !5
; CHECK-NEXT: %1 = insertvalue %preserve_debug_info undef, i32 %arg0, 0, !dbg !11
; CHECK-NEXT: ret %preserve_debug_info %1, !dbg !11

; CHECK-LABEL: define void @preserve_debug_info(i32, i32*) #2 !dbg !6 {
; CHECK-NEXT: %3 = call %preserve_debug_info @preserve_debug_info.body(i32 %0, i32* undef){{$}}
; CHECK-NEXT: %4 = extractvalue %preserve_debug_info %3, 0{{$}}
; CHECK-NEXT: store i32 %4, i32* %1, align 4{{$}}
; CHECK-NEXT: ret void
define void @preserve_debug_info(i32 %arg0, i32* %val) #0 !dbg !5 {
  call void @may.clobber(), !dbg !10
  store i32 %arg0, i32* %val, !dbg !11
  ret void, !dbg !12
}

define void @preserve_metadata(i32 %arg0, i32* %val) #0 !kernel_arg_access_qual !13 {
  call void @may.clobber()
  store i32 %arg0, i32* %val
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind noinline optnone }
attributes #2 = { alwaysinline nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "code-object-metadata-kernel-debug-props.cl", directory: "/some/random/directory")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 2, column: 3, scope: !5)
!11 = !DILocation(line: 2, column: 8, scope: !5)
!12 = !DILocation(line: 3, column: 3, scope: !5)
!13 = !{!"none"}
