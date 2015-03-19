; RUN: opt < %s -S -nvvm-reflect -nvvm-reflect-list USE_MUL=0 -O2 | FileCheck %s --check-prefix=USE_MUL_0
; RUN: opt < %s -S -nvvm-reflect -nvvm-reflect-list USE_MUL=1 -O2 | FileCheck %s --check-prefix=USE_MUL_1

@str = private unnamed_addr addrspace(4) constant [8 x i8] c"USE_MUL\00"

declare i32 @__nvvm_reflect(i8*)
declare i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)*)

define float @foo(float %a, float %b) {
; USE_MUL_0: define float @foo
; USE_MUL_0-NOT: call i32 @__nvvm_reflect
; USE_MUL_1: define float @foo
; USE_MUL_1-NOT: call i32 @__nvvm_reflect
  %ptr = tail call i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)* getelementptr inbounds ([8 x i8], [8 x i8] addrspace(4)* @str, i32 0, i32 0))
  %reflect = tail call i32 @__nvvm_reflect(i8* %ptr)
  %cmp = icmp ugt i32 %reflect, 0
  br i1 %cmp, label %use_mul, label %use_add

use_mul:
; USE_MUL_1: fmul float %a, %b
; USE_MUL_0-NOT: fadd float %a, %b
  %ret1 = fmul float %a, %b
  br label %exit

use_add:
; USE_MUL_0: fadd float %a, %b
; USE_MUL_1-NOT: fmul float %a, %b
  %ret2 = fadd float %a, %b
  br label %exit

exit:
  %ret = phi float [%ret1, %use_mul], [%ret2, %use_add]
  ret float %ret
}

declare i32 @llvm.nvvm.reflect.p0i8(i8*)

; USE_MUL_0: define i32 @intrinsic
; USE_MUL_1: define i32 @intrinsic
define i32 @intrinsic() {
; USE_MUL_0-NOT: call i32 @llvm.nvvm.reflect
; USE_MUL_0: ret i32 0
; USE_MUL_1-NOT: call i32 @llvm.nvvm.reflect
; USE_MUL_1: ret i32 1
  %ptr = tail call i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)* getelementptr inbounds ([8 x i8], [8 x i8] addrspace(4)* @str, i32 0, i32 0))
  %reflect = tail call i32 @llvm.nvvm.reflect.p0i8(i8* %ptr)
  ret i32 %reflect
}

; CUDA-7.0 passes __nvvm_reflect argument slightly differently.
; Verify that it works, too

@"$str" = private addrspace(1) constant [8 x i8] c"USE_MUL\00"

define float @bar(float %a, float %b) {
; USE_MUL_0: define float @bar
; USE_MUL_0-NOT: call i32 @__nvvm_reflect
; USE_MUL_1: define float @bar
; USE_MUL_1-NOT: call i32 @__nvvm_reflect
  %reflect = call i32 @__nvvm_reflect(i8* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([8 x i8], [8 x i8] addrspace(1)* @"$str", i32 0, i32 0) to i8*))
  %cmp = icmp ne i32 %reflect, 0
  br i1 %cmp, label %use_mul, label %use_add

use_mul:
; USE_MUL_1: fmul float %a, %b
; USE_MUL_0-NOT: fadd float %a, %b
  %ret1 = fmul float %a, %b
  br label %exit

use_add:
; USE_MUL_0: fadd float %a, %b
; USE_MUL_1-NOT: fmul float %a, %b
  %ret2 = fadd float %a, %b
  br label %exit

exit:
  %ret = phi float [%ret1, %use_mul], [%ret2, %use_add]
  ret float %ret
}
