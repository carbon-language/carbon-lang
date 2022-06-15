; We run nvvm-reflect (and then optimize) this module twice, once with metadata
; that enables FTZ, and again with metadata that disables it.

; RUN: cat %s > %t.noftz
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}' >> %t.noftz
; RUN: opt %t.noftz -S -mtriple=nvptx-nvidia-cuda -passes='default<O2>' \
; RUN:   | FileCheck %s --check-prefix=USE_FTZ_0 --check-prefix=CHECK

; RUN: cat %s > %t.ftz
; RUN: echo '!0 = !{i32 4, !"nvvm-reflect-ftz", i32 1}' >> %t.ftz
; RUN: opt %t.ftz -S -mtriple=nvptx-nvidia-cuda -passes='default<O2>' \
; RUN:   | FileCheck %s --check-prefix=USE_FTZ_1 --check-prefix=CHECK

@str = private unnamed_addr addrspace(4) constant [11 x i8] c"__CUDA_FTZ\00"

declare i32 @__nvvm_reflect(ptr)
declare ptr @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(ptr addrspace(4))

; CHECK-LABEL: @foo
define float @foo(float %a, float %b) {
; CHECK-NOT: call i32 @__nvvm_reflect
  %ptr = tail call ptr @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(ptr addrspace(4) @str)
  %reflect = tail call i32 @__nvvm_reflect(ptr %ptr)
  %cmp = icmp ugt i32 %reflect, 0
  br i1 %cmp, label %use_mul, label %use_add

use_mul:
; USE_FTZ_1: fmul float %a, %b
; USE_FTZ_0-NOT: fadd float %a, %b
  %ret1 = fmul float %a, %b
  br label %exit

use_add:
; USE_FTZ_0: fadd float %a, %b
; USE_FTZ_1-NOT: fmul float %a, %b
  %ret2 = fadd float %a, %b
  br label %exit

exit:
  %ret = phi float [%ret1, %use_mul], [%ret2, %use_add]
  ret float %ret
}

declare i32 @llvm.nvvm.reflect.p0i8(ptr)

; CHECK-LABEL: define i32 @intrinsic
define i32 @intrinsic() {
; CHECK-NOT: call i32 @llvm.nvvm.reflect
; USE_FTZ_0: ret i32 0
; USE_FTZ_1: ret i32 1
  %ptr = tail call ptr @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(ptr addrspace(4) @str)
  %reflect = tail call i32 @llvm.nvvm.reflect.p0i8(ptr %ptr)
  ret i32 %reflect
}

; CUDA-7.0 passes __nvvm_reflect argument slightly differently.
; Verify that it works, too

@"$str" = private addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"

; CHECK-LABEL: @bar
define float @bar(float %a, float %b) {
; CHECK-NOT: call i32 @__nvvm_reflect
  %reflect = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @"$str" to ptr))
  %cmp = icmp ne i32 %reflect, 0
  br i1 %cmp, label %use_mul, label %use_add

use_mul:
; USE_FTZ_1: fmul float %a, %b
; USE_FTZ_0-NOT: fadd float %a, %b
  %ret1 = fmul float %a, %b
  br label %exit

use_add:
; USE_FTZ_0: fadd float %a, %b
; USE_FTZ_1-NOT: fmul float %a, %b
  %ret2 = fadd float %a, %b
  br label %exit

exit:
  %ret = phi float [%ret1, %use_mul], [%ret2, %use_add]
  ret float %ret
}

@str0 = private constant [11 x i8] c"__CUDA_FTZ\00"

; CHECK-LABEL: @baz
define float @baz(float %a, float %b) {
; CHECK-NOT: call i32 @__nvvm_reflect
  %reflect = call i32 @__nvvm_reflect(ptr @str0)
  %cmp = icmp ne i32 %reflect, 0
  br i1 %cmp, label %use_mul, label %use_add

use_mul:
; USE_FTZ_1: fmul float %a, %b
; USE_FTZ_0-NOT: fadd float %a, %b
  %ret1 = fmul float %a, %b
  br label %exit

use_add:
; USE_FTZ_0: fadd float %a, %b
; USE_FTZ_1-NOT: fmul float %a, %b
  %ret2 = fadd float %a, %b
  br label %exit

exit:
  %ret = phi float [%ret1, %use_mul], [%ret2, %use_add]
  ret float %ret
}

!llvm.module.flags = !{!0}
; A module flag is added to the end of this file by the RUN lines at the top.
