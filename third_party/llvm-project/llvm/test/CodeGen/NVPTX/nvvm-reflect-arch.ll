; Libdevice in recent CUDA versions relies on __CUDA_ARCH reflecting GPU type.
; Verify that __nvvm_reflect() is replaced with an appropriate value.
;
; RUN: opt %s -S -passes='default<O2>' -mtriple=nvptx64 \
; RUN:   | FileCheck %s --check-prefixes=COMMON,SM20
; RUN: opt %s -S -passes='default<O2>' -mtriple=nvptx64 -mcpu=sm_35 \
; RUN:   | FileCheck %s --check-prefixes=COMMON,SM35

@"$str" = private addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"

declare i32 @__nvvm_reflect(i8*)

; COMMON-LABEL: @foo
define i32 @foo(float %a, float %b) {
; COMMON-NOT: call i32 @__nvvm_reflect
  %reflect = call i32 @__nvvm_reflect(i8* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([12 x i8], [12 x i8] addrspace(1)* @"$str", i32 0, i32 0) to i8*))
; SM20: ret i32 200  
; SM35: ret i32 350  
  ret i32 %reflect
}

