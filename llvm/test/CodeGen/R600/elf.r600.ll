; RUN: llc < %s -march=r600 -mcpu=redwood -filetype=obj | llvm-readobj -s - | FileCheck --check-prefix=ELF-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=redwood -o - | FileCheck --check-prefix=CONFIG-CHECK %s

; ELF-CHECK: Format: ELF32
; ELF-CHECK: Name: .AMDGPU.config

; CONFIG-CHECK: .section .AMDGPU.config
; CONFIG-CHECK-NEXT: .long   166100
; CONFIG-CHECK-NEXT: .long   258
; CONFIG-CHECK-NEXT: .long   165900
; CONFIG-CHECK-NEXT: .long   0
define void @test(float addrspace(1)* %out, i32 %p) {
   %i = add i32 %p, 2
   %r = bitcast i32 %i to float
   store float %r, float addrspace(1)* %out
   ret void
}
