; RUN: llvm-as < %s > %t1
; RUN: llvm-as < %p/Inputs/remangle_intrinsics.ll > %t2
; RUN: llvm-lto %t1 %t2 | FileCheck %s

; We have "struct.rtx_def" type in both modules being LTOed. Both modules use
; an overloaded intrinsic which has this type in its signature/name. When
; modules are loaded one of the types is renamed to "struct.rtx_def.0".
; The intrinsic which uses this type should be remangled/renamed as well.
; If we didn't do that verifier would complain.

; CHECK: Wrote native object file

target triple = "x86_64-unknown-linux-gnu"

%struct.rtx_def = type { i16 }

define void @foo(%struct.rtx_def* %a, i8 %b, i32 %c) {
  call void  @llvm.memset.p0struct.rtx_def.i32(%struct.rtx_def* align 4 %a, i8 %b, i32 %c, i1 true)
  ret void
}

declare void @llvm.memset.p0struct.rtx_def.i32(%struct.rtx_def*, i8, i32, i1)
