; RUN: llc < %s -march=r600 -mcpu=SI -filetype=obj | llvm-readobj -s - | FileCheck %s

; CHECK: Format: ELF32
define void @test(i32 %p) {
   %i = add i32 %p, 2
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
