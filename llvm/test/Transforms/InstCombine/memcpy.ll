; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define void @test4(i8* %a) {
        tail call void @llvm.memcpy.i32( i8* %a, i8* %a, i32 100, i32 1 )
        ret void
}
; CHECK: define void @test4
; CHECK-NEXT: ret void
