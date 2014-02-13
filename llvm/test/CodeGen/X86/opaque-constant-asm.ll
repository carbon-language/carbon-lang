; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; This tests makes sure that we not mistake the bitcast inside the asm statement
; as an opaque constant. If we do, then the compilation will simply fail.

%struct2 = type <{ i32, i32, i32, i32 }>
%union.anon = type { [2 x i64], [4 x i32] }
%struct1 = type { i32, %union.anon }

define void @test() {
; CHECK: #ASM $16
  call void asm sideeffect "#ASM $0", "n"(i32 ptrtoint (i32* getelementptr inbounds (%struct2* bitcast (%union.anon* getelementptr inbounds (%struct1* null, i32 0, i32 1) to %struct2*), i32 0, i32 2) to i32))
  ret void
}
