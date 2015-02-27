; RUN: llc -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 < %s | FileCheck %s
; Test that we do not introduce vector operations with noimplicitfloat.
; rdar://12879313

%struct1 = type { i32*, i32* }

define void @test() nounwind noimplicitfloat {
entry:
; CHECK-NOT: xmm
; CHECK: ret
  %0 = load %struct1** undef, align 8
  %1 = getelementptr inbounds %struct1, %struct1* %0, i64 0, i32 0
  store i32* null, i32** %1, align 8
  %2 = getelementptr inbounds %struct1, %struct1* %0, i64 0, i32 1
  store i32* null, i32** %2, align 8
  ret void
}
