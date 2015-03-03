; RUN: opt < %s -argpromotion -S | FileCheck %s
; CHECK: call void @test(i32 %
; CHECK: !MDSubprogram(name: "test",{{.*}} function: void (i32)* @test

declare void @sink(i32)

define internal void @test(i32** %X) {
  %1 = load i32*, i32** %X, align 8
  %2 = load i32, i32* %1, align 8
  call void @sink(i32 %2)
  ret void
}

define void @caller(i32** %Y) {
  call void @test(i32** %Y)
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !MDLocation(line: 8, scope: !2)
!2 = !MDSubprogram(name: "test", line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, scope: null, function: void (i32**)* @test)
!3 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: 2, file: null, subprograms: !4)
!4 = !{!2}
