; RUN: opt < %s -verify -S | FileCheck %s

; CHECK: DW_LANG_Mips_Assembler

define void @Foo(i32 %a, i32 %b) {
entry:
  call void @llvm.dbg.declare(metadata i32* null, metadata !1, metadata !MDExpression())
  ret void
}
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5}
!2 = !MDCompileUnit(language: DW_LANG_Mips_Assembler, producer: "clang version 3.3 ", isOptimized: false, emissionKind: 1, file: !4, enums: !3, retainedTypes: !3, subprograms: !3, globals: !3, imports:  !3)
!3 = !{}
!0 = !MDLocation(line: 662302, column: 26, scope: !1)
!1 = !{i32 4, !"foo"}
!4 = !MDFile(filename: "scratch.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
!5 = !{i32 1, !"Debug Info Version", i32 3}
