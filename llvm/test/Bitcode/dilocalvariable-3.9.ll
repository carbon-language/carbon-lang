; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: !9 = !DILocalVariable(name: "i", scope: !6, file: !1, line: 3, type: !10)
; CHECK: !10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

source_filename = "test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @_Z1fv() #0 !dbg !6 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %i, metadata !10, metadata !12), !dbg !13
  store i32 42, i32* %i, align 4, !dbg !13
  ret void, !dbg !14
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.1 (http://llvm.org/git/clang.git c3709e72d22432f53f8e2f14354def31a96734fe)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.1 (http://llvm.org/git/clang.git c3709e72d22432f53f8e2f14354def31a96734fe)"}
!6 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !7, file: !7, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DIFile(filename: "test.cpp", directory: "/tmp")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "i", scope: !6, file: !7, line: 3, type: !11)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DIExpression()
!13 = !DILocation(line: 3, column: 7, scope: !6)
!14 = !DILocation(line: 4, column: 1, scope: !6)
