; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

; CHECK:      .long   .Lline_table_start0          # DW_AT_stmt_list

; CHECK:      .section        .debug_line,"",@progbits
; CHECK-NEXT: .Lsection_line:
; CHECK-NEXT: .Lline_table_start0:

define void @f() {
entry:
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7}
!5 = !{!0}

!0 = !MDSubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 1, file: !6, scope: !1, type: !3, function: void ()* @f)
!1 = !MDFile(filename: "test2.c", directory: "/home/espindola/llvm")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 ()", isOptimized: true, emissionKind: 0, file: !6, enums: !4, retainedTypes: !4, subprograms: !5)
!3 = !MDSubroutineType(types: !4)
!4 = !{null}
!6 = !MDFile(filename: "test2.c", directory: "/home/espindola/llvm")
!7 = !{i32 1, !"Debug Info Version", i32 3}
