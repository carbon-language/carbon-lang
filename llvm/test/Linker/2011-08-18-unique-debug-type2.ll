; This file is for use with 2011-08-10-unique-debug-type.ll
; RUN: true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

define i32 @bar() nounwind uwtable ssp {
entry:
  ret i32 2, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 137954)", isOptimized: true, emissionKind: 0, file: !12, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2)
!1 = !{!2}
!2 = !{i32 0}
!3 = !{!5}
!5 = !MDSubprogram(name: "bar", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !12, scope: !6, type: !7, function: i32 ()* @bar)
!6 = !MDFile(filename: "two.c", directory: "/private/tmp")
!7 = !MDSubroutineType(types: !8)
!8 = !{!9}
!9 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !MDLocation(line: 1, column: 13, scope: !11)
!11 = distinct !MDLexicalBlock(line: 1, column: 11, file: !12, scope: !5)
!12 = !MDFile(filename: "two.c", directory: "/private/tmp")
!13 = !{i32 1, !"Debug Info Version", i32 3}
