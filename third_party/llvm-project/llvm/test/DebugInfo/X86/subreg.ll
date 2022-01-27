; RUN: llc %s -mtriple=x86_64-pc-linux-gnu -O0 -o - | FileCheck %s

; We are testing that a value in a 16 bit register gets reported as
; being in its superregister.

; CHECK: .byte   80                      # super-register DW_OP_reg0
; No need to a piece at offset 0.
; CHECK-NOT: DW_OP_piece
; CHECK-NOT: DW_OP_bit_piece

define i16 @f(i16 signext %zzz) nounwind !dbg !1 {
entry:
  call void @llvm.dbg.value(metadata i16 %zzz, metadata !0, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  %conv = sext i16 %zzz to i32, !dbg !7
  %conv1 = trunc i32 %conv to i16
  ret i16 %conv1
}

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!11}
!9 = !{!1}

!0 = !DILocalVariable(name: "zzz", line: 3, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "f", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !3, scopeLine: 3, file: !10, scope: !2, type: !4)
!2 = !DIFile(filename: "/home/espindola/llvm/test.c", directory: "/home/espindola/tmpfs/build")
!3 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 ()", isOptimized: false, emissionKind: FullDebug, file: !10, enums: !{}, retainedTypes: !{}, imports:  null)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!7 = !DILocation(line: 4, column: 22, scope: !8)
!8 = distinct !DILexicalBlock(line: 3, column: 19, file: !10, scope: !1)
!10 = !DIFile(filename: "/home/espindola/llvm/test.c", directory: "/home/espindola/tmpfs/build")
!11 = !{i32 1, !"Debug Info Version", i32 3}
