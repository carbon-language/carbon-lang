; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android24"

; CHECK:      DW_TAG_variable
; CHECK-NOT:  DW_TAG
; CHECK:        DW_AT_LLVM_tag_offset (0x80)
; CHECK-NEXT:   DW_AT_name    ("y")

; CHECK:      DW_TAG_variable
; CHECK-NOT:  DW_TAG
; CHECK:        DW_AT_LLVM_tag_offset (0x00)
; CHECK-NEXT:   DW_AT_name    ("x")

define dso_local void @f() !dbg !14 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = bitcast i32* %1 to i8*, !dbg !21
  %4 = bitcast i32* %2 to i8*, !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !20, metadata !DIExpression()), !dbg !22
  store i32 1, i32* %2, align 4, !dbg !23, !tbaa !24
  call void @llvm.dbg.value(metadata i32* %1, metadata !18, metadata !DIExpression(DW_OP_LLVM_tag_offset, 0, DW_OP_deref)), !dbg !22
  call void @use(i8* nonnull %3), !dbg !28
  call void @llvm.dbg.value(metadata i32* %2, metadata !20, metadata !DIExpression(DW_OP_LLVM_tag_offset, 128, DW_OP_deref)), !dbg !22
  call void @use(i8* nonnull %4), !dbg !29
  ret void, !dbg !30
}

declare !dbg !5 void @use(i8*)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git 5560dd08b99a0e8b0c55116376624e4f967caec5)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "dbg.cc", directory: "/tmp")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!5 = !DISubprogram(name: "use", scope: !1, file: !1, line: 2, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !4}
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"PIC Level", i32 2}
!12 = !{i32 7, !"PIE Level", i32 2}
!13 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 5560dd08b99a0e8b0c55116376624e4f967caec5)"}
!14 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !15, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18, !20}
!18 = !DILocalVariable(name: "x", scope: !14, file: !1, line: 5, type: !19)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DILocalVariable(name: "y", scope: !14, file: !1, line: 5, type: !19)
!21 = !DILocation(line: 5, column: 3, scope: !14)
!22 = !DILocation(line: 0, scope: !14)
!23 = !DILocation(line: 5, column: 10, scope: !14)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !DILocation(line: 6, column: 3, scope: !14)
!29 = !DILocation(line: 7, column: 3, scope: !14)
!30 = !DILocation(line: 8, column: 1, scope: !14)
