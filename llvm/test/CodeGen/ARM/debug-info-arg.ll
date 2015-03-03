; RUN: llc < %s | FileCheck %s
; Test to check argument y's debug info uses FI
; Radar 10048772
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.tag_s = type { i32, i32, i32 }

define void @foo(%struct.tag_s* nocapture %this, %struct.tag_s* %c, i64 %x, i64 %y, %struct.tag_s* nocapture %ptr1, %struct.tag_s* nocapture %ptr2) nounwind ssp {
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %this, i64 0, metadata !5, metadata !MDExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %c, i64 0, metadata !13, metadata !MDExpression()), !dbg !21
  tail call void @llvm.dbg.value(metadata i64 %x, i64 0, metadata !14, metadata !MDExpression()), !dbg !22
  tail call void @llvm.dbg.value(metadata i64 %y, i64 0, metadata !17, metadata !MDExpression()), !dbg !23
;CHECK:	@DEBUG_VALUE: foo:y <- [R7+8]
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %ptr1, i64 0, metadata !18, metadata !MDExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata %struct.tag_s* %ptr2, i64 0, metadata !19, metadata !MDExpression()), !dbg !25
  %1 = icmp eq %struct.tag_s* %c, null, !dbg !26
  br i1 %1, label %3, label %2, !dbg !26

; <label>:2                                       ; preds = %0
  tail call void @foobar(i64 %x, i64 %y) nounwind, !dbg !28
  br label %3, !dbg !28

; <label>:3                                       ; preds = %0, %2
  ret void, !dbg !29
}

declare void @foobar(i64, i64)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "Apple clang version 3.0 (tags/Apple/clang-211.10.1) (based on LLVM 3.0svn)", isOptimized: true, emissionKind: 1, file: !32, enums: !4, retainedTypes: !4, subprograms: !30, imports:  null)
!1 = !MDSubprogram(name: "foo", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 11, file: !2, scope: !2, type: !3, function: void (%struct.tag_s*, %struct.tag_s*, i64, i64, %struct.tag_s*, %struct.tag_s*)* @foo, variables: !31)
!2 = !MDFile(filename: "one.c", directory: "/Volumes/Athwagate/R10048772")
!3 = !MDSubroutineType(types: !4)
!4 = !{null}
!5 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", line: 11, arg: 1, scope: !1, file: !2, type: !6)
!6 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !0, baseType: !7)
!7 = !MDCompositeType(tag: DW_TAG_structure_type, name: "tag_s", line: 5, size: 96, align: 32, file: !32, scope: !0, elements: !8)
!8 = !{!9, !11, !12}
!9 = !MDDerivedType(tag: DW_TAG_member, name: "x", line: 6, size: 32, align: 32, file: !32, scope: !7, baseType: !10)
!10 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !MDDerivedType(tag: DW_TAG_member, name: "y", line: 7, size: 32, align: 32, offset: 32, file: !32, scope: !7, baseType: !10)
!12 = !MDDerivedType(tag: DW_TAG_member, name: "z", line: 8, size: 32, align: 32, offset: 64, file: !32, scope: !7, baseType: !10)
!13 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "c", line: 11, arg: 2, scope: !1, file: !2, type: !6)
!14 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "x", line: 11, arg: 3, scope: !1, file: !2, type: !15)
!15 = !MDDerivedType(tag: DW_TAG_typedef, name: "UInt64", line: 1, file: !32, scope: !0, baseType: !16)
!16 = !MDBasicType(tag: DW_TAG_base_type, name: "long long unsigned int", size: 64, align: 32, encoding: DW_ATE_unsigned)
!17 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "y", line: 11, arg: 4, scope: !1, file: !2, type: !15)
!18 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "ptr1", line: 11, arg: 5, scope: !1, file: !2, type: !6)
!19 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "ptr2", line: 11, arg: 6, scope: !1, file: !2, type: !6)
!20 = !MDLocation(line: 11, column: 24, scope: !1)
!21 = !MDLocation(line: 11, column: 44, scope: !1)
!22 = !MDLocation(line: 11, column: 54, scope: !1)
!23 = !MDLocation(line: 11, column: 64, scope: !1)
!24 = !MDLocation(line: 11, column: 81, scope: !1)
!25 = !MDLocation(line: 11, column: 101, scope: !1)
!26 = !MDLocation(line: 12, column: 3, scope: !27)
!27 = distinct !MDLexicalBlock(line: 11, column: 107, file: !2, scope: !1)
!28 = !MDLocation(line: 13, column: 5, scope: !27)
!29 = !MDLocation(line: 14, column: 1, scope: !27)
!30 = !{!1}
!31 = !{!5, !13, !14, !17, !18, !19}
!32 = !MDFile(filename: "one.c", directory: "/Volumes/Athwagate/R10048772")
!33 = !{i32 1, !"Debug Info Version", i32 3}
