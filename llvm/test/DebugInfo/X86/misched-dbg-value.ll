; RUN: llc %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -enable-misched
; RUN: llvm-dwarfdump %t | FileCheck %s

; rdar://13183203
; Make sure when misched is enabled, we still have location information for
; function parameters.
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "Proc8"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "Array1Par"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "Array2Par"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "IntParI1"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK:       DW_AT_name {{.*}} "IntParI2"

%struct.Record = type { %struct.Record*, i32, i32, i32, [31 x i8] }

@Version = global [4 x i8] c"1.1\00", align 1
@IntGlob = common global i32 0, align 4
@BoolGlob = common global i32 0, align 4
@Char1Glob = common global i8 0, align 1
@Char2Glob = common global i8 0, align 1
@Array1Glob = common global [51 x i32] zeroinitializer, align 16
@Array2Glob = common global [51 x [51 x i32]] zeroinitializer, align 16
@PtrGlb = common global %struct.Record* null, align 8
@PtrGlbNext = common global %struct.Record* null, align 8

define void @Proc8(i32* nocapture %Array1Par, [51 x i32]* nocapture %Array2Par, i32 %IntParI1, i32 %IntParI2) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata i32* %Array1Par, i64 0, metadata !23, metadata !MDExpression()), !dbg !64
  tail call void @llvm.dbg.value(metadata [51 x i32]* %Array2Par, i64 0, metadata !24, metadata !MDExpression()), !dbg !65
  tail call void @llvm.dbg.value(metadata i32 %IntParI1, i64 0, metadata !25, metadata !MDExpression()), !dbg !66
  tail call void @llvm.dbg.value(metadata i32 %IntParI2, i64 0, metadata !26, metadata !MDExpression()), !dbg !67
  %add = add i32 %IntParI1, 5, !dbg !68
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !27, metadata !MDExpression()), !dbg !68
  %idxprom = sext i32 %add to i64, !dbg !69
  %arrayidx = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom, !dbg !69
  store i32 %IntParI2, i32* %arrayidx, align 4, !dbg !69
  %add3 = add nsw i32 %IntParI1, 6, !dbg !73
  %idxprom4 = sext i32 %add3 to i64, !dbg !73
  %arrayidx5 = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom4, !dbg !73
  store i32 %IntParI2, i32* %arrayidx5, align 4, !dbg !73
  %add6 = add nsw i32 %IntParI1, 35, !dbg !74
  %idxprom7 = sext i32 %add6 to i64, !dbg !74
  %arrayidx8 = getelementptr inbounds i32, i32* %Array1Par, i64 %idxprom7, !dbg !74
  store i32 %add, i32* %arrayidx8, align 4, !dbg !74
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !28, metadata !MDExpression()), !dbg !75
  br label %for.body, !dbg !75

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %idxprom, %entry ], [ %indvars.iv.next, %for.body ]
  %IntIndex.046 = phi i32 [ %add, %entry ], [ %inc, %for.body ]
  %arrayidx13 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom, i64 %indvars.iv, !dbg !77
  store i32 %add, i32* %arrayidx13, align 4, !dbg !77
  %inc = add nsw i32 %IntIndex.046, 1, !dbg !75
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !28, metadata !MDExpression()), !dbg !75
  %cmp = icmp sgt i32 %inc, %add3, !dbg !75
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !75
  br i1 %cmp, label %for.end, label %for.body, !dbg !75

for.end:                                          ; preds = %for.body
  %sub = add nsw i32 %IntParI1, 4, !dbg !78
  %idxprom14 = sext i32 %sub to i64, !dbg !78
  %arrayidx17 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom, i64 %idxprom14, !dbg !78
  %0 = load i32, i32* %arrayidx17, align 4, !dbg !78
  %inc18 = add nsw i32 %0, 1, !dbg !78
  store i32 %inc18, i32* %arrayidx17, align 4, !dbg !78
  %1 = load i32, i32* %arrayidx, align 4, !dbg !79
  %add22 = add nsw i32 %IntParI1, 25, !dbg !79
  %idxprom23 = sext i32 %add22 to i64, !dbg !79
  %arrayidx25 = getelementptr inbounds [51 x i32], [51 x i32]* %Array2Par, i64 %idxprom23, i64 %idxprom, !dbg !79
  store i32 %1, i32* %arrayidx25, align 4, !dbg !79
  store i32 5, i32* @IntGlob, align 4, !dbg !80
  ret void, !dbg !81
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

attributes #0 = { nounwind optsize ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!83}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 175015)", isOptimized: true, emissionKind: 1, file: !82, enums: !1, retainedTypes: !10, subprograms: !11, globals: !29, imports:  !10)
!1 = !{!2}
!2 = !MDCompositeType(tag: DW_TAG_enumeration_type, line: 128, size: 32, align: 32, file: !82, elements: !4)
!3 = !MDFile(filename: "dry.c", directory: "/Users/manmanren/test-Nov/rdar_13183203/test2")
!4 = !{!5, !6, !7, !8, !9}
!5 = !MDEnumerator(name: "Ident1", value: 0) ; [ DW_TAG_enumerator ] [Ident1 :: 0]
!6 = !MDEnumerator(name: "Ident2", value: 10000) ; [ DW_TAG_enumerator ] [Ident2 :: 10000]
!7 = !MDEnumerator(name: "Ident3", value: 10001) ; [ DW_TAG_enumerator ] [Ident3 :: 10001]
!8 = !MDEnumerator(name: "Ident4", value: 10002) ; [ DW_TAG_enumerator ] [Ident4 :: 10002]
!9 = !MDEnumerator(name: "Ident5", value: 10003) ; [ DW_TAG_enumerator ] [Ident5 :: 10003]
!10 = !{}
!11 = !{!12}
!12 = !MDSubprogram(name: "Proc8", line: 180, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 185, file: !82, scope: !3, type: !13, function: void (i32*, [51 x i32]*, i32, i32)* @Proc8, variables: !22)
!13 = !MDSubroutineType(types: !14)
!14 = !{null, !15, !17, !21, !21}
!15 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !16)
!16 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !18)
!18 = !MDCompositeType(tag: DW_TAG_array_type, size: 1632, align: 32, baseType: !16, elements: !19)
!19 = !{!20}
!20 = !MDSubrange(count: 51)
!21 = !MDDerivedType(tag: DW_TAG_typedef, name: "OneToFifty", line: 132, file: !82, baseType: !16)
!22 = !{!23, !24, !25, !26, !27, !28}
!23 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "Array1Par", line: 181, arg: 1, scope: !12, file: !3, type: !15)
!24 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "Array2Par", line: 182, arg: 2, scope: !12, file: !3, type: !17)
!25 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "IntParI1", line: 183, arg: 3, scope: !12, file: !3, type: !21)
!26 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "IntParI2", line: 184, arg: 4, scope: !12, file: !3, type: !21)
!27 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "IntLoc", line: 186, scope: !12, file: !3, type: !21)
!28 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "IntIndex", line: 187, scope: !12, file: !3, type: !21)
!29 = !{!30, !35, !36, !38, !39, !40, !42, !46, !63}
!30 = !MDGlobalVariable(name: "Version", line: 111, isLocal: false, isDefinition: true, scope: null, file: !3, type: !31, variable: [4 x i8]* @Version)
!31 = !MDCompositeType(tag: DW_TAG_array_type, size: 32, align: 8, baseType: !32, elements: !33)
!32 = !MDBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!33 = !{!34}
!34 = !MDSubrange(count: 4)
!35 = !MDGlobalVariable(name: "IntGlob", line: 171, isLocal: false, isDefinition: true, scope: null, file: !3, type: !16, variable: i32* @IntGlob)
!36 = !MDGlobalVariable(name: "BoolGlob", line: 172, isLocal: false, isDefinition: true, scope: null, file: !3, type: !37, variable: i32* @BoolGlob)
!37 = !MDDerivedType(tag: DW_TAG_typedef, name: "boolean", line: 149, file: !82, baseType: !16)
!38 = !MDGlobalVariable(name: "Char1Glob", line: 173, isLocal: false, isDefinition: true, scope: null, file: !3, type: !32, variable: i8* @Char1Glob)
!39 = !MDGlobalVariable(name: "Char2Glob", line: 174, isLocal: false, isDefinition: true, scope: null, file: !3, type: !32, variable: i8* @Char2Glob)
!40 = !MDGlobalVariable(name: "Array1Glob", line: 175, isLocal: false, isDefinition: true, scope: null, file: !3, type: !41, variable: [51 x i32]* @Array1Glob)
!41 = !MDDerivedType(tag: DW_TAG_typedef, name: "Array1Dim", line: 135, file: !82, baseType: !18)
!42 = !MDGlobalVariable(name: "Array2Glob", line: 176, isLocal: false, isDefinition: true, scope: null, file: !3, type: !43, variable: [51 x [51 x i32]]* @Array2Glob)
!43 = !MDDerivedType(tag: DW_TAG_typedef, name: "Array2Dim", line: 136, file: !82, baseType: !44)
!44 = !MDCompositeType(tag: DW_TAG_array_type, size: 83232, align: 32, baseType: !16, elements: !45)
!45 = !{!20, !20}
!46 = !MDGlobalVariable(name: "PtrGlb", line: 177, isLocal: false, isDefinition: true, scope: null, file: !3, type: !47, variable: %struct.Record** @PtrGlb)
!47 = !MDDerivedType(tag: DW_TAG_typedef, name: "RecordPtr", line: 148, file: !82, baseType: !48)
!48 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !49)
!49 = !MDDerivedType(tag: DW_TAG_typedef, name: "RecordType", line: 147, file: !82, baseType: !50)
!50 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Record", line: 138, size: 448, align: 64, file: !82, elements: !51)
!51 = !{!52, !54, !56, !57, !58}
!52 = !MDDerivedType(tag: DW_TAG_member, name: "PtrComp", line: 140, size: 64, align: 64, file: !82, scope: !50, baseType: !53)
!53 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !50)
!54 = !MDDerivedType(tag: DW_TAG_member, name: "Discr", line: 141, size: 32, align: 32, offset: 64, file: !82, scope: !50, baseType: !55)
!55 = !MDDerivedType(tag: DW_TAG_typedef, name: "Enumeration", line: 128, file: !82, baseType: !2)
!56 = !MDDerivedType(tag: DW_TAG_member, name: "EnumComp", line: 142, size: 32, align: 32, offset: 96, file: !82, scope: !50, baseType: !55)
!57 = !MDDerivedType(tag: DW_TAG_member, name: "IntComp", line: 143, size: 32, align: 32, offset: 128, file: !82, scope: !50, baseType: !21)
!58 = !MDDerivedType(tag: DW_TAG_member, name: "StringComp", line: 144, size: 248, align: 8, offset: 160, file: !82, scope: !50, baseType: !59)
!59 = !MDDerivedType(tag: DW_TAG_typedef, name: "String30", line: 134, file: !82, baseType: !60)
!60 = !MDCompositeType(tag: DW_TAG_array_type, size: 248, align: 8, baseType: !32, elements: !61)
!61 = !{!62}
!62 = !MDSubrange(count: 31)
!63 = !MDGlobalVariable(name: "PtrGlbNext", line: 178, isLocal: false, isDefinition: true, scope: null, file: !3, type: !47, variable: %struct.Record** @PtrGlbNext)
!64 = !MDLocation(line: 181, scope: !12)
!65 = !MDLocation(line: 182, scope: !12)
!66 = !MDLocation(line: 183, scope: !12)
!67 = !MDLocation(line: 184, scope: !12)
!68 = !MDLocation(line: 189, scope: !12)
!69 = !MDLocation(line: 190, scope: !12)
!73 = !MDLocation(line: 191, scope: !12)
!74 = !MDLocation(line: 192, scope: !12)
!75 = !MDLocation(line: 193, scope: !76)
!76 = distinct !MDLexicalBlock(line: 193, column: 0, file: !82, scope: !12)
!77 = !MDLocation(line: 194, scope: !76)
!78 = !MDLocation(line: 195, scope: !12)
!79 = !MDLocation(line: 196, scope: !12)
!80 = !MDLocation(line: 197, scope: !12)
!81 = !MDLocation(line: 198, scope: !12)
!82 = !MDFile(filename: "dry.c", directory: "/Users/manmanren/test-Nov/rdar_13183203/test2")
!83 = !{i32 1, !"Debug Info Version", i32 3}
