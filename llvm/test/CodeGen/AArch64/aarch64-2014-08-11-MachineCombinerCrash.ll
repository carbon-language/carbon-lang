; RUN: llc < %s -O2 -mtriple=aarch64-none-linux-gnu 

; Bug 20598


define void @test() #0 {
entry:
  br label %for.body, !dbg !39

for.body:                                         ; preds = %for.body, %entry
  %arrayidx5 = getelementptr inbounds i32, i32* null, i64 1, !dbg !43
  %0 = load i32, i32* null, align 4, !dbg !45, !tbaa !46
  %s1 = sub nsw i32 0, %0, !dbg !50
  %n1 = sext i32 %s1 to i64, !dbg !50
  %arrayidx21 = getelementptr inbounds i32, i32* null, i64 3, !dbg !51
  %add53 = add nsw i64 %n1, 0, !dbg !52
  %add55 = add nsw i64 %n1, 0, !dbg !53
  %mul63 = mul nsw i64 %add53, -20995, !dbg !54
  tail call void @llvm.dbg.value(metadata i64 %mul63, i64 0, metadata !30, metadata !MDExpression()), !dbg !55
  %mul65 = mul nsw i64 %add55, -3196, !dbg !56
  %add67 = add nsw i64 0, %mul65, !dbg !57
  %add80 = add i64 0, 1024, !dbg !58
  %add81 = add i64 %add80, %mul63, !dbg !58
  %add82 = add i64 %add81, 0, !dbg !58
  %shr83351 = lshr i64 %add82, 11, !dbg !58
  %conv84 = trunc i64 %shr83351 to i32, !dbg !58
  store i32 %conv84, i32* %arrayidx21, align 4, !dbg !58, !tbaa !46
  %add86 = add i64 0, 1024, !dbg !59
  %add87 = add i64 %add86, 0, !dbg !59
  %add88 = add i64 %add87, %add67, !dbg !59
  %shr89352 = lshr i64 %add88, 11, !dbg !59
  %n2 = trunc i64 %shr89352 to i32, !dbg !59
  store i32 %n2, i32* %arrayidx5, align 4, !dbg !59, !tbaa !46
  br label %for.body, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!36, !37}
!llvm.ident = !{!38}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !MDFile(filename: "test.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !MDSubprogram(name: "", line: 140, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 141, file: !1, scope: !1, type: !6, function: void ()* @test, variables: !12)
!6 = !MDSubroutineType(types: !7)
!7 = !{null, !8}
!8 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !MDDerivedType(tag: DW_TAG_typedef, line: 30, file: !1, baseType: !11)
!11 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35}
!13 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "", line: 140, arg: 1, scope: !4, file: !1, type: !8)
!14 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!15 = !MDDerivedType(tag: DW_TAG_typedef, line: 183, file: !1, baseType: !17)
!17 = !MDBasicType(tag: DW_TAG_base_type, size: 64, align: 64, encoding: DW_ATE_signed)
!18 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!19 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!20 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!21 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!22 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!23 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!24 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 142, scope: !4, file: !1, type: !15)
!25 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 143, scope: !4, file: !1, type: !15)
!26 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 143, scope: !4, file: !1, type: !15)
!27 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 143, scope: !4, file: !1, type: !15)
!28 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 143, scope: !4, file: !1, type: !15)
!29 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 144, scope: !4, file: !1, type: !15)
!30 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 144, scope: !4, file: !1, type: !15)
!31 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 144, scope: !4, file: !1, type: !15)
!32 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 144, scope: !4, file: !1, type: !15)
!33 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 144, scope: !4, file: !1, type: !15)
!34 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 145, scope: !4, file: !1, type: !8)
!35 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "", line: 146, scope: !4, file: !1, type: !11)
!36 = !{i32 2, !"Dwarf Version", i32 4}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{!"clang version 3.6.0 "}
!39 = !MDLocation(line: 154, column: 8, scope: !40)
!40 = distinct !MDLexicalBlock(line: 154, column: 8, file: !1, scope: !41)
!41 = distinct !MDLexicalBlock(line: 154, column: 8, file: !1, scope: !42)
!42 = distinct !MDLexicalBlock(line: 154, column: 3, file: !1, scope: !4)
!43 = !MDLocation(line: 157, column: 5, scope: !44)
!44 = distinct !MDLexicalBlock(line: 154, column: 42, file: !1, scope: !42)
!45 = !MDLocation(line: 159, column: 5, scope: !44)
!46 = !{!47, !47, i64 0}
!47 = !{!"int", !48, i64 0}
!48 = !{!"omnipotent char", !49, i64 0}
!49 = !{!"Simple C/C++ TBAA"}
!50 = !MDLocation(line: 160, column: 5, scope: !44)
!51 = !MDLocation(line: 161, column: 5, scope: !44)
!52 = !MDLocation(line: 188, column: 5, scope: !44)
!53 = !MDLocation(line: 190, column: 5, scope: !44)
!54 = !MDLocation(line: 198, column: 5, scope: !44)
!55 = !MDLocation(line: 144, column: 13, scope: !4)
!56 = !MDLocation(line: 200, column: 5, scope: !44)
!57 = !MDLocation(line: 203, column: 5, scope: !44)
!58 = !MDLocation(line: 207, column: 5, scope: !44)
!59 = !MDLocation(line: 208, column: 5, scope: !44)
