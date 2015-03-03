; RUN: llc < %s - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

;CHECK: sub-register DW_OP_regx
;CHECK-NEXT: 256
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: 8
;CHECK-NEXT: sub-register DW_OP_regx
;CHECK-NEXT: 257
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: 8

@.str = external constant [13 x i8]

declare <4 x float> @test0001(float) nounwind readnone ssp

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind ssp {
entry:
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %entry
  %add19 = fadd <4 x float> undef, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, !dbg !39
  br i1 undef, label %for.end54, label %for.body9, !dbg !44

for.end54:                                        ; preds = %for.body9
  tail call void @llvm.dbg.value(metadata <4 x float> %add19, i64 0, metadata !27, metadata !MDExpression()), !dbg !39
  %tmp115 = extractelement <4 x float> %add19, i32 1
  %conv6.i75 = fpext float %tmp115 to double, !dbg !45
  %call.i82 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([13 x i8]* @.str, i32 0, i32 0), double undef, double %conv6.i75, double undef, double undef) nounwind, !dbg !45
  ret i32 0, !dbg !49
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!56}

!0 = !MDSubprogram(name: "test0001", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !54, scope: !1, type: !3, function: <4 x float> (float)* @test0001, variables: !51)
!1 = !MDFile(filename: "build2.c", directory: "/private/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 129915)", isOptimized: true, emissionKind: 1, file: !54, enums: !17, retainedTypes: !17, subprograms: !50, imports:  null)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDDerivedType(tag: DW_TAG_typedef, name: "v4f32", line: 14, file: !54, scope: !2, baseType: !6)
!6 = !MDCompositeType(tag: DW_TAG_array_type, size: 128, align: 128, file: !2, baseType: !7, elements: !8)
!7 = !MDBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!8 = !{!9}
!9 = !MDSubrange(count: 4)
!10 = !MDSubprogram(name: "main", line: 59, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 59, file: !54, scope: !1, type: !11, function: i32 (i32, i8**)* @main, variables: !52)
!11 = !MDSubroutineType(types: !12)
!12 = !{!13}
!13 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !MDSubprogram(name: "printFV", line: 41, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 41, file: !55, scope: !15, type: !16, variables: !53)
!15 = !MDFile(filename: "/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", directory: "/private/tmp")
!16 = !MDSubroutineType(types: !17)
!17 = !{null}
!18 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 3, arg: 1, scope: !0, file: !1, type: !7)
!19 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "argc", line: 59, arg: 1, scope: !10, file: !1, type: !13)
!20 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "argv", line: 59, arg: 2, scope: !10, file: !1, type: !21)
!21 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !22)
!22 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !23)
!23 = !MDBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!24 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "i", line: 60, scope: !25, file: !1, type: !13)
!25 = distinct !MDLexicalBlock(line: 59, column: 33, file: !54, scope: !10)
!26 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "j", line: 60, scope: !25, file: !1, type: !13)
!27 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "x", line: 61, scope: !25, file: !1, type: !5)
!28 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "y", line: 62, scope: !25, file: !1, type: !5)
!29 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "z", line: 63, scope: !25, file: !1, type: !5)
!30 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "F", line: 41, arg: 1, scope: !14, file: !15, type: !31)
!31 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !32)
!32 = !MDDerivedType(tag: DW_TAG_typedef, name: "FV", line: 25, file: !55, scope: !2, baseType: !33)
!33 = !MDCompositeType(tag: DW_TAG_union_type, line: 22, size: 128, align: 128, file: !55, scope: !2, elements: !34)
!34 = !{!35, !37}
!35 = !MDDerivedType(tag: DW_TAG_member, name: "V", line: 23, size: 128, align: 128, file: !55, scope: !15, baseType: !36)
!36 = !MDDerivedType(tag: DW_TAG_typedef, name: "v4sf", line: 3, file: !55, scope: !2, baseType: !6)
!37 = !MDDerivedType(tag: DW_TAG_member, name: "A", line: 24, size: 128, align: 32, file: !55, scope: !15, baseType: !38)
!38 = !MDCompositeType(tag: DW_TAG_array_type, size: 128, align: 32, scope: !2, baseType: !7, elements: !8)
!39 = !MDLocation(line: 79, column: 7, scope: !40)
!40 = distinct !MDLexicalBlock(line: 75, column: 35, file: !54, scope: !41)
!41 = distinct !MDLexicalBlock(line: 75, column: 5, file: !54, scope: !42)
!42 = distinct !MDLexicalBlock(line: 71, column: 32, file: !54, scope: !43)
!43 = distinct !MDLexicalBlock(line: 71, column: 3, file: !54, scope: !25)
!44 = !MDLocation(line: 75, column: 5, scope: !42)
!45 = !MDLocation(line: 42, column: 2, scope: !46, inlinedAt: !48)
!46 = distinct !MDLexicalBlock(line: 42, column: 2, file: !55, scope: !47)
!47 = distinct !MDLexicalBlock(line: 41, column: 28, file: !55, scope: !14)
!48 = !MDLocation(line: 95, column: 3, scope: !25)
!49 = !MDLocation(line: 99, column: 3, scope: !25)
!50 = !{!0, !10, !14}
!51 = !{!18}
!52 = !{!19, !20, !24, !26, !27, !28, !29}
!53 = !{!30}
!54 = !MDFile(filename: "build2.c", directory: "/private/tmp")
!55 = !MDFile(filename: "/Volumes/Lalgate/work/llvm/projects/llvm-test/SingleSource/UnitTests/Vector/helpers.h", directory: "/private/tmp")
!56 = !{i32 1, !"Debug Info Version", i32 3}
