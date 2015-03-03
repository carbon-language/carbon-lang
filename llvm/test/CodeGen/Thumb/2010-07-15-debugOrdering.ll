; RUN: llc -mtriple=thumbv6-apple-darwin10 < %s | FileCheck %s
; RUN: opt -strip-debug < %s | llc -mtriple=thumbv6-apple-darwin10 | FileCheck %s
; Stripping out debug info formerly caused the last two multiplies to be emitted in
; the other order.  7797940 (part of it dated 6/29/2010..7/15/2010).

%0 = type { [3 x double] }

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%0*, i32, i32)* @_Z19getClosestDiagonal3ii to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define void @_Z19getClosestDiagonal3ii(%0* noalias sret, i32, i32) nounwind {
; CHECK: blx ___muldf3
; CHECK: blx ___muldf3
; CHECK: beq LBB0
; CHECK: blx ___muldf3
; <label>:3
  switch i32 %1, label %4 [
    i32 0, label %5
    i32 3, label %5
  ]

; <label>:4                                       ; preds = %3
  br label %5, !dbg !0

; <label>:5                                       ; preds = %4, %3, %3
  %storemerge = phi double [ -1.000000e+00, %4 ], [ 1.000000e+00, %3 ], [ 1.000000e+00, %3 ] ; <double> [#uses=1]
  %v_6 = icmp slt i32 %1, 2                         ; <i1> [#uses=1]
  %storemerge1 = select i1 %v_6, double 1.000000e+00, double -1.000000e+00 ; <double> [#uses=3]
  call void @llvm.dbg.value(metadata double %storemerge, i64 0, metadata !91, metadata !MDExpression()), !dbg !0
  %v_7 = icmp eq i32 %2, 1, !dbg !92                ; <i1> [#uses=1]
  %storemerge2 = select i1 %v_7, double 1.000000e+00, double -1.000000e+00 ; <double> [#uses=3]
  %v_8 = getelementptr inbounds %0, %0* %0, i32 0, i32 0, i32 0 ; <double*> [#uses=1]
  %v_10 = getelementptr inbounds %0, %0* %0, i32 0, i32 0, i32 2 ; <double*> [#uses=1]
  %v_11 = fmul double %storemerge1, %storemerge1, !dbg !93 ; <double> [#uses=1]
  %v_15 = tail call double @sqrt(double %v_11) nounwind readonly, !dbg !93 ; <double> [#uses=1]
  %v_16 = fdiv double 1.000000e+00, %v_15, !dbg !93   ; <double> [#uses=3]
  %v_17 = fmul double %storemerge, %v_16, !dbg !97    ; <double> [#uses=1]
  store double %v_17, double* %v_8, align 4, !dbg !97
  %v_19 = fmul double %storemerge2, %v_16, !dbg !97   ; <double> [#uses=1]
  store double %v_19, double* %v_10, align 4, !dbg !97
  ret void, !dbg !98
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare double @sqrt(double) nounwind readonly

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!104}
!0 = !MDLocation(line: 46, scope: !1)
!1 = distinct !MDLexicalBlock(line: 44, column: 0, file: !101, scope: !2)
!2 = distinct !MDLexicalBlock(line: 44, column: 0, file: !101, scope: !3)
!3 = !MDSubprogram(name: "getClosestDiagonal3", linkageName: "_Z19getClosestDiagonal3ii", line: 44, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !101, scope: null, type: !6)
!4 = !MDFile(filename: "ggEdgeDiscrepancy.cc", directory: "/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src")
!5 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build 00)", isOptimized: true, emissionKind: 0, file: !101, enums: !102, retainedTypes: !102, subprograms: !103)
!6 = !MDSubroutineType(types: !7)
!7 = !{!8, !22, !22}
!8 = !MDCompositeType(tag: DW_TAG_structure_type, name: "ggVector3", line: 66, size: 192, align: 32, file: !99, elements: !10)
!9 = !MDFile(filename: "ggVector3.h", directory: "/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src")
!99 = !MDFile(filename: "ggVector3.h", directory: "/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src")
!10 = !{!11, !16, !23, !26, !29, !30, !35, !36, !37, !41, !42, !43, !46, !47, !48, !52, !53, !54, !57, !60, !63, !66, !70, !71, !74, !75, !76, !77, !78, !81, !82, !83, !84, !85, !88, !89, !90}
!11 = !MDDerivedType(tag: DW_TAG_member, name: "e", line: 160, size: 192, align: 32, file: !99, scope: !8, baseType: !12)
!12 = !MDCompositeType(tag: DW_TAG_array_type, size: 192, align: 32, file: !101, scope: !4, baseType: !13, elements: !14)
!13 = !MDBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 32, encoding: DW_ATE_float)
!14 = !{!15}
!15 = !MDSubrange(count: 3)
!16 = !MDSubprogram(name: "ggVector3", line: 72, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !17)
!17 = !MDSubroutineType(types: !18)
!18 = !{null, !19, !20}
!19 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, flags: DIFlagArtificial, file: !101, scope: !4, baseType: !8)
!20 = !MDDerivedType(tag: DW_TAG_typedef, name: "ggBoolean", line: 478, file: !100, baseType: !22)
!21 = !MDFile(filename: "math.h", directory: "/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm")
!100 = !MDFile(filename: "math.h", directory: "/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS4.2.Internal.sdk/usr/include/architecture/arm")
!22 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!23 = !MDSubprogram(name: "ggVector3", line: 73, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !24)
!24 = !MDSubroutineType(types: !25)
!25 = !{null, !19}
!26 = !MDSubprogram(name: "ggVector3", line: 74, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !27)
!27 = !MDSubroutineType(types: !28)
!28 = !{null, !19, !13, !13, !13}
!29 = !MDSubprogram(name: "Set", linkageName: "_ZN9ggVector33SetEddd", line: 81, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !27)
!30 = !MDSubprogram(name: "x", linkageName: "_ZNK9ggVector31xEv", line: 82, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!31 = !MDSubroutineType(types: !32)
!32 = !{!13, !33}
!33 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, flags: DIFlagArtificial, file: !101, scope: !4, baseType: !34)
!34 = !MDDerivedType(tag: DW_TAG_const_type, size: 192, align: 32, file: !101, scope: !4, baseType: !8)
!35 = !MDSubprogram(name: "y", linkageName: "_ZNK9ggVector31yEv", line: 83, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!36 = !MDSubprogram(name: "z", linkageName: "_ZNK9ggVector31zEv", line: 84, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!37 = !MDSubprogram(name: "x", linkageName: "_ZN9ggVector31xEv", line: 85, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !38)
!38 = !MDSubroutineType(types: !39)
!39 = !{!40, !19}
!40 = !MDDerivedType(tag: DW_TAG_reference_type, name: "double", size: 32, align: 32, file: !101, scope: !4, baseType: !13)
!41 = !MDSubprogram(name: "y", linkageName: "_ZN9ggVector31yEv", line: 86, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !38)
!42 = !MDSubprogram(name: "z", linkageName: "_ZN9ggVector31zEv", line: 87, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !38)
!43 = !MDSubprogram(name: "SetX", linkageName: "_ZN9ggVector34SetXEd", line: 88, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !44)
!44 = !MDSubroutineType(types: !45)
!45 = !{null, !19, !13}
!46 = !MDSubprogram(name: "SetY", linkageName: "_ZN9ggVector34SetYEd", line: 89, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !44)
!47 = !MDSubprogram(name: "SetZ", linkageName: "_ZN9ggVector34SetZEd", line: 90, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !44)
!48 = !MDSubprogram(name: "ggVector3", line: 92, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !49)
!49 = !MDSubroutineType(types: !50)
!50 = !{null, !19, !51}
!51 = !MDDerivedType(tag: DW_TAG_reference_type, size: 32, align: 32, file: !101, scope: !4, baseType: !34)
!52 = !MDSubprogram(name: "tolerance", linkageName: "_ZNK9ggVector39toleranceEv", line: 100, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!53 = !MDSubprogram(name: "tolerance", linkageName: "_ZN9ggVector39toleranceEv", line: 101, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !38)
!54 = !MDSubprogram(name: "operator+", linkageName: "_ZNK9ggVector3psEv", line: 107, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !55)
!55 = !MDSubroutineType(types: !56)
!56 = !{!51, !33}
!57 = !MDSubprogram(name: "operator-", linkageName: "_ZNK9ggVector3ngEv", line: 108, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !58)
!58 = !MDSubroutineType(types: !59)
!59 = !{!8, !33}
!60 = !MDSubprogram(name: "operator[]", linkageName: "_ZNK9ggVector3ixEi", line: 290, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !61)
!61 = !MDSubroutineType(types: !62)
!62 = !{!13, !33, !22}
!63 = !MDSubprogram(name: "operator[]", linkageName: "_ZN9ggVector3ixEi", line: 278, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !64)
!64 = !MDSubroutineType(types: !65)
!65 = !{!40, !19, !22}
!66 = !MDSubprogram(name: "operator+=", linkageName: "_ZN9ggVector3pLERKS_", line: 303, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !67)
!67 = !MDSubroutineType(types: !68)
!68 = !{!69, !19, !51}
!69 = !MDDerivedType(tag: DW_TAG_reference_type, name: "ggVector3", size: 32, align: 32, file: !101, scope: !4, baseType: !8)
!70 = !MDSubprogram(name: "operator-=", linkageName: "_ZN9ggVector3mIERKS_", line: 310, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !67)
!71 = !MDSubprogram(name: "operator*=", linkageName: "_ZN9ggVector3mLEd", line: 317, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !72)
!72 = !MDSubroutineType(types: !73)
!73 = !{!69, !19, !13}
!74 = !MDSubprogram(name: "operator/=", linkageName: "_ZN9ggVector3dVEd", line: 324, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !72)
!75 = !MDSubprogram(name: "length", linkageName: "_ZNK9ggVector36lengthEv", line: 121, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!76 = !MDSubprogram(name: "squaredLength", linkageName: "_ZNK9ggVector313squaredLengthEv", line: 122, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!77 = !MDSubprogram(name: "MakeUnitVector", linkageName: "_ZN9ggVector314MakeUnitVectorEv", line: 217, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !24)
!78 = !MDSubprogram(name: "Perturb", linkageName: "_ZNK9ggVector37PerturbEdd", line: 126, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !79)
!79 = !MDSubroutineType(types: !80)
!80 = !{!8, !33, !13, !13}
!81 = !MDSubprogram(name: "maxComponent", linkageName: "_ZNK9ggVector312maxComponentEv", line: 128, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!82 = !MDSubprogram(name: "minComponent", linkageName: "_ZNK9ggVector312minComponentEv", line: 129, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!83 = !MDSubprogram(name: "maxAbsComponent", linkageName: "_ZNK9ggVector315maxAbsComponentEv", line: 131, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!84 = !MDSubprogram(name: "minAbsComponent", linkageName: "_ZNK9ggVector315minAbsComponentEv", line: 132, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !31)
!85 = !MDSubprogram(name: "indexOfMinComponent", linkageName: "_ZNK9ggVector319indexOfMinComponentEv", line: 133, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !86)
!86 = !MDSubroutineType(types: !87)
!87 = !{!22, !33}
!88 = !MDSubprogram(name: "indexOfMinAbsComponent", linkageName: "_ZNK9ggVector322indexOfMinAbsComponentEv", line: 137, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !86)
!89 = !MDSubprogram(name: "indexOfMaxComponent", linkageName: "_ZNK9ggVector319indexOfMaxComponentEv", line: 146, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !86)
!90 = !MDSubprogram(name: "indexOfMaxAbsComponent", linkageName: "_ZNK9ggVector322indexOfMaxAbsComponentEv", line: 150, isLocal: false, isDefinition: false, virtualIndex: 6, isOptimized: false, file: !9, scope: !8, type: !86)
!91 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "vx", line: 46, scope: !1, file: !4, type: !13)
!92 = !MDLocation(line: 48, scope: !1)
!93 = !MDLocation(line: 218, scope: !94, inlinedAt: !96)
!94 = distinct !MDLexicalBlock(line: 217, column: 0, file: !101, scope: !95)
!95 = distinct !MDLexicalBlock(line: 217, column: 0, file: !101, scope: !77)
!96 = !MDLocation(line: 51, scope: !1)
!97 = !MDLocation(line: 227, scope: !94, inlinedAt: !96)
!98 = !MDLocation(line: 52, scope: !1)
!101 = !MDFile(filename: "ggEdgeDiscrepancy.cc", directory: "/Volumes/Home/grosbaj/sources/llvm-externals/speccpu2000/benchspec/CINT2000/252.eon/src")
!102 = !{i32 0}
!103 = !{!3, !77}
!104 = !{i32 1, !"Debug Info Version", i32 3}
