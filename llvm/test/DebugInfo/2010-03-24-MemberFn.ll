; RUN: %llc_dwarf -O0 < %s | grep AT_decl_file |  grep 2
; Here _ZN1S3fooEv is defined in header file identified as AT_decl_file no. 2 in debug info.
%struct.S = type <{ i8 }>

define i32 @_Z3barv() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %s1 = alloca %struct.S                          ; <%struct.S*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.S* %s1, metadata !0, metadata !MDExpression()), !dbg !16
  %1 = call i32 @_ZN1S3fooEv(%struct.S* %s1) nounwind, !dbg !17 ; <i32> [#uses=1]
  store i32 %1, i32* %0, align 4, !dbg !17
  %2 = load i32, i32* %0, align 4, !dbg !17            ; <i32> [#uses=1]
  store i32 %2, i32* %retval, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %entry
  %retval1 = load i32, i32* %retval, !dbg !17          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !16
}

define linkonce_odr i32 @_ZN1S3fooEv(%struct.S* %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca %struct.S*                  ; <%struct.S**> [#uses=1]
  %retval = alloca i32                            ; <i32*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.S** %this_addr, metadata !18, metadata !MDExpression()), !dbg !21
  store %struct.S* %this, %struct.S** %this_addr
  br label %return, !dbg !21

return:                                           ; preds = %entry
  %retval1 = load i32, i32* %retval, !dbg !21          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!28}

!0 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "s1", line: 3, scope: !1, file: !4, type: !9)
!1 = distinct !MDLexicalBlock(line: 3, column: 0, file: !25, scope: !2)
!2 = distinct !MDLexicalBlock(line: 3, column: 0, file: !25, scope: !3)
!3 = !MDSubprogram(name: "bar", linkageName: "_Z3barv", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 3, file: !25, scope: !4, type: !6, function: i32 ()* @_Z3barv)
!4 = !MDFile(filename: "one.cc", directory: "/tmp/")
!5 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: 0, file: !25, enums: !27, retainedTypes: !27, subprograms: !24, imports:  null)
!6 = !MDSubroutineType(types: !7)
!7 = !{!8}
!8 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !MDCompositeType(tag: DW_TAG_structure_type, name: "S", line: 2, size: 8, align: 8, file: !26, scope: !4, elements: !11)
!10 = !MDFile(filename: "one.h", directory: "/tmp/")
!11 = !{!12}
!12 = !MDSubprogram(name: "foo", linkageName: "_ZN1S3fooEv", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 3, file: !26, scope: !9, type: !13, function: i32 (%struct.S*)* @_ZN1S3fooEv)
!13 = !MDSubroutineType(types: !14)
!14 = !{!8, !15}
!15 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !25, scope: !4, baseType: !9)
!16 = !MDLocation(line: 3, scope: !1)
!17 = !MDLocation(line: 3, scope: !3)
!18 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "this", line: 3, arg: 0, scope: !12, file: !10, type: !19)
!19 = !MDDerivedType(tag: DW_TAG_const_type, size: 64, align: 64, flags: DIFlagArtificial, file: !25, scope: !4, baseType: !20)
!20 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !25, scope: !4, baseType: !9)
!21 = !MDLocation(line: 3, scope: !12)
!22 = !MDLocation(line: 3, scope: !23)
!23 = distinct !MDLexicalBlock(line: 3, column: 0, file: !26, scope: !12)
!24 = !{!3, !12}
!25 = !MDFile(filename: "one.cc", directory: "/tmp/")
!26 = !MDFile(filename: "one.h", directory: "/tmp/")
!27 = !{i32 0}
!28 = !{i32 1, !"Debug Info Version", i32 3}
