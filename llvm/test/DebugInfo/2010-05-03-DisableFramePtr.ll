; RUN: llc  -o /dev/null < %s
; Radar 7937664
%struct.AppleEvent = type opaque

define void @DisposeDMNotificationUPP(void (%struct.AppleEvent*)* %userUPP) "no-frame-pointer-elim-non-leaf" nounwind ssp {
entry:
  %userUPP_addr = alloca void (%struct.AppleEvent*)* ; <void (%struct.AppleEvent*)**> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata void (%struct.AppleEvent*)** %userUPP_addr, metadata !0, metadata !MDExpression()), !dbg !13
  store void (%struct.AppleEvent*)* %userUPP, void (%struct.AppleEvent*)** %userUPP_addr
  br label %return, !dbg !14

return:                                           ; preds = %entry
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!19}
!0 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "userUPP", line: 7, arg: 0, scope: !1, file: !2, type: !6)
!1 = !MDSubprogram(name: "DisposeDMNotificationUPP", linkageName: "DisposeDMNotificationUPP", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !16, scope: null, type: !4)
!2 = !MDFile(filename: "t.c", directory: "/Users/echeng/LLVM/radars/r7937664/")
!3 = !MDCompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)", isOptimized: true, emissionKind: 0, file: !16, enums: !17, retainedTypes: !17, subprograms: !18)
!4 = !MDSubroutineType(types: !5)
!5 = !{null, !6}
!6 = !MDDerivedType(tag: DW_TAG_typedef, name: "DMNotificationUPP", line: 6, file: !16, scope: !2, baseType: !7)
!7 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !16, scope: !2, baseType: !8)
!8 = !MDSubroutineType(types: !9)
!9 = !{null, !10}
!10 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !16, scope: !2, baseType: !11)
!11 = !MDDerivedType(tag: DW_TAG_typedef, name: "AppleEvent", line: 4, file: !16, scope: !2, baseType: !12)
!12 = !MDCompositeType(tag: DW_TAG_structure_type, name: "AEDesc", line: 1, flags: DIFlagFwdDecl, file: !16, scope: !2)
!13 = !MDLocation(line: 7, scope: !1)
!14 = !MDLocation(line: 8, scope: !15)
!15 = distinct !MDLexicalBlock(line: 7, column: 0, file: !16, scope: !1)
!16 = !MDFile(filename: "t.c", directory: "/Users/echeng/LLVM/radars/r7937664/")
!17 = !{i32 0}
!18 = !{!1}
!19 = !{i32 1, !"Debug Info Version", i32 3}
