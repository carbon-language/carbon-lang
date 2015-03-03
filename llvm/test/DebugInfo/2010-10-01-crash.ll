; RUN: llc -O0 %s -o /dev/null
; XFAIL: hexagon
; PR 8235

define void @CGRectStandardize(i32* sret %agg.result, i32* byval %rect) nounwind ssp {
entry:
  call void @llvm.dbg.declare(metadata i32* %rect, metadata !23, metadata !MDExpression()), !dbg !24
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind


!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!27}
!0 = !MDSubprogram(name: "CGRectStandardize", linkageName: "CGRectStandardize", line: 54, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !1, scope: null, function: void (i32*, i32*)* @CGRectStandardize)
!1 = !MDFile(filename: "GSFusedSilica.m", directory: "/Volumes/Data/Users/sabre/Desktop")
!2 = !MDCompileUnit(language: DW_LANG_ObjC, producer: "clang version 2.9 (trunk 115292)", isOptimized: true, runtimeVersion: 1, emissionKind: 0, file: !25, enums: !26, retainedTypes: !26)
!5 = !MDDerivedType(tag: DW_TAG_typedef, name: "CGRect", line: 49, file: !25, baseType: null)
!23 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "rect", line: 53, arg: 0, scope: !0, file: !1, type: !5)
!24 = !MDLocation(line: 53, column: 33, scope: !0)
!25 = !MDFile(filename: "GSFusedSilica.m", directory: "/Volumes/Data/Users/sabre/Desktop")
!26 = !{i32 0}
!27 = !{i32 1, !"Debug Info Version", i32 3}
