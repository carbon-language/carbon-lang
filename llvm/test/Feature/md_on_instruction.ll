; RUN: llvm-as < %s | llvm-dis | grep " !dbg " | count 4
define i32 @foo() nounwind ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  call void @llvm.dbg.func.start(metadata !0)
  store i32 42, i32* %retval, !dbg !3
  br label %0, !dbg !3

; <label>:0                                       ; preds = %entry
  call void @llvm.dbg.region.end(metadata !0)
  %1 = load i32, i32* %retval, !dbg !3                  ; <i32> [#uses=1]
  ret i32 %1, !dbg !3
}

declare void @llvm.dbg.func.start(metadata) nounwind readnone

declare void @llvm.dbg.region.end(metadata) nounwind readnone

!llvm.module.flags = !{!6}

!0 = !MDSubprogram(name: "foo", linkageName: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scope: !1, type: !2)
!1 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang 1.0", isOptimized: true, emissionKind: 0, file: !4, enums: !5, retainedTypes: !5, subprograms: !4)
!2 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!3 = !MDLocation(line: 1, column: 13, scope: !1, inlinedAt: !1)
!4 = !MDFile(filename: "foo.c", directory: "/tmp")
!5 = !{i32 0}
!6 = !{i32 1, !"Debug Info Version", i32 3}
