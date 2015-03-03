; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s
; RUN: llc -mtriple=x86_64-linux-gnu < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=LINUX %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj -regalloc=basic \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "nums"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "nums"

;CHECK: DW_TAG_inlined_subroutine
;CHECK-NEXT: DW_AT_abstract_origin {{.*}} "foo"
;CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]
;CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]
;CHECK-NEXT: DW_AT_call_file
;CHECK-NEXT: DW_AT_call_line

;CHECK: DW_TAG_formal_parameter
;FIXME: Linux shouldn't drop this parameter either...
;CHECK-NOT: DW_TAG
;DARWIN:   DW_AT_abstract_origin {{.*}} "sp"
;DARWIN: DW_TAG_formal_parameter
;CHECK: DW_AT_abstract_origin {{.*}} "nums"
;CHECK-NOT: DW_TAG_formal_parameter

%struct.S1 = type { float*, i32 }

@p = common global %struct.S1 zeroinitializer, align 8

define i32 @foo(%struct.S1* nocapture %sp, i32 %nums) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata %struct.S1* %sp, i64 0, metadata !9, metadata !MDExpression()), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 %nums, i64 0, metadata !18, metadata !MDExpression()), !dbg !21
  %tmp2 = getelementptr inbounds %struct.S1, %struct.S1* %sp, i64 0, i32 1, !dbg !22
  store i32 %nums, i32* %tmp2, align 4, !dbg !22
  %call = tail call float* @bar(i32 %nums) nounwind optsize, !dbg !27
  %tmp5 = getelementptr inbounds %struct.S1, %struct.S1* %sp, i64 0, i32 0, !dbg !27
  store float* %call, float** %tmp5, align 8, !dbg !27
  %cmp = icmp ne float* %call, null, !dbg !29
  %cond = zext i1 %cmp to i32, !dbg !29
  ret i32 %cond, !dbg !29
}

declare float* @bar(i32) optsize

define void @foobar() nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata %struct.S1* @p, i64 0, metadata !9, metadata !MDExpression()) nounwind, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !18, metadata !MDExpression()) nounwind, !dbg !35
  store i32 1, i32* getelementptr inbounds (%struct.S1* @p, i64 0, i32 1), align 8, !dbg !36
  %call.i = tail call float* @bar(i32 1) nounwind optsize, !dbg !37
  store float* %call.i, float** getelementptr inbounds (%struct.S1* @p, i64 0, i32 0), align 8, !dbg !37
  ret void, !dbg !38
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!43}

!0 = !MDSubprogram(name: "foo", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 8, file: !1, scope: !1, type: !3, function: i32 (%struct.S1*, i32)* @foo, variables: !41)
!1 = !MDFile(filename: "nm2.c", directory: "/private/tmp")
!2 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 125693)", isOptimized: true, emissionKind: 1, file: !42, enums: !8, retainedTypes: !8, subprograms: !39, globals: !40, imports:  !44)
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDSubprogram(name: "foobar", line: 15, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !1, scope: !1, type: !7, function: void ()* @foobar)
!7 = !MDSubroutineType(types: !8)
!8 = !{null}
!9 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "sp", line: 7, arg: 1, scope: !0, file: !1, type: !10, inlinedAt: !32)
!10 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !2, baseType: !11)
!11 = !MDDerivedType(tag: DW_TAG_typedef, name: "S1", line: 4, file: !42, scope: !2, baseType: !12)
!12 = !MDCompositeType(tag: DW_TAG_structure_type, name: "S1", line: 1, size: 128, align: 64, file: !42, scope: !2, elements: !13)
!13 = !{!14, !17}
!14 = !MDDerivedType(tag: DW_TAG_member, name: "m", line: 2, size: 64, align: 64, file: !42, scope: !1, baseType: !15)
!15 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, scope: !2, baseType: !16)
!16 = !MDBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!17 = !MDDerivedType(tag: DW_TAG_member, name: "nums", line: 3, size: 32, align: 32, offset: 64, file: !42, scope: !1, baseType: !5)
!18 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "nums", line: 7, arg: 2, scope: !0, file: !1, type: !5, inlinedAt: !32)
!19 = !MDGlobalVariable(name: "p", line: 14, isLocal: false, isDefinition: true, scope: !2, file: !1, type: !11, variable: %struct.S1* @p)
!20 = !MDLocation(line: 7, column: 13, scope: !0)
!21 = !MDLocation(line: 7, column: 21, scope: !0)
!22 = !MDLocation(line: 9, column: 3, scope: !23)
!23 = distinct !MDLexicalBlock(line: 8, column: 1, file: !1, scope: !0)
!27 = !MDLocation(line: 10, column: 3, scope: !23)
!29 = !MDLocation(line: 11, column: 3, scope: !23)
!30 = !{%struct.S1* @p}
!31 = !MDLocation(line: 7, column: 13, scope: !0, inlinedAt: !32)
!32 = !MDLocation(line: 16, column: 3, scope: !33)
!33 = distinct !MDLexicalBlock(line: 15, column: 15, file: !1, scope: !6)
!34 = !{i32 1}
!35 = !MDLocation(line: 7, column: 21, scope: !0, inlinedAt: !32)
!36 = !MDLocation(line: 9, column: 3, scope: !23, inlinedAt: !32)
!37 = !MDLocation(line: 10, column: 3, scope: !23, inlinedAt: !32)
!38 = !MDLocation(line: 17, column: 1, scope: !33)
!39 = !{!0, !6}
!40 = !{!19}
!41 = !{!9, !18}
!42 = !MDFile(filename: "nm2.c", directory: "/private/tmp")
!43 = !{i32 1, !"Debug Info Version", i32 3}
!44 = !{}
