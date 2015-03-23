; RUN: llc -O3 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Check debug info output for merged global.
; DW_AT_location
; 0x03 DW_OP_addr
; 0x.. .long __MergedGlobals
; 0x10 DW_OP_constu
; 0x.. offset
; 0x22 DW_OP_plus

; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}} "x1"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x8> 03 [[ADDR:.. .. .. ..]] 10 00 22  )
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:    DW_AT_name {{.*}} "x2"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:    DW_AT_location [DW_FORM_exprloc]        (<0x8> 03 [[ADDR]] 10 04 22  )

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.7.0"

@x1 = internal unnamed_addr global i32 1, align 4
@x2 = internal unnamed_addr global i32 2, align 4
@x3 = internal unnamed_addr global i32 3, align 4
@x4 = internal unnamed_addr global i32 4, align 4
@x5 = global i32 0, align 4

define i32 @get1(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !10, metadata !MDExpression()), !dbg !30
  %1 = load i32, i32* @x1, align 4, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !11, metadata !MDExpression()), !dbg !31
  store i32 %a, i32* @x1, align 4, !dbg !31
  ret i32 %1, !dbg !31
}

define i32 @get2(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !13, metadata !MDExpression()), !dbg !32
  %1 = load i32, i32* @x2, align 4, !dbg !33
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !14, metadata !MDExpression()), !dbg !33
  store i32 %a, i32* @x2, align 4, !dbg !33
  ret i32 %1, !dbg !33
}

define i32 @get3(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !16, metadata !MDExpression()), !dbg !34
  %1 = load i32, i32* @x3, align 4, !dbg !35
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !17, metadata !MDExpression()), !dbg !35
  store i32 %a, i32* @x3, align 4, !dbg !35
  ret i32 %1, !dbg !35
}

define i32 @get4(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !19, metadata !MDExpression()), !dbg !36
  %1 = load i32, i32* @x4, align 4, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !20, metadata !MDExpression()), !dbg !37
  store i32 %a, i32* @x4, align 4, !dbg !37
  ret i32 %1, !dbg !37
}

define i32 @get5(i32 %a) nounwind optsize ssp {
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !27, metadata !MDExpression()), !dbg !38
  %1 = load i32, i32* @x5, align 4, !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !28, metadata !MDExpression()), !dbg !39
  store i32 %a, i32* @x5, align 4, !dbg !39
  ret i32 %1, !dbg !39
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!49}

!0 = !MDCompileUnit(language: DW_LANG_C99, producer: "clang", isOptimized: true, emissionKind: 1, file: !47, enums: !48, retainedTypes: !48, subprograms: !40, globals: !41, imports:  !48)
!1 = !MDSubprogram(name: "get1", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 5, file: !47, scope: !2, type: !3, function: i32 (i32)* @get1, variables: !42)
!2 = !MDFile(filename: "ss3.c", directory: "/private/tmp")
!3 = !MDSubroutineType(types: !4)
!4 = !{!5}
!5 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !MDSubprogram(name: "get2", line: 8, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 8, file: !47, scope: !2, type: !3, function: i32 (i32)* @get2, variables: !43)
!7 = !MDSubprogram(name: "get3", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 11, file: !47, scope: !2, type: !3, function: i32 (i32)* @get3, variables: !44)
!8 = !MDSubprogram(name: "get4", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 14, file: !47, scope: !2, type: !3, function: i32 (i32)* @get4, variables: !45)
!9 = !MDSubprogram(name: "get5", line: 17, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 17, file: !47, scope: !2, type: !3, function: i32 (i32)* @get5, variables: !46)
!10 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 5, arg: 1, scope: !1, file: !2, type: !5)
!11 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 5, scope: !12, file: !2, type: !5)
!12 = distinct !MDLexicalBlock(line: 5, column: 19, file: !47, scope: !1)
!13 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 8, arg: 1, scope: !6, file: !2, type: !5)
!14 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 8, scope: !15, file: !2, type: !5)
!15 = distinct !MDLexicalBlock(line: 8, column: 17, file: !47, scope: !6)
!16 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 11, arg: 1, scope: !7, file: !2, type: !5)
!17 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 11, scope: !18, file: !2, type: !5)
!18 = distinct !MDLexicalBlock(line: 11, column: 19, file: !47, scope: !7)
!19 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 14, arg: 1, scope: !8, file: !2, type: !5)
!20 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 14, scope: !21, file: !2, type: !5)
!21 = distinct !MDLexicalBlock(line: 14, column: 19, file: !47, scope: !8)
!25 = !MDGlobalVariable(name: "x1", line: 4, isLocal: true, isDefinition: true, scope: !0, file: !2, type: !5, variable: i32* @x1)
!26 = !MDGlobalVariable(name: "x2", line: 7, isLocal: true, isDefinition: true, scope: !0, file: !2, type: !5, variable: i32* @x2)
!27 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 17, arg: 1, scope: !9, file: !2, type: !5)
!28 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "b", line: 17, scope: !29, file: !2, type: !5)
!29 = distinct !MDLexicalBlock(line: 17, column: 19, file: !47, scope: !9)
!30 = !MDLocation(line: 5, column: 16, scope: !1)
!31 = !MDLocation(line: 5, column: 32, scope: !12)
!32 = !MDLocation(line: 8, column: 14, scope: !6)
!33 = !MDLocation(line: 8, column: 29, scope: !15)
!34 = !MDLocation(line: 11, column: 16, scope: !7)
!35 = !MDLocation(line: 11, column: 32, scope: !18)
!36 = !MDLocation(line: 14, column: 16, scope: !8)
!37 = !MDLocation(line: 14, column: 32, scope: !21)
!38 = !MDLocation(line: 17, column: 16, scope: !9)
!39 = !MDLocation(line: 17, column: 32, scope: !29)
!40 = !{!1, !6, !7, !8, !9}
!41 = !{!25, !26}
!42 = !{!10, !11}
!43 = !{!13, !14}
!44 = !{!16, !17}
!45 = !{!19, !20}
!46 = !{!27, !28}
!47 = !MDFile(filename: "ss3.c", directory: "/private/tmp")
!48 = !{}
!49 = !{i32 1, !"Debug Info Version", i32 3}
