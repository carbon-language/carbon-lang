; RUN: llc %s -o - -stop-before=phi-node-elimination \
; RUN:     -experimental-debug-variable-locations=true \
; RUN:    | FileCheck %s --check-prefixes=DYN_LOWERED

;; Copy of instr-ref-dyn-alloca-win32.ll, targetting 64 bit Windows. Here,
;; _chkstk doesn't return with a modified stack pointer, instead we have to
;; edit it with our own subtract operation. Check that it's labelled and the
;; substitution is correct. This also covers the code paths for non-Windows on
;; x86.

; DYN_LOWERED:      debugValueSubstitutions:
; DYN_LOWERED-NEXT: - { srcinst: 1, srcop: 2, dstinst: 2, dstop: 0, subreg: 0 }
; DYN_LOWERED:      SUB64rr $rsp, killed $rax, {{.*}} debug-instr-number 2,

source_filename = "test/DebugInfo/COFF/types-array-advanced.ll"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24210"

%struct.incomplete_struct = type { i32 }

@"\01?multi_dim_arr@@3PAY146DA" = global [2 x [5 x [7 x i8]]] zeroinitializer, align 1, !dbg !0
@"\01?p_incomplete_struct_arr@@3PAY02Uincomplete_struct@@A" = global [3 x i8]* null, align 4, !dbg !6
@"\01?incomplete_struct_arr@@3PAUincomplete_struct@@A" = global [3 x %struct.incomplete_struct] zeroinitializer, align 4, !dbg !16
@"\01?typedef_arr@@3SDHD" = constant [4 x i32] zeroinitializer, align 4, !dbg !18

; Function Attrs: nounwind
define void @"\01?foo@@YAXH@Z"(i32 %x) #0 !dbg !35 {
entry:
  %x.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !38, metadata !39), !dbg !40
  %0 = load i32, i32* %x.addr, align 4, !dbg !41
  %1 = call i8* @llvm.stacksave(), !dbg !42
  store i8* %1, i8** %saved_stack, align 4, !dbg !42
  %vla = alloca i32, i32 %0, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !43, metadata !47), !dbg !48
  %arrayidx = getelementptr inbounds i32, i32* %vla, i32 0, !dbg !49
  store i32 0, i32* %arrayidx, align 4, !dbg !50
  %2 = load i8*, i8** %saved_stack, align 4, !dbg !51
  call void @llvm.stackrestore(i8* %2), !dbg !51
  ret void, !dbg !51
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind
declare i8* @llvm.stacksave()

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!32, !33}
!llvm.ident = !{!34}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "multi_dim_arr", linkageName: "\01?multi_dim_arr@@3PAY146DA", scope: !2, file: !3, line: 1, type: !26, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 273874)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/")
!4 = !{}
!5 = !{!0, !6, !16, !18}
!6 = distinct !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "p_incomplete_struct_arr", linkageName: "\01?p_incomplete_struct_arr@@3PAY02Uincomplete_struct@@A", scope: !2, file: !3, line: 3, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 32, align: 32)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, elements: !14)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "incomplete_struct", file: !3, line: 4, size: 32, align: 32, elements: !11, identifier: ".?AUincomplete_struct@@")
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !10, file: !3, line: 5, baseType: !13, size: 32, align: 32)
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DISubrange(count: 3)
!16 = distinct !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = !DIGlobalVariable(name: "incomplete_struct_arr", linkageName: "\01?incomplete_struct_arr@@3PAUincomplete_struct@@A", scope: !2, file: !3, line: 6, type: !9, isLocal: false, isDefinition: true)
!18 = distinct !DIGlobalVariableExpression(var: !19, expr: !DIExpression())
!19 = !DIGlobalVariable(name: "typedef_arr", linkageName: "\01?typedef_arr@@3SDHD", scope: !2, file: !3, line: 14, type: !20, isLocal: false, isDefinition: true)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 128, align: 32, elements: !24)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "T_INT", file: !3, line: 13, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !13)
!24 = !{!25}
!25 = !DISubrange(count: 4)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !27, size: 560, align: 8, elements: !28)
!27 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!28 = !{!29, !30, !31}
!29 = !DISubrange(count: 2)
!30 = !DISubrange(count: 5)
!31 = !DISubrange(count: 7)
!32 = !{i32 2, !"CodeView", i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{!"clang version 3.9.0 (trunk 273874)"}
!35 = distinct !DISubprogram(name: "foo", linkageName: "\01?foo@@YAXH@Z", scope: !3, file: !3, line: 8, type: !36, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!36 = !DISubroutineType(types: !37)
!37 = !{null, !13}
!38 = !DILocalVariable(name: "x", arg: 1, scope: !35, file: !3, line: 8, type: !13)
!39 = !DIExpression()
!40 = !DILocation(line: 8, column: 14, scope: !35)
!41 = !DILocation(line: 9, column: 21, scope: !35)
!42 = !DILocation(line: 9, column: 4, scope: !35)
!43 = !DILocalVariable(name: "dyn_size_arr", scope: !35, file: !3, line: 9, type: !44)
!44 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, align: 32, elements: !45)
!45 = !{!46}
!46 = !DISubrange(count: -1)
!47 = !DIExpression(DW_OP_deref)
!48 = !DILocation(line: 9, column: 8, scope: !35)
!49 = !DILocation(line: 10, column: 4, scope: !35)
!50 = !DILocation(line: 10, column: 20, scope: !35)
!51 = !DILocation(line: 11, column: 1, scope: !35)

