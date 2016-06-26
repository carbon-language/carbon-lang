; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; char multi_dim_arr[2][5][7];
;
; struct incomplete_struct(*p_incomplete_struct_arry)[3];
; struct incomplete_struct {
;   int s1;
; } incomplete_struct_arry[3];
;
; void foo(int x) {
;   int dyn_size_arr[x];
;   dyn_size_arr[0] = 0;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:   Section: .debug$T (5)
; CHECK:   Magic: 0x4
; CHECK:   ArgList (0x1000) {
; CHECK:     TypeLeafKind: LF_ARGLIST (0x1201)
; CHECK:     NumArgs: 1
; CHECK:     Arguments [
; CHECK:       ArgType: int (0x74)
; CHECK:     ]
; CHECK:   }
; CHECK:   Procedure (0x1001) {
; CHECK:     TypeLeafKind: LF_PROCEDURE (0x1008)
; CHECK:     ReturnType: void (0x3)
; CHECK:     CallingConvention: NearC (0x0)
; CHECK:     FunctionOptions [ (0x0)
; CHECK:     ]
; CHECK:     NumParameters: 1
; CHECK:     ArgListType: (int) (0x1000)
; CHECK:   }
; CHECK:   FuncId (0x1002) {
; CHECK:     TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK:     ParentScope: 0x0
; CHECK:     FunctionType: void (int) (0x1001)
; CHECK:     Name: foo
; CHECK:   }
; CHECK:   Array (0x1003) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: int (0x74)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 4
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Array (0x1004) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: char (0x70)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 7
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Array (0x1005) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: 0x1004
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 35
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Array (0x1006) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: 0x1005
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 70
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Struct (0x1007) {
; CHECK:     TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:     MemberCount: 0
; CHECK:     Properties [ (0x280)
; CHECK:       ForwardReference (0x80)
; CHECK:       HasUniqueName (0x200)
; CHECK:     ]
; CHECK:     FieldList: 0x0
; CHECK:     DerivedFrom: 0x0
; CHECK:     VShape: 0x0
; CHECK:     SizeOf: 0
; CHECK:     Name: incomplete_struct
; CHECK:     LinkageName: .?AUincomplete_struct@@
; CHECK:   }
; CHECK:   Array (0x1008) {
; CHECK:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK:     ElementType: incomplete_struct (0x1007)
; CHECK:     IndexType: unsigned long (0x22)
; CHECK:     SizeOf: 12
; CHECK:     Name: 
; CHECK:   }
; CHECK:   Pointer (0x1009) {
; CHECK:     TypeLeafKind: LF_POINTER (0x1002)
; CHECK:     PointeeType: 0x1008
; CHECK:     PointerAttributes: 0x800A
; CHECK:     PtrType: Near32 (0xA)
; CHECK:     PtrMode: Pointer (0x0)
; CHECK:     IsFlat: 0
; CHECK:     IsConst: 0
; CHECK:     IsVolatile: 0
; CHECK:     IsUnaligned: 0
; CHECK:     SizeOf: 4
; CHECK:   }
; CHECK: ]

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.31101"

%struct.incomplete_struct = type { i32 }

@"\01?multi_dim_arr@@3PAY146DA" = global [2 x [5 x [7 x i8]]] zeroinitializer, align 1
@"\01?p_incomplete_struct_arry@@3PAY02Uincomplete_struct@@A" = global [3 x i8]* null, align 4
@"\01?incomplete_struct_arry@@3PAUincomplete_struct@@A" = global [3 x %struct.incomplete_struct] zeroinitializer, align 4

; Function Attrs: nounwind
define void @"\01?foo@@YAXH@Z"(i32 %x) #0 !dbg !24 {
entry:
  %x.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !27, metadata !28), !dbg !29
  %0 = load i32, i32* %x.addr, align 4, !dbg !30
  %1 = call i8* @llvm.stacksave(), !dbg !31
  store i8* %1, i8** %saved_stack, align 4, !dbg !31
  %vla = alloca i32, i32 %0, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !32, metadata !36), !dbg !37
  %arrayidx = getelementptr inbounds i32, i32* %vla, i32 0, !dbg !38
  store i32 0, i32* %arrayidx, align 4, !dbg !39
  %2 = load i8*, i8** %saved_stack, align 4, !dbg !40
  call void @llvm.stackrestore(i8* %2), !dbg !40
  ret void, !dbg !40
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #2

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 273084)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{}
!3 = !{!4, !11, !20}
!4 = distinct !DIGlobalVariable(name: "multi_dim_arr", linkageName: "\01?multi_dim_arr@@3PAY146DA", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, variable: [2 x [5 x [7 x i8]]]* @"\01?multi_dim_arr@@3PAY146DA")
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 560, align: 8, elements: !7)
!6 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!7 = !{!8, !9, !10}
!8 = !DISubrange(count: 2)
!9 = !DISubrange(count: 5)
!10 = !DISubrange(count: 7)
!11 = distinct !DIGlobalVariable(name: "p_incomplete_struct_arry", linkageName: "\01?p_incomplete_struct_arry@@3PAY02Uincomplete_struct@@A", scope: !0, file: !1, line: 3, type: !12, isLocal: false, isDefinition: true, variable: [3 x i8]** @"\01?p_incomplete_struct_arry@@3PAY02Uincomplete_struct@@A")
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 32, align: 32)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, elements: !18)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "incomplete_struct", file: !1, line: 4, size: 32, align: 32, elements: !15, identifier: ".?AUincomplete_struct@@")
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !14, file: !1, line: 5, baseType: !17, size: 32, align: 32)
!17 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!18 = !{!19}
!19 = !DISubrange(count: 3)
!20 = distinct !DIGlobalVariable(name: "incomplete_struct_arry", linkageName: "\01?incomplete_struct_arry@@3PAUincomplete_struct@@A", scope: !0, file: !1, line: 6, type: !13, isLocal: false, isDefinition: true, variable: [3 x %struct.incomplete_struct]* @"\01?incomplete_struct_arry@@3PAUincomplete_struct@@A")
!21 = !{i32 2, !"CodeView", i32 1}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{!"clang version 3.9.0 (trunk 273084)"}
!24 = distinct !DISubprogram(name: "foo", linkageName: "\01?foo@@YAXH@Z", scope: !1, file: !1, line: 8, type: !25, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !17}
!27 = !DILocalVariable(name: "x", arg: 1, scope: !24, file: !1, line: 8, type: !17)
!28 = !DIExpression()
!29 = !DILocation(line: 8, column: 14, scope: !24)
!30 = !DILocation(line: 9, column: 20, scope: !24)
!31 = !DILocation(line: 9, column: 3, scope: !24)
!32 = !DILocalVariable(name: "dyn_size_arr", scope: !24, file: !1, line: 9, type: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, align: 32, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: -1)
!36 = !DIExpression(DW_OP_deref)
!37 = !DILocation(line: 9, column: 7, scope: !24)
!38 = !DILocation(line: 10, column: 3, scope: !24)
!39 = !DILocation(line: 10, column: 19, scope: !24)
!40 = !DILocation(line: 11, column: 1, scope: !24)
