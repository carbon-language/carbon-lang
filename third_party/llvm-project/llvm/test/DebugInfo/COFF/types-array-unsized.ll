; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s
; RUN: llc < %s | llvm-mc -filetype=obj --triple=x86_64-windows | llvm-readobj --codeview - | FileCheck %s

; We should emit two array types: one used to describe the static data member,
; and the other used by the S_GDATA32 for the definition.

; // Build with: clang-cl a.cpp /c /Z7 /clang:-S /clang:-emit-llvm
; struct Foo {
;   static const char str[];
; };
; const char Foo::str[] = "asdf";
; Foo f; // FIXME: only needed to force emit 'Foo'

; CHECK:      CodeViewTypes [
; CHECK:        Array ([[ARRAY_COMPLETE:0x.*]]) {
; CHECK-NEXT:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK-NEXT:     ElementType: const char ({{.*}})
; CHECK-NEXT:     IndexType: unsigned __int64 (0x23)
; CHECK-NEXT:     SizeOf: 5
; CHECK-NEXT:     Name:
; CHECK-NEXT:   }
; CHECK:      ]
; CHECK:        Array ([[ARRAY_FWD:0x.*]]) {
; CHECK-NEXT:     TypeLeafKind: LF_ARRAY (0x1503)
; CHECK-NEXT:     ElementType: const char ({{.*}})
; CHECK-NEXT:     IndexType: unsigned __int64 (0x23)
; CHECK-NEXT:     SizeOf: 0
; CHECK-NEXT:     Name:
; CHECK-NEXT:   }
; CHECK:        FieldList (0x1004) {
; CHECK-NEXT:     TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:     StaticDataMember {
; CHECK-NEXT:       TypeLeafKind: LF_STMEMBER (0x150E)
; CHECK-NEXT:       AccessSpecifier: Public (0x3)
; CHECK-NEXT:       Type: [[ARRAY_FWD]]
; CHECK-NEXT:       Name: str
; CHECK-NEXT:     }
; CHECK-NEXT:   }

; CHECK:          GlobalData {
; CHECK-NEXT:       Kind: S_GDATA32 (0x110D)
; CHECK-NEXT:       DataOffset: ?str@Foo@@2QBDB+0x0
; CHECK-NEXT:       Type: [[ARRAY_COMPLETE]]
; CHECK-NEXT:       DisplayName: Foo::str
; CHECK-NEXT:       LinkageName: ?str@Foo@@2QBDB
; CHECK-NEXT:     }

; ModuleID = 'a.cpp'
source_filename = "a.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.25.28614"

%struct.Foo = type { i8 }

@"?str@Foo@@2QBDB" = dso_local constant [5 x i8] c"asdf\00", align 1, !dbg !0
@"?f@@3UFoo@@A" = dso_local global %struct.Foo zeroinitializer, align 1, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.linker.options = !{!19, !20}
!llvm.module.flags = !{!21, !22, !23, !24}
!llvm.ident = !{!25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "str", linkageName: "?str@Foo@@2QBDB", scope: !2, file: !3, line: 4, type: !16, isLocal: false, isDefinition: true, declaration: !10)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 202f144bffd0be254a829924195e1b8ebabcbb79)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "a.cpp", directory: "F:\\llvm-project\\__test", checksumkind: CSK_MD5, checksum: "c2b2d28fd8aaa040c4141cea420d0648")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "f", linkageName: "?f@@3UFoo@@A", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !9, identifier: ".?AUFoo@@")
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "str", scope: !8, file: !3, line: 2, baseType: !11, flags: DIFlagStaticMember)
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, elements: !14)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !{!15}
!15 = !DISubrange(count: -1)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 40, elements: !17)
!17 = !{!18}
!18 = !DISubrange(count: 5)
!19 = !{!"/DEFAULTLIB:libcmt.lib"}
!20 = !{!"/DEFAULTLIB:oldnames.lib"}
!21 = !{i32 2, !"CodeView", i32 1}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 2}
!24 = !{i32 7, !"PIC Level", i32 2}
!25 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 202f144bffd0be254a829924195e1b8ebabcbb79)"}
