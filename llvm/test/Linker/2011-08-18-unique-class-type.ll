; RUN: llvm-link %s %p/2011-08-18-unique-class-type2.ll -S -o - | FileCheck %s
; CHECK: DW_TAG_class_type
; CHECK-NOT: DW_TAG_class_type
; Test to check there is only one MDNode for class A after linking.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%"class.N1::A" = type { i8 }

define void @_Z3fooN2N11AE() nounwind uwtable ssp {
entry:
  %mya = alloca %"class.N1::A", align 1
  call void @llvm.dbg.declare(metadata %"class.N1::A"* %mya, metadata !9, metadata !MDExpression()), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.0 (trunk 137954)", isOptimized: true, emissionKind: 0, file: !16, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2)
!1 = !{!2}
!2 = !{i32 0}
!3 = !{!5}
!5 = !MDSubprogram(name: "foo", linkageName: "_Z3fooN2N11AE", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, file: !16, scope: !6, type: !7, function: void ()* @_Z3fooN2N11AE)
!6 = !MDFile(filename: "n1.c", directory: "/private/tmp")
!7 = !MDSubroutineType(types: !8)
!8 = !{null}
!9 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "mya", line: 4, arg: 1, scope: !5, file: !6, type: !10)
!10 = !MDCompositeType(tag: DW_TAG_class_type, name: "A", line: 3, size: 8, align: 8, file: !17, scope: !11, elements: !2)
!11 = !MDNamespace(name: "N1", line: 2, file: !17, scope: null)
!12 = !MDFile(filename: "./n.h", directory: "/private/tmp")
!13 = !MDLocation(line: 4, column: 12, scope: !5)
!14 = !MDLocation(line: 4, column: 18, scope: !15)
!15 = distinct !MDLexicalBlock(line: 4, column: 17, file: !16, scope: !5)
!16 = !MDFile(filename: "n1.c", directory: "/private/tmp")
!17 = !MDFile(filename: "./n.h", directory: "/private/tmp")
!18 = !{i32 1, !"Debug Info Version", i32 3}
