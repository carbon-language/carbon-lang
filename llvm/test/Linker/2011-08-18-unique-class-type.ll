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
  call void @llvm.dbg.declare(metadata !{%"class.N1::A"* %mya}, metadata !9, metadata !{metadata !"0x102"}), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = metadata !{metadata !"0x11\004\00clang version 3.0 (trunk 137954)\001\00\000\00\000", metadata !16, metadata !2, metadata !2, metadata !3, metadata !2, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3fooN2N11AE\004\000\001\000\006\00256\000\000", metadata !16, metadata !6, metadata !7, null, void ()* @_Z3fooN2N11AE, null, null, null} ; [ DW_TAG_subprogram ] [line 4] [def] [scope 0] [foo]
!6 = metadata !{metadata !"0x29", metadata !16} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !16, metadata !6, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{metadata !"0x101\00mya\0016777220\000", metadata !5, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!10 = metadata !{metadata !"0x2\00A\003\008\008\000\000\000", metadata !17, metadata !11, null, metadata !2, null, null, null} ; [ DW_TAG_class_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!11 = metadata !{metadata !"0x39\00N1\002", metadata !17, null} ; [ DW_TAG_namespace ]
!12 = metadata !{metadata !"0x29", metadata !17} ; [ DW_TAG_file_type ]
!13 = metadata !{i32 4, i32 12, metadata !5, null}
!14 = metadata !{i32 4, i32 18, metadata !15, null}
!15 = metadata !{metadata !"0xb\004\0017\000", metadata !16, metadata !5} ; [ DW_TAG_lexical_block ]
!16 = metadata !{metadata !"n1.c", metadata !"/private/tmp"}
!17 = metadata !{metadata !"./n.h", metadata !"/private/tmp"}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
