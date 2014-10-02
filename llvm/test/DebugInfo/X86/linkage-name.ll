; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_subprogram [9] *
; CHECK-NOT: DW_AT_MIPS_linkage_name
; CHECK: DW_AT_specification

%class.A = type { i8 }

@a = global %class.A zeroinitializer, align 1

define i32 @_ZN1A1aEi(%class.A* %this, i32 %b) nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  %b.addr = alloca i32, align 4
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !21, metadata !{metadata !"0x102"}), !dbg !23
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %b.addr}, metadata !24, metadata !{metadata !"0x102"}), !dbg !25
  %this1 = load %class.A** %this.addr
  %0 = load i32* %b.addr, align 4, !dbg !26
  ret i32 %0, !dbg !26
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = metadata !{metadata !"0x11\004\00clang version 3.1 (trunk 152691) (llvm/trunk 152692)\000\00\000\00\000", metadata !28, metadata !1, metadata !1, metadata !3, metadata !18,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00a\00a\00_ZN1A1aEi\005\000\001\000\006\00256\000\005", metadata !6, null, metadata !7, null, i32 (%class.A*, i32)* @_ZN1A1aEi, null, metadata !13, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !28} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10, metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\0064", i32 0, null, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0x2\00A\001\008\008\000\000\000", metadata !28, null, null, metadata !12, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 8, align 8, offset 0] [def] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x2e\00a\00a\00_ZN1A1aEi\002\000\000\000\006\00257\000\000", metadata !6, metadata !11, metadata !7, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ]
!18 = metadata !{metadata !20}
!20 = metadata !{metadata !"0x34\00a\00a\00\009\000\001", null, metadata !6, metadata !11, %class.A* @a, null} ; [ DW_TAG_variable ]
!21 = metadata !{metadata !"0x101\00this\0016777221\0064", metadata !5, metadata !6, metadata !22} ; [ DW_TAG_arg_variable ]
!22 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ]
!23 = metadata !{i32 5, i32 8, metadata !5, null}
!24 = metadata !{metadata !"0x101\00b\0033554437\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!25 = metadata !{i32 5, i32 14, metadata !5, null}
!26 = metadata !{i32 6, i32 4, metadata !27, null}
!27 = metadata !{metadata !"0xb\005\0017\000", metadata !6, metadata !5} ; [ DW_TAG_lexical_block ]
!28 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo"}
!29 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
