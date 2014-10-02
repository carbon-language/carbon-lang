; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; Checks that we don't emit a size for a pointer type.
; CHECK: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type
; CHECK-NOT: DW_AT_byte_size
; CHECK: DW_TAG
; CHECK: .debug_info contents

%struct.A = type { i32 }

define i32 @_Z3fooP1A(%struct.A* %a) nounwind uwtable ssp {
entry:
  %a.addr = alloca %struct.A*, align 8
  store %struct.A* %a, %struct.A** %a.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.A** %a.addr}, metadata !16, metadata !{metadata !"0x102"}), !dbg !17
  %0 = load %struct.A** %a.addr, align 8, !dbg !18
  %b = getelementptr inbounds %struct.A* %0, i32 0, i32 0, !dbg !18
  %1 = load i32* %b, align 4, !dbg !18
  ret i32 %1, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{metadata !"0x11\004\00clang version 3.1 (trunk 150996)\000\00\000\00\000", metadata !20, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3fooP1A\003\000\001\000\006\00256\000\003", metadata !20, metadata !6, metadata !7, null, i32 (%struct.A*)* @_Z3fooP1A, null, null, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0x2\00A\001\0032\0032\000\000\000", metadata !20, null, null, metadata !12, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0xd\00b\001\0032\0032\000\000", metadata !20, metadata !11, metadata !9} ; [ DW_TAG_member ]
!16 = metadata !{metadata !"0x101\00a\0016777219\000", metadata !5, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!17 = metadata !{i32 3, i32 13, metadata !5, null}
!18 = metadata !{i32 4, i32 3, metadata !19, null}
!19 = metadata !{metadata !"0xb\003\0016\000", metadata !20, metadata !5} ; [ DW_TAG_lexical_block ]
!20 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
