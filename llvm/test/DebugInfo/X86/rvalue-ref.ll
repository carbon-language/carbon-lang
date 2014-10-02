; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj -O0
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_rvalue_reference_type

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

define void @_Z3fooOi(i32* %i) uwtable ssp {
entry:
  %i.addr = alloca i32*, align 8
  store i32* %i, i32** %i.addr, align 8
  call void @llvm.dbg.declare(metadata !{i32** %i.addr}, metadata !11, metadata !{metadata !"0x102"}), !dbg !12
  %0 = load i32** %i.addr, align 8, !dbg !13
  %1 = load i32* %0, align 4, !dbg !13
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32 %1), !dbg !13
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17}

!0 = metadata !{metadata !"0x11\004\00clang version 3.2 (trunk 157054) (llvm/trunk 157060)\000\00\000\00\000", metadata !16, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3fooOi\004\000\001\000\006\00256\000\005", metadata !16, metadata !6, metadata !7, null, void (i32*)* @_Z3fooOi, null, null, metadata !1} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !16} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{metadata !"0x42\00\000\000\000\000\000", null, null, metadata !10} ; [ DW_TAG_rvalue_reference_type ]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!11 = metadata !{metadata !"0x101\00i\0016777220\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 4, i32 17, metadata !5, null}
!13 = metadata !{i32 6, i32 3, metadata !14, null}
!14 = metadata !{metadata !"0xb\005\001\000", metadata !16, metadata !5} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 7, i32 1, metadata !14, null}
!16 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo/tmp"}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
