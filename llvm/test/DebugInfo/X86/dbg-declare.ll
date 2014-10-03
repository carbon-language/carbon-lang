; RUN: llc < %s -O0 -mtriple x86_64-apple-darwin
; <rdar://problem/11134152>

define i32 @foo(i32* %x) nounwind uwtable ssp {
entry:
  %x.addr = alloca i32*, align 8
  %saved_stack = alloca i8*
  %cleanup.dest.slot = alloca i32
  store i32* %x, i32** %x.addr, align 8
  call void @llvm.dbg.declare(metadata !{i32** %x.addr}, metadata !14, metadata !{metadata !"0x102"}), !dbg !15
  %0 = load i32** %x.addr, align 8, !dbg !16
  %1 = load i32* %0, align 4, !dbg !16
  %2 = zext i32 %1 to i64, !dbg !16
  %3 = call i8* @llvm.stacksave(), !dbg !16
  store i8* %3, i8** %saved_stack, !dbg !16
  %vla = alloca i8, i64 %2, align 16, !dbg !16
  call void @llvm.dbg.declare(metadata !{i8* %vla}, metadata !18, metadata !{metadata !"0x102"}), !dbg !23
  store i32 1, i32* %cleanup.dest.slot
  %4 = load i8** %saved_stack, !dbg !24
  call void @llvm.stackrestore(i8* %4), !dbg !24
  ret i32 0, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1 (trunk 153698)\000\00\000\00\000", metadata !26, metadata !1, metadata !1, metadata !3, metadata !1, null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00\006\000\001\000\006\00256\000\000", metadata !26, metadata !0, metadata !7, null, i32 (i32*)* @foo, null, null, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !26} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !9} ; [ DW_TAG_const_type ]
!14 = metadata !{metadata !"0x101\00x\0016777221\000", metadata !5, metadata !6, metadata !10} ; [ DW_TAG_arg_variable ]
!15 = metadata !{i32 5, i32 21, metadata !5, null}
!16 = metadata !{i32 7, i32 13, metadata !17, null}
!17 = metadata !{metadata !"0xb\006\001\000", metadata !26, metadata !5} ; [ DW_TAG_lexical_block ]
!18 = metadata !{metadata !"0x100\00a\007\000", metadata !17, metadata !6, metadata !19} ; [ DW_TAG_auto_variable ]
!19 = metadata !{metadata !"0x1\00\000\000\008\000\000", null, null, metadata !20, metadata !21, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 8, offset 0] [from char]
!20 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ]
!21 = metadata !{metadata !22}
!22 = metadata !{metadata !"0x21\000\00-1"}        ; [ DW_TAG_subrange_type ]
!23 = metadata !{i32 7, i32 8, metadata !17, null}
!24 = metadata !{i32 9, i32 1, metadata !17, null}
!25 = metadata !{i32 8, i32 3, metadata !17, null}
!26 = metadata !{metadata !"20020104-2.c", metadata !"/Volumes/Sandbox/llvm"}
!27 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
