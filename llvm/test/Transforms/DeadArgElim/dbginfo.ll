; RUN: opt -deadargelim -S < %s | FileCheck %s
; PR14016

; Check that debug info metadata for subprograms stores pointers to
; updated LLVM functions.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = global i32 0, align 4

define void @_Z3runv() uwtable {
entry:
  call void @_ZN12_GLOBAL__N_18dead_argEPv(i8* null), !dbg !10
  call void (...)* @_ZN12_GLOBAL__N_111dead_varargEz(), !dbg !12
  ret void, !dbg !13
}

; Argument will be deleted
define internal void @_ZN12_GLOBAL__N_18dead_argEPv(i8* %foo) nounwind uwtable {
entry:
  %0 = load i32* @x, align 4, !dbg !14
  %inc = add nsw i32 %0, 1, !dbg !14
  store i32 %inc, i32* @x, align 4, !dbg !14
  ret void, !dbg !16
}

; Vararg will be deleted
define internal void @_ZN12_GLOBAL__N_111dead_varargEz(...) nounwind uwtable {
entry:
  %0 = load i32* @x, align 4, !dbg !17
  %inc = add nsw i32 %0, 1, !dbg !17
  store i32 %inc, i32* @x, align 4, !dbg !17
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = metadata !{metadata !"0x11\004\00clang version 3.2 (trunk 165305)\000\00\000\00\000", metadata !20, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ] [/home/samsonov/tmp/clang-di/test.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !8, metadata !9}
!5 = metadata !{metadata !"0x2e\00run\00run\00\008\000\001\000\006\00256\000\008", metadata !20, metadata !6, metadata !7, null, void ()* @_Z3runv, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 8] [def] [run]
!6 = metadata !{metadata !"0x29", metadata !20} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !1, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !"0x2e\00dead_vararg\00dead_vararg\00\005\001\001\000\006\00256\000\005", metadata !20, metadata !6, metadata !7, null, void (...)* @_ZN12_GLOBAL__N_111dead_varargEz, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 5] [local] [def] [dead_vararg]

; CHECK: metadata !"0x2e\00dead_vararg\00{{[^"]+}}"{{.*}}void ()* @_ZN12_GLOBAL__N_111dead_varargEz

!9 = metadata !{metadata !"0x2e\00dead_arg\00dead_arg\00\004\001\001\000\006\00256\000\004", metadata !20, metadata !6, metadata !7, null, void (i8*)* @_ZN12_GLOBAL__N_18dead_argEPv, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 4] [local] [def] [dead_arg]

; CHECK: metadata !"0x2e\00dead_arg\00{{[^"]+}}"{{.*}}void ()* @_ZN12_GLOBAL__N_18dead_argEPv

!10 = metadata !{i32 8, i32 14, metadata !11, null}
!11 = metadata !{metadata !"0xb\008\0012\000", metadata !20, metadata !5} ; [ DW_TAG_lexical_block ] [/home/samsonov/tmp/clang-di/test.cc]
!12 = metadata !{i32 8, i32 27, metadata !11, null}
!13 = metadata !{i32 8, i32 42, metadata !11, null}
!14 = metadata !{i32 4, i32 28, metadata !15, null}
!15 = metadata !{metadata !"0xb\004\0026\002", metadata !20, metadata !9} ; [ DW_TAG_lexical_block ] [/home/samsonov/tmp/clang-di/test.cc]
!16 = metadata !{i32 4, i32 33, metadata !15, null}
!17 = metadata !{i32 5, i32 25, metadata !18, null}
!18 = metadata !{metadata !"0xb\005\0023\001", metadata !20, metadata !8} ; [ DW_TAG_lexical_block ] [/home/samsonov/tmp/clang-di/test.cc]
!19 = metadata !{i32 5, i32 30, metadata !18, null}
!20 = metadata !{metadata !"test.cc", metadata !"/home/samsonov/tmp/clang-di"}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
