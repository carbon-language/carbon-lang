; RUN: llc -O0 -mtriple=x86_64-apple-darwin -filetype=asm %s -o - | FileCheck %s
; ModuleID = 'aggregate-indirect-arg.cpp'
; extracted from debuginfo-tests/aggregate-indirect-arg.cpp

; v should not be a pointer.
; CHECK: ##DEBUG_VALUE: foo:v <- RSI
; rdar://problem/13658587

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%class.SVal = type { i8*, i32 }
%class.A = type { i8 }

declare void @_Z3barR4SVal(%class.SVal* %v)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1
declare i32 @main()
; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1A3fooE4SVal(%class.A* %this, %class.SVal* %v) nounwind ssp uwtable align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !59, metadata !{metadata !"0x102"}), !dbg !61
  call void @llvm.dbg.declare(metadata !{%class.SVal* %v}, metadata !62, metadata !{metadata !"0x102"}), !dbg !61
  %this1 = load %class.A** %this.addr
  call void @_Z3barR4SVal(%class.SVal* %v), !dbg !61
  ret void, !dbg !61
}
declare void @_ZN4SValD1Ev(%class.SVal* %this)
declare void @_ZN4SValD2Ev(%class.SVal* %this)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!47, !68}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [aggregate-indirect-arg.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"aggregate-indirect-arg.cpp", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !29, metadata !33, metadata !34, metadata !35}
!4 = metadata !{metadata !"0x2e\00bar\00bar\00_Z3barR4SVal\0019\000\001\000\006\00256\000\0019", metadata !1, metadata !5, metadata !6, null, void (%class.SVal*)* @_Z3barR4SVal, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 19] [def] [bar]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [aggregate-indirect-arg.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !9} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from SVal]
!9 = metadata !{metadata !"0x2\00SVal\0012\00128\0064\000\000\000", metadata !1, null, null, metadata !10, null, null, null} ; [ DW_TAG_class_type ] [SVal] [line 12, size 128, align 64, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !14, metadata !16, metadata !21, metadata !23}
!11 = metadata !{metadata !"0xd\00Data\0015\0064\0064\000\000", metadata !1, metadata !9, metadata !12} ; [ DW_TAG_member ] [Data] [line 15, size 64, align 64, offset 0] [from ]
!12 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, null} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{metadata !"0xd\00Kind\0016\0032\0032\0064\000", metadata !1, metadata !9, metadata !15} ; [ DW_TAG_member ] [Kind] [line 16, size 32, align 32, offset 64] [from unsigned int]
!15 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!16 = metadata !{metadata !"0x2e\00~SVal\00~SVal\00\0014\000\000\000\006\00256\000\0014", metadata !1, metadata !9, metadata !17, null, null, null, i32 0, metadata !20} ; [ DW_TAG_subprogram ] [line 14] [~SVal]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from SVal]
!20 = metadata !{i32 786468}
!21 = metadata !{metadata !"0x2e\00SVal\00SVal\00\0012\000\000\000\006\00320\000\0012", metadata !1, metadata !9, metadata !17, null, null, null, i32 0, metadata !22} ; [ DW_TAG_subprogram ] [line 12] [SVal]
!22 = metadata !{i32 786468}
!23 = metadata !{metadata !"0x2e\00SVal\00SVal\00\0012\000\000\000\006\00320\000\0012", metadata !1, metadata !9, metadata !24, null, null, null, i32 0, metadata !28} ; [ DW_TAG_subprogram ] [line 12] [SVal]
!24 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !25, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{null, metadata !19, metadata !26}
!26 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !27} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !9} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from SVal]
!28 = metadata !{i32 786468}
!29 = metadata !{metadata !"0x2e\00main\00main\00\0025\000\001\000\006\00256\000\0025", metadata !1, metadata !5, metadata !30, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 25] [def] [main]
!30 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !31, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = metadata !{metadata !32}
!32 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!33 = metadata !{metadata !"0x2e\00~SVal\00~SVal\00_ZN4SValD1Ev\0014\000\001\000\006\00256\000\0014", metadata !1, null, metadata !17, null, void (%class.SVal*)* @_ZN4SValD1Ev, null, metadata !16, metadata !2} ; [ DW_TAG_subprogram ] [line 14] [def] [~SVal]
!34 = metadata !{metadata !"0x2e\00~SVal\00~SVal\00_ZN4SValD2Ev\0014\000\001\000\006\00256\000\0014", metadata !1, null, metadata !17, null, void (%class.SVal*)* @_ZN4SValD2Ev, null, metadata !16, metadata !2} ; [ DW_TAG_subprogram ] [line 14] [def] [~SVal]
!35 = metadata !{metadata !"0x2e\00foo\00foo\00_ZN1A3fooE4SVal\0022\000\001\000\006\00256\000\0022", metadata !1, null, metadata !36, null, void (%class.A*, %class.SVal*)* @_ZN1A3fooE4SVal, null, metadata !41, metadata !2} ; [ DW_TAG_subprogram ] [line 22] [def] [foo]
!36 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !37, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!37 = metadata !{null, metadata !38, metadata !9}
!38 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !39} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!39 = metadata !{metadata !"0x2\00A\0020\008\008\000\000\000", metadata !1, null, null, metadata !40, null, null, null} ; [ DW_TAG_class_type ] [A] [line 20, size 8, align 8, offset 0] [def] [from ]
!40 = metadata !{metadata !41, metadata !43}
!41 = metadata !{metadata !"0x2e\00foo\00foo\00_ZN1A3fooE4SVal\0022\000\000\000\006\00256\000\0022", metadata !1, metadata !39, metadata !36, null, null, null, i32 0, metadata !42} ; [ DW_TAG_subprogram ] [line 22] [foo]
!42 = metadata !{i32 786468}
!43 = metadata !{metadata !"0x2e\00A\00A\00\0020\000\000\000\006\00320\000\0020", metadata !1, metadata !39, metadata !44, null, null, null, i32 0, metadata !46} ; [ DW_TAG_subprogram ] [line 20] [A]
!44 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !45, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{null, metadata !38}
!46 = metadata !{i32 786468}
!47 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!48 = metadata !{metadata !"0x101\00v\0016777235\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [v] [line 19]
!49 = metadata !{i32 19, i32 0, metadata !4, null}
!50 = metadata !{metadata !"0x100\00v\0026\000", metadata !29, metadata !5, metadata !9} ; [ DW_TAG_auto_variable ] [v] [line 26]
!51 = metadata !{i32 26, i32 0, metadata !29, null}
!52 = metadata !{i32 27, i32 0, metadata !29, null}
!53 = metadata !{i32 28, i32 0, metadata !29, null}
!54 = metadata !{metadata !"0x100\00a\0029\000", metadata !29, metadata !5, metadata !39} ; [ DW_TAG_auto_variable ] [a] [line 29]
!55 = metadata !{i32 29, i32 0, metadata !29, null}
!56 = metadata !{i32 30, i32 0, metadata !29, null}
!57 = metadata !{i32 31, i32 0, metadata !29, null}
!58 = metadata !{i32 32, i32 0, metadata !29, null}
!59 = metadata !{metadata !"0x101\00this\0016777238\001088", metadata !35, metadata !5, metadata !60} ; [ DW_TAG_arg_variable ] [this] [line 22]
!60 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !39} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!61 = metadata !{i32 22, i32 0, metadata !35, null}
!62 = metadata !{metadata !"0x101\00v\0033554454\008192", metadata !35, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [v] [line 22]
!63 = metadata !{metadata !"0x101\00this\0016777230\001088", metadata !33, metadata !5, metadata !64} ; [ DW_TAG_arg_variable ] [this] [line 14]
!64 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from SVal]
!65 = metadata !{i32 14, i32 0, metadata !33, null}
!66 = metadata !{metadata !"0x101\00this\0016777230\001088", metadata !34, metadata !5, metadata !64} ; [ DW_TAG_arg_variable ] [this] [line 14]
!67 = metadata !{i32 14, i32 0, metadata !34, null}
!68 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
