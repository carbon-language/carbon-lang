; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_formal_parameter [
; CHECK-NOT: ""
; CHECK: DW_TAG
; CHECK: DW_TAG_class_type
; CHECK: DW_AT_object_pointer [DW_FORM_ref4]     (cu + 0x{{[0-9a-f]*}} => {[[PARAM:0x[0-9a-f]*]]})
; CHECK: [[PARAM]]:     DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x{{[0-9a-f]*}}] = "this")

%class.A = type { i32 }

define i32 @_Z3fooi(i32) nounwind uwtable ssp {
entry:
  %.addr = alloca i32, align 4
  %a = alloca %class.A, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %.addr}, metadata !36, metadata !{metadata !"0x102"}), !dbg !35
  call void @llvm.dbg.declare(metadata !{%class.A* %a}, metadata !21, metadata !{metadata !"0x102"}), !dbg !23
  call void @_ZN1AC1Ev(%class.A* %a), !dbg !24
  %m_a = getelementptr inbounds %class.A* %a, i32 0, i32 0, !dbg !25
  %1 = load i32* %m_a, align 4, !dbg !25
  ret i32 %1, !dbg !25
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN1AC1Ev(%class.A* %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !26, metadata !{metadata !"0x102"}), !dbg !28
  %this1 = load %class.A** %this.addr
  call void @_ZN1AC2Ev(%class.A* %this1), !dbg !29
  ret void, !dbg !29
}

define linkonce_odr void @_ZN1AC2Ev(%class.A* %this) unnamed_addr nounwind uwtable ssp align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !30, metadata !{metadata !"0x102"}), !dbg !31
  %this1 = load %class.A** %this.addr
  %m_a = getelementptr inbounds %class.A* %this1, i32 0, i32 0, !dbg !32
  store i32 0, i32* %m_a, align 4, !dbg !32
  ret void, !dbg !34
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38}

!0 = metadata !{metadata !"0x11\004\00clang version 3.2 (trunk 163586) (llvm/trunk 163570)\000\00\000\00\000", metadata !37, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ] [/Users/echristo/debug-tests/bar.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{}
!3 = metadata !{metadata !5, metadata !10, metadata !20}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3fooi\007\000\001\000\006\00256\000\007", metadata !6, metadata !6, metadata !7, null, i32 (i32)* @_Z3fooi, null, null, metadata !1} ; [ DW_TAG_subprogram ] [line 7] [def] [foo]
!6 = metadata !{metadata !"0x29", metadata !37} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !"0x2e\00A\00A\00_ZN1AC1Ev\003\000\001\000\006\00256\000\003", metadata !6, null, metadata !11, null, void (%class.A*)* @_ZN1AC1Ev, null, metadata !17, metadata !1} ; [ DW_TAG_subprogram ] [line 3] [def] [A]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!14 = metadata !{metadata !"0x2\00A\001\0032\0032\000\000\000", metadata !37, null, null, metadata !15, null, null, null} ; [ DW_TAG_class_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!15 = metadata !{metadata !16, metadata !17}
!16 = metadata !{metadata !"0xd\00m_a\004\0032\0032\000\000", metadata !37, metadata !14, metadata !9} ; [ DW_TAG_member ] [m_a] [line 4, size 32, align 32, offset 0] [from int]
!17 = metadata !{metadata !"0x2e\00A\00A\00\003\000\000\000\006\00256\000\003", metadata !6, metadata !14, metadata !11, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ] [line 3] [A]
!18 = metadata !{metadata !19}
!19 = metadata !{metadata !"0x24"}                      ; [ DW_TAG_base_type ] [line 0, size 0, align 0, offset 0]
!20 = metadata !{metadata !"0x2e\00A\00A\00_ZN1AC2Ev\003\000\001\000\006\00256\000\003", metadata !6, null, metadata !11, null, void (%class.A*)* @_ZN1AC2Ev, null, metadata !17, metadata !1} ; [ DW_TAG_subprogram ] [line 3] [def] [A]
!21 = metadata !{metadata !"0x100\00a\008\000", metadata !22, metadata !6, metadata !14} ; [ DW_TAG_auto_variable ] [a] [line 8]
!22 = metadata !{metadata !"0xb\007\0011\000", metadata !6, metadata !5} ; [ DW_TAG_lexical_block ] [/Users/echristo/debug-tests/bar.cpp]
!23 = metadata !{i32 8, i32 5, metadata !22, null}
!24 = metadata !{i32 8, i32 6, metadata !22, null}
!25 = metadata !{i32 9, i32 3, metadata !22, null}
!26 = metadata !{metadata !"0x101\00this\0016777219\001088", metadata !10, metadata !6, metadata !27} ; [ DW_TAG_arg_variable ] [this] [line 3]
!27 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!28 = metadata !{i32 3, i32 3, metadata !10, null}
!29 = metadata !{i32 3, i32 18, metadata !10, null}
!30 = metadata !{metadata !"0x101\00this\0016777219\001088", metadata !20, metadata !6, metadata !27} ; [ DW_TAG_arg_variable ] [this] [line 3]
!31 = metadata !{i32 3, i32 3, metadata !20, null}
!32 = metadata !{i32 3, i32 9, metadata !33, null}
!33 = metadata !{metadata !"0xb\003\007\001", metadata !6, metadata !20} ; [ DW_TAG_lexical_block ] [/Users/echristo/debug-tests/bar.cpp]
!34 = metadata !{i32 3, i32 18, metadata !33, null}
!35 = metadata !{i32 7, i32 0, metadata !5, null}
!36 = metadata !{metadata !"0x101\00\0016777223\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ] [line 7]
!37 = metadata !{metadata !"bar.cpp", metadata !"/Users/echristo/debug-tests"}
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
