; RUN: llc -mtriple=x86_64-apple-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; rdar://problem/16362674

; Test that we do not emit a linkage name for the declaration of a destructor.
; Test that we do emit a linkage name for a specific instance of it.

; CHECK: DW_TAG_subprogram
; CHECK: [[A:.*]]:     DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}} "~A"
; CHECK-NOT: DW_AT_MIPS_linkage_name
; CHECK: DW_TAG_subprogram
; CHECK-NEXT:  DW_AT_specification {{.*}}[[A]]
; CHECK-NEXT: DW_AT_MIPS_linkage_name {{.*}} "_ZN1AD2Ev"


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.A = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_ZN1AD2Ev(%struct.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.A** %this.addr}, metadata !26), !dbg !28
  %this1 = load %struct.A** %this.addr
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_ZN1AD1Ev(%struct.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.A** %this.addr}, metadata !30), !dbg !31
  %this1 = load %struct.A** %this.addr
  call void @_ZN1AD2Ev(%struct.A* %this1), !dbg !32
  ret void, !dbg !33
}

; Function Attrs: ssp uwtable
define void @_Z3foov() #2 {
entry:
  %a = alloca %struct.A, align 1
  call void @llvm.dbg.declare(metadata !{%struct.A* %a}, metadata !34), !dbg !35
  call void @_ZN1AC1Ei(%struct.A* %a, i32 1), !dbg !35
  call void @_ZN1AD1Ev(%struct.A* %a), !dbg !36
  ret void, !dbg !36
}

declare void @_ZN1AC1Ei(%struct.A*, i32)

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !16, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [linkage-name.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"linkage-name.cpp", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"A", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !12}
!6 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"", i32 2, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !11, i32 2} ; [ DW_TAG_subprogram ] [line 2] [A]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !10}
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 786468}
!12 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"~A", metadata !"~A", metadata !"", i32 3, metadata !13, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !15, i32 3} ; [ DW_TAG_subprogram ] [line 3] [~A]
!13 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !14, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{null, metadata !9}
!15 = metadata !{i32 786468}
!16 = metadata !{metadata !17, metadata !18, metadata !19}
!17 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"~A", metadata !"~A", metadata !"_ZN1AD2Ev", i32 6, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.A*)* @_ZN1AD2Ev, null, metadata !12, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [~A]
!18 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"~A", metadata !"~A", metadata !"_ZN1AD1Ev", i32 6, metadata !13, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.A*)* @_ZN1AD1Ev, null, metadata !12, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [~A]
!19 = metadata !{i32 786478, metadata !1, metadata !20, metadata !"foo", metadata !"foo", metadata !"_Z3foov", i32 10, metadata !21, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z3foov, null, null, metadata !2, i32 10} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!20 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [linkage-name.cpp]
!21 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !22, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!22 = metadata !{null}
!23 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!25 = metadata !{metadata !"clang version 3.5.0 "}
!26 = metadata !{i32 786689, metadata !17, metadata !"this", null, i32 16777216, metadata !27, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!27 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!28 = metadata !{i32 0, i32 0, metadata !17, null}
!29 = metadata !{i32 8, i32 0, metadata !17, null} ; [ DW_TAG_imported_declaration ]
!30 = metadata !{i32 786689, metadata !18, metadata !"this", null, i32 16777216, metadata !27, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!31 = metadata !{i32 0, i32 0, metadata !18, null}
!32 = metadata !{i32 6, i32 0, metadata !18, null}
!33 = metadata !{i32 8, i32 0, metadata !18, null} ; [ DW_TAG_imported_declaration ]
!34 = metadata !{i32 786688, metadata !19, metadata !"a", metadata !20, i32 11, metadata !"_ZTS1A", i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 11]
!35 = metadata !{i32 11, i32 0, metadata !19, null}
!36 = metadata !{i32 12, i32 0, metadata !19, null}
