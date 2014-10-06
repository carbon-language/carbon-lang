; RUN: llc -mtriple=x86_64-apple-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; struct A {
;   A(int i);
;   ~A();
; };
;
; A::~A() {}
;
; void foo() {
;   A a(1);
; }
;
; rdar://problem/16362674
;
; Test that we do not emit a linkage name for the declaration of a destructor.
; Test that we do emit a linkage name for a specific instance of it.

; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}} "~A"
; CHECK-NOT: DW_AT_MIPS_linkage_name
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name {{.*}} "_ZN1AD2Ev"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_specification {{.*}} "~A"


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.A = type { i8 }

; Function Attrs: nounwind ssp uwtable
define void @_ZN1AD2Ev(%struct.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.A** %this.addr}, metadata !26, metadata !{metadata !"0x102"}), !dbg !28
  %this1 = load %struct.A** %this.addr
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_ZN1AD1Ev(%struct.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.A*, align 8
  store %struct.A* %this, %struct.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.A** %this.addr}, metadata !30, metadata !{metadata !"0x102"}), !dbg !31
  %this1 = load %struct.A** %this.addr
  call void @_ZN1AD2Ev(%struct.A* %this1), !dbg !32
  ret void, !dbg !33
}

; Function Attrs: ssp uwtable
define void @_Z3foov() #2 {
entry:
  %a = alloca %struct.A, align 1
  call void @llvm.dbg.declare(metadata !{%struct.A* %a}, metadata !34, metadata !{metadata !"0x102"}), !dbg !35
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

!0 = metadata !{metadata !"0x11\004\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !16, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [linkage-name.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"linkage-name.cpp", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00A\001\008\008\000\000\000", metadata !1, null, null, metadata !5, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !12}
!6 = metadata !{metadata !"0x2e\00A\00A\00\002\000\000\000\006\00256\000\002", metadata !1, metadata !"_ZTS1A", metadata !7, null, null, null, i32 0, metadata !11} ; [ DW_TAG_subprogram ] [line 2] [A]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !10}
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!10 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!11 = metadata !{i32 786468}
!12 = metadata !{metadata !"0x2e\00~A\00~A\00\003\000\000\000\006\00256\000\003", metadata !1, metadata !"_ZTS1A", metadata !13, null, null, null, i32 0, metadata !15} ; [ DW_TAG_subprogram ] [line 3] [~A]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{null, metadata !9}
!15 = metadata !{i32 786468}
!16 = metadata !{metadata !17, metadata !18, metadata !19}
!17 = metadata !{metadata !"0x2e\00~A\00~A\00_ZN1AD2Ev\006\000\001\000\006\00256\000\006", metadata !1, metadata !"_ZTS1A", metadata !13, null, void (%struct.A*)* @_ZN1AD2Ev, null, metadata !12, metadata !2} ; [ DW_TAG_subprogram ] [line 6] [def] [~A]
!18 = metadata !{metadata !"0x2e\00~A\00~A\00_ZN1AD1Ev\006\000\001\000\006\00256\000\006", metadata !1, metadata !"_ZTS1A", metadata !13, null, void (%struct.A*)* @_ZN1AD1Ev, null, metadata !12, metadata !2} ; [ DW_TAG_subprogram ] [line 6] [def] [~A]
!19 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3foov\0010\000\001\000\006\00256\000\0010", metadata !1, metadata !20, metadata !21, null, void ()* @_Z3foov, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!20 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [linkage-name.cpp]
!21 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !22, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!22 = metadata !{null}
!23 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!25 = metadata !{metadata !"clang version 3.5.0 "}
!26 = metadata !{metadata !"0x101\00this\0016777216\001088", metadata !17, null, metadata !27} ; [ DW_TAG_arg_variable ] [this] [line 0]
!27 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!28 = metadata !{i32 0, i32 0, metadata !17, null}
!29 = metadata !{i32 8, i32 0, metadata !17, null}
!30 = metadata !{metadata !"0x101\00this\0016777216\001088", metadata !18, null, metadata !27} ; [ DW_TAG_arg_variable ] [this] [line 0]
!31 = metadata !{i32 0, i32 0, metadata !18, null}
!32 = metadata !{i32 6, i32 0, metadata !18, null}
!33 = metadata !{i32 8, i32 0, metadata !18, null}
!34 = metadata !{metadata !"0x100\00a\0011\000", metadata !19, metadata !20, metadata !"_ZTS1A"} ; [ DW_TAG_auto_variable ] [a] [line 11]
!35 = metadata !{i32 11, i32 0, metadata !19, null}
!36 = metadata !{i32 12, i32 0, metadata !19, null}
