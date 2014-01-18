; REQUIRES: object-emission
;
; RUN: llvm-link %s %p/type-unique-simple2-b.ll -S -o - | llc -filetype=obj -O0 | llvm-dwarfdump -debug-dump=info - | FileCheck %s
;
; Tests for a merge error where attributes are inserted twice into the same DIE.
;
; $ cat ab.h
; typedef int foo_t;
; class A {
; public:
;   virtual void setFoo();
;   virtual const foo_t getFoo();
; };
;
; $ cat a.cpp
; #include "ab.h"
; foo_t bar() {
;     return A().getFoo();
; }
;
; CHECK: _ZN1A6setFooEv
; CHECK: DW_AT_accessibility [DW_FORM_data1]   (0x01)
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG

; ModuleID = 'a.cpp'

%class.A = type { i32 (...)** }

@_ZTV1A = external unnamed_addr constant [4 x i8*]

; Function Attrs: nounwind
define i32 @_Z3barv() #0 {
entry:
  %tmp = alloca %class.A, align 8
  %0 = bitcast %class.A* %tmp to i8*, !dbg !38
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 8, i32 8, i1 false), !dbg !38
  call void @_ZN1AC1Ev(%class.A* %tmp) #1, !dbg !38
  %call = call i32 @_ZN1A6getFooEv(%class.A* %tmp), !dbg !38
  ret i32 %call, !dbg !38
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

; Function Attrs: inlinehint nounwind
define linkonce_odr void @_ZN1AC1Ev(%class.A* %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !39), !dbg !41
  %this1 = load %class.A** %this.addr
  call void @_ZN1AC2Ev(%class.A* %this1) #1, !dbg !42
  ret void, !dbg !42
}

declare i32 @_ZN1A6getFooEv(%class.A*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #4

; Function Attrs: inlinehint nounwind
define linkonce_odr void @_ZN1AC2Ev(%class.A* %this) unnamed_addr #2 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !44), !dbg !45
  %this1 = load %class.A** %this.addr
  %0 = bitcast %class.A* %this1 to i8***, !dbg !46
  store i8** getelementptr inbounds ([4 x i8*]* @_ZTV1A, i64 0, i64 2), i8*** %0, !dbg !46
  ret void, !dbg !46
}

attributes #0 = { nounwind }
attributes #1 = { nounwind }
attributes #2 = { inlinehint nounwind }
attributes #4 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36}
!llvm.ident = !{!37}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !26, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<unknown>", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786434, metadata !5, null, metadata !"A", i32 2, i64 64, i64 64, i32 0, i32 0, null, metadata !6, i32 0, metadata !"_ZTS1A", null, metadata !"_ZTS1A"} ; [ DW_TAG_class_type ] [A] [line 2, size 64, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !"./ab.h", metadata !""}
!6 = metadata !{metadata !7, metadata !14, metadata !19}
!7 = metadata !{i32 786445, metadata !5, metadata !8, metadata !"_vptr$A", i32 0, i64 64, i64 0, i64 0, i32 64, metadata !9} ; [ DW_TAG_member ] [_vptr$A] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!8 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/./ab.h]
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!10 = metadata !{i32 786447, null, null, metadata !"__vtbl_ptr_type", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"setFoo", metadata !"setFoo", metadata !"_ZN1A6setFooEv", i32 4, metadata !15, i1 false, i1 false, i32 1, i32 0, metadata !"_ZTS1A", i32 256, i1 false, null, null, i32 0, metadata !18, i32 4} ; [ DW_TAG_subprogram ] [line 4] [setFoo]
!15 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !17}
!17 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!18 = metadata !{i32 786468}
!19 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"getFoo", metadata !"getFoo", metadata !"_ZN1A6getFooEv", i32 5, metadata !20, i1 false, i1 false, i32 1, i32 1, metadata !"_ZTS1A", i32 256, i1 false, null, null, i32 0, metadata !25, i32 5} ; [ DW_TAG_subprogram ] [line 5] [getFoo]
!20 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !22, metadata !17}
!22 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !23} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo_t]
!23 = metadata !{i32 786454, metadata !24, null, metadata !"foo_t", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_typedef ] [foo_t] [line 1, size 0, align 0, offset 0] [from int]
!24 = metadata !{metadata !"a.cpp", metadata !""}
!25 = metadata !{i32 786468}
!26 = metadata !{metadata !27, metadata !31, metadata !34}
!27 = metadata !{i32 786478, metadata !24, metadata !28, metadata !"bar", metadata !"bar", metadata !"_Z3barv", i32 2, metadata !29, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z3barv, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [bar]
!28 = metadata !{i32 786473, metadata !24}        ; [ DW_TAG_file_type ] [/a.cpp]
!29 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !30, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{metadata !23}
!31 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"_ZN1AC1Ev", i32 2, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 320, i1 false, void (%class.A*)* @_ZN1AC1Ev, null, metadata !32, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [A]
!32 = metadata !{i32 786478, null, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"", i32 0, metadata !15, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !33, i32 0} ; [ DW_TAG_subprogram ] [line 0] [A]
!33 = metadata !{i32 786468}
!34 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"A", metadata !"A", metadata !"_ZN1AC2Ev", i32 2, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 320, i1 false, void (%class.A*)* @_ZN1AC2Ev, null, metadata !32, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [A]
!35 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!36 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!37 = metadata !{metadata !"clang version 3.5 "}
!38 = metadata !{i32 3, i32 0, metadata !27, null}
!39 = metadata !{i32 786689, metadata !31, metadata !"this", null, i32 16777216, metadata !40, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!40 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!41 = metadata !{i32 0, i32 0, metadata !31, null}
!42 = metadata !{i32 2, i32 0, metadata !43, null}
!43 = metadata !{i32 786443, metadata !5, metadata !31} ; [ DW_TAG_lexical_block ] [/./ab.h]
!44 = metadata !{i32 786689, metadata !34, metadata !"this", null, i32 16777216, metadata !40, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!45 = metadata !{i32 0, i32 0, metadata !34, null}
!46 = metadata !{i32 2, i32 0, metadata !34, null}
