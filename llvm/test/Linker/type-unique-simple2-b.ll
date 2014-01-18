; RUN: true
; This file belongs to type-unique-simple2-a.ll.
;
; $ cat b.cpp
; #include "ab.h"
; void A::setFoo() {}
; const
; foo_t A::getFoo() { return 1; }
; ModuleID = 'b.cpp'
; target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
; target triple = "x86_64-apple-macosx10.9.0"

%class.A = type { i32 (...)** }

@_ZTV1A = unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%class.A*)* @_ZN1A6setFooEv to i8*), i8* bitcast (i32 (%class.A*)* @_ZN1A6getFooEv to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1A = constant [3 x i8] c"1A\00"
@_ZTI1A = unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1A, i32 0, i32 0) }

; Function Attrs: nounwind
define void @_ZN1A6setFooEv(%class.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !32), !dbg !34
  %this1 = load %class.A** %this.addr
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind
define i32 @_ZN1A6getFooEv(%class.A* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %class.A*, align 8
  store %class.A* %this, %class.A** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.A** %this.addr}, metadata !36), !dbg !37
  %this1 = load %class.A** %this.addr
  ret i32 1, !dbg !38
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29, !30}
!llvm.ident = !{!31}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !25, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/<unknown>] [DW_LANG_C_plus_plus]
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
!19 = metadata !{i32 786478, metadata !5, metadata !"_ZTS1A", metadata !"getFoo", metadata !"getFoo", metadata !"_ZN1A6getFooEv", i32 5, metadata !20, i1 false, i1 false, i32 1, i32 1, metadata !"_ZTS1A", i32 256, i1 false, null, null, i32 0, metadata !24, i32 5} ; [ DW_TAG_subprogram ] [line 5] [getFoo]
!20 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !22, metadata !17}
!22 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !23} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo_t]
!23 = metadata !{i32 786454, metadata !5, null, metadata !"foo_t", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_typedef ] [foo_t] [line 1, size 0, align 0, offset 0] [from int]
!24 = metadata !{i32 786468}
!25 = metadata !{metadata !26, metadata !28}
!26 = metadata !{i32 786478, metadata !27, metadata !"_ZTS1A", metadata !"setFoo", metadata !"setFoo", metadata !"_ZN1A6setFooEv", i32 2, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*)* @_ZN1A6setFooEv, null, metadata !14, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [setFoo]
!27 = metadata !{metadata !"b.cpp", metadata !""}
!28 = metadata !{i32 786478, metadata !27, metadata !"_ZTS1A", metadata !"getFoo", metadata !"getFoo", metadata !"_ZN1A6getFooEv", i32 4, metadata !20, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%class.A*)* @_ZN1A6getFooEv, null, metadata !19, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [getFoo]
!29 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!30 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!31 = metadata !{metadata !"clang version 3.5 "}
!32 = metadata !{i32 786689, metadata !26, metadata !"this", null, i32 16777216, metadata !33, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!33 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!34 = metadata !{i32 0, i32 0, metadata !26, null}
!35 = metadata !{i32 2, i32 0, metadata !26, null}
!36 = metadata !{i32 786689, metadata !28, metadata !"this", null, i32 16777216, metadata !33, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!37 = metadata !{i32 0, i32 0, metadata !28, null}
!38 = metadata !{i32 4, i32 0, metadata !28, null}
