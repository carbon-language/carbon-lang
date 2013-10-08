; REQUIRES: object-emission

; RUN: llc -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: [[TYPE:.*]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_containing_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE]]})
; CHECK: [[SP:.*]]: DW_TAG_subprogram
; CHECK: DW_AT_containing_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE]]})
; CHECK: DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + {{.*}} => {[[TYPE]]})
; IR generated from clang -g with the following source:
; struct C {
;   virtual void foo();
; };
; void C::foo() {
; }

%struct.C = type { i32 (...)** }

@_ZTV1C = unnamed_addr constant [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1C to i8*), i8* bitcast (void (%struct.C*)* @_ZN1C3fooEv to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1C = constant [3 x i8] c"1C\00"
@_ZTI1C = unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1C, i32 0, i32 0) }

; Function Attrs: nounwind ssp uwtable
define void @_ZN1C3fooEv(%struct.C* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.C*, align 8
  store %struct.C* %this, %struct.C** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%struct.C** %this.addr}, metadata !21), !dbg !23
  %this1 = load %struct.C** %this.addr
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 190115)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !18, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [tmp.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tmp.cpp", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"C", i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !5, i32 0, metadata !"_ZTS1C", null, metadata !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] [line 1, size 64, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !13}
!6 = metadata !{i32 786445, metadata !1, metadata !7, metadata !"_vptr$C", i32 0, i64 64, i64 0, i64 0, i32 64, metadata !8} ; [ DW_TAG_member ] [_vptr$C] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!7 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [tmp.cpp]
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!9 = metadata !{i32 786447, null, null, metadata !"__vtbl_ptr_type", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!10 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{i32 786478, metadata !1, metadata !4, metadata !"foo", metadata !"foo", metadata !"_ZN1C3fooEv", i32 2, metadata !14, i1 false, i1 false, i32 1, i32 0, metadata !"_ZTS1C", i32 256, i1 false, null, null, i32 0, metadata !17, i32 2} ; [ DW_TAG_subprogram ] [line 2] [foo]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{null, metadata !16}
!16 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1C]
!17 = metadata !{i32 786468}
!18 = metadata !{metadata !19}
!19 = metadata !{i32 786478, metadata !1, null, metadata !"foo", metadata !"foo", metadata !"_ZN1C3fooEv", i32 4, metadata !14, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.C*)* @_ZN1C3fooEv, null, metadata !13, metadata !2, i32 4} ; [ DW_TAG_subprogram ] [line 4] [def] [foo]
!20 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!21 = metadata !{i32 786689, metadata !19, metadata !"this", null, i32 16777216, metadata !22, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!22 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1C"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1C]
!23 = metadata !{i32 0, i32 0, metadata !19, null}
!24 = metadata !{i32 5, i32 0, metadata !19, null}
