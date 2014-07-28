; RUN: true
; This file belongs to type-unique-type-array-a.ll.
;
; rdar://problem/17628609
;
; cat -n b.cpp
;     1	struct SA {
;     2	  int a;
;     3	};
;     4	
;     5	class B {
;     6	public:
;     7	  void testB(SA sa) {
;     8	  }
;     9	};
;    10	
;    11	void topB(B* b, SA sa) {
;    12	  b->testB(sa);
;    13	}

%class.B = type { i8 }
%struct.SA = type { i32 }

; Function Attrs: ssp uwtable
define void @_Z4topBP1B2SA(%class.B* %b, i32 %sa.coerce) #0 {
entry:
  %sa = alloca %struct.SA, align 4
  %b.addr = alloca %class.B*, align 8
  %agg.tmp = alloca %struct.SA, align 4
  %coerce.dive = getelementptr %struct.SA* %sa, i32 0, i32 0
  store i32 %sa.coerce, i32* %coerce.dive
  store %class.B* %b, %class.B** %b.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.B** %b.addr}, metadata !24), !dbg !25
  call void @llvm.dbg.declare(metadata !{%struct.SA* %sa}, metadata !26), !dbg !27
  %0 = load %class.B** %b.addr, align 8, !dbg !28
  %1 = bitcast %struct.SA* %agg.tmp to i8*, !dbg !28
  %2 = bitcast %struct.SA* %sa to i8*, !dbg !28
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* %2, i64 4, i32 4, i1 false), !dbg !28
  %coerce.dive1 = getelementptr %struct.SA* %agg.tmp, i32 0, i32 0, !dbg !28
  %3 = load i32* %coerce.dive1, !dbg !28
  call void @_ZN1B5testBE2SA(%class.B* %0, i32 %3), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN1B5testBE2SA(%class.B* %this, i32 %sa.coerce) #2 align 2 {
entry:
  %sa = alloca %struct.SA, align 4
  %this.addr = alloca %class.B*, align 8
  %coerce.dive = getelementptr %struct.SA* %sa, i32 0, i32 0
  store i32 %sa.coerce, i32* %coerce.dive
  store %class.B* %this, %class.B** %this.addr, align 8
  call void @llvm.dbg.declare(metadata !{%class.B** %this.addr}, metadata !30), !dbg !31
  call void @llvm.dbg.declare(metadata !{%struct.SA* %sa}, metadata !32), !dbg !33
  %this1 = load %class.B** %this.addr
  ret void, !dbg !34
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #3

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 214102:214113M) (llvm/trunk 214102:214115M)", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !14, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [b.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"b.cpp", metadata !"/Users/manmanren/test-Nov/type_unique/rdar_di_array"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !10}
!4 = metadata !{i32 786434, metadata !1, null, metadata !"B", i32 5, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 5, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"testB", metadata !"testB", metadata !"_ZN1B5testBE2SA", i32 7, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 7} ; [ DW_TAG_subprogram ] [line 7] [testB]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !"_ZTS2SA"}
!9 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!10 = metadata !{i32 786451, metadata !1, null, metadata !"SA", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !11, i32 0, null, null, metadata !"_ZTS2SA"} ; [ DW_TAG_structure_type ] [SA] [line 1, size 32, align 32, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786445, metadata !1, metadata !"_ZTS2SA", metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!13 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{metadata !15, metadata !20}
!15 = metadata !{i32 786478, metadata !1, metadata !16, metadata !"topB", metadata !"topB", metadata !"_Z4topBP1B2SA", i32 11, metadata !17, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.B*, i32)* @_Z4topBP1B2SA, null, null, metadata !2, i32 11} ; [ DW_TAG_subprogram ] [line 11] [def] [topB]
!16 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [b.cpp]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19, metadata !"_ZTS2SA"}
!19 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1B]
!20 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"testB", metadata !"testB", metadata !"_ZN1B5testBE2SA", i32 7, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.B*, i32)* @_ZN1B5testBE2SA, null, metadata !6, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [testB]
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!22 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!23 = metadata !{metadata !"clang version 3.5.0 (trunk 214102:214113M) (llvm/trunk 214102:214115M)"}
!24 = metadata !{i32 786689, metadata !15, metadata !"b", metadata !16, i32 16777227, metadata !19, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 11]
!25 = metadata !{i32 11, i32 14, metadata !15, null}
!26 = metadata !{i32 786689, metadata !15, metadata !"sa", metadata !16, i32 33554443, metadata !"_ZTS2SA", i32 0, i32 0} ; [ DW_TAG_arg_variable ] [sa] [line 11]
!27 = metadata !{i32 11, i32 20, metadata !15, null}
!28 = metadata !{i32 12, i32 3, metadata !15, null}
!29 = metadata !{i32 13, i32 1, metadata !15, null}
!30 = metadata !{i32 786689, metadata !20, metadata !"this", null, i32 16777216, metadata !19, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!31 = metadata !{i32 0, i32 0, metadata !20, null}
!32 = metadata !{i32 786689, metadata !20, metadata !"sa", metadata !16, i32 33554439, metadata !"_ZTS2SA", i32 0, i32 0} ; [ DW_TAG_arg_variable ] [sa] [line 7]
!33 = metadata !{i32 7, i32 17, metadata !20, null}
!34 = metadata !{i32 8, i32 3, metadata !20, null} ; [ DW_TAG_imported_declaration ]
