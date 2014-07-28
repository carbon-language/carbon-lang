; REQUIRES: object-emission
;
; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Make sure we can handle enums with the same identifier but in enum types of
; different compile units.
; rdar://17628609

; CHECK: DW_TAG_compile_unit
; CHECK: 0x[[ENUM:.*]]: DW_TAG_enumeration_type
; CHECK-NEXT:   DW_AT_name {{.*}} "EA"
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_MIPS_linkage_name {{.*}} "_Z4topA2EA"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref4] (cu + 0x{{.*}} => {0x[[ENUM]]})

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_Z4topB2EA"
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[ENUM]]

; Function Attrs: nounwind ssp uwtable
define void @_Z4topA2EA(i32 %sa) #0 {
entry:
  %sa.addr = alloca i32, align 4
  store i32 %sa, i32* %sa.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %sa.addr}, metadata !22), !dbg !23
  ret void, !dbg !24
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
define void @_Z4topB2EA(i32 %sa) #0 {
entry:
  %sa.addr = alloca i32, align 4
  store i32 %sa, i32* %sa.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %sa.addr}, metadata !25), !dbg !26
  ret void, !dbg !27
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !12}
!llvm.module.flags = !{!19, !20}
!llvm.ident = !{!21, !21}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 214102:214133) (llvm/trunk 214102:214132)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !6, metadata !11, metadata !11, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [<unknown>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"a.cpp", metadata !""}
!2 = metadata !{metadata !3}
!3 = metadata !{i32 786436, metadata !1, null, metadata !"EA", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null, metadata !"_ZTS2EA"} ; [ DW_TAG_enumeration_type ] [EA] [line 1, size 32, align 32, offset 0] [def] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786472, metadata !"EA_0", i64 0} ; [ DW_TAG_enumerator ] [EA_0 :: 0]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"topA", metadata !"topA", metadata !"_Z4topA2EA", i32 5, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @_Z4topA2EA, null, null, metadata !11, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [topA]
!8 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [a.cpp]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !"_ZTS2EA"}
!11 = metadata !{}
!12 = metadata !{i32 786449, metadata !13, i32 4, metadata !"clang version 3.5.0 (trunk 214102:214133) (llvm/trunk 214102:214132)", i1 false, metadata !"", i32 0, metadata !14, metadata !14, metadata !16, metadata !11, metadata !11, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [b.cpp] [DW_LANG_C_plus_plus]
!13 = metadata !{metadata !"b.cpp", metadata !""}
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786436, metadata !13, null, metadata !"EA", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null, metadata !"_ZTS2EA"} ; [ DW_TAG_enumeration_type ] [EA] [line 1, size 32, align 32, offset 0] [def] [from ]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786478, metadata !13, metadata !18, metadata !"topB", metadata !"topB", metadata !"_Z4topB2EA", i32 5, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @_Z4topB2EA, null, null, metadata !11, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [topB]
!18 = metadata !{i32 786473, metadata !13}        ; [ DW_TAG_file_type ] [b.cpp]
!19 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!20 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!21 = metadata !{metadata !"clang version 3.5.0 (trunk 214102:214133) (llvm/trunk 214102:214132)"}
!22 = metadata !{i32 786689, metadata !7, metadata !"sa", metadata !8, i32 16777221, metadata !"_ZTS2EA", i32 0, i32 0} ; [ DW_TAG_arg_variable ] [sa] [line 5]
!23 = metadata !{i32 5, i32 14, metadata !7, null}
!24 = metadata !{i32 6, i32 1, metadata !7, null}
!25 = metadata !{i32 786689, metadata !17, metadata !"sa", metadata !18, i32 16777221, metadata !"_ZTS2EA", i32 0, i32 0} ; [ DW_TAG_arg_variable ] [sa] [line 5]
!26 = metadata !{i32 5, i32 14, metadata !17, null}
!27 = metadata !{i32 6, i32 1, metadata !17, null}
