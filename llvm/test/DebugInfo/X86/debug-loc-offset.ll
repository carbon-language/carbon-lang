; RUN: llc %s -filetype=obj -O0 -mtriple=i386-unknown-linux-gnu -dwarf-version=4 -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; From the code:

; bar.cpp
; int bar (int b) {
;   return b+4;
; }

; foo.cpp
; struct A {
;   int a;
;   int b;
;   int c;
; };

; int a (struct A var) {
;   return var.a;
; }

; Compiled separately for i386-pc-linux-gnu and linked together.
; This ensures that we have multiple compile units so that we can verify that
; debug_loc entries are relative to the low_pc of the CU. The loc entry for
; the byval argument in foo.cpp is in the second CU and so should have
; an offset relative to that CU rather than from the beginning of the text
; section.

; Checking that we have two compile units with two sets of high/lo_pc.
; CHECK: .debug_info contents
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_low_pc
; CHECK: DW_AT_high_pc

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_low_pc
; CHECK: DW_AT_high_pc

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name [DW_FORM_strp]{{.*}}"_Z1a1A"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name [DW_FORM_strp]{{.*}}"var"
; CHECK: DW_AT_location [DW_FORM_sec_offset]   (0x00000000)
; CHECK-NOT: DW_AT_location

; CHECK: .debug_loc contents:
; CHECK: 0x00000000: Beginning address offset: 0x0000000000000000
; CHECK:                Ending address offset: 0x0000000000000009


%struct.A = type { i32, i32, i32 }

; Function Attrs: nounwind
define i32 @_Z3bari(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %b.addr}, metadata !25), !dbg !26
  %0 = load i32* %b.addr, align 4, !dbg !27
  %add = add nsw i32 %0, 4, !dbg !27
  ret i32 %add, !dbg !27
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind
define i32 @_Z1a1A(%struct.A* byval align 4 %var) #0 {
entry:
  call void @llvm.dbg.declare(metadata !{%struct.A* %var}, metadata !28), !dbg !29
  %a = getelementptr inbounds %struct.A* %var, i32 0, i32 0, !dbg !30
  %0 = load i32* %a, align 4, !dbg !30
  ret i32 %0, !dbg !30
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24, !24}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (trunk 204264) (llvm/trunk 204286)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1}
!1 = metadata !{metadata !"bar.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"_Z3bari", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z3bari, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/bar.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786449, metadata !10, i32 4, metadata !"clang version 3.5.0 (trunk 204264) (llvm/trunk 204286)", i1 false, metadata !"", i32 0, metadata !2, metadata !11, metadata !17, metadata !2, metadata !2, metadata !"", i32 1}
!10 = metadata !{metadata !"foo.cpp", metadata !"/usr/local/google/home/echristo/tmp"}
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786451, metadata !10, null, metadata !"A", i32 1, i64 96, i64 32, i32 0, i32 0, null, metadata !13, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 96, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !15, metadata !16}
!14 = metadata !{i32 786445, metadata !10, metadata !"_ZTS1A", metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!15 = metadata !{i32 786445, metadata !10, metadata !"_ZTS1A", metadata !"b", i32 3, i64 32, i64 32, i64 32, i32 0, metadata !8} ; [ DW_TAG_member ] [b] [line 3, size 32, align 32, offset 32] [from int]
!16 = metadata !{i32 786445, metadata !10, metadata !"_ZTS1A", metadata !"c", i32 4, i64 32, i64 32, i64 64, i32 0, metadata !8} ; [ DW_TAG_member ] [c] [line 4, size 32, align 32, offset 64] [from int]
!17 = metadata !{metadata !18}
!18 = metadata !{i32 786478, metadata !10, metadata !19, metadata !"a", metadata !"a", metadata !"_Z1a1A", i32 7, metadata !20, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%struct.A*)* @_Z1a1A, null, null, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [a]
!19 = metadata !{i32 786473, metadata !10}        ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/foo.cpp]
!20 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !21, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !8, metadata !12}
!22 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!23 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!24 = metadata !{metadata !"clang version 3.5.0 (trunk 204264) (llvm/trunk 204286)"}
!25 = metadata !{i32 786689, metadata !4, metadata !"b", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 1]
!26 = metadata !{i32 1, i32 0, metadata !4, null}
!27 = metadata !{i32 2, i32 0, metadata !4, null}
!28 = metadata !{i32 786689, metadata !18, metadata !"var", metadata !19, i32 16777223, metadata !"_ZTS1A", i32 0, i32 0}
!29 = metadata !{i32 7, i32 0, metadata !18, null}
!30 = metadata !{i32 8, i32 0, metadata !18, null} ; [ DW_TAG_imported_declaration ]
