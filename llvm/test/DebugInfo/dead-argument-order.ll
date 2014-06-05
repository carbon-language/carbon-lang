; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Built from the following source with clang -O1
; struct S { int i; };
; int function(struct S s, int i) { return s.i + i; }

; Due to the X86_64 ABI, 's' is passed in registers and once optimized, the
; entirety of 's' is never reconstituted, since only the int is required, and
; thus the variable's location is unknown/dead to debug info.

; Future/current work should enable us to describe partial variables, which, in
; this case, happens to be the entire variable.

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "function"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "s"
; CHECK-NOT: DW_TAG
; FIXME: Even though 's' is never reconstituted into a struct, the one member
; variable is still live and used, and so we should be able to describe 's's
; location as the location of that int.
; CHECK-NOT: DW_AT_location
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "i"
; CHECK:     DW_AT_location


%struct.S = type { i32 }

; Function Attrs: nounwind readnone uwtable
define i32 @_Z8function1Si(i32 %s.coerce, i32 %i) #0 {
entry:
  tail call void @llvm.dbg.declare(metadata !19, metadata !14), !dbg !20
  tail call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata !15), !dbg !20
  %add = add nsw i32 %i, %s.coerce, !dbg !20
  ret i32 %add, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17}
!llvm.ident = !{!18}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !3, metadata !8, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/dead-argument-order.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"dead-argument-order.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"S", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1S"} ; [ DW_TAG_structure_type ] [S] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1S", metadata !"i", i32 1, i64 32, i64 32, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ] [i] [line 1, size 32, align 32, offset 0] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"function", metadata !"function", metadata !"_Z8function1Si", i32 2, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i32, i32)* @_Z8function1Si, null, null, metadata !13, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [function]
!10 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/dead-argument-order.cpp]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !7, metadata !4, metadata !7}
!13 = metadata !{metadata !14, metadata !15}
!14 = metadata !{i32 786689, metadata !9, metadata !"s", metadata !10, i32 16777218, metadata !"_ZTS1S", i32 0, i32 0} ; [ DW_TAG_arg_variable ] [s] [line 2]
!15 = metadata !{i32 786689, metadata !9, metadata !"i", metadata !10, i32 33554434, metadata !7, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [i] [line 2]
!16 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!17 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!18 = metadata !{metadata !"clang version 3.5.0 "}
!19 = metadata !{%struct.S* undef}
!20 = metadata !{i32 2, i32 0, metadata !9, null}

