; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from the following code compiled with clang -g:
; enum e1 { I, J = 0xffffffffU, K = 0xf000000000000000ULL } a;
; enum e2 { X };
; void func() {
;   int b = X;
; }

; These values were previously being truncated to -1 and 0 respectively.

; CHECK: debug_info contents
; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name{{.*}} = "e1"
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "J"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]     (4294967295)
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "K"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]     (-1152921504606846976)

; Check that we retain enums that aren't referenced by any variables, etc
; CHECK: DW_TAG_enumeration_type
; CHECK-NEXT: DW_AT_name{{.*}} = "e2"
; CHECK-NOT: NULL
; CHECK: DW_TAG_enumerator
; CHECK-NEXT: DW_AT_name{{.*}} = "X"

@a = global i64 0, align 8

; Function Attrs: nounwind uwtable
define void @_Z4funcv() #0 {
entry:
  %b = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %b}, metadata !20, metadata !{metadata !"0x102"}), !dbg !22
  store i32 0, i32* %b, align 4, !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !24}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \000\00\000\00\000", metadata !1, metadata !2, metadata !11, metadata !12, metadata !17, metadata !11} ; [ DW_TAG_compile_unit ] [/tmp/enum.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"enum.cpp", metadata !"/tmp"}
!2 = metadata !{metadata !3, metadata !8}
!3 = metadata !{metadata !"0x4\00e1\001\0064\0064\000\000\000", metadata !1, null, null, metadata !4, null, null, null} ; [ DW_TAG_enumeration_type ] [e1] [line 1, size 64, align 64, offset 0] [def] [from ]
!4 = metadata !{metadata !5, metadata !6, metadata !7}
!5 = metadata !{metadata !"0x28\00I\000"} ; [ DW_TAG_enumerator ] [I :: 0]
!6 = metadata !{metadata !"0x28\00J\004294967295"} ; [ DW_TAG_enumerator ] [J :: 4294967295]
!7 = metadata !{metadata !"0x28\00K\00-1152921504606846976"} ; [ DW_TAG_enumerator ] [K :: 17293822569102704640]
!8 = metadata !{metadata !"0x4\00e2\002\0032\0032\000\000\000", metadata !1, null, null, metadata !9, null, null, null} ; [ DW_TAG_enumeration_type ] [e2] [line 2, size 32, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x28\00X\000"} ; [ DW_TAG_enumerator ] [X :: 0]
!11 = metadata !{}
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x2e\00func\00func\00_Z4funcv\003\000\001\000\006\00256\000\003", metadata !1, metadata !14, metadata !15, null, void ()* @_Z4funcv, null, null, metadata !11} ; [ DW_TAG_subprogram ] [line 3] [def] [func]
!14 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/tmp/enum.cpp]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null}
!17 = metadata !{metadata !18}
!18 = metadata !{metadata !"0x34\00a\00a\00\001\000\001", null, metadata !14, metadata !3, i64* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!19 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!20 = metadata !{metadata !"0x100\00b\004\000", metadata !13, metadata !14, metadata !21} ; [ DW_TAG_auto_variable ] [b] [line 4]
!21 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!22 = metadata !{i32 4, i32 0, metadata !13, null}
!23 = metadata !{i32 5, i32 0, metadata !13, null}
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
