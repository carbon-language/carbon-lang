; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Also test that the null streamer doesn't crash with debug info.
; RUN: %llc_dwarf -O0 -filetype=null < %s

; generated from the following source compiled to bitcode with clang -g -O1
; static int i;
; int main() {
;   (void)&i;
; }

; CHECK: debug_info contents
; CHECK: DW_TAG_variable

; Function Attrs: nounwind readnone uwtable
define i32 @main() #0 {
entry:
  ret i32 0, !dbg !12
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !13}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \001\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !9, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/global.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"global.cpp", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00main\00main\00\002\000\001\000\006\00256\001\002", metadata !1, metadata !5, metadata !6, null, i32 ()* @main, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [def] [main]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/tmp/global.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x34\00i\00i\00_ZL1i\001\001\001", null, metadata !5, metadata !8, null, null} ; [ DW_TAG_variable ]
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!12 = metadata !{i32 4, i32 0, metadata !4, null}
!13 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
