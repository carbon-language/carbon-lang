; RUN: opt -deadargelim -S < %s | FileCheck %s
; PR14016

; Built with clang (then manually running -mem2reg with opt) from the following source:
; static void f1(int, ...) {
; }
;
; void f2() {
;   f1(1);
; }

; Test both varargs removal and removal of a traditional dead arg together, to
; test both the basic functionality, and a particular wrinkle involving updating
; the function->debug info mapping on update to ensure it's accurate when used
; again for the next removal.

; CHECK: void ()* @_ZL2f1iz, {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [f1]

; Check that debug info metadata for subprograms stores pointers to
; updated LLVM functions.

; Function Attrs: uwtable
define void @_Z2f2v() #0 {
entry:
  call void (i32, ...)* @_ZL2f1iz(i32 1), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind uwtable
define internal void @_ZL2f1iz(i32, ...) #1 {
entry:
  call void @llvm.dbg.value(metadata !{i32 %0}, i64 0, metadata !17, metadata !18), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/dbg.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"dbg.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{metadata !"0x2e\00f2\00f2\00_Z2f2v\004\000\001\000\000\00256\000\004", metadata !1, metadata !5, metadata !6, null, void ()* @_Z2f2v, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 4] [def] [f2]
!5 = metadata !{metadata !"0x29", metadata !1}    ; [ DW_TAG_file_type ] [/tmp/dbginfo/dbg.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !"0x2e\00f1\00f1\00_ZL2f1iz\001\001\001\000\000\00256\000\001", metadata !1, metadata !5, metadata !9, null, void (i32, ...)* @_ZL2f1iz, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [local] [def] [f1]
!9 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11, null}
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!13 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!14 = metadata !{metadata !"clang version 3.6.0 "}
!15 = metadata !{i32 5, i32 3, metadata !4, null}
!16 = metadata !{i32 6, i32 1, metadata !4, null}
!17 = metadata !{metadata !"0x101\00\0016777217\000", metadata !8, metadata !5, metadata !11} ; [ DW_TAG_arg_variable ] [line 1]
!18 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!19 = metadata !{i32 1, i32 19, metadata !8, null}
!20 = metadata !{i32 2, i32 1, metadata !8, null}
