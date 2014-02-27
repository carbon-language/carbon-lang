; RUN: llc -split-dwarf=Enable -O0 %s -ffunction-sections -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck --check-prefix=FUNCTION-SECTIONS %s
; RUN: llvm-readobj --relocations %t | FileCheck --check-prefix=FUNCTION-SECTIONS-RELOCS %s

; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck --check-prefix=NO-FUNCTION-SECTIONS %s

; From:
; int foo (int a) {
;   return a+1;
; }

; With function sections enabled make sure that we have a DW_AT_ranges attribute.
; FUNCTION-SECTIONS: DW_AT_ranges

; Check that we have a relocation against the .debug_ranges section.
; FUNCTION-SECTIONS-RELOCS: R_X86_64_32 .debug_ranges 0x0

; Without function sections enabled make sure that we have no DW_AT_ranges attribute.
; NO-FUNCTION-SECTIONS-NOT: DW_AT_ranges

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %a.addr}, metadata !12), !dbg !13
  %0 = load i32* %a.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 (trunk 199256) (llvm/trunk 199254)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = metadata !{metadata !"foo.c", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @foo, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/foo.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!11 = metadata !{metadata !"clang version 3.5 (trunk 199256) (llvm/trunk 199254)"}
!12 = metadata !{i32 786689, metadata !4, metadata !"a", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 1]
!13 = metadata !{i32 1, i32 0, metadata !4, null}
!14 = metadata !{i32 2, i32 0, metadata !4, null}
