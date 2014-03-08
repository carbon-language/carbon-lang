; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; CHECK: DW_TAG_lexical_block
; CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]
; CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]

; Test case produced from:
; void b() {
;   if (int i = 3)
;     return;
; }

; Function Attrs: nounwind uwtable
define void @_Z1bv() #0 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !11), !dbg !14
  store i32 3, i32* %i, align 4, !dbg !14
  %0 = load i32* %i, align 4, !dbg !14
  %tobool = icmp ne i32 %0, 0, !dbg !14
  br i1 %tobool, label %if.then, label %if.end, !dbg !14

if.then:                                          ; preds = %entry
  br label %if.end, !dbg !15

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/lexical_block.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"lexical_block.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"b", metadata !"b", metadata !"_Z1bv", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z1bv, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [b]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/lexical_block.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.5.0 "}
!11 = metadata !{i32 786688, metadata !12, metadata !"i", metadata !5, i32 2, metadata !13, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 2]
!12 = metadata !{i32 786443, metadata !1, metadata !4, i32 2, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/tmp/dbginfo/lexical_block.cpp]
!13 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{i32 2, i32 0, metadata !12, null}
!15 = metadata !{i32 3, i32 0, metadata !12, null}
!16 = metadata !{i32 4, i32 0, metadata !4, null}
