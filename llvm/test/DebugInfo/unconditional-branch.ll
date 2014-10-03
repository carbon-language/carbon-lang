; REQUIRES: object-emission
; PR 19261

; RUN: %llc_dwarf -fast-isel=false -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: {{0x[0-9a-f]+}}      1      0      1   0             0  is_stmt
; CHECK: {{0x[0-9a-f]+}}      2      0      1   0             0  is_stmt
; CHECK: {{0x[0-9a-f]+}}      4      0      1   0             0  is_stmt

; IR generated from clang -O0 -g with the following source:
;void foo(int i){
;  switch(i){
;  default:
;    break;
;  }
;  return;
;}

; Function Attrs: nounwind
define void @foo(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !12, metadata !{metadata !"0x102"}), !dbg !13
  %0 = load i32* %i.addr, align 4, !dbg !14
  switch i32 %0, label %sw.default [
  ], !dbg !14

sw.default:                                       ; preds = %entry
  br label %sw.epilog, !dbg !15

sw.epilog:                                        ; preds = %sw.default
  ret void, !dbg !17
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 (204712)\000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [D:\work\EPRs\396363/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"D:\5Cwork\5CEPRs\5C396363"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !1, metadata !5, metadata !6, null, void (i32)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [D:\work\EPRs\396363/test.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!11 = metadata !{metadata !"clang version 3.5.0 (204712)"}
!12 = metadata !{metadata !"0x101\00i\0016777217\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [i] [line 1]
!13 = metadata !{i32 1, i32 0, metadata !4, null}
!14 = metadata !{i32 2, i32 0, metadata !4, null}
!15 = metadata !{i32 4, i32 0, metadata !16, null}
!16 = metadata !{metadata !"0xb\002\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [D:\work\EPRs\396363/test.c]
!17 = metadata !{i32 6, i32 0, metadata !4, null}
