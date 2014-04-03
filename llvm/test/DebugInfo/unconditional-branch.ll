; REQUIRES: object-emission
; PR 19261

; FIXME: It is broken for targeting x86_64-cygming.
; RUN: llc -mtriple=x86_64-unknown-unknown -fast-isel=false -O0 -filetype=obj %s -o %t
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
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !12), !dbg !13
  %0 = load i32* %i.addr, align 4, !dbg !14
  switch i32 %0, label %sw.default [
  ], !dbg !14

sw.default:                                       ; preds = %entry
  br label %sw.epilog, !dbg !15

sw.epilog:                                        ; preds = %sw.default
  ret void, !dbg !17
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 (204712)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [D:\work\EPRs\396363/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"D:\5Cwork\5CEPRs\5C396363"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @foo, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [D:\work\EPRs\396363/test.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!11 = metadata !{metadata !"clang version 3.5.0 (204712)"}
!12 = metadata !{i32 786689, metadata !4, metadata !"i", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [i] [line 1]
!13 = metadata !{i32 1, i32 0, metadata !4, null}
!14 = metadata !{i32 2, i32 0, metadata !4, null}
!15 = metadata !{i32 4, i32 0, metadata !16, null}
!16 = metadata !{i32 786443, metadata !1, metadata !4, i32 2, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [D:\work\EPRs\396363/test.c]
!17 = metadata !{i32 6, i32 0, metadata !4, null}
