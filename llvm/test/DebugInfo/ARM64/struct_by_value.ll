; A by-value struct is a register-indirect value (breg).
; RUN: llc %s -filetype=asm -o - | FileCheck %s

; CHECK: DW_OP_breg0

; rdar://problem/13658587
;
; Generated from
;
; struct five
; {
;   int a;
;   int b;
;   int c;
;   int d;
;   int e;
; };
;
; int
; return_five_int (struct five f)
; {
;   return f.a;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"
target triple = "arm64-apple-ios3.0.0"

%struct.five = type { i32, i32, i32, i32, i32 }

; Function Attrs: nounwind ssp
define i32 @return_five_int(%struct.five* %f) #0 {
entry:
  call void @llvm.dbg.declare(metadata !{%struct.five* %f}, metadata !17), !dbg !18
  %a = getelementptr inbounds %struct.five* %f, i32 0, i32 0, !dbg !19
  %0 = load i32* %a, align 4, !dbg !19
  ret i32 %0, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !20}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"LLVM version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [struct_by_value.c] [DW_LANG_C99]
!1 = metadata !{metadata !"struct_by_value.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"return_five_int", metadata !"return_five_int", metadata !"", i32 13, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (%struct.five*)* @return_five_int, null, null, metadata !2, i32 14} ; [ DW_TAG_subprogram ] [line 13] [def] [scope 14] [return_five_int]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [struct_by_value.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786451, metadata !1, null, metadata !"five", i32 1, i64 160, i64 32, i32 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [five] [line 1, size 160, align 32, offset 0] [def] [from ]
!10 = metadata !{metadata !11, metadata !12, metadata !13, metadata !14, metadata !15}
!11 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"a", i32 3, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 3, size 32, align 32, offset 0] [from int]
!12 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"b", i32 4, i64 32, i64 32, i64 32, i32 0, metadata !8} ; [ DW_TAG_member ] [b] [line 4, size 32, align 32, offset 32] [from int]
!13 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"c", i32 5, i64 32, i64 32, i64 64, i32 0, metadata !8} ; [ DW_TAG_member ] [c] [line 5, size 32, align 32, offset 64] [from int]
!14 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"d", i32 6, i64 32, i64 32, i64 96, i32 0, metadata !8} ; [ DW_TAG_member ] [d] [line 6, size 32, align 32, offset 96] [from int]
!15 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"e", i32 7, i64 32, i64 32, i64 128, i32 0, metadata !8} ; [ DW_TAG_member ] [e] [line 7, size 32, align 32, offset 128] [from int]
!16 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!17 = metadata !{i32 786689, metadata !4, metadata !"f", metadata !5, i32 16777229, metadata !9, i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [f] [line 13]
!18 = metadata !{i32 13, i32 0, metadata !4, null}
!19 = metadata !{i32 16, i32 0, metadata !4, null}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
