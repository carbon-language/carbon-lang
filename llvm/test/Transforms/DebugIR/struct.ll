; ModuleID = 'struct.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.blah = type { i32, float, i8 }

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %b = alloca %struct.blah, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata !{%struct.blah* %b}, metadata !9), !dbg !22
  %a = getelementptr inbounds %struct.blah* %b, i32 0, i32 0, !dbg !23
  %0 = load i32* %a, align 4, !dbg !23
  ret i32 %0, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/struct.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"struct.cpp", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 8, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 8} ; [ DW_TAG_subprogram ] [line 8] [def] [main]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/struct.cpp]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786688, metadata !4, metadata !"b", metadata !5, i32 9, metadata !10, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [b] [line 9]
!10 = metadata !{i32 786451, metadata !1, null, metadata !"blah", i32 2, i64 96, i64 32, i32 0, i32 0, null, metadata !11, i32 0, null, null} ; [ DW_TAG_structure_type ] [blah] [line 2, size 96, align 32, offset 0] [from ]
!11 = metadata !{metadata !12, metadata !13, metadata !15, metadata !17}
!12 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"a", i32 3, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 3, size 32, align 32, offset 0] [from int]
!13 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"b", i32 4, i64 32, i64 32, i64 32, i32 0, metadata !14} ; [ DW_TAG_member ] [b] [line 4, size 32, align 32, offset 32] [from float]
!14 = metadata !{i32 786468, null, null, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!15 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"c", i32 5, i64 8, i64 8, i64 64, i32 0, metadata !16} ; [ DW_TAG_member ] [c] [line 5, size 8, align 8, offset 64] [from char]
!16 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!17 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"blah", metadata !"blah", metadata !"", i32 2, metadata !18, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !21, i32 2} ; [ DW_TAG_subprogram ] [line 2] [blah]
!18 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !19, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = metadata !{null, metadata !20}
!20 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from blah]
!21 = metadata !{i32 786468}
!22 = metadata !{i32 9, i32 0, metadata !4, null}
!23 = metadata !{i32 10, i32 0, metadata !4, null}
; RUN: opt < %s -debug-ir -S | FileCheck %s.check
