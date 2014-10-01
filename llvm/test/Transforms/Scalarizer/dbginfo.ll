; RUN: opt %s -scalarizer -scalarize-load-store -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @f1(<4 x i32>* nocapture %a, <4 x i32>* nocapture readonly %b, <4 x i32>* nocapture readonly %c) #0 {
; CHECK: @f1(
; CHECK: %a.i0 = bitcast <4 x i32>* %a to i32*
; CHECK: %a.i1 = getelementptr i32* %a.i0, i32 1
; CHECK: %a.i2 = getelementptr i32* %a.i0, i32 2
; CHECK: %a.i3 = getelementptr i32* %a.i0, i32 3
; CHECK: %c.i0 = bitcast <4 x i32>* %c to i32*
; CHECK: %c.i1 = getelementptr i32* %c.i0, i32 1
; CHECK: %c.i2 = getelementptr i32* %c.i0, i32 2
; CHECK: %c.i3 = getelementptr i32* %c.i0, i32 3
; CHECK: %b.i0 = bitcast <4 x i32>* %b to i32*
; CHECK: %b.i1 = getelementptr i32* %b.i0, i32 1
; CHECK: %b.i2 = getelementptr i32* %b.i0, i32 2
; CHECK: %b.i3 = getelementptr i32* %b.i0, i32 3
; CHECK: tail call void @llvm.dbg.value(metadata !{<4 x i32>* %a}, i64 0, metadata !{{[0-9]+}}), !dbg !{{[0-9]+}}
; CHECK: tail call void @llvm.dbg.value(metadata !{<4 x i32>* %b}, i64 0, metadata !{{[0-9]+}}), !dbg !{{[0-9]+}}
; CHECK: tail call void @llvm.dbg.value(metadata !{<4 x i32>* %c}, i64 0, metadata !{{[0-9]+}}), !dbg !{{[0-9]+}}
; CHECK: %bval.i0 = load i32* %b.i0, align 16, !dbg ![[TAG1:[0-9]+]], !tbaa ![[TAG2:[0-9]+]]
; CHECK: %bval.i1 = load i32* %b.i1, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %bval.i2 = load i32* %b.i2, align 8, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %bval.i3 = load i32* %b.i3, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i0 = load i32* %c.i0, align 16, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i1 = load i32* %c.i1, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i2 = load i32* %c.i2, align 8, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %cval.i3 = load i32* %c.i3, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: %add.i0 = add i32 %bval.i0, %cval.i0, !dbg ![[TAG1]]
; CHECK: %add.i1 = add i32 %bval.i1, %cval.i1, !dbg ![[TAG1]]
; CHECK: %add.i2 = add i32 %bval.i2, %cval.i2, !dbg ![[TAG1]]
; CHECK: %add.i3 = add i32 %bval.i3, %cval.i3, !dbg ![[TAG1]]
; CHECK: store i32 %add.i0, i32* %a.i0, align 16, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: store i32 %add.i1, i32* %a.i1, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: store i32 %add.i2, i32* %a.i2, align 8, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: store i32 %add.i3, i32* %a.i3, align 4, !dbg ![[TAG1]], !tbaa ![[TAG2]]
; CHECK: ret void
entry:
  tail call void @llvm.dbg.value(metadata !{<4 x i32>* %a}, i64 0, metadata !15), !dbg !20
  tail call void @llvm.dbg.value(metadata !{<4 x i32>* %b}, i64 0, metadata !16), !dbg !20
  tail call void @llvm.dbg.value(metadata !{<4 x i32>* %c}, i64 0, metadata !17), !dbg !20
  %bval = load <4 x i32>* %b, align 16, !dbg !21, !tbaa !22
  %cval = load <4 x i32>* %c, align 16, !dbg !21, !tbaa !22
  %add = add <4 x i32> %bval, %cval, !dbg !21
  store <4 x i32> %add, <4 x i32>* %a, align 16, !dbg !21, !tbaa !22
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18, !26}
!llvm.ident = !{!19}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 194134) (llvm/trunk 194126)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/home/richards/llvm/build//tmp/add.c] [DW_LANG_C99]
!1 = metadata !{metadata !"/tmp/add.c", metadata !"/home/richards/llvm/build"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f1", metadata !"f1", metadata !"", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (<4 x i32>*, <4 x i32>*, <4 x i32>*)* @f1, null, null, metadata !14, i32 4} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [f]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/home/richards/llvm/build//tmp/add.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8, metadata !8, metadata !8}
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from V4SI]
!9 = metadata !{i32 786454, metadata !1, null, metadata !"V4SI", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ] [V4SI] [line 1, size 0, align 0, offset 0] [from ]
!10 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 128, i64 128, i32 0, i32 2048, metadata !11, metadata !12, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 128, offset 0] [vector] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786465, i64 0, i64 4}        ; [ DW_TAG_subrange_type ] [0, 3]
!14 = metadata !{metadata !15, metadata !16, metadata !17}
!15 = metadata !{i32 786689, metadata !4, metadata !"a", metadata !5, i32 16777219, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 3]
!16 = metadata !{i32 786689, metadata !4, metadata !"b", metadata !5, i32 33554435, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 3]
!17 = metadata !{i32 786689, metadata !4, metadata !"c", metadata !5, i32 50331651, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [c] [line 3]
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!19 = metadata !{metadata !"clang version 3.4 (trunk 194134) (llvm/trunk 194126)"}
!20 = metadata !{i32 3, i32 0, metadata !4, null}
!21 = metadata !{i32 5, i32 0, metadata !4, null}
!22 = metadata !{metadata !23, metadata !23, i64 0}
!23 = metadata !{metadata !"omnipotent char", metadata !24, i64 0}
!24 = metadata !{metadata !"Simple C/C++ TBAA"}
!25 = metadata !{i32 6, i32 0, metadata !4, null}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
