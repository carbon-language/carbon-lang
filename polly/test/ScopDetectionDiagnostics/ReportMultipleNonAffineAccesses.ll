; RUN: opt %loadPolly -basicaa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s
; RUN: opt %loadPolly -basicaa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -polly-detect-keep-going -analyze < %s 2>&1| FileCheck %s -check-prefix=ALL
; RUN: opt %loadPolly -basicaa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -polly-delinearize -analyze < %s 2>&1| FileCheck %s -check-prefix=DELIN
; RUN: opt %loadPolly -basicaa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -polly-delinearize -polly-detect-keep-going -analyze < %s 2>&1| FileCheck %s -check-prefix=DELIN-ALL
; RUN: opt %loadPolly -basicaa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -polly-allow-nonaffine -analyze < %s 2>&1| FileCheck %s -check-prefix=NONAFFINE
; RUN: opt %loadPolly -basicaa -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -polly-delinearize -polly-allow-nonaffine -analyze < %s 2>&1| FileCheck %s -check-prefix=NONAFFINE

;  1 void manyaccesses(float A[restrict], long n, float B[restrict][n])
;  2 {
;  3   for (long i = 0; i < 1024; ++i) {
;  4     float a1 = A[2 * i * i];
;  5     float a2 = A[2 * i * i + 1];
;  6     float b1 = B[0][0];
;  7     float b2 = B[i][i];
;  8     float b3 = B[i * i][i];
;  9     float b4 = B[i][0];
; 10     float b5 = B[0][i];
; 11     float b6 = B[0][i*i];
; 12
; 13     A[i * i] = a1 + a2 + b1 + b2 + b3 + b4 + b5 + b6;
; 14   }
; 15 }

; CHECK: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; CHECK-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; CHECK-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; ALL: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; ALL-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; ALL-NEXT: remark: /tmp/test.c:5:16: The array subscript of "A" is not affine
; -> B[0][0] is affine
; ALL-NEXT: remark: /tmp/test.c:7:16: The array subscript of "B" is not affine
; ALL-NEXT: remark: /tmp/test.c:8:16: The array subscript of "B" is not affine
; ALL-NEXT: remark: /tmp/test.c:9:16: The array subscript of "B" is not affine
; -> B[0][i] is affine
; ALL-NEXT: remark: /tmp/test.c:11:16: The array subscript of "B" is not affine
; ALL-NEXT: remark: /tmp/test.c:13:5: The array subscript of "A" is not affine
; ALL-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; DELIN: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; DELIN-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; DELIN-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; DELIN-ALL: remark: /tmp/test.c:3:20: The following errors keep this region from being a Scop.
; DELIN-ALL-NEXT: remark: /tmp/test.c:4:16: The array subscript of "A" is not affine
; DELIN-ALL-NEXT: remark: /tmp/test.c:5:16: The array subscript of "A" is not affine
; DELIN-ALL-NEXT: remark: /tmp/test.c:13:5: The array subscript of "A" is not affine
; -> B[0][0] is affine if delinearized
; -> B[i][i] is affine if delinearized
; DELIN-ALL-NEXT: remark: /tmp/test.c:8:16: The array subscript of "B" is not affine
; -> B[i][0] is affine if delinearized
; -> B[0][i] is affine if delinearized
; DELIN-ALL-NEXT: remark: /tmp/test.c:11:16: The array subscript of "B" is not affine
; DELIN-ALL-NEXT: remark: /tmp/test.c:13:51: Invalid Scop candidate ends here.

; NONAFFINE-NOT: remark: The following errors keep this region from being a Scop.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @manyaccesses(float* noalias %A, i64 %n, float* noalias %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %tmp = add i64 %n, 1, !dbg !10
  br label %for.body, !dbg !10

for.body:                                         ; preds = %entry.split, %for.body
  %tmp3 = phi i64 [ 0, %entry.split ], [ %tmp14, %for.body ], !dbg !15
  %mul = mul i64 %tmp3, 2, !dbg !17
  %tmp4 = mul i64 %tmp, %tmp3, !dbg !18
  %arrayidx8 = getelementptr float* %B, i64 %tmp4, !dbg !19
  %mul9 = mul i64 %n, %tmp3, !dbg !15
  %arrayidx12 = getelementptr float* %B, i64 %mul9, !dbg !20
  %arrayidx15 = getelementptr float* %B, i64 %tmp3, !dbg !21
  %mul1 = mul nsw i64 %mul, %tmp3, !dbg !17
  %arrayidx = getelementptr inbounds float* %A, i64 %mul1, !dbg !22
  %tmp5 = load float* %arrayidx, align 4, !dbg !22
  %mul3 = mul nsw i64 %mul, %tmp3, !dbg !27
  %add1 = or i64 %mul3, 1, !dbg !27
  %arrayidx4 = getelementptr inbounds float* %A, i64 %add1, !dbg !28
  %tmp6 = load float* %arrayidx4, align 4, !dbg !28
  %tmp7 = load float* %B, align 4, !dbg !29
  %tmp8 = load float* %arrayidx8, align 4, !dbg !19
  %tmp9 = mul i64 %mul9, %tmp3, !dbg !15
  %arrayidx10.sum = add i64 %tmp9, %tmp3, !dbg !15
  %arrayidx11 = getelementptr inbounds float* %B, i64 %arrayidx10.sum, !dbg !15
  %tmp10 = load float* %arrayidx11, align 4, !dbg !15
  %tmp11 = load float* %arrayidx12, align 4, !dbg !20
  %tmp12 = load float* %arrayidx15, align 4, !dbg !21
  %mul16 = mul nsw i64 %tmp3, %tmp3, !dbg !30
  %arrayidx18 = getelementptr inbounds float* %B, i64 %mul16, !dbg !31
  %tmp13 = load float* %arrayidx18, align 4, !dbg !31
  %add19 = fadd float %tmp5, %tmp6, !dbg !32
  %add20 = fadd float %add19, %tmp7, !dbg !33
  %add21 = fadd float %add20, %tmp8, !dbg !34
  %add22 = fadd float %add21, %tmp10, !dbg !35
  %add23 = fadd float %add22, %tmp11, !dbg !36
  %add24 = fadd float %add23, %tmp12, !dbg !37
  %add25 = fadd float %add24, %tmp13, !dbg !38
  %mul26 = mul nsw i64 %tmp3, %tmp3, !dbg !39
  %arrayidx27 = getelementptr inbounds float* %A, i64 %mul26, !dbg !40
  store float %add25, float* %arrayidx27, align 4, !dbg !40
  %tmp14 = add nsw i64 %tmp3, 1, !dbg !41
  %exitcond = icmp ne i64 %tmp14, 1024, !dbg !10
  br i1 %exitcond, label %for.body, label %for.end, !dbg !10

for.end:                                          ; preds = %for.body
  ret void, !dbg !42
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.6.0 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 2} ; [ DW_TAG_compile_unit ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"/tmp/test.c", metadata !"/home/grosser/Projects/polly/git/tools/polly/test"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"manyaccesses", metadata !"manyaccesses", metadata !"", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (float*, i64, float*)* @manyaccesses, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [manyaccesses]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!8 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!9 = metadata !{metadata !"clang version 3.6.0 "}
!10 = metadata !{i32 3, i32 20, metadata !11, null}
!11 = metadata !{i32 786443, metadata !1, metadata !12, i32 2} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c]
!12 = metadata !{i32 786443, metadata !1, metadata !13, i32 1} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c]
!13 = metadata !{i32 786443, metadata !1, metadata !14, i32 3, i32 3, i32 1} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c]
!14 = metadata !{i32 786443, metadata !1, metadata !4, i32 3, i32 3, i32 0} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c]
!15 = metadata !{i32 8, i32 16, metadata !16, null} ; [ DW_TAG_imported_declaration ]
!16 = metadata !{i32 786443, metadata !1, metadata !13, i32 3, i32 35, i32 2} ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly/test//tmp/test.c]
!17 = metadata !{i32 4, i32 26, metadata !16, null}
!18 = metadata !{i32 4, i32 22, metadata !16, null}
!19 = metadata !{i32 7, i32 16, metadata !16, null}
!20 = metadata !{i32 9, i32 16, metadata !16, null}
!21 = metadata !{i32 10, i32 16, metadata !16, null}
!22 = metadata !{i32 4, i32 16, metadata !16, null}
!27 = metadata !{i32 5, i32 26, metadata !16, null}
!28 = metadata !{i32 5, i32 16, metadata !16, null}
!29 = metadata !{i32 6, i32 16, metadata !16, null}
!30 = metadata !{i32 11, i32 23, metadata !16, null} ; [ DW_TAG_lexical_block ] [/]
!31 = metadata !{i32 11, i32 16, metadata !16, null} ; [ DW_TAG_lexical_block ] [/]
!32 = metadata !{i32 13, i32 21, metadata !16, null}
!33 = metadata !{i32 13, i32 26, metadata !16, null}
!34 = metadata !{i32 13, i32 31, metadata !16, null}
!35 = metadata !{i32 13, i32 36, metadata !16, null}
!36 = metadata !{i32 13, i32 41, metadata !16, null}
!37 = metadata !{i32 13, i32 46, metadata !16, null}
!38 = metadata !{i32 13, i32 51, metadata !16, null}
!39 = metadata !{i32 13, i32 11, metadata !16, null}
!40 = metadata !{i32 13, i32 5, metadata !16, null}
!41 = metadata !{i32 3, i32 30, metadata !13, null}
!42 = metadata !{i32 15, i32 1, metadata !4, null}
