; RUN: opt %loadPolly -pass-remarks-missed="polly-detect" -polly-detect-track-failures -polly-detect -analyze < %s 2>&1| FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; void onlyWrite(float *A) {
;   for (long i = 0; i < 100; i++)
;     A[i] = 0;
; }
; 
; void onlyRead(float *A) {
;   for (long i = 0; i < 100; i++)
;     A[i];
; }

; CHECK: remark: /tmp/test.c:2:3: The following errors keep this region from being a Scop.
; CHECK: remark: /tmp/test.c:3:10: Invalid Scop candidate ends here.

; CHECK: remark: /tmp/test.c:7:3: The following errors keep this region from being a Scop.
; CHECK: remark: /tmp/test.c:8:10: Invalid Scop candidate ends here.


; Function Attrs: nounwind uwtable
define void @onlyWrite(float* %A) #0 {
entry:
  call void @llvm.dbg.value(metadata float* %A, i64 0, metadata !14, metadata !15), !dbg !16
  call void @llvm.dbg.value(metadata i64 0, i64 0, metadata !17, metadata !15), !dbg !20
  br label %for.cond, !dbg !21

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 100, !dbg !22
  br i1 %exitcond, label %for.body, label %for.end, !dbg !22

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0, !dbg !23
  store float 0.000000e+00, float* %arrayidx, align 4, !dbg !25
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1, !dbg !26
  call void @llvm.dbg.value(metadata i64 %inc, i64 0, metadata !17, metadata !15), !dbg !20
  br label %for.cond, !dbg !27

for.end:                                          ; preds = %for.cond
  ret void, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define void @onlyRead(float* %A) #0 {
entry:
  call void @llvm.dbg.value(metadata float* %A, i64 0, metadata !29, metadata !15), !dbg !30
  call void @llvm.dbg.value(metadata i64 0, i64 0, metadata !31, metadata !15), !dbg !33
  br label %for.cond, !dbg !34

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 100, !dbg !35
  br i1 %exitcond, label %for.body, label %for.end, !dbg !35

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0, !dbg !36
  %val = load float* %arrayidx, align 4, !dbg !38
  br label %for.inc, !dbg !36

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1, !dbg !39
  call void @llvm.dbg.value(metadata i64 %inc, i64 0, metadata !31, metadata !15), !dbg !33
  br label %for.cond, !dbg !40

for.end:                                          ; preds = %for.cond
  ret void, !dbg !41
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !{!"0x11\0012\00clang version 3.7.0  (llvm/trunk 229257)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/grosser/Projects/polly/git/tools/polly//tmp/test.c] [DW_LANG_C99]
!1 = !{!"/tmp/test.c", !"/home/grosser/Projects/polly/git/tools/polly"}
!2 = !{}
!3 = !{!4, !10}
!4 = !{!"0x2e\00onlyWrite\00onlyWrite\00\001\000\001\000\000\00256\000\001", !1, !5, !6, null, void (float*)* @onlyWrite, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [onlyWrite]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/home/grosser/Projects/polly/git/tools/polly//tmp/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from float]
!9 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!10 = !{!"0x2e\00onlyRead\00onlyRead\00\006\000\001\000\000\00256\000\006", !1, !5, !6, null, void (float*)* @onlyRead, null, null, !2} ; [ DW_TAG_subprogram ] [line 6] [def] [onlyRead]
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 2}
!13 = !{!"clang version 3.7.0  (llvm/trunk 229257)"}
!14 = !{!"0x101\00A\0016777217\000", !4, !5, !8}  ; [ DW_TAG_arg_variable ] [A] [line 1]
!15 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!16 = !MDLocation(line: 1, column: 23, scope: !4)
!17 = !{!"0x100\00i\002\000", !18, !5, !19}       ; [ DW_TAG_auto_variable ] [i] [line 2]
!18 = !{!"0xb\002\003\000", !1, !4}               ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly//tmp/test.c]
!19 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!20 = !MDLocation(line: 2, column: 13, scope: !18)
!21 = !MDLocation(line: 2, column: 8, scope: !18)
!22 = !MDLocation(line: 2, column: 3, scope: !18)
!23 = !MDLocation(line: 3, column: 5, scope: !24)
!24 = !{!"0xb\002\003\001", !1, !18}              ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly//tmp/test.c]
!25 = !MDLocation(line: 3, column: 10, scope: !24)
!26 = !MDLocation(line: 2, column: 30, scope: !24)
!27 = !MDLocation(line: 2, column: 3, scope: !24)
!28 = !MDLocation(line: 4, column: 1, scope: !4)
!29 = !{!"0x101\00A\0016777222\000", !10, !5, !8} ; [ DW_TAG_arg_variable ] [A] [line 6]
!30 = !MDLocation(line: 6, column: 22, scope: !10)
!31 = !{!"0x100\00i\007\000", !32, !5, !19}       ; [ DW_TAG_auto_variable ] [i] [line 7]
!32 = !{!"0xb\007\003\002", !1, !10}              ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly//tmp/test.c]
!33 = !MDLocation(line: 7, column: 13, scope: !32)
!34 = !MDLocation(line: 7, column: 8, scope: !32)
!35 = !MDLocation(line: 7, column: 3, scope: !32)
!36 = !MDLocation(line: 8, column: 5, scope: !37)
!37 = !{!"0xb\007\003\003", !1, !32}              ; [ DW_TAG_lexical_block ] [/home/grosser/Projects/polly/git/tools/polly//tmp/test.c]
!38 = !MDLocation(line: 8, column: 10, scope: !37)
!39 = !MDLocation(line: 7, column: 30, scope: !37)
!40 = !MDLocation(line: 7, column: 3, scope: !37)
!41 = !MDLocation(line: 9, column: 1, scope: !10)
