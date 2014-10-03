; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/branch.prof | opt -analyze -branch-prob | FileCheck %s

; Original C++ code for this test case:
;
; #include <stdio.h>
; #include <stdlib.h>
;
; int main(int argc, char *argv[]) {
;   if (argc < 2)
;     return 1;
;   double result;
;   int limit = atoi(argv[1]);
;   if (limit > 100) {
;     double s = 23.041968;
;     for (int u = 0; u < limit; u++) {
;       double x = s;
;       s = x + 3.049 + (double)u;
;       s -= s + 3.94 / x * 0.32;
;     }
;     result = s;
;   } else {
;     result = 0;
;   }
;   printf("result is %lf\n", result);
;   return 0;
; }

@.str = private unnamed_addr constant [15 x i8] c"result is %lf\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readonly %argv) #0 {
; CHECK: Printing analysis 'Branch Probability Analysis' for function 'main':

entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !13, metadata !{}), !dbg !27
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !14, metadata !{}), !dbg !27
  %cmp = icmp slt i32 %argc, 2, !dbg !28
  br i1 %cmp, label %return, label %if.end, !dbg !28
; CHECK: edge entry -> return probability is 1 / 2 = 50%
; CHECK: edge entry -> if.end probability is 1 / 2 = 50%

if.end:                                           ; preds = %entry
  %arrayidx = getelementptr inbounds i8** %argv, i64 1, !dbg !30
  %0 = load i8** %arrayidx, align 8, !dbg !30, !tbaa !31
  %call = tail call i32 @atoi(i8* %0) #4, !dbg !30
  tail call void @llvm.dbg.value(metadata !{i32 %call}, i64 0, metadata !17, metadata !{}), !dbg !30
  %cmp1 = icmp sgt i32 %call, 100, !dbg !35
  br i1 %cmp1, label %for.body, label %if.end6, !dbg !35
; CHECK: edge if.end -> for.body probability is 1 / 2 = 50%
; CHECK: edge if.end -> if.end6 probability is 1 / 2 = 50%

for.body:                                         ; preds = %if.end, %for.body
  %u.016 = phi i32 [ %inc, %for.body ], [ 0, %if.end ]
  %s.015 = phi double [ %sub, %for.body ], [ 0x40370ABE6A337A81, %if.end ]
  %add = fadd double %s.015, 3.049000e+00, !dbg !36
  %conv = sitofp i32 %u.016 to double, !dbg !36
  %add4 = fadd double %add, %conv, !dbg !36
  tail call void @llvm.dbg.value(metadata !{double %add4}, i64 0, metadata !18, metadata !{}), !dbg !36
  %div = fdiv double 3.940000e+00, %s.015, !dbg !37
  %mul = fmul double %div, 3.200000e-01, !dbg !37
  %add5 = fadd double %add4, %mul, !dbg !37
  %sub = fsub double %add4, %add5, !dbg !37
  tail call void @llvm.dbg.value(metadata !{double %sub}, i64 0, metadata !18, metadata !{}), !dbg !37
  %inc = add nsw i32 %u.016, 1, !dbg !38
  tail call void @llvm.dbg.value(metadata !{i32 %inc}, i64 0, metadata !21, metadata !{}), !dbg !38
  %exitcond = icmp eq i32 %inc, %call, !dbg !38
  br i1 %exitcond, label %if.end6, label %for.body, !dbg !38
; CHECK: edge for.body -> if.end6 probability is 1 / 10227 = 0.00977804
; CHECK: edge for.body -> for.body probability is 10226 / 10227 = 99.9902% [HOT edge]

if.end6:                                          ; preds = %for.body, %if.end
  %result.0 = phi double [ 0.000000e+00, %if.end ], [ %sub, %for.body ]
  %call7 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([15 x i8]* @.str, i64 0, i64 0), double %result.0), !dbg !39
  br label %return, !dbg !40
; CHECK: edge if.end6 -> return probability is 16 / 16 = 100% [HOT edge]

return:                                           ; preds = %entry, %if.end6
  %retval.0 = phi i32 [ 0, %if.end6 ], [ 1, %entry ]
  ret i32 %retval.0, !dbg !41
}

; Function Attrs: nounwind readonly
declare i32 @atoi(i8* nocapture) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #3

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind readonly }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !42}
!llvm.ident = !{!26}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 (trunk 192896) (llvm/trunk 192895)\001\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [./branch.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"branch.cc", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00main\00main\00\004\000\001\000\006\00256\001\004", metadata !1, metadata !5, metadata !6, null, i32 (i32, i8**)* @main, null, null, metadata !12} ; [ DW_TAG_subprogram ] [line 4] [def] [main]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [./branch.cc]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8, metadata !9}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!11 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!12 = metadata !{metadata !13, metadata !14, metadata !15, metadata !17, metadata !18, metadata !21, metadata !23}
!13 = metadata !{metadata !"0x101\00argc\0016777220\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [argc] [line 4]
!14 = metadata !{metadata !"0x101\00argv\0033554436\000", metadata !4, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [argv] [line 4]
!15 = metadata !{metadata !"0x100\00result\007\000", metadata !4, metadata !5, metadata !16} ; [ DW_TAG_auto_variable ] [result] [line 7]
!16 = metadata !{metadata !"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!17 = metadata !{metadata !"0x100\00limit\008\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_auto_variable ] [limit] [line 8]
!18 = metadata !{metadata !"0x100\00s\0010\000", metadata !19, metadata !5, metadata !16} ; [ DW_TAG_auto_variable ] [s] [line 10]
!19 = metadata !{metadata !"0xb\009\000\000", metadata !1, metadata !20} ; [ DW_TAG_lexical_block ] [./branch.cc]
!20 = metadata !{metadata !"0xb\009\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [./branch.cc]
!21 = metadata !{metadata !"0x100\00u\0011\000", metadata !22, metadata !5, metadata !8} ; [ DW_TAG_auto_variable ] [u] [line 11]
!22 = metadata !{metadata !"0xb\0011\000\000", metadata !1, metadata !19} ; [ DW_TAG_lexical_block ] [./branch.cc]
!23 = metadata !{metadata !"0x100\00x\0012\000", metadata !24, metadata !5, metadata !16} ; [ DW_TAG_auto_variable ] [x] [line 12]
!24 = metadata !{metadata !"0xb\0011\000\000", metadata !1, metadata !22} ; [ DW_TAG_lexical_block ] [./branch.cc]
!25 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!26 = metadata !{metadata !"clang version 3.4 (trunk 192896) (llvm/trunk 192895)"}
!27 = metadata !{i32 4, i32 0, metadata !4, null}
!28 = metadata !{i32 5, i32 0, metadata !29, null}
!29 = metadata !{metadata !"0xb\005\000\000", metadata !1, metadata !4} ; [ DW_TAG_lexical_block ] [./branch.cc]
!30 = metadata !{i32 8, i32 0, metadata !4, null}
!31 = metadata !{metadata !32, metadata !32, i64 0}
!32 = metadata !{metadata !"any pointer", metadata !33, i64 0}
!33 = metadata !{metadata !"omnipotent char", metadata !34, i64 0}
!34 = metadata !{metadata !"Simple C/C++ TBAA"}
!35 = metadata !{i32 9, i32 0, metadata !20, null}
!36 = metadata !{i32 13, i32 0, metadata !24, null}
!37 = metadata !{i32 14, i32 0, metadata !24, null}
!38 = metadata !{i32 11, i32 0, metadata !22, null}
!39 = metadata !{i32 20, i32 0, metadata !4, null}
!40 = metadata !{i32 21, i32 0, metadata !4, null}
!41 = metadata !{i32 22, i32 0, metadata !4, null}
!42 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
