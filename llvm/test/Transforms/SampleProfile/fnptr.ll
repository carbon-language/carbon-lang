; The two profiles used in this test are the same but encoded in different
; formats. This checks that we produce the same profile annotations regardless
; of the profile format.
;
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/fnptr.prof | opt -analyze -branch-prob | FileCheck %s
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/fnptr.binprof | opt -analyze -branch-prob | FileCheck %s

; CHECK:   edge for.body3 -> if.then probability is 534 / 2598 = 20.5543%
; CHECK:   edge for.body3 -> if.else probability is 2064 / 2598 = 79.4457%
; CHECK:   edge for.inc -> for.inc12 probability is 1052 / 2598 = 40.4927%
; CHECK:   edge for.inc -> for.body3 probability is 1546 / 2598 = 59.5073%
; CHECK:   edge for.inc12 -> for.end14 probability is 518 / 1052 = 49.2395%
; CHECK:   edge for.inc12 -> for.cond1.preheader probability is 534 / 1052 = 50.7605%

; Original C++ test case.
;
; #include <stdlib.h>
; #include <math.h>
; #include <stdio.h>
;
; #define N 10000
; #define M 6000
;
; double foo(int x) {
;   return x * sin((double)x);
; }
;
; double bar(int x) {
;   return x - cos((double)x);
; }
;
; int main() {
;   double (*fptr)(int);
;   double S = 0;
;   for (int i = 0; i < N; i++)
;     for (int j = 0; j < M; j++) {
;       fptr = (rand() % 100 < 30) ? foo : bar;
;       if (rand() % 100 < 10)
;         S += (*fptr)(i + j * 300);
;       else
;         S += (*fptr)(i - j / 840);
;     }
;   printf("S = %lf\n", S);
;   return 0;
; }

@.str = private unnamed_addr constant [9 x i8] c"S = %lf\0A\00", align 1

define double @_Z3fooi(i32 %x) #0 {
entry:
  %conv = sitofp i32 %x to double, !dbg !2
  %call = tail call double @sin(double %conv) #3, !dbg !8
  %mul = fmul double %conv, %call, !dbg !8
  ret double %mul, !dbg !8
}

declare double @sin(double) #1

define double @_Z3bari(i32 %x) #0 {
entry:
  %conv = sitofp i32 %x to double, !dbg !9
  %call = tail call double @cos(double %conv) #3, !dbg !11
  %sub = fsub double %conv, %call, !dbg !11
  ret double %sub, !dbg !11
}

declare double @cos(double) #1

define i32 @main() #2 {
entry:
  br label %for.cond1.preheader, !dbg !12

for.cond1.preheader:                              ; preds = %for.inc12, %entry
  %i.025 = phi i32 [ 0, %entry ], [ %inc13, %for.inc12 ]
  %S.024 = phi double [ 0.000000e+00, %entry ], [ %S.2.lcssa, %for.inc12 ]
  br label %for.body3, !dbg !14

for.body3:                                        ; preds = %for.inc, %for.cond1.preheader
  %j.023 = phi i32 [ 0, %for.cond1.preheader ], [ %inc, %for.inc ]
  %S.122 = phi double [ %S.024, %for.cond1.preheader ], [ %S.2, %for.inc ]
  %call = tail call i32 @rand() #3, !dbg !15
  %rem = srem i32 %call, 100, !dbg !15
  %cmp4 = icmp slt i32 %rem, 30, !dbg !15
  %_Z3fooi._Z3bari = select i1 %cmp4, double (i32)* @_Z3fooi, double (i32)* @_Z3bari, !dbg !15
  %call5 = tail call i32 @rand() #3, !dbg !16
  %rem6 = srem i32 %call5, 100, !dbg !16
  %cmp7 = icmp slt i32 %rem6, 10, !dbg !16
  br i1 %cmp7, label %if.then, label %if.else, !dbg !16, !prof !17

if.then:                                          ; preds = %for.body3
  %mul = mul nsw i32 %j.023, 300, !dbg !18
  %add = add nsw i32 %mul, %i.025, !dbg !18
  %call8 = tail call double %_Z3fooi._Z3bari(i32 %add), !dbg !18
  br label %for.inc, !dbg !18

if.else:                                          ; preds = %for.body3
  %div = sdiv i32 %j.023, 840, !dbg !19
  %sub = sub nsw i32 %i.025, %div, !dbg !19
  %call10 = tail call double %_Z3fooi._Z3bari(i32 %sub), !dbg !19
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %call8.pn = phi double [ %call8, %if.then ], [ %call10, %if.else ]
  %S.2 = fadd double %S.122, %call8.pn, !dbg !18
  %inc = add nsw i32 %j.023, 1, !dbg !20
  %exitcond = icmp eq i32 %j.023, 5999, !dbg !14
  br i1 %exitcond, label %for.inc12, label %for.body3, !dbg !14, !prof !21

for.inc12:                                        ; preds = %for.inc
  %S.2.lcssa = phi double [ %S.2, %for.inc ]
  %inc13 = add nsw i32 %i.025, 1, !dbg !22
  %exitcond26 = icmp eq i32 %i.025, 9999, !dbg !12
  br i1 %exitcond26, label %for.end14, label %for.cond1.preheader, !dbg !12, !prof !23

for.end14:                                        ; preds = %for.inc12
  %S.2.lcssa.lcssa = phi double [ %S.2.lcssa, %for.inc12 ]
  %call15 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str, i64 0, i64 0), double %S.2.lcssa.lcssa), !dbg !24
  ret i32 0, !dbg !25
}

; Function Attrs: nounwind
declare i32 @rand() #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #1

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!1 = metadata !{metadata !"clang version 3.6.0 "}
!2 = metadata !{i32 9, i32 3, metadata !3, null}
!3 = metadata !{metadata !"0x2e\00foo\00foo\00\008\000\001\000\000\00256\001\008", metadata !4, metadata !5, metadata !6, null, double (i32)* @_Z3fooi, null, null, metadata !7} ; [ DW_TAG_subprogram ] [line 8] [def] [foo]
!4 = metadata !{metadata !"fnptr.cc", metadata !"."}
!5 = metadata !{metadata !"0x29", metadata !4}    ; [ DW_TAG_file_type ] [./fnptr.cc]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{}
!8 = metadata !{i32 9, i32 14, metadata !3, null}
!9 = metadata !{i32 13, i32 3, metadata !10, null}
!10 = metadata !{metadata !"0x2e\00bar\00bar\00\0012\000\001\000\000\00256\001\0012", metadata !4, metadata !5, metadata !6, null, double (i32)* @_Z3bari, null, null, metadata !7} ; [ DW_TAG_subprogram ] [line 12] [def] [bar]
!11 = metadata !{i32 13, i32 14, metadata !10, null}
!12 = metadata !{i32 19, i32 3, metadata !13, null}
!13 = metadata !{metadata !"0x2e\00main\00main\00\0016\000\001\000\000\00256\001\0016", metadata !4, metadata !5, metadata !6, null, i32 ()* @main, null, null, metadata !7} ; [ DW_TAG_subprogram ] [line 16] [def] [main]
!14 = metadata !{i32 20, i32 5, metadata !13, null}
!15 = metadata !{i32 21, i32 15, metadata !13, null}
!16 = metadata !{i32 22, i32 11, metadata !13, null}
!17 = metadata !{metadata !"branch_weights", i32 534, i32 2064}
!18 = metadata !{i32 23, i32 14, metadata !13, null}
!19 = metadata !{i32 25, i32 14, metadata !13, null}
!20 = metadata !{i32 20, i32 28, metadata !13, null}
!21 = metadata !{metadata !"branch_weights", i32 0, i32 1075}
!22 = metadata !{i32 19, i32 26, metadata !13, null}
!23 = metadata !{metadata !"branch_weights", i32 0, i32 534}
!24 = metadata !{i32 27, i32 3, metadata !13, null}
!25 = metadata !{i32 28, i32 3, metadata !13, null}
