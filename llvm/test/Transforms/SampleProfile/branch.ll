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
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !13, metadata !DIExpression()), !dbg !27
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !14, metadata !DIExpression()), !dbg !27
  %cmp = icmp slt i32 %argc, 2, !dbg !28
  br i1 %cmp, label %return, label %if.end, !dbg !28
; CHECK: edge entry -> return probability is 0 / 1 = 0%
; CHECK: edge entry -> if.end probability is 1 / 1 = 100%

if.end:                                           ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1, !dbg !30
  %0 = load i8*, i8** %arrayidx, align 8, !dbg !30, !tbaa !31
  %call = tail call i32 @atoi(i8* %0) #4, !dbg !30
  tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !17, metadata !DIExpression()), !dbg !30
  %cmp1 = icmp sgt i32 %call, 100, !dbg !35
  br i1 %cmp1, label %for.body, label %if.end6, !dbg !35
; CHECK: edge if.end -> for.body probability is 0 / 1 = 0%
; CHECK: edge if.end -> if.end6 probability is 1 / 1 = 100%

for.body:                                         ; preds = %if.end, %for.body
  %u.016 = phi i32 [ %inc, %for.body ], [ 0, %if.end ]
  %s.015 = phi double [ %sub, %for.body ], [ 0x40370ABE6A337A81, %if.end ]
  %add = fadd double %s.015, 3.049000e+00, !dbg !36
  %conv = sitofp i32 %u.016 to double, !dbg !36
  %add4 = fadd double %add, %conv, !dbg !36
  tail call void @llvm.dbg.value(metadata double %add4, i64 0, metadata !18, metadata !DIExpression()), !dbg !36
  %div = fdiv double 3.940000e+00, %s.015, !dbg !37
  %mul = fmul double %div, 3.200000e-01, !dbg !37
  %add5 = fadd double %add4, %mul, !dbg !37
  %sub = fsub double %add4, %add5, !dbg !37
  tail call void @llvm.dbg.value(metadata double %sub, i64 0, metadata !18, metadata !DIExpression()), !dbg !37
  %inc = add nsw i32 %u.016, 1, !dbg !38
  tail call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !21, metadata !DIExpression()), !dbg !38
  %exitcond = icmp eq i32 %inc, %call, !dbg !38
  br i1 %exitcond, label %if.end6, label %for.body, !dbg !38
; CHECK: edge for.body -> if.end6 probability is 0 / 10226 = 0%
; CHECK: edge for.body -> for.body probability is 10226 / 10226 = 100% [HOT edge]

if.end6:                                          ; preds = %for.body, %if.end
  %result.0 = phi double [ 0.000000e+00, %if.end ], [ %sub, %for.body ]
  %call7 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), double %result.0), !dbg !39
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

!0 = !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (trunk 192896) (llvm/trunk 192895)", isOptimized: true, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "branch.cc", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "main", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 4, file: !1, scope: !5, type: !6, function: i32 (i32, i8**)* @main, variables: !12)
!5 = !DIFile(filename: "branch.cc", directory: ".")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!12 = !{!13, !14, !15, !17, !18, !21, !23}
!13 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "argc", line: 4, arg: 1, scope: !4, file: !5, type: !8)
!14 = !DILocalVariable(tag: DW_TAG_arg_variable, name: "argv", line: 4, arg: 2, scope: !4, file: !5, type: !9)
!15 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "result", line: 7, scope: !4, file: !5, type: !16)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!17 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "limit", line: 8, scope: !4, file: !5, type: !8)
!18 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "s", line: 10, scope: !19, file: !5, type: !16)
!19 = distinct !DILexicalBlock(line: 9, column: 0, file: !1, scope: !20)
!20 = distinct !DILexicalBlock(line: 9, column: 0, file: !1, scope: !4)
!21 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "u", line: 11, scope: !22, file: !5, type: !8)
!22 = distinct !DILexicalBlock(line: 11, column: 0, file: !1, scope: !19)
!23 = !DILocalVariable(tag: DW_TAG_auto_variable, name: "x", line: 12, scope: !24, file: !5, type: !16)
!24 = distinct !DILexicalBlock(line: 11, column: 0, file: !1, scope: !22)
!25 = !{i32 2, !"Dwarf Version", i32 4}
!26 = !{!"clang version 3.4 (trunk 192896) (llvm/trunk 192895)"}
!27 = !DILocation(line: 4, scope: !4)
!28 = !DILocation(line: 5, scope: !29)
!29 = distinct !DILexicalBlock(line: 5, column: 0, file: !1, scope: !4)
!30 = !DILocation(line: 8, scope: !4)
!31 = !{!32, !32, i64 0}
!32 = !{!"any pointer", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 9, scope: !20)
!36 = !DILocation(line: 13, scope: !24)
!37 = !DILocation(line: 14, scope: !24)
!38 = !DILocation(line: 11, scope: !22)
!39 = !DILocation(line: 20, scope: !4)
!40 = !DILocation(line: 21, scope: !4)
!41 = !DILocation(line: 22, scope: !4)
!42 = !{i32 1, !"Debug Info Version", i32 3}
