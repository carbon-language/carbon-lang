; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/calls.prof | opt -analyze -branch-prob | FileCheck %s

; Original C++ test case
;
; #include <stdio.h>
;
; int sum(int x, int y) {
;   return x + y;
; }
;
; int main() {
;   int s, i = 0;
;   while (i++ < 20000 * 20000)
;     if (i != 100) s = sum(i, s); else s = 30;
;   printf("sum is %d\n", s);
;   return 0;
; }

@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @_Z3sumii(i32 %x, i32 %y) {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32* %x.addr, align 4, !dbg !11
  %1 = load i32* %y.addr, align 4, !dbg !11
  %add = add nsw i32 %0, %1, !dbg !11
  ret i32 %add, !dbg !11
}

; Function Attrs: uwtable
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 0, i32* %i, align 4, !dbg !12
  br label %while.cond, !dbg !13

while.cond:                                       ; preds = %if.end, %entry
  %0 = load i32* %i, align 4, !dbg !13
  %inc = add nsw i32 %0, 1, !dbg !13
  store i32 %inc, i32* %i, align 4, !dbg !13
  %cmp = icmp slt i32 %0, 400000000, !dbg !13
  br i1 %cmp, label %while.body, label %while.end, !dbg !13
; CHECK: edge while.cond -> while.body probability is 11143 / 16696 = 66.7405%
; CHECK: edge while.cond -> while.end probability is 5553 / 16696 = 33.2595%

while.body:                                       ; preds = %while.cond
  %1 = load i32* %i, align 4, !dbg !14
  %cmp1 = icmp ne i32 %1, 100, !dbg !14
  br i1 %cmp1, label %if.then, label %if.else, !dbg !14
; NOTE: These weights are wrong. We currently do not handle multiple
;       control flow paths in the same line. The edge while.body -> if.then
;       should be much heavier than the other one. Support for DWARF
;       discriminators will fix this.
; CHECK: edge while.body -> if.then probability is 11143 / 22286 = 50%
; CHECK: edge while.body -> if.else probability is 11143 / 22286 = 50%

if.then:                                          ; preds = %while.body
  %2 = load i32* %i, align 4, !dbg !14
  %3 = load i32* %s, align 4, !dbg !14
  %call = call i32 @_Z3sumii(i32 %2, i32 %3), !dbg !14
  store i32 %call, i32* %s, align 4, !dbg !14
  br label %if.end, !dbg !14

if.else:                                          ; preds = %while.body
  store i32 30, i32* %s, align 4, !dbg !14
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !dbg !14

while.end:                                        ; preds = %while.cond
  %4 = load i32* %s, align 4, !dbg !16
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i32 %4), !dbg !16
  ret i32 0, !dbg !17
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [./calls.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"calls.cc", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !7}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"sum", metadata !"sum", metadata !"", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32, i32)* @_Z3sumii, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [sum]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [./calls.cc]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 7, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.5 "}
!11 = metadata !{i32 4, i32 0, metadata !4, null}
!12 = metadata !{i32 8, i32 0, metadata !7, null} ; [ DW_TAG_imported_declaration ]
!13 = metadata !{i32 9, i32 0, metadata !7, null}
!14 = metadata !{i32 10, i32 0, metadata !15, null}
!15 = metadata !{i32 786443, metadata !1, metadata !7, i32 10, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [./calls.cc]
!16 = metadata !{i32 11, i32 0, metadata !7, null}
!17 = metadata !{i32 12, i32 0, metadata !7, null}
