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
;
; Note that this test is missing the llvm.dbg.cu annotation. This emulates
; the effect of the user having only used -fprofile-sample-use without
; -gmlt when invoking the driver. In those cases, we need to track source
; location information but we do not have to generate debug info in the
; final binary.
@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @_Z3sumii(i32 %x, i32 %y) {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4, !dbg !11
  %1 = load i32, i32* %y.addr, align 4, !dbg !11
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
  %0 = load i32, i32* %i, align 4, !dbg !14
  %inc = add nsw i32 %0, 1, !dbg !14
  store i32 %inc, i32* %i, align 4, !dbg !14
  %cmp = icmp slt i32 %0, 400000000, !dbg !14
  br i1 %cmp, label %while.body, label %while.end, !dbg !14
; CHECK: edge while.cond -> while.body probability is 5391 / 5392 = 99.9815% [HOT edge]
; CHECK: edge while.cond -> while.end probability is 1 / 5392 = 0.018546%

while.body:                                       ; preds = %while.cond
  %1 = load i32, i32* %i, align 4, !dbg !16
  %cmp1 = icmp ne i32 %1, 100, !dbg !16
  br i1 %cmp1, label %if.then, label %if.else, !dbg !16
; Without discriminator information, the profiler used to think that
; both branches out of while.body had the same weight. In reality,
; the edge while.body->if.then is taken most of the time.
;
; CHECK: edge while.body -> if.then probability is 5752 / 5753 = 99.9826% [HOT edge]
; CHECK: edge while.body -> if.else probability is 1 / 5753 = 0.0173822%


if.then:                                          ; preds = %while.body
  %2 = load i32, i32* %i, align 4, !dbg !18
  %3 = load i32, i32* %s, align 4, !dbg !18
  %call = call i32 @_Z3sumii(i32 %2, i32 %3), !dbg !18
  store i32 %call, i32* %s, align 4, !dbg !18
  br label %if.end, !dbg !18

if.else:                                          ; preds = %while.body
  store i32 30, i32* %s, align 4, !dbg !20
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !dbg !22

while.end:                                        ; preds = %while.cond
  %4 = load i32, i32* %s, align 4, !dbg !24
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 %4), !dbg !24
  ret i32 0, !dbg !25
}

declare i32 @printf(i8*, ...) #2

!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !MDFile(filename: "calls.cc", directory: ".")
!2 = !{}
!3 = !{!4, !7}
!4 = !MDSubprogram(name: "sum", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !5, type: !6, function: i32 (i32, i32)* @_Z3sumii, variables: !2)
!5 = !MDFile(filename: "calls.cc", directory: ".")
!6 = !MDSubroutineType(types: !2)
!7 = !MDSubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 7, file: !1, scope: !5, type: !6, function: i32 ()* @main, variables: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5 "}
!11 = !MDLocation(line: 4, scope: !4)
!12 = !MDLocation(line: 8, scope: !7)
!13 = !MDLocation(line: 9, scope: !7)
!14 = !MDLocation(line: 9, scope: !15)
!15 = !MDLexicalBlockFile(discriminator: 1, file: !1, scope: !7)
!16 = !MDLocation(line: 10, scope: !17)
!17 = distinct !MDLexicalBlock(line: 10, column: 0, file: !1, scope: !7)
!18 = !MDLocation(line: 10, scope: !19)
!19 = !MDLexicalBlockFile(discriminator: 1, file: !1, scope: !17)
!20 = !MDLocation(line: 10, scope: !21)
!21 = !MDLexicalBlockFile(discriminator: 2, file: !1, scope: !17)
!22 = !MDLocation(line: 10, scope: !23)
!23 = !MDLexicalBlockFile(discriminator: 3, file: !1, scope: !17)
!24 = !MDLocation(line: 11, scope: !7)
!25 = !MDLocation(line: 12, scope: !7)
