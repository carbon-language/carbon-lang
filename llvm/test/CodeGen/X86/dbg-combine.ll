; RUN: llc -mtriple x86_64-pc-linux -O0 < %s | FileCheck %s

; Make sure that the sequence of debug locations for function foo is correctly
; generated. More specifically, .loc entries for lines 4,5,6,7 must appear in
; the correct sequence.

; $ clang -emit-llvm -S -g dbg-combine.c
; 1.  int foo()
; 2.  {
; 3.     int elems = 3;
; 4.     int array1[elems];
; 5.     array1[0]=0;
; 6.     array1[1]=1;
; 7.     array1[2]=2;
; 8.     int array2[elems];
; 9.     array2[0]=1;
; 10.    return array2[0];
; 11. }

; CHECK: .loc    1 4
; CHECK: .loc    1 5
; CHECK: .loc    1 6
; CHECK: .loc    1 7

; ModuleID = 'dbg-combine.c'
; Function Attrs: nounwind uwtable
define i32 @foo() #0 {
entry:
  %elems = alloca i32, align 4
  %saved_stack = alloca i8*
  %cleanup.dest.slot = alloca i32
  call void @llvm.dbg.declare(metadata i32* %elems, metadata !12, metadata !13), !dbg !14
  store i32 3, i32* %elems, align 4, !dbg !14
  %0 = load i32* %elems, align 4, !dbg !15
  %1 = zext i32 %0 to i64, !dbg !16
  %2 = call i8* @llvm.stacksave(), !dbg !16
  store i8* %2, i8** %saved_stack, !dbg !16
  %vla = alloca i32, i64 %1, align 16, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !17, metadata !21), !dbg !22
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 0, !dbg !23
  store i32 0, i32* %arrayidx, align 4, !dbg !24
  %arrayidx1 = getelementptr inbounds i32, i32* %vla, i64 1, !dbg !25
  store i32 1, i32* %arrayidx1, align 4, !dbg !26
  %arrayidx2 = getelementptr inbounds i32, i32* %vla, i64 2, !dbg !27
  store i32 2, i32* %arrayidx2, align 4, !dbg !28
  %3 = load i32* %elems, align 4, !dbg !29
  %4 = zext i32 %3 to i64, !dbg !30
  %vla3 = alloca i32, i64 %4, align 16, !dbg !30
  call void @llvm.dbg.declare(metadata i32* %vla3, metadata !31, metadata !21), !dbg !32
  %arrayidx4 = getelementptr inbounds i32, i32* %vla3, i64 0, !dbg !33
  store i32 1, i32* %arrayidx4, align 4, !dbg !34
  %arrayidx5 = getelementptr inbounds i32, i32* %vla3, i64 0, !dbg !35
  %5 = load i32* %arrayidx5, align 4, !dbg !35
  store i32 1, i32* %cleanup.dest.slot
  %6 = load i8** %saved_stack, !dbg !36
  call void @llvm.stackrestore(i8* %6), !dbg !36
  ret i32 %5, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #2

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #2

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.7.0 (trunk 227074)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/home/probinson/projects/scratch/dbg-combine.c] [DW_LANG_C99]
!1 = !{!"dbg-combine.c", !"/home/probinson/projects/scratch"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\000\000\000\002", !1, !5, !6, null, i32 ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [foo]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/home/probinson/projects/scratch/dbg-combine.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 2}
!11 = !{!"clang version 3.7.0 (trunk 227074)"}
!12 = !{!"0x100\00elems\003\000", !4, !5, !8}     ; [ DW_TAG_auto_variable ] [elems] [line 3]
!13 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!14 = !MDLocation(line: 3, column: 8, scope: !4)
!15 = !MDLocation(line: 4, column: 15, scope: !4)
!16 = !MDLocation(line: 4, column: 4, scope: !4)
!17 = !{!"0x100\00array1\004\000", !4, !5, !18}   ; [ DW_TAG_auto_variable ] [array1] [line 4]
!18 = !{!"0x1\00\000\000\0032\000\000\000", null, null, !8, !19, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!19 = !{!20}
!20 = !{!"0x21\000\00-1"}                         ; [ DW_TAG_subrange_type ] [unbounded]
!21 = !{!"0x102\006"}                             ; [ DW_TAG_expression ] [DW_OP_deref]
!22 = !MDLocation(line: 4, column: 8, scope: !4)
!23 = !MDLocation(line: 5, column: 4, scope: !4)
!24 = !MDLocation(line: 5, column: 13, scope: !4)
!25 = !MDLocation(line: 6, column: 4, scope: !4)
!26 = !MDLocation(line: 6, column: 13, scope: !4)
!27 = !MDLocation(line: 7, column: 4, scope: !4)
!28 = !MDLocation(line: 7, column: 13, scope: !4)
!29 = !MDLocation(line: 8, column: 15, scope: !4)
!30 = !MDLocation(line: 8, column: 4, scope: !4)
!31 = !{!"0x100\00array2\008\000", !4, !5, !18}   ; [ DW_TAG_auto_variable ] [array2] [line 8]
!32 = !MDLocation(line: 8, column: 8, scope: !4)
!33 = !MDLocation(line: 9, column: 4, scope: !4)
!34 = !MDLocation(line: 9, column: 13, scope: !4)
!35 = !MDLocation(line: 10, column: 11, scope: !4)
!36 = !MDLocation(line: 11, column: 1, scope: !4)
