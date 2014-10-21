; RUN: opt < %s -always-inline -S | FileCheck %s
;
; Generated from the following C++ source with:
; clang -cc1 -disable-llvm-optzns -emit-llvm -g -stack-protector 2 test.cpp
;
; /* BEGIN SOURCE */
; int __attribute__((always_inline)) foo()
; {
;    int arr[10];
;    arr[0] = 5;
;    int sum = 4;
;    return sum;
; }
; 
; extern void bar();
; 
; int main()
; {
;   bar();
;   int i = foo();
;   return i;
; }
; /* END SOURCE */

; The patch that includes this test case, is addressing the following issue:
; 
; When functions are inlined, instructions without debug information 
; are attributed with the call site's DebugLoc. After inlining, inlined static
; allocas are moved to the caller's entry block, adjacent to the caller's original
; static alloca instructions. By retaining the call site's DebugLoc, these instructions
; may cause instructions that are subsequently inserted at the entry block to pick
; up the same DebugLoc.
;
; In the offending case stack protection inserts an instruction at the caller's
; entry block, which inadvertently picks up the inlined call's DebugLoc, because
; the entry block's first instruction is the recently moved inlined alloca instruction. 
;
; The stack protection instruction then becomes part of the function prologue, with the
; result that the line number that is associated with the stack protection instruction
; is deemed to be the end of the function prologue. Since this line number is the
; call site's line number, setting a breakpoint at the function in the debugger
; will make the user stop at the line of the inlined call.

; Note that without the stack protection instruction this effect would not occur
; because the allocas all get collapsed into a single instruction that reserves
; stack space and have no further influence on the prologue's line number information.


; The selected solution is to not attribute static allocas with the call site's
; DebugLoc. 

; At some point in the future, it may be desirable to describe the inlining 
; in the alloca instructions, but then the code that handles prologues must
; be able to handle this correctly, including the late insertion of instructions
; into it.

; In this context it is also important to distingush between functions
; with the "nodebug" attribute and those without it. Alloca instructions from
; nodebug functions should continue to have no DebugLoc, whereas those from
; non-nodebug functions (i.e. functions with debug information) may want to
; have their DebugLocs augmented with inlining information.


; Make sure that after inlining the call to foo() the alloca instructions for
; arr.i and sum.i do not retain debug information.

; CHECK: %arr.i = alloca [10 x i32], align {{[0-9]*$}}
; CHECK: %sum.i = alloca i32, align {{[0-9]*$}}


; ModuleID = 'test.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline nounwind sspstrong
define i32 @_Z3foov() #0 {
entry:
  %arr = alloca [10 x i32], align 16
  %sum = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{[10 x i32]* %arr}, metadata !14), !dbg !18
  %arrayidx = getelementptr inbounds [10 x i32]* %arr, i32 0, i64 0, !dbg !19
  store i32 5, i32* %arrayidx, align 4, !dbg !19
  call void @llvm.dbg.declare(metadata !{i32* %sum}, metadata !20), !dbg !21
  store i32 4, i32* %sum, align 4, !dbg !21
  %0 = load i32* %sum, align 4, !dbg !22
  ret i32 %0, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind sspstrong
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  call void @_Z3barv(), !dbg !23
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !24), !dbg !25
  %call = call i32 @_Z3foov(), !dbg !25
  store i32 %call, i32* %i, align 4, !dbg !25
  %0 = load i32* %i, align 4, !dbg !26
  ret i32 %0, !dbg !26
}

declare void @_Z3barv() #3

attributes #0 = { alwaysinline nounwind sspstrong "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind sspstrong "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.6.0 (217844)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/home/user/test/<stdin>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<stdin>", metadata !"/home/user/test"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !10}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"foo", metadata !"foo", metadata !"_Z3foov", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z3foov, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [foo]
!5 = metadata !{metadata !"test.cpp", metadata !"/home/user/test"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/home/user/test/test.cpp]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 11, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 12} ; [ DW_TAG_subprogram ] [line 11] [def] [scope 12] [main]
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!12 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!13 = metadata !{metadata !"clang version 3.6.0 (217844)"}
!14 = metadata !{i32 786688, metadata !4, metadata !"arr", metadata !6, i32 3, metadata !15, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [arr] [line 3]
!15 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 320, i64 32, i32 0, i32 0, metadata !9, metadata !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 320, align 32, offset 0] [from int]
!16 = metadata !{metadata !17}
!17 = metadata !{i32 786465, i64 0, i64 10}       ; [ DW_TAG_subrange_type ] [0, 9]
!18 = metadata !{i32 3, i32 0, metadata !4, null}
!19 = metadata !{i32 4, i32 0, metadata !4, null}
!20 = metadata !{i32 786688, metadata !4, metadata !"sum", metadata !6, i32 5, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [sum] [line 5]
!21 = metadata !{i32 5, i32 0, metadata !4, null}
!22 = metadata !{i32 6, i32 0, metadata !4, null}
!23 = metadata !{i32 13, i32 0, metadata !10, null}
!24 = metadata !{i32 786688, metadata !10, metadata !"i", metadata !6, i32 14, metadata !9, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [i] [line 14]
!25 = metadata !{i32 14, i32 0, metadata !10, null}
!26 = metadata !{i32 15, i32 0, metadata !10, null}
