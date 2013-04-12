; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux %s -o - | llvm-readobj -t | FileCheck %s

; Check that the appropriate symbols were created.

; CHECK: Symbols [
; CHECK:   Symbol {
; CHECK:     Name: $.str
; CHECK:   }
; CHECK:   Symbol {
; CHECK:     Name: $.str1
; CHECK:   }
; CHECK:   Symbol {
; CHECK:     Name: $CPI0_0
; CHECK:   }
; CHECK:   Symbol {
; CHECK:     Name: $CPI0_1
; CHECK:   }
; CHECK: ]

@.str = private unnamed_addr constant [6 x i8] c"abcde\00", align 1
@gc1 = external global i8*
@.str1 = private unnamed_addr constant [5 x i8] c"fghi\00", align 1
@gc2 = external global i8*
@gd1 = external global double
@gd2 = external global double

define void @foo1() nounwind {
entry:
  store i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i8** @gc1, align 4
  store i8* getelementptr inbounds ([5 x i8]* @.str1, i32 0, i32 0), i8** @gc2, align 4
  %0 = load double* @gd1, align 8
  %add = fadd double %0, 2.500000e+00
  store double %add, double* @gd1, align 8
  %1 = load double* @gd2, align 8
  %add1 = fadd double %1, 4.500000e+00
  store double %add1, double* @gd2, align 8
  ret void
}

