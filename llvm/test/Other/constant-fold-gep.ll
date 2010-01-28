; RUN: opt -S -o - < %s | FileCheck --check-prefix=PLAIN %s
; RUN: opt -S -o - -instcombine -globalopt < %s | FileCheck --check-prefix=OPT %s

; The automatic constant folder in opt does not have targetdata access, so
; it can't fold gep arithmetic, in general. However, the constant folder run
; from instcombine and global opt does, and can.

; PLAIN: @G8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
; PLAIN: @G1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
; PLAIN: @F8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
; PLAIN: @F1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
; PLAIN: @H8 = global i8* getelementptr (i8* null, i32 -1)
; PLAIN: @H1 = global i1* getelementptr (i1* null, i32 -1)
; PLAIN: define i8* @goo8() nounwind {
; PLAIN:   ret i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
; PLAIN: }
; PLAIN: define i1* @goo1() nounwind {
; PLAIN:   ret i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
; PLAIN: }
; PLAIN: define i8* @foo8() nounwind {
; PLAIN:   ret i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
; PLAIN: }
; PLAIN: define i1* @foo1() nounwind {
; PLAIN:   ret i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
; PLAIN: }
; PLAIN: define i8* @hoo8() nounwind {
; PLAIN:   ret i8* getelementptr (i8* null, i32 -1)
; PLAIN: }
; PLAIN: define i1* @hoo1() nounwind {
; PLAIN:   ret i1* getelementptr (i1* null, i32 -1)
; PLAIN: }

; OPT: @G8 = global i8* null
; OPT: @G1 = global i1* null
; OPT: @F8 = global i8* inttoptr (i64 -1 to i8*)
; OPT: @F1 = global i1* inttoptr (i64 -1 to i1*)
; OPT: @H8 = global i8* inttoptr (i64 -1 to i8*)
; OPT: @H1 = global i1* inttoptr (i64 -1 to i1*)
; OPT: define i8* @goo8() nounwind {
; OPT:   ret i8* null
; OPT: }
; OPT: define i1* @goo1() nounwind {
; OPT:   ret i1* null
; OPT: }
; OPT: define i8* @foo8() nounwind {
; OPT:   ret i8* inttoptr (i64 -1 to i8*)
; OPT: }
; OPT: define i1* @foo1() nounwind {
; OPT:   ret i1* inttoptr (i64 -1 to i1*)
; OPT: }
; OPT: define i8* @hoo8() nounwind {
; OPT:   ret i8* inttoptr (i64 -1 to i8*)
; OPT: }
; OPT: define i1* @hoo1() nounwind {
; OPT:   ret i1* inttoptr (i64 -1 to i1*)
; OPT: }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

@G8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
@G1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
@F8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
@F1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
@H8 = global i8* getelementptr (i8* inttoptr (i32 0 to i8*), i32 -1)
@H1 = global i1* getelementptr (i1* inttoptr (i32 0 to i1*), i32 -1)

define i8* @goo8() nounwind {
  ret i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
}
define i1* @goo1() nounwind {
  ret i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
}
define i8* @foo8() nounwind {
  ret i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
}
define i1* @foo1() nounwind {
  ret i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
}
define i8* @hoo8() nounwind {
  ret i8* getelementptr (i8* inttoptr (i32 0 to i8*), i32 -1)
}
define i1* @hoo1() nounwind {
  ret i1* getelementptr (i1* inttoptr (i32 0 to i1*), i32 -1)
}
