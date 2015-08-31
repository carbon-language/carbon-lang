; RUN: opt -S -functionattrs %s | FileCheck %s
declare nonnull i8* @ret_nonnull()

; Return a pointer trivially nonnull (call return attribute)
define i8* @test1() {
; CHECK: define nonnull i8* @test1
  %ret = call i8* @ret_nonnull()
  ret i8* %ret
}

; Return a pointer trivially nonnull (argument attribute)
define i8* @test2(i8* nonnull %p) {
; CHECK: define nonnull i8* @test2
  ret i8* %p
}

; Given an SCC where one of the functions can not be marked nonnull,
; can we still mark the other one which is trivially nonnull
define i8* @scc_binder() {
; CHECK: define i8* @scc_binder
  call i8* @test3()
  ret i8* null
}

define i8* @test3() {
; CHECK: define nonnull i8* @test3
  call i8* @scc_binder()
  %ret = call i8* @ret_nonnull()
  ret i8* %ret
}

; Given a mutual recursive set of functions, we can mark them
; nonnull if neither can ever return null.  (In this case, they
; just never return period.)
define i8* @test4_helper() {
; CHECK: define noalias nonnull i8* @test4_helper
  %ret = call i8* @test4()
  ret i8* %ret
}

define i8* @test4() {
; CHECK: define noalias nonnull i8* @test4
  %ret = call i8* @test4_helper()
  ret i8* %ret
}

; Given a mutual recursive set of functions which *can* return null
; make sure we haven't marked them as nonnull.
define i8* @test5_helper() {
; CHECK: define noalias i8* @test5_helper
  %ret = call i8* @test5()
  ret i8* null
}

define i8* @test5() {
; CHECK: define noalias i8* @test5
  %ret = call i8* @test5_helper()
  ret i8* %ret
}

; Local analysis, but going through a self recursive phi
define i8* @test6() {
entry:
; CHECK: define nonnull i8* @test6
  %ret = call i8* @ret_nonnull()
  br label %loop
loop:
  %phi = phi i8* [%ret, %entry], [%phi, %loop]
  br i1 undef, label %loop, label %exit
exit:
  ret i8* %phi
}


