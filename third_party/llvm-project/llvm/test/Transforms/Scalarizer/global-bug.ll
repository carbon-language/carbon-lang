; RUN: opt < %s -passes='function(scalarizer)' -S | FileCheck %s

@a = dso_local global i16 0, align 1
@b = dso_local local_unnamed_addr global i16 0, align 1

; The scalarizer used to take the name of the extractelement instruction
; ("extract") and put that on the extracted value, which in this test case is
; the global variable @a. That was wrong, as we must not change the name of
; the global variable. So make sure we find "@a" in the ptrtoint.
define dso_local void @test1() local_unnamed_addr {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = ptrtoint i16* @a to i16
; CHECK-NEXT:    store i16 [[T0]], i16* @b, align 1
; CHECK-NEXT:    ret void
;
entry:
  %extract = extractelement <4 x i16*> <i16* @a, i16* @a, i16* @a, i16* @a>, i32 1
  %t0 = ptrtoint i16* %extract to i16
  store i16 %t0, i16* @b, align 1
  ret void
}
