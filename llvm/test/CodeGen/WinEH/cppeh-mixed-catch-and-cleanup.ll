; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; void test()
; {
;   try {
;     Obj o;
;     may_throw();
;   } catch (...) {
;   }
; }
;
; The purpose of this test is to verify that we create separate catch and
; cleanup handlers.  When compiling for the C++ 11 standard, this isn't
; strictly necessary, since calling the destructor from the catch handler
; would be logically equivalent to calling it from a cleanup handler.
; However, if the -std=c++98 option is used, an exception in the cleanup
; code should terminate the process (the MSVCRT runtime will do that) but
; if the destructor is called from the catch handler, it wouldn't terminate
; the process


; ModuleID = 'cppeh-mixed-catch-and-cleanup.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%class.Obj = type { i8 }

; This just verifies that the function was processed by WinEHPrepare.
;
; CHECK-LABEL: define void @"\01?test@@YAXXZ"()
; CHECK: entry:
; CHECK:   call void (...) @llvm.frameescape
; CHECK: }

; Function Attrs: nounwind uwtable
define void @"\01?test@@YAXXZ"() #0 {
entry:
  %o = alloca %class.Obj, align 1
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @"\01??1Obj@@QEAA@XZ"(%class.Obj* %o) #3
  br label %try.cont

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot
  call void @"\01??1Obj@@QEAA@XZ"(%class.Obj* %o) #3
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #3
  call void @llvm.eh.endcatch() #3
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void
}

; Verify that a cleanup handler was created and that it calls ~Obj().
; CHECK-LABEL: define internal void @"\01?test@@YAXXZ.cleanup"(i8*, i8*)
; CHECK: entry:
; CHECK:   @llvm.framerecover(i8* bitcast (void ()* @"\01?test@@YAXXZ" to i8*), i8* %1, i32 0)
; CHECK:   call void @"\01??1Obj@@QEAA@XZ"
; CHECK:   ret void
; CHECK: }

; Verify that a catch handler was created and that it does not call ~Obj().
; CHECK-LABEL: define internal i8* @"\01?test@@YAXXZ.catch"(i8*, i8*)
; CHECK: entry:
; CHECK-NOT:  call void @"\01??1Obj@@QEAA@XZ"
; CHECK:   ret i8* blockaddress(@"\01?test@@YAXXZ", %try.cont)
; CHECK: }



declare void @"\01?may_throw@@YAXXZ"() #1

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @"\01??1Obj@@QEAA@XZ"(%class.Obj*) #2

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #3

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.7.0 (trunk 235779) (llvm/trunk 235769)"}
