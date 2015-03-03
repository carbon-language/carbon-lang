; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; void test()
; {
;   try {
;     may_throw();
;   } catch (...) {
;     handle_exception();
;   }
; }
;
; Parts of the IR have been hand-edited to simplify the test case.
; The full IR will be restored when Windows C++ EH support is complete.

; ModuleID = 'catch-all.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: uwtable
define void @_Z4testv() #0 {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %tmp = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* null
  %tmp1 = extractvalue { i8*, i32 } %tmp, 0
  store i8* %tmp1, i8** %exn.slot
  %tmp2 = extractvalue { i8*, i32 } %tmp, 1
  store i32 %tmp2, i32* %ehselector.slot
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #2
  call void @_Z16handle_exceptionv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %invoke.cont
  ret void
}

; CHECK: define i8* @_Z4testv.catch(i8*, i8*) {
; CHECK: catch.entry:
; CHECK:   %eh.alloc = call i8* @llvm.framerecover(i8* bitcast (void ()* @_Z4testv to i8*), i8* %1)
; CHECK:   %eh.data = bitcast i8* %eh.alloc to %struct._Z4testv.ehdata*
; CHECK:   %eh.obj.ptr = getelementptr inbounds %struct._Z4testv.ehdata, %struct._Z4testv.ehdata* %eh.data, i32 0, i32 1
; CHECK:   %eh.obj = load i8*, i8** %eh.obj.ptr
; CHECK:   call void @_Z16handle_exceptionv()
; CHECK:   ret i8* blockaddress(@_Z4testv, %try.cont)
; CHECK: }

declare void @_Z9may_throwv() #1

declare i32 @__CxxFrameHandler3(...)

declare void @llvm.eh.begincatch(i8*, i8*)

declare void @_Z16handle_exceptionv() #1

declare void @llvm.eh.endcatch()

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline noreturn nounwind }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 226027)"}
