; RUN: opt -mtriple=x86_64-pc-windows-msvc -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; void test()
; {
;   try {
;     may_throw();
;   } catch (int) {
;     handle_int();
;   }
; }
;
; Parts of the IR have been hand-edited to simplify the test case.
; The full IR will be restored when Windows C++ EH support is complete.

;ModuleID = 'cppeh-catch-scalar.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@_ZTIi = external constant i8*

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
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32* %ehselector.slot
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #3
  %matches = icmp eq i32 %sel, %3
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %catch.dispatch
  %exn11 = load i8** %exn.slot
  %4 = call i8* @llvm.eh.begincatch(i8* %exn11) #3
  %5 = bitcast i8* %4 to i32*
  call void @_Z10handle_intv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch() #3
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %invoke.cont
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  %exn3 = load i8** %exn.slot
  %sel4 = load i32* %ehselector.slot
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn3, 0
  %lpad.val5 = insertvalue { i8*, i32 } %lpad.val, i32 %sel4, 1
  resume { i8*, i32 } %lpad.val5
}

; CHECK: define i8* @_Z4testv.catch(i8*, i8*) {
; CHECK: catch.entry:
; CHECK:   %eh.alloc = call i8* @llvm.framerecover(i8* bitcast (void ()* @_Z4testv to i8*), i8* %1)
; CHECK:   %ehdata = bitcast i8* %eh.alloc to %struct._Z4testv.ehdata*
; CHECK:   %eh.obj.ptr = getelementptr inbounds %struct._Z4testv.ehdata* %ehdata, i32 0, i32 1
; CHECK:   %eh.obj = load i8** %eh.obj.ptr
; CHECK:   %2 = bitcast i8* %eh.obj to i32*
; CHECK:   call void @_Z10handle_intv()
; CHECK:   ret i8* blockaddress(@_Z4testv, %try.cont)
; CHECK: }

declare void @_Z9may_throwv() #1

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

declare i8* @llvm.eh.begincatch(i8*)

declare void @llvm.eh.endcatch()

declare void @_Z10handle_intv() #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 227474) (llvm/trunk 227508)"}
