; RUN: opt -winehprepare -S -o - < %s | FileCheck %s

; This test is based on the following code:
;
; extern "C" void may_throw();
; extern "C" void handle_exception();
; extern "C" void test() {
;   try {
;     may_throw();
;   } catch (...) {
;     handle_exception();
;   }
; }

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

; The function entry in this case remains unchanged.
; CHECK: define void @test()
; CHECK: entry:
; CHECK:   invoke void @may_throw()
; CHECK:           to label %invoke.cont unwind label %[[LPAD_LABEL:lpad[0-9]*]]

define void @test() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  invoke void @may_throw()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

; CHECK: [[LPAD_LABEL]]:{{[ ]+}}; preds = %entry
; CHECK:   landingpad { i8*, i32 }
; CHECK-NEXT:           catch i8* null
; CHECK-NEXT:   [[RECOVER:\%.+]] = call i8* (...) @llvm.eh.actions(i32 1, i8* null, i32 -1, i8* ()* @test.catch)
; CHECK-NEXT:   indirectbr i8* [[RECOVER]], [label %try.cont]

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  store i8* %1, i8** %exn.slot
  %2 = extractvalue { i8*, i32 } %0, 1
  store i32 %2, i32* %ehselector.slot
  br label %catch

; CHECK-NOT: catch:
; CHECK-NOT: @handle_exception()

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot
  call void @llvm.eh.begincatch(i8* %exn, i8* null) #1
  call void @handle_exception()
  call void @llvm.eh.endcatch() #1
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void

; CHECK: }
}

; CHECK: define internal i8* @test.catch()
; CHECK:   call i8* @llvm.frameaddress(i32 1)
; CHECK:   call i8* @llvm.x86.seh.recoverfp(i8* bitcast (void ()* @test to i8*), i8* %{{.*}})
; CHECK:   call void @handle_exception()
; CHECK:   ret i8* blockaddress(@test, %try.cont)
; CHECK: }


declare void @may_throw() #0

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
declare void @llvm.eh.begincatch(i8* nocapture, i8* nocapture) #1

declare void @handle_exception() #0

; Function Attrs: nounwind
declare void @llvm.eh.endcatch() #1

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
