; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

; Don't remove invokes of nounwind functions if the personality handles async
; exceptions. The @div function in this test can fault, even though it can't
; throw a synchronous exception.

define i32 @div(i32 %n, i32 %d) nounwind {
entry:
  %div = sdiv i32 %n, %d
  ret i32 %div
}

define i32 @main() nounwind personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %call = invoke i32 @div(i32 10, i32 0)
          to label %__try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  br label %__try.cont

__try.cont:
  %retval.0 = phi i32 [ %call, %entry ], [ 0, %lpad ]
  ret i32 %retval.0
}

; CHECK-LABEL: define i32 @main()
; CHECK: invoke i32 @div(i32 10, i32 0)

declare i32 @__C_specific_handler(...)
