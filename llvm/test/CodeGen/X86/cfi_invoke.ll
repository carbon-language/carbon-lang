; RUN: llc <%s -fcfi -cfi-type=sub | FileCheck %s
; ModuleID = 'test.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @__gxx_personality_v0(...)

@_ZTIPKc = external constant i8*
@_ZTIi = external constant i8*

define void @f() unnamed_addr jumptable {
  ret void
}

@a = global void ()* @f

; Make sure invoke gets targeted as well as regular calls
define void @_Z3foov(void ()* %f) uwtable ssp {
; CHECK-LABEL: _Z3foov:
 entry:
   invoke void %f()
           to label %try.cont unwind label %lpad
; CHECK: callq __llvm_cfi_pointer_warning
; CHECK: callq *%rbx

 lpad:
   %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
                                  catch i8* bitcast (i8** @_ZTIi to i8*)
                                  filter [1 x i8*] [i8* bitcast (i8** @_ZTIPKc to i8*)]
   ret void

 try.cont:
   ret void
}

