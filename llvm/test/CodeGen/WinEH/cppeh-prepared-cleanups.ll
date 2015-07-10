; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }
%struct.S = type { i8 }

$"\01??_DS@@QEAA@XZ" = comdat any

$"\01??_R0H@8" = comdat any

$"_CT??_R0H@84" = comdat any

$_CTA1H = comdat any

$_TI1H = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@__ImageBase = external constant i8
@"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0H@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @_CTA1H to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat


; CHECK-LABEL: "?test1@@YAXXZ":
; CHECK:             .seh_handlerdata
; CHECK-NEXT:        .long   ("$cppxdata$?test1@@YAXXZ")@IMGREL
; CHECK-NEXT: .align 4
; CHECK-NEXT:"$cppxdata$?test1@@YAXXZ":
; CHECK-NEXT:        .long   429065506
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   ("$stateUnwindMap$?test1@@YAXXZ")@IMGREL
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   ("$ip2state$?test1@@YAXXZ")@IMGREL
; CHECK-NEXT:        .long   32
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1
; CHECK-NEXT:"$stateUnwindMap$?test1@@YAXXZ":
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   "?test1@@YAXXZ.cleanup"@IMGREL
; CHECK-NEXT:"$ip2state$?test1@@YAXXZ":
; CHECK-NEXT:        .long   .Lfunc_begin0@IMGREL
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   .Ltmp0@IMGREL
; CHECK-NEXT:        .long   0

define void @"\01?test1@@YAXXZ"() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %unwindhelp = alloca i64
  %tmp = alloca i32, align 4
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i32 0, i32* %tmp
  %0 = bitcast i32* %tmp to i8*
  call void (...) @llvm.localescape()
  store volatile i64 -2, i64* %unwindhelp
  %1 = bitcast i64* %unwindhelp to i8*
  call void @llvm.eh.unwindhelp(i8* %1)
  invoke void @_CxxThrowException(i8* %0, %eh.ThrowInfo* @_TI1H) #8
          to label %unreachable unwind label %lpad1

lpad1:                                            ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
  %recover = call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @"\01?test1@@YAXXZ.cleanup")
  indirectbr i8* %recover, []

unreachable:                                      ; preds = %entry
  unreachable
}

declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind
define linkonce_odr void @"\01??_DS@@QEAA@XZ"(%struct.S* %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.S*, align 8
  store %struct.S* %this, %struct.S** %this.addr, align 8
  %this1 = load %struct.S*, %struct.S** %this.addr
  call void @"\01??1S@@QEAA@XZ"(%struct.S* %this1) #4
  ret void
}

; CHECK-LABEL: "?test2@@YAX_N@Z":
; CHECK:             .seh_handlerdata
; CHECK-NEXT:        .long   ("$cppxdata$?test2@@YAX_N@Z")@IMGREL
; CHECK-NEXT: .align 4
; CHECK-NEXT:"$cppxdata$?test2@@YAX_N@Z":
; CHECK-NEXT:        .long   429065506
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   ("$stateUnwindMap$?test2@@YAX_N@Z")@IMGREL
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   ("$ip2state$?test2@@YAX_N@Z")@IMGREL
; CHECK-NEXT:        .long   40
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   1
; CHECK-NEXT:"$stateUnwindMap$?test2@@YAX_N@Z":
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   "?test2@@YAX_N@Z.cleanup"@IMGREL
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   "?test2@@YAX_N@Z.cleanup1"@IMGREL
; CHECK-NEXT:"$ip2state$?test2@@YAX_N@Z":
; CHECK-NEXT:        .long   .Lfunc_begin1@IMGREL
; CHECK-NEXT:        .long   -1
; CHECK-NEXT:        .long   .Ltmp7@IMGREL
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   .Ltmp9@IMGREL
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp12@IMGREL
; CHECK-NEXT:        .long   0

define void @"\01?test2@@YAX_N@Z"(i1 zeroext %b) #2 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
  %b.addr = alloca i8, align 1
  %s = alloca %struct.S, align 1
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  %s1 = alloca %struct.S, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void (...) @llvm.localescape(%struct.S* %s, %struct.S* %s1)
  call void @"\01?may_throw@@YAXXZ"()
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont unwind label %lpad1

invoke.cont:                                      ; preds = %entry
  %1 = load i8, i8* %b.addr, align 1
  %tobool = trunc i8 %1 to i1
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %invoke.cont
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont3 unwind label %lpad3

invoke.cont3:                                     ; preds = %if.then
  call void @"\01??_DS@@QEAA@XZ"(%struct.S* %s1) #4
  br label %if.end

lpad1:                                            ; preds = %entry, %if.end
  %2 = landingpad { i8*, i32 }
          cleanup
  %recover = call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @"\01?test2@@YAX_N@Z.cleanup")
  indirectbr i8* %recover, []

lpad3:                                            ; preds = %if.then
  %3 = landingpad { i8*, i32 }
          cleanup
  %recover4 = call i8* (...) @llvm.eh.actions(i32 0, void (i8*, i8*)* @"\01?test2@@YAX_N@Z.cleanup1", i32 0, void (i8*, i8*)* @"\01?test2@@YAX_N@Z.cleanup")
  indirectbr i8* %recover4, []

if.else:                                          ; preds = %invoke.cont
  call void @"\01?dont_throw@@YAXXZ"() #4
  br label %if.end

if.end:                                           ; preds = %if.else, %invoke.cont3
  invoke void @"\01?may_throw@@YAXXZ"()
          to label %invoke.cont4 unwind label %lpad1

invoke.cont4:                                     ; preds = %if.end
  call void @"\01??_DS@@QEAA@XZ"(%struct.S* %s) #4
  ret void
}

declare void @"\01?may_throw@@YAXXZ"() #3

; Function Attrs: nounwind
declare void @"\01?dont_throw@@YAXXZ"() #1

; Function Attrs: nounwind
declare void @"\01??1S@@QEAA@XZ"(%struct.S*) #1

; Function Attrs: nounwind
declare i8* @llvm.eh.actions(...) #4

define internal void @"\01?test1@@YAXXZ.cleanup"(i8*, i8*) #5 {
entry:
  %s = alloca %struct.S, align 1
  call void @"\01??_DS@@QEAA@XZ"(%struct.S* %s) #4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.localescape(...) #4

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #6

; Function Attrs: nounwind
declare void @llvm.eh.unwindhelp(i8*) #4

define internal void @"\01?test2@@YAX_N@Z.cleanup"(i8*, i8*) #7 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %s.i8 = call i8* @llvm.localrecover(i8* bitcast (void (i1)* @"\01?test2@@YAX_N@Z" to i8*), i8* %1, i32 0)
  %s = bitcast i8* %s.i8 to %struct.S*
  call void @"\01??_DS@@QEAA@XZ"(%struct.S* %s) #4
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret void

stub:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
  unreachable
}

define internal void @"\01?test2@@YAX_N@Z.cleanup1"(i8*, i8*) #7 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %s1.i8 = call i8* @llvm.localrecover(i8* bitcast (void (i1)* @"\01?test2@@YAX_N@Z" to i8*), i8* %1, i32 1)
  %s1 = bitcast i8* %s1.i8 to %struct.S*
  call void @"\01??_DS@@QEAA@XZ"(%struct.S* %s1) #4
  invoke void @llvm.donothing()
          to label %entry.split unwind label %stub

entry.split:                                      ; preds = %entry
  ret void

stub:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
  unreachable
}

declare void @llvm.donothing()

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" "wineh-parent"="?test1@@YAXXZ" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" "wineh-parent"="?test2@@YAX_N@Z" }
attributes #3 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { "wineh-parent"="?test1@@YAXXZ" }
attributes #6 = { nounwind readnone }
attributes #7 = { "wineh-parent"="?test2@@YAX_N@Z" }
attributes #8 = { noreturn }
