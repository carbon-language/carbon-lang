; RUN: opt -globals-aa -gvn -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@deallocCalled = internal global i8 0, align 1
@.objc_method_list = internal global { i8* ()* } { i8* ()*  @_i_Associated__dealloc }, align 8
@._OBJC_CLASS_Associated = global { i8* } { i8* bitcast ({ i8* ()* }* @.objc_method_list to i8*) }, align 8
@._OBJC_INIT_CLASS_Associated = global { i8* }* @._OBJC_CLASS_Associated
@llvm.used = appending global [1 x i8*] [i8* bitcast ({ i8* }** @._OBJC_INIT_CLASS_Associated to i8*)]

define internal i8* @_i_Associated__dealloc() {
entry:
  store i8 1, i8* @deallocCalled, align 1
  ret i8* null
}

; CHECK-LABEL: @main()
define dso_local i32 @main() {
entry:
  %tmp0 = call i8* @llvm.objc.autoreleasePoolPush() #1
  %tmp6 = load i8, i8* @deallocCalled, align 1
  %tobool = icmp ne i8 %tmp6, 0
  br i1 %tobool, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  call void @__assert_fail() #0
  unreachable

; CHECK-LABEL: if.end:
; CHECK-NEXT: call void @llvm.objc.autoreleasePoolPop
if.end:                                           ; preds = %entry
  call void @llvm.objc.autoreleasePoolPop(i8* %tmp0)
  %tmp7 = load i8, i8* @deallocCalled, align 1
  %tobool3 = icmp ne i8 %tmp7, 0
  br i1 %tobool3, label %if.end6, label %if.else5

if.else5:                                         ; preds = %if.end
  call void @__assert_fail() #0
  unreachable

if.end6:                                          ; preds = %if.end
  store i8 0, i8* @deallocCalled, align 1
  ret i32 0
}

declare i8* @llvm.objc.autoreleasePoolPush() #1
declare void @llvm.objc.autoreleasePoolPop(i8*) #1
declare dso_local void @__assert_fail() #0

attributes #0 = { noreturn nounwind }
attributes #1 = { nounwind }
