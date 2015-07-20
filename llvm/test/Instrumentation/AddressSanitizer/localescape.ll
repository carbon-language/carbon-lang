; RUN: opt < %s -asan -asan-module -asan-use-after-return -asan-stack-dynamic-alloca -S | FileCheck %s
; RUN: opt < %s -asan -asan-module -asan-use-after-return=0 -asan-stack-dynamic-alloca=0 -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

declare i32 @llvm.eh.typeid.for(i8*) #2
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.x86.seh.recoverfp(i8*, i8*)
declare i8* @llvm.localrecover(i8*, i8*, i32)
declare void @llvm.localescape(...) #1

declare i32 @_except_handler3(...)
declare void @may_throw(i32* %r)

define i32 @main() sanitize_address personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) {
entry:
  %r = alloca i32, align 4
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(i32* nonnull %__exception_code)
  %0 = bitcast i32* %r to i8*
  store i32 0, i32* %r, align 4
  invoke void @may_throw(i32* nonnull %r) #4
          to label %__try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i32 ()* @"\01?filt$0@0@main@@" to i8*)
  %2 = extractvalue { i8*, i32 } %1, 1
  %3 = call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ()* @"\01?filt$0@0@main@@" to i8*)) #1
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %__except, label %eh.resume

__except:                                         ; preds = %lpad
  store i32 1, i32* %r, align 4
  br label %__try.cont

__try.cont:                                       ; preds = %entry, %__except
  %4 = load i32, i32* %r, align 4
  ret i32 %4

eh.resume:                                        ; preds = %lpad
  resume { i8*, i32 } %1
}

; Check that the alloca remains static and the localescape call remains in the
; entry block.

; CHECK-LABEL: define i32 @main()
; CHECK-NOT: br {{.*}}label
; CHECK: %__exception_code = alloca i32, align 4
; CHECK-NOT: br {{.*}}label
; CHECK: call void (...) @llvm.localescape(i32* nonnull %__exception_code)

; Function Attrs: nounwind
define internal i32 @"\01?filt$0@0@main@@"() #1 {
entry:
  %0 = tail call i8* @llvm.frameaddress(i32 1)
  %1 = tail call i8* @llvm.x86.seh.recoverfp(i8* bitcast (i32 ()* @main to i8*), i8* %0)
  %2 = tail call i8* @llvm.localrecover(i8* bitcast (i32 ()* @main to i8*), i8* %1, i32 0)
  %__exception_code = bitcast i8* %2 to i32*
  %3 = getelementptr inbounds i8, i8* %0, i32 -20
  %4 = bitcast i8* %3 to { i32*, i8* }**
  %5 = load { i32*, i8* }*, { i32*, i8* }** %4, align 4
  %6 = getelementptr inbounds { i32*, i8* }, { i32*, i8* }* %5, i32 0, i32 0
  %7 = load i32*, i32** %6, align 4
  %8 = load i32, i32* %7, align 4
  store i32 %8, i32* %__exception_code, align 4
  ret i32 1
}

; CHECK-LABEL: define internal i32 @"\01?filt$0@0@main@@"()
; CHECK: tail call i8* @llvm.localrecover(i8* bitcast (i32 ()* @main to i8*), i8* {{.*}}, i32 0)
