; RUN: opt -S -winehprepare -sehprepare -mtriple=x86_64-windows-msvc < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @cleanup()
declare i32 @filt()
declare void @might_crash()
declare i32 @__C_specific_handler(...)
declare i32 @llvm.eh.typeid.for(i8*)

define i32 @simple_except_store() {
entry:
  %retval = alloca i32
  store i32 0, i32* %retval
  invoke void @might_crash()
          to label %return unwind label %lpad

lpad:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__C_specific_handler
          catch i32 ()* @filt
  %sel = extractvalue { i8*, i32 } %ehvals, 1
  %filt_sel = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ()* @filt to i8*))
  %matches = icmp eq i32 %sel, %filt_sel
  br i1 %matches, label %__except, label %eh.resume

__except:
  store i32 1, i32* %retval
  br label %return

return:
  %r = load i32, i32* %retval
  ret i32 %r

eh.resume:
  resume { i8*, i32 } %ehvals
}

; CHECK-LABEL: define i32 @simple_except_store()
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: catch i32 ()* @filt
; CHECK-NEXT: call i8* (...)* @llvm.eh.actions({{.*}}, i32 0, i8* bitcast (i32 ()* @filt to i8*), i8* null, i8* blockaddress(@simple_except_store, %__except))
; CHECK-NEXT: indirectbr {{.*}} [label %__except]

define i32 @catch_all() {
entry:
  %retval = alloca i32
  store i32 0, i32* %retval
  invoke void @might_crash()
          to label %return unwind label %lpad

lpad:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__C_specific_handler
          catch i8* null
  store i32 1, i32* %retval
  br label %return

return:
  %r = load i32, i32* %retval
  ret i32 %r
}

; CHECK-LABEL: define i32 @catch_all()
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: catch i8* null
; CHECK-NEXT: call i8* (...)* @llvm.eh.actions({{.*}}, i32 0, i8* null, i8* null, i8* blockaddress(@catch_all, %catch.all))
; CHECK-NEXT: indirectbr {{.*}} [label %catch.all]
;
; CHECK: catch.all:
; CHECK: store i32 1, i32* %retval


define i32 @except_phi() {
entry:
  invoke void @might_crash()
          to label %return unwind label %lpad

lpad:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__C_specific_handler
          catch i32 ()* @filt
  %sel = extractvalue { i8*, i32 } %ehvals, 1
  %filt_sel = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ()* @filt to i8*))
  %matches = icmp eq i32 %sel, %filt_sel
  br i1 %matches, label %return, label %eh.resume

return:
  %r = phi i32 [0, %entry], [1, %lpad]
  ret i32 %r

eh.resume:
  resume { i8*, i32 } %ehvals
}

; CHECK-LABEL: define i32 @except_phi()
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: catch i32 ()* @filt
; CHECK-NEXT: call i8* (...)* @llvm.eh.actions({{.*}}, i32 0, i8* bitcast (i32 ()* @filt to i8*), i8* null, i8* blockaddress(@except_phi, %return))
; CHECK-NEXT: indirectbr {{.*}} [label %return]
;
; CHECK: return:
; CHECK-NEXT: %r = phi i32 [ 0, %entry ], [ 1, %lpad1 ]
; CHECK-NEXT: ret i32 %r

define i32 @cleanup_and_except() {
entry:
  invoke void @might_crash()
          to label %return unwind label %lpad

lpad:
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__C_specific_handler
          cleanup
          catch i32 ()* @filt
  call void @cleanup()
  %sel = extractvalue { i8*, i32 } %ehvals, 1
  %filt_sel = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ()* @filt to i8*))
  %matches = icmp eq i32 %sel, %filt_sel
  br i1 %matches, label %return, label %eh.resume

return:
  %r = phi i32 [0, %entry], [1, %lpad]
  ret i32 %r

eh.resume:
  resume { i8*, i32 } %ehvals
}

; CHECK-LABEL: define i32 @cleanup_and_except()
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: catch i32 ()* @filt
; CHECK-NEXT: call i8* (...)* @llvm.eh.actions(
; CHECK: i32 1, i8* bitcast (void (i8*, i8*)* @cleanup_and_except.cleanup to i8*),
; CHECK: i32 0, i8* bitcast (i32 ()* @filt to i8*), i8* null, i8* blockaddress(@cleanup_and_except, %return))
; CHECK-NEXT: indirectbr {{.*}} [label %return]
;
; CHECK: return:
; CHECK-NEXT: %r = phi i32 [ 0, %entry ], [ 1, %lpad1 ]
; CHECK-NEXT: ret i32 %r
