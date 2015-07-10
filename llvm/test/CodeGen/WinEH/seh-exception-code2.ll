; RUN: opt -winehprepare -S < %s | FileCheck %s

; WinEHPrepare was crashing during phi demotion.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

declare i32 @__C_specific_handler(...)

@str = linkonce_odr unnamed_addr constant [16 x i8] c"caught it! %lx\0A\00", align 1

declare void @maycrash()
declare void @finally(i1 %abnormal)
declare i32 @printf(i8* nocapture readonly, ...)
declare i32 @llvm.eh.typeid.for(i8*)

; Function Attrs: nounwind uwtable
define void @doit() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @maycrash()
          to label %invoke.cont unwind label %lpad.1

invoke.cont:                                      ; preds = %entry
  invoke void @maycrash()
          to label %__try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %lp0 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@doit@@" to i8*)
  %ehptr.0 = extractvalue { i8*, i32 } %lp0, 0
  %ehsel.0 = extractvalue { i8*, i32 } %lp0, 1
  call void @finally(i1 true)
  br label %ehdispatch

lpad.1:                                           ; preds = %invoke.cont, %lpad
  %lp1 = landingpad { i8*, i32 }
          catch i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@doit@@" to i8*)
  %ehptr.1 = extractvalue { i8*, i32 } %lp1, 0
  %ehsel.1 = extractvalue { i8*, i32 } %lp1, 1
  br label %ehdispatch

ehdispatch:
  %ehptr.2 = phi i8* [ %ehptr.0, %lpad ], [ %ehptr.1, %lpad.1 ]
  %ehsel.2 = phi i32 [ %ehsel.0, %lpad ], [ %ehsel.1, %lpad.1 ]
  %mysel = call i32 @llvm.eh.typeid.for(i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@doit@@" to i8*))
  %matches = icmp eq i32 %ehsel.2, %mysel
  br i1 %matches, label %__except, label %eh.resume

__except:                                         ; preds = %lpad, %lpad.1
  %t4 = ptrtoint i8* %ehptr.2 to i64
  %t5 = trunc i64 %t4 to i32
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @str, i64 0, i64 0), i32 %t5)
  br label %__try.cont

__try.cont:                                       ; preds = %invoke.cont, %__except
  call void @finally(i1 false)
  ret void

eh.resume:
  %ehvals0 = insertvalue { i8*, i32 } undef, i8* %ehptr.2, 0
  %ehvals = insertvalue { i8*, i32 } %ehvals0, i32 %ehsel.2, 1
  resume { i8*, i32 } %ehvals
}

define internal i32 @"\01?filt$0@0@doit@@"(i8* %exception_pointers, i8* %frame_pointer) #1 {
entry:
  %0 = bitcast i8* %exception_pointers to { i32*, i8* }*
  %1 = getelementptr inbounds { i32*, i8* }, { i32*, i8* }* %0, i32 0, i32 0
  %2 = load i32*, i32** %1
  %3 = load i32, i32* %2
  %cmp = icmp eq i32 %3, -1073741819
  %4 = zext i1 %cmp to i32
  ret i32 %4
}

; CHECK-LABEL: define void @doit()
; CHECK: %lp0 = landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: catch i8*
; CHECK-NEXT: call i8* (...) @llvm.eh.actions({{.*}})
; CHECK-NEXT: indirectbr i8* %{{[^,]*}}, [label %__except]
;
; CHECK: %lp1 = landingpad { i8*, i32 }
; CHECK-NEXT: catch i8*
; CHECK-NEXT: call i8* (...) @llvm.eh.actions({{.*}})
; CHECK-NEXT: indirectbr i8* %{{[^,]*}}, [label %__except]
;
; CHECK: __except:
; CHECK: call i32 @llvm.eh.exceptioncode()
; CHECK: call i32 (i8*, ...) @printf
