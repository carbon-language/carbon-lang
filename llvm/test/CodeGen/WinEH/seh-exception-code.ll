; RUN: opt -winehprepare -S < %s | FileCheck %s

; WinEHPrepare was crashing during phi demotion.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

declare i32 @__C_specific_handler(...)

@str = linkonce_odr unnamed_addr constant [16 x i8] c"caught it! %lx\0A\00", align 1

; Function Attrs: nounwind uwtable
declare void @maycrash()

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...)

; Function Attrs: nounwind uwtable
define void @doit() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  invoke void @maycrash()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  invoke void @maycrash()
          to label %__try.cont unwind label %lpad.1

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  br label %__except

lpad.1:                                           ; preds = %invoke.cont, %lpad
  %2 = landingpad { i8*, i32 }
          catch i8* null
  %3 = extractvalue { i8*, i32 } %2, 0
  br label %__except

__except:                                         ; preds = %lpad, %lpad.1
  %exn.slot.0 = phi i8* [ %3, %lpad.1 ], [ %1, %lpad ]
  %4 = ptrtoint i8* %exn.slot.0 to i64
  %5 = trunc i64 %4 to i32
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @str, i64 0, i64 0), i32 %5)
  br label %__try.cont

__try.cont:                                       ; preds = %invoke.cont, %__except
  ret void
}

; CHECK-LABEL: define void @doit()
; CHECK: landingpad
; CHECK: indirectbr i8* %{{[^,]*}}, [label %[[except_split1:.*]]]
; CHECK: [[except_split1]]:
; CHECK: call i32 @llvm.eh.exceptioncode()
; CHECK: br label %__except
;
; CHECK: landingpad
; CHECK: indirectbr i8* %{{[^,]*}}, [label %[[except_split2:.*]]]
; CHECK: [[except_split2]]:
; CHECK: call i32 @llvm.eh.exceptioncode()
; CHECK: br label %__except
;
; CHECK: __except:
; CHECK: phi
; CHECK: call i32 (i8*, ...) @printf
