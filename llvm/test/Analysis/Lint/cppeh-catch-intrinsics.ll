; RUN: opt -lint -disable-output < %s 2>&1 | FileCheck %s

; This test is meant to prove that the Verifier is able to identify a variety
; of errors with the llvm.eh.begincatch and llvm.eh.endcatch intrinsics.
; See cppeh-catch-intrinsics-clean for correct uses.

target triple = "x86_64-pc-windows-msvc"

declare void @llvm.eh.begincatch(i8*, i8*)

declare void @llvm.eh.endcatch()

@_ZTIi = external constant i8*

; Function Attrs: uwtable
define void @test_missing_endcatch() {
; CHECK: Some paths from llvm.eh.begincatch may not reach llvm.eh.endcatch
; CHECK-NEXT: call void @llvm.eh.begincatch(i8* %exn, i8* null)
entry:
  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  call void @llvm.eh.begincatch(i8* %exn, i8* null)
  call void @_Z10handle_intv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  resume { i8*, i32 } %0
}

; Function Attrs: uwtable
define void @test_missing_begincatch() {
; CHECK: llvm.eh.endcatch may be reachable without passing llvm.eh.begincatch
; CHECK-NEXT:  call void @llvm.eh.endcatch()
entry:
  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  call void @_Z10handle_intv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  resume { i8*, i32 } %0
}

; Function Attrs: uwtable
define void @test_multiple_begin() {
; CHECK: llvm.eh.begincatch may be called a second time before llvm.eh.endcatch
; CHECK-NEXT: call void @llvm.eh.begincatch(i8* %exn, i8* null)
; CHECK-NEXT: call void @llvm.eh.begincatch(i8* %exn, i8* null)
entry:
  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  call void @llvm.eh.begincatch(i8* %exn, i8* null)
  call void @_Z10handle_intv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.begincatch(i8* %exn, i8* null)
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  resume { i8*, i32 } %0
}

; Function Attrs: uwtable
define void @test_multiple_end() {
; CHECK: llvm.eh.endcatch may be called a second time after llvm.eh.begincatch
; CHECK-NEXT:  call void @llvm.eh.endcatch()
; CHECK-NEXT:  call void @llvm.eh.endcatch()
entry:
  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  call void @llvm.eh.begincatch(i8* %exn, i8* null)
  call void @_Z10handle_intv()
  call void @llvm.eh.endcatch()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  resume { i8*, i32 } %0
}


; Function Attrs: uwtable
define void @test_begincatch_without_lpad() {
; CHECK: llvm.eh.begincatch may be reachable without passing a landingpad
; CHECK-NEXT: call void @llvm.eh.begincatch(i8* %exn, i8* null)
entry:
  %exn = alloca i8
  call void @llvm.eh.begincatch(i8* %exn, i8* null)
  call void @_Z10handle_intv()
  br label %invoke.cont2

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void
}

; Function Attrs: uwtable
define void @test_branch_to_begincatch_with_no_lpad(i32 %fake.sel) {
; CHECK: llvm.eh.begincatch may be reachable without passing a landingpad
; CHECK-NEXT: call void @llvm.eh.begincatch(i8* %exn2, i8* null)
entry:
  %fake.exn = alloca i8
  invoke void @_Z9may_throwv()
          to label %catch unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

catch:                                            ; preds = %lpad, %entry
  %exn2 = phi i8* [%exn, %lpad], [%fake.exn, %entry]
  %sel2 = phi i32 [%sel, %lpad], [%fake.sel, %entry]
  call void @llvm.eh.begincatch(i8* %exn2, i8* null)
  call void @_Z10handle_intv()
  %matches1 = icmp eq i32 %sel2, 0
  br i1 %matches1, label %invoke.cont2, label %invoke.cont3

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

invoke.cont3:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %eh.resume

try.cont:                                         ; preds = %invoke.cont2
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  %lpad.val = insertvalue { i8*, i32 } undef, i32 0, 1
  resume { i8*, i32 } %lpad.val
}

; Function Attrs: uwtable
define void @test_branch_missing_endcatch() {
; CHECK: Some paths from llvm.eh.begincatch may not reach llvm.eh.endcatch
; CHECK-NEXT: call void @llvm.eh.begincatch(i8* %exn2, i8* null)
entry:
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad1

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad1:                                            ; preds = %entry
  %l1.0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*)
		  cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn1 = extractvalue { i8*, i32 } %l1.0, 0
  %sel1 = extractvalue { i8*, i32 } %l1.0, 1
  %l1.1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matchesl1 = icmp eq i32 %sel1, %l1.1
  br i1 %matchesl1, label %catch, label %eh.resume

catch:                                            ; preds = %lpad, %lpad1
  %exn2 = phi i8* [%exn, %lpad], [%exn1, %lpad1]
  %sel2 = phi i32 [%sel, %lpad], [%sel1, %lpad1]
  call void @llvm.eh.begincatch(i8* %exn2, i8* null)
  call void @_Z10handle_intv()
  %matches1 = icmp eq i32 %sel2, 0
  br i1 %matches1, label %invoke.cont2, label %invoke.cont3

invoke.cont2:                                     ; preds = %catch
  call void @llvm.eh.endcatch()
  br label %try.cont

invoke.cont3:                                     ; preds = %catch
  br label %eh.resume

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  %lpad.val = insertvalue { i8*, i32 } undef, i32 0, 1
  resume { i8*, i32 } %lpad.val
}

declare void @_Z9may_throwv()

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*)

declare void @_Z10handle_intv()

