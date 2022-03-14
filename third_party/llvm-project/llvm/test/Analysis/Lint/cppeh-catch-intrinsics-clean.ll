; RUN: opt -lint -disable-output < %s

; This test is meant to prove that the verifier does not report errors for correct
; use of the llvm.eh.begincatch and llvm.eh.endcatch intrinsics.

target triple = "x86_64-pc-windows-msvc"

declare void @llvm.eh.begincatch(i8*, i8*)

declare void @llvm.eh.endcatch()

@_ZTIi = external constant i8*

; Function Attrs: uwtable
define void @test_ref_clean() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
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
  call void @llvm.eh.endcatch()
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret void

eh.resume:                                        ; preds = %catch.dispatch
  resume { i8*, i32 } %0
}

; Function Attrs: uwtable
define void @test_ref_clean_multibranch() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @_Z9may_throwv()
          to label %invoke.cont unwind label %lpad1

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %exn = extractvalue { i8*, i32 } %0, 0
  %sel = extractvalue { i8*, i32 } %0, 1
  %1 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %sel, %1
  br i1 %matches, label %catch, label %eh.resume

  invoke void @_Z9may_throwv()
          to label %try.cont unwind label %lpad

lpad1:                                            ; preds = %entry
  %l1.0 = landingpad { i8*, i32 }
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
  call void @llvm.eh.endcatch()
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

