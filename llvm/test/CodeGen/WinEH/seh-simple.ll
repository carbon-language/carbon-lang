; RUN: opt -S -winehprepare -mtriple=x86_64-windows-msvc < %s \
; RUN: 		| FileCheck %s --check-prefix=CHECK --check-prefix=X64

; This test should also pass in 32-bit using _except_handler3.
; RUN: sed -e 's/__C_specific_handler/_except_handler3/' %s \
; RUN: 		| opt -S -winehprepare -mtriple=i686-windows-msvc \
; RUN: 		| FileCheck %s --check-prefix=CHECK --check-prefix=X86

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
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (i32 ()* @filt to i8*), i32 -1, i8* blockaddress(@simple_except_store, %__except))
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
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 1, i8* null, i32 -1, i8* blockaddress(@catch_all, %lpad.split))
; CHECK-NEXT: indirectbr {{.*}} [label %lpad.split]
;
; CHECK: lpad.split:
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
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 1, i8* bitcast (i32 ()* @filt to i8*), i32 -1, i8* blockaddress(@except_phi, %lpad.return_crit_edge))
; CHECK-NEXT: indirectbr {{.*}} [label %lpad.return_crit_edge]
;
; CHECK: lpad.return_crit_edge:
; CHECK: br label %return
;
; CHECK: return:
; CHECK-NEXT: %r = phi i32 [ 0, %entry ], [ 1, %lpad.return_crit_edge ]
; CHECK-NEXT: ret i32 %r

define i32 @lpad_phi() {
entry:
  invoke void @might_crash()
          to label %cont unwind label %lpad

cont:
  invoke void @might_crash()
          to label %return unwind label %lpad

lpad:
  %ncalls.1 = phi i32 [ 0, %entry ], [ 1, %cont ]
  %ehvals = landingpad { i8*, i32 } personality i32 (...)* @__C_specific_handler
          catch i32 ()* @filt
  %sel = extractvalue { i8*, i32 } %ehvals, 1
  %filt_sel = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i32 ()* @filt to i8*))
  %matches = icmp eq i32 %sel, %filt_sel
  br i1 %matches, label %return, label %eh.resume

return:
  %r = phi i32 [2, %cont], [%ncalls.1, %lpad]
  ret i32 %r

eh.resume:
  resume { i8*, i32 } %ehvals
}

; CHECK-LABEL: define i32 @lpad_phi()
; CHECK: alloca i32
; CHECK: store i32 0, i32*
; CHECK: invoke void @might_crash()
; CHECK: store i32 1, i32*
; CHECK: invoke void @might_crash()
; CHECK: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: catch i32 ()* @filt
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(i32 0, void ({{.*}})* @lpad_phi.cleanup, i32 1, i8* bitcast (i32 ()* @filt to i8*), i32 -1, i8* blockaddress(@lpad_phi, %lpad.return_crit_edge))
; CHECK-NEXT: indirectbr {{.*}} [label %lpad.return_crit_edge]
;
; CHECK: lpad.return_crit_edge:
; CHECK: load i32, i32*
; CHECK: br label %return
;
; CHECK: return:
; CHECK-NEXT: %r = phi i32 [ 2, %cont ], [ %{{.*}}, %lpad.return_crit_edge ]
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
; CHECK-NEXT: call i8* (...) @llvm.eh.actions(
; CHECK: i32 0, void ({{.*}})* @cleanup_and_except.cleanup,
; CHECK: i32 1, i8* bitcast (i32 ()* @filt to i8*), i32 -1, i8* blockaddress(@cleanup_and_except, %lpad.return_crit_edge))
; CHECK-NEXT: indirectbr {{.*}} [label %lpad.return_crit_edge]
;
; CHECK: lpad.return_crit_edge:
; CHECK: br label %return
;
; CHECK: return:
; CHECK-NEXT: %r = phi i32 [ 0, %entry ], [ 1, %lpad.return_crit_edge ]
; CHECK-NEXT: ret i32 %r

; FIXME: This cleanup is an artifact of bad demotion.
; X64-LABEL: define internal void @lpad_phi.cleanup(i8*, i8*)
; X86-LABEL: define internal void @lpad_phi.cleanup()
; X86: call i8* @llvm.frameaddress(i32 1)
; CHECK: call i8* @llvm.framerecover({{.*}})
; CHECK: load i32
; CHECK: store i32 %{{.*}}, i32*
