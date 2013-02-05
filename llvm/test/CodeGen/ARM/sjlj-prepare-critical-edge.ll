; RUN: llc < %s -O1 -mtriple thumbv7-apple-ios6
; Just make sure no one tries to make the assumption that the normal edge of an
; invoke is never a critical edge.  Previously, this code would assert.

%struct.__CFString = type opaque

declare void @bar(%struct.__CFString*, %struct.__CFString*)

define noalias i8* @foo(i8* nocapture %inRefURL) noreturn ssp {
entry:
  %call = tail call %struct.__CFString* @bar3()
  %call2 = invoke i8* @bar2()
          to label %for.cond unwind label %lpad

for.cond:                                         ; preds = %entry, %for.cond
  invoke void @bar(%struct.__CFString* undef, %struct.__CFString* null)
          to label %for.cond unwind label %lpad5

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          cleanup
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  br label %ehcleanup

lpad5:                                            ; preds = %for.cond
  %3 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          cleanup
  %4 = extractvalue { i8*, i32 } %3, 0
  %5 = extractvalue { i8*, i32 } %3, 1
  invoke void @release(i8* %call2)
          to label %ehcleanup unwind label %terminate.lpad.i.i16

terminate.lpad.i.i16:                             ; preds = %lpad5
  %6 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          catch i8* null
  tail call void @terminatev() noreturn nounwind
  unreachable

ehcleanup:                                        ; preds = %lpad5, %lpad
  %exn.slot.0 = phi i8* [ %1, %lpad ], [ %4, %lpad5 ]
  %ehselector.slot.0 = phi i32 [ %2, %lpad ], [ %5, %lpad5 ]
  %7 = bitcast %struct.__CFString* %call to i8*
  invoke void @release(i8* %7)
          to label %_ZN5SmartIPK10__CFStringED1Ev.exit unwind label %terminate.lpad.i.i

terminate.lpad.i.i:                               ; preds = %ehcleanup
  %8 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*)
          catch i8* null
  tail call void @terminatev() noreturn nounwind
  unreachable

_ZN5SmartIPK10__CFStringED1Ev.exit:               ; preds = %ehcleanup
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val12 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val12
}

declare %struct.__CFString* @bar3()

declare i8* @bar2()

declare i32 @__gxx_personality_sj0(...)

declare void @release(i8*)

declare void @terminatev()
