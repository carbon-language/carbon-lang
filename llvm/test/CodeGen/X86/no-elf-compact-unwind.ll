; RUN: llc < %s -mtriple x86_64-apple-macosx10.8.0 -disable-cfi | FileCheck -check-prefix=MACHO %s
; RUN: llc < %s -mtriple x86_64-unknown-linux -disable-cfi | FileCheck -check-prefix=ELF %s

; Make sure we don't generate a compact unwind for ELF.

; MACHO-LABEL: _Z3barv:
; MACHO:       __compact_unwind

; ELF-LABEL:   _Z3barv:
; ELF-NOT:     __compact_unwind

@_ZTIi = external constant i8*

define void @_Z3barv() uwtable {
entry:
  invoke void @_Z3foov()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %1 = extractvalue { i8*, i32 } %0, 1
  %2 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %1, %2
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = extractvalue { i8*, i32 } %0, 0
  %4 = tail call i8* @__cxa_begin_catch(i8* %3)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %catch
  ret void

eh.resume:                                        ; preds = %lpad
  resume { i8*, i32 } %0
}

declare void @_Z3foov()

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()
