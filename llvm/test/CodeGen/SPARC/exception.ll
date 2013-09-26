; RUN: llc < %s -march=sparc | FileCheck %s


%struct.__fundamental_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { i8*, i8* }

@_ZTIi = external constant %struct.__fundamental_type_info_pseudo
@_ZTIf = external constant %struct.__fundamental_type_info_pseudo
@.cst = linker_private unnamed_addr constant [12 x i8] c"catched int\00", align 64
@.cst1 = linker_private unnamed_addr constant [14 x i8] c"catched float\00", align 64

; CHECK-LABEL: main:
; CHECK:       .cfi_startproc
; CHECK:       .cfi_def_cfa_register 30
; CHECK:       .cfi_window_save
; CHECK:       .cfi_register 15, 31

; CHECK:        call __cxa_throw
; CHECK:        call __cxa_throw

; CHECK:        call __cxa_begin_catch
; CHECK:        call __cxa_end_catch

; CHECK:        call __cxa_begin_catch
; CHECK:        call __cxa_end_catch

; CHECK:        .cfi_endproc

define i32 @main(i32 %argc, i8** nocapture readnone %argv) unnamed_addr #0 {
entry:
  %0 = icmp eq i32 %argc, 2
  %1 = tail call i8* @__cxa_allocate_exception(i32 4) #1
  br i1 %0, label %"3", label %"4"

"3":                                              ; preds = %entry
  %2 = bitcast i8* %1 to i32*
  store i32 0, i32* %2, align 4
  invoke void @__cxa_throw(i8* %1, i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIi to i8*), void (i8*)* null) #2
          to label %3 unwind label %"8"

; <label>:3                                       ; preds = %"3"
  unreachable

"4":                                              ; preds = %entry
  %4 = bitcast i8* %1 to float*
  store float 1.000000e+00, float* %4, align 4


  invoke void @__cxa_throw(i8* %1, i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIf to i8*), void (i8*)* null) #2
          to label %5 unwind label %"8"

; <label>:5                                       ; preds = %"4"
  unreachable

"5":                                              ; preds = %"13", %"11"
  %6 = phi i32 [ 2, %"13" ], [ 0, %"11" ]
  ret i32 %6

"8":                                              ; preds = %"4", %"3"
  %exc = landingpad { i8*, i32 } personality i32 (i32, i64, i8*, i8*)* @__gxx_personality_v0
          catch %struct.__fundamental_type_info_pseudo* @_ZTIi
          catch %struct.__fundamental_type_info_pseudo* @_ZTIf
  %exc_ptr12 = extractvalue { i8*, i32 } %exc, 0
  %filter13 = extractvalue { i8*, i32 } %exc, 1
  %typeid = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIi to i8*))
  %7 = icmp eq i32 %filter13, %typeid
  br i1 %7, label %"11", label %8

; <label>:8                                       ; preds = %"8"
  %typeid8 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIf to i8*))
  %9 = icmp eq i32 %filter13, %typeid8
  br i1 %9, label %"13", label %"9"

"9":                                              ; preds = %8
  resume { i8*, i32 } %exc

"11":                                             ; preds = %"8"
  %10 = tail call i8* @__cxa_begin_catch(i8* %exc_ptr12) #1
  %11 = tail call i32 @puts(i8* getelementptr inbounds ([12 x i8]* @.cst, i32 0, i32 0))
  tail call void @__cxa_end_catch() #1
  br label %"5"

"13":                                             ; preds = %8
  %12 = tail call i8* @__cxa_begin_catch(i8* %exc_ptr12) #1
  %13 = tail call i32 @puts(i8* getelementptr inbounds ([14 x i8]* @.cst1, i32 0, i32 0))
  tail call void @__cxa_end_catch() #1
  br label %"5"
}

; Function Attrs: nounwind
declare i8* @__cxa_allocate_exception(i32) #1

; Function Attrs: noreturn
declare void @__cxa_throw(i8*, i8*, void (i8*)*) #2

declare void @__cxa_end_catch()

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #3

; Function Attrs: nounwind
declare i8* @__cxa_begin_catch(i8*) #1

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) #1

declare i32 @__gxx_personality_v0(i32, i64, i8*, i8*)

attributes #0 = { "no-frame-pointer-elim-non-leaf"="false" }
attributes #1 = { nounwind }
attributes #2 = { noreturn }
attributes #3 = { nounwind readnone }
