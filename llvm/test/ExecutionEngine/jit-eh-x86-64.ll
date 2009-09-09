; RUN: llvm-as %s -o %t.bc
; RUN: lli -march=x86-64 %t.bc > /dev/null
; PR4865

%struct.__fundamental_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { i8*, i8* }

@_ZTIi = external constant %struct.__fundamental_type_info_pseudo ; <%struct.__fundamental_type_info_pseudo*> [#uses=1]
@.str = private constant [3 x i8] c"ok\00", align 1 ; <[3 x i8]*> [#uses=1]

define i32 @main() ssp {
entry:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %save_filt.5 = alloca i32                       ; <i32*> [#uses=2]
  %save_eptr.4 = alloca i8*                       ; <i8**> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %eh_exception = alloca i8*                      ; <i8**> [#uses=10]
  %eh_selector = alloca i32                       ; <i32*> [#uses=5]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  invoke void @_ZL3foov() ssp
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %entry
  br label %bb6

bb:                                               ; preds = %ppad
  %eh_value = load i8** %eh_exception             ; <i8*> [#uses=1]
  %1 = call i8* @__cxa_begin_catch(i8* %eh_value) nounwind ; <i8*> [#uses=0]
  %2 = invoke i32 @puts(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0))
          to label %invcont1 unwind label %lpad10 ; <i32> [#uses=0]

invcont1:                                         ; preds = %bb
  call void @__cxa_end_catch()
  br label %bb6

bb2:                                              ; preds = %ppad18
  %eh_select = load i32* %eh_selector             ; <i32> [#uses=1]
  store i32 %eh_select, i32* %save_filt.5, align 4
  %eh_value3 = load i8** %eh_exception            ; <i8*> [#uses=1]
  store i8* %eh_value3, i8** %save_eptr.4, align 4
  invoke void @__cxa_end_catch()
          to label %invcont4 unwind label %lpad14

invcont4:                                         ; preds = %bb2
  %3 = load i8** %save_eptr.4, align 4            ; <i8*> [#uses=1]
  store i8* %3, i8** %eh_exception, align 4
  %4 = load i32* %save_filt.5, align 4            ; <i32> [#uses=1]
  store i32 %4, i32* %eh_selector, align 4
  br label %Unwind

bb5:                                              ; preds = %ppad19
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable

bb6:                                              ; preds = %invcont1, %invcont
  store i32 0, i32* %0, align 4
  %5 = load i32* %0, align 4                      ; <i32> [#uses=1]
  store i32 %5, i32* %retval, align 4
  br label %return

return:                                           ; preds = %bb6
  %retval7 = load i32* %retval                    ; <i32> [#uses=1]
  ret i32 %retval7

lpad:                                             ; preds = %entry
  %eh_ptr = call i8* @llvm.eh.exception()         ; <i8*> [#uses=1]
  store i8* %eh_ptr, i8** %eh_exception
  %eh_ptr8 = load i8** %eh_exception              ; <i8*> [#uses=1]
  %eh_select9 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr8, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null) ; <i32> [#uses=1]
  store i32 %eh_select9, i32* %eh_selector
  br label %ppad

lpad10:                                           ; preds = %bb
  %eh_ptr11 = call i8* @llvm.eh.exception()       ; <i8*> [#uses=1]
  store i8* %eh_ptr11, i8** %eh_exception
  %eh_ptr12 = load i8** %eh_exception             ; <i8*> [#uses=1]
  %eh_select13 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr12, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null) ; <i32> [#uses=1]
  store i32 %eh_select13, i32* %eh_selector
  br label %ppad18

lpad14:                                           ; preds = %bb2
  %eh_ptr15 = call i8* @llvm.eh.exception()       ; <i8*> [#uses=1]
  store i8* %eh_ptr15, i8** %eh_exception
  %eh_ptr16 = load i8** %eh_exception             ; <i8*> [#uses=1]
  %eh_select17 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr16, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1) ; <i32> [#uses=1]
  store i32 %eh_select17, i32* %eh_selector
  br label %ppad19

ppad:                                             ; preds = %lpad
  br label %bb

ppad18:                                           ; preds = %lpad10
  br label %bb2

ppad19:                                           ; preds = %lpad14
  br label %bb5

Unwind:                                           ; preds = %invcont4
  %eh_ptr20 = load i8** %eh_exception             ; <i8*> [#uses=1]
  call void @_Unwind_Resume_or_Rethrow(i8* %eh_ptr20)
  unreachable
}

define internal void @_ZL3foov() ssp {
entry:
  %0 = alloca i8*                                 ; <i8**> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %1 = call i8* @__cxa_allocate_exception(i32 4) nounwind ; <i8*> [#uses=1]
  store i8* %1, i8** %0, align 4
  %2 = load i8** %0, align 4                      ; <i8*> [#uses=1]
  %3 = bitcast i8* %2 to i32*                     ; <i32*> [#uses=1]
  store i32 42, i32* %3, align 4
  %4 = load i8** %0, align 4                      ; <i8*> [#uses=1]
  call void @__cxa_throw(i8* %4, i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIi to i8*), void (i8*)* null) noreturn
  unreachable

return:                                           ; No predecessors!
  ret void
}

declare i8* @__cxa_allocate_exception(i32) nounwind

declare void @__cxa_throw(i8*, i8*, void (i8*)*) noreturn

declare i8* @__cxa_begin_catch(i8*) nounwind

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare i32 @llvm.eh.typeid.for.i32(i8*) nounwind

declare i32 @puts(i8*)

declare void @__cxa_end_catch()

declare void @_ZSt9terminatev() noreturn nounwind

declare i32 @__gxx_personality_v0(...)

declare void @_Unwind_Resume_or_Rethrow(i8*)
