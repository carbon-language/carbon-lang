; RUN: llc < %s -mtriple=arm-apple-darwin9 -march=arm | FileCheck %s

%struct.A = type { i32* }

define void @"\01-[MyFunction Name:]"() {
entry:
  %save_filt.1 = alloca i32                       ; <i32*> [#uses=2]
  %save_eptr.0 = alloca i8*                       ; <i8**> [#uses=2]
  %a = alloca %struct.A                           ; <%struct.A*> [#uses=3]
  %eh_exception = alloca i8*                      ; <i8**> [#uses=5]
  %eh_selector = alloca i32                       ; <i32*> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call  void @_ZN1AC1Ev(%struct.A* %a)
  invoke  void @_Z3barv()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %entry
  call  void @_ZN1AD1Ev(%struct.A* %a) nounwind
  br label %return

bb:                                               ; preds = %ppad
  %eh_select = load i32* %eh_selector             ; <i32> [#uses=1]
  store i32 %eh_select, i32* %save_filt.1, align 4
  %eh_value = load i8** %eh_exception             ; <i8*> [#uses=1]
  store i8* %eh_value, i8** %save_eptr.0, align 4
  call  void @_ZN1AD1Ev(%struct.A* %a) nounwind
  %0 = load i8** %save_eptr.0, align 4            ; <i8*> [#uses=1]
  store i8* %0, i8** %eh_exception, align 4
  %1 = load i32* %save_filt.1, align 4            ; <i32> [#uses=1]
  store i32 %1, i32* %eh_selector, align 4
  br label %Unwind

return:                                           ; preds = %invcont
  ret void

lpad:                                             ; preds = %entry
  %eh_ptr = call i8* @llvm.eh.exception()         ; <i8*> [#uses=1]
  store i8* %eh_ptr, i8** %eh_exception
  %eh_ptr1 = load i8** %eh_exception              ; <i8*> [#uses=1]
  %eh_select2 = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr1, i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*), i32 0) ; <i32> [#uses=1]
  store i32 %eh_select2, i32* %eh_selector
  br label %ppad

ppad:                                             ; preds = %lpad
  br label %bb

Unwind:                                           ; preds = %bb
  %eh_ptr3 = load i8** %eh_exception              ; <i8*> [#uses=1]
  call  void @_Unwind_SjLj_Resume(i8* %eh_ptr3)
  unreachable
}

define linkonce_odr void @_ZN1AC1Ev(%struct.A* %this) {
entry:
  %this_addr = alloca %struct.A*                  ; <%struct.A**> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store %struct.A* %this, %struct.A** %this_addr
  %0 = call  i8* @_Znwm(i32 4)         ; <i8*> [#uses=1]
  %1 = bitcast i8* %0 to i32*                     ; <i32*> [#uses=1]
  %2 = load %struct.A** %this_addr, align 4       ; <%struct.A*> [#uses=1]
  %3 = getelementptr inbounds %struct.A* %2, i32 0, i32 0 ; <i32**> [#uses=1]
  store i32* %1, i32** %3, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare i8* @_Znwm(i32)

define linkonce_odr void @_ZN1AD1Ev(%struct.A* %this) nounwind {
entry:
  %this_addr = alloca %struct.A*                  ; <%struct.A**> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store %struct.A* %this, %struct.A** %this_addr
  %0 = load %struct.A** %this_addr, align 4       ; <%struct.A*> [#uses=1]
  %1 = getelementptr inbounds %struct.A* %0, i32 0, i32 0 ; <i32**> [#uses=1]
  %2 = load i32** %1, align 4                     ; <i32*> [#uses=1]
  %3 = bitcast i32* %2 to i8*                     ; <i8*> [#uses=1]
  call  void @_ZdlPv(i8* %3) nounwind
  br label %bb

bb:                                               ; preds = %entry
  br label %return

return:                                           ; preds = %bb
  ret void
}
;CHECK: L_LSDA_0:

declare void @_ZdlPv(i8*) nounwind

declare void @_Z3barv()

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare i32 @llvm.eh.typeid.for.i32(i8*) nounwind

declare i32 @__gxx_personality_sj0(...)

declare void @_Unwind_SjLj_Resume(i8*)
