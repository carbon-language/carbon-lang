; RUN: llvm-as %s -o %t.bc
; RUN: lli -enable-eh %t.bc > /dev/null

%struct.__fundamental_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { i8*, i8* }

@_ZTIi = external constant %struct.__fundamental_type_info_pseudo ; <%struct.__fundamental_type_info_pseudo*> [#uses=1]
@.llvm.eh.catch.all.value = linkonce constant i8* null, section "llvm.metadata" ; <i8**> [#uses=1]

define i32 @main() optsize ssp {
entry:
  %0 = tail call i8* @__cxa_allocate_exception(i64 4) nounwind ; <i8*> [#uses=2]
  %1 = bitcast i8* %0 to i32*                     ; <i32*> [#uses=1]
  store i32 1, i32* %1, align 4
  invoke void @__cxa_throw(i8* %0, i8* bitcast (%struct.__fundamental_type_info_pseudo* @_ZTIi to i8*), void (i8*)* null) noreturn
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %entry
  unreachable

lpad:                                             ; preds = %entry
  %eh_ptr = tail call i8* @llvm.eh.exception()    ; <i8*> [#uses=2]
  %eh_select = tail call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8** @.llvm.eh.catch.all.value) ; <i32> [#uses=0]
  %2 = tail call i8* @__cxa_begin_catch(i8* %eh_ptr) nounwind ; <i8*> [#uses=0]
  tail call void @__cxa_end_catch()
  ret i32 0
}

declare i8* @__cxa_allocate_exception(i64) nounwind

declare void @__cxa_throw(i8*, i8*, void (i8*)*) noreturn

declare i8* @__cxa_begin_catch(i8*) nounwind

declare i8* @llvm.eh.exception() nounwind readonly

declare i32 @llvm.eh.selector(i8*, i8*, ...) nounwind

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)
