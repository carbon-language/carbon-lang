; RUN: llvm-as < %s | opt -std-compile-opts | llc -enable-eh
;; Formerly crashed, see PR 1508
; ModuleID = '5550437.c'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-apple-darwin8"
  %struct.Range = type { i64, i64 }

define void @Bork(i64 %range.0.0, i64 %range.0.1, i64 %size) {
entry:
  %range_addr = alloca %struct.Range    ; <%struct.Range*> [#uses=5]
  %size_addr = alloca i64   ; <i64*> [#uses=2]
  %size.0 = alloca i64, align 8   ; <i64*> [#uses=6]
  %additionalKeys.5 = alloca i8**   ; <i8***> [#uses=2]
  %saved_stack.7 = alloca i8*   ; <i8**> [#uses=3]
  %eh_exception = alloca i8*    ; <i8**> [#uses=3]
  %eh_selector = alloca i64   ; <i64*> [#uses=1]
  %effectiveRange = alloca %struct.Range, align 8   ; <%struct.Range*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32    ; <i32> [#uses=0]
  %tmp = bitcast %struct.Range* %range_addr to { [2 x i64] }*   ; <{ [2 x i64] }*> [#uses=1]
  %tmp1 = getelementptr { [2 x i64] }* %tmp, i32 0, i32 0   ; <[2 x i64]*> [#uses=2]
  %tmp2 = getelementptr [2 x i64]* %tmp1, i32 0, i32 0    ; <i64*> [#uses=1]
  store i64 %range.0.0, i64* %tmp2
  %tmp3 = getelementptr [2 x i64]* %tmp1, i32 0, i32 1    ; <i64*> [#uses=1]
  store i64 %range.0.1, i64* %tmp3
  store i64 %size, i64* %size_addr
  %tmp4 = call i8* @llvm.stacksave( )   ; <i8*> [#uses=1]
  store i8* %tmp4, i8** %saved_stack.7, align 8
  %tmp5 = load i64* %size_addr, align 8   ; <i64> [#uses=1]
  store i64 %tmp5, i64* %size.0, align 8
  %tmp6 = load i64* %size.0, align 8    ; <i64> [#uses=1]
  %tmp7 = sub i64 %tmp6, 1    ; <i64> [#uses=0]
  %tmp8 = load i64* %size.0, align 8    ; <i64> [#uses=1]
  %tmp9 = mul i64 %tmp8, 64   ; <i64> [#uses=0]
  %tmp10 = load i64* %size.0, align 8   ; <i64> [#uses=1]
  %tmp11 = mul i64 %tmp10, 8    ; <i64> [#uses=0]
  %tmp12 = load i64* %size.0, align 8   ; <i64> [#uses=1]
  %tmp13 = mul i64 %tmp12, 64   ; <i64> [#uses=0]
  %tmp14 = load i64* %size.0, align 8   ; <i64> [#uses=1]
  %tmp15 = mul i64 %tmp14, 8    ; <i64> [#uses=1]
  %tmp1516 = trunc i64 %tmp15 to i32    ; <i32> [#uses=1]
  %tmp17 = alloca i8, i32 %tmp1516    ; <i8*> [#uses=1]
  %tmp1718 = bitcast i8* %tmp17 to i8**   ; <i8**> [#uses=1]
  store i8** %tmp1718, i8*** %additionalKeys.5, align 8
  %tmp19 = load i8*** %additionalKeys.5, align 8    ; <i8**> [#uses=1]
  invoke void @Foo( i8** %tmp19 )
      to label %invcont unwind label %unwind

unwind:   ; preds = %bb, %entry
  %eh_ptr = call i8* @llvm.eh.exception( )    ; <i8*> [#uses=1]
  store i8* %eh_ptr, i8** %eh_exception
  %eh_ptr20 = load i8** %eh_exception   ; <i8*> [#uses=1]
  %eh_select = call i64 (i8*, i8*, ...)* @llvm.eh.selector.i64( i8* %eh_ptr20, i8* bitcast (void ()* @__gxx_personality_v0 to i8*), i8* null )    ; <i64> [#uses=1]
  store i64 %eh_select, i64* %eh_selector
  br label %cleanup37

invcont:    ; preds = %entry
  br label %bb30

bb:   ; preds = %cond_true
  %tmp21 = getelementptr %struct.Range* %range_addr, i32 0, i32 0   ; <i64*> [#uses=1]
  %tmp22 = load i64* %tmp21, align 8    ; <i64> [#uses=1]
  invoke void @Bar( i64 %tmp22, %struct.Range* %effectiveRange )
      to label %invcont23 unwind label %unwind

invcont23:    ; preds = %bb
  %tmp24 = getelementptr %struct.Range* %range_addr, i32 0, i32 1   ; <i64*> [#uses=1]
  %tmp25 = load i64* %tmp24, align 8    ; <i64> [#uses=1]
  %tmp26 = getelementptr %struct.Range* %effectiveRange, i32 0, i32 1   ; <i64*> [#uses=1]
  %tmp27 = load i64* %tmp26, align 8    ; <i64> [#uses=1]
  %tmp28 = sub i64 %tmp25, %tmp27   ; <i64> [#uses=1]
  %tmp29 = getelementptr %struct.Range* %range_addr, i32 0, i32 1   ; <i64*> [#uses=1]
  store i64 %tmp28, i64* %tmp29, align 8
  br label %bb30

bb30:   ; preds = %invcont23, %invcont
  %tmp31 = getelementptr %struct.Range* %range_addr, i32 0, i32 1   ; <i64*> [#uses=1]
  %tmp32 = load i64* %tmp31, align 8    ; <i64> [#uses=1]
  %tmp33 = icmp ne i64 %tmp32, 0    ; <i1> [#uses=1]
  %tmp3334 = zext i1 %tmp33 to i8   ; <i8> [#uses=1]
  %toBool = icmp ne i8 %tmp3334, 0    ; <i1> [#uses=1]
  br i1 %toBool, label %cond_true, label %cond_false

cond_true:    ; preds = %bb30
  br label %bb

cond_false:   ; preds = %bb30
  br label %bb35

cond_next:    ; No predecessors!
  br label %bb35

bb35:   ; preds = %cond_next, %cond_false
  br label %cleanup

cleanup:    ; preds = %bb35
  %tmp36 = load i8** %saved_stack.7, align 8    ; <i8*> [#uses=1]
  call void @llvm.stackrestore( i8* %tmp36 )
  br label %finally

cleanup37:    ; preds = %unwind
  %tmp38 = load i8** %saved_stack.7, align 8    ; <i8*> [#uses=1]
  call void @llvm.stackrestore( i8* %tmp38 )
  br label %Unwind

finally:    ; preds = %cleanup
  br label %return

return:   ; preds = %finally
  ret void

Unwind:   ; preds = %cleanup37
  %eh_ptr39 = load i8** %eh_exception   ; <i8*> [#uses=1]
  call void @_Unwind_Resume( i8* %eh_ptr39 )
  unreachable
}

declare i8* @llvm.stacksave()

declare void @Foo(i8**)

declare i8* @llvm.eh.exception()

declare i64 @llvm.eh.selector.i64(i8*, i8*, ...)

declare i64 @llvm.eh.typeid.for.i64(i8*)

declare void @__gxx_personality_v0()

declare void @_Unwind_Resume(i8*)

declare void @Bar(i64, %struct.Range*)

declare void @llvm.stackrestore(i8*)
