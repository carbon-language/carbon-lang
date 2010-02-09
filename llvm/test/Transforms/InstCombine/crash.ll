; RUN: opt < %s -instcombine -S
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128:n8:16:32"
target triple = "i386-apple-darwin10.0"

define i32 @test0(i8 %tmp2) ssp {
entry:
  %tmp3 = zext i8 %tmp2 to i32
  %tmp8 = lshr i32 %tmp3, 6 
  %tmp9 = lshr i32 %tmp3, 7 
  %tmp10 = xor i32 %tmp9, 67108858
  %tmp11 = xor i32 %tmp10, %tmp8 
  %tmp12 = xor i32 %tmp11, 0     
  ret i32 %tmp12
}

; PR4905
define <2 x i64> @test1(<2 x i64> %x, <2 x i64> %y) nounwind {
entry:
  %conv.i94 = bitcast <2 x i64> %y to <4 x i32>   ; <<4 x i32>> [#uses=1]
  %sub.i97 = sub <4 x i32> %conv.i94, undef       ; <<4 x i32>> [#uses=1]
  %conv3.i98 = bitcast <4 x i32> %sub.i97 to <2 x i64> ; <<2 x i64>> [#uses=2]
  %conv2.i86 = bitcast <2 x i64> %conv3.i98 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %cmp.i87 = icmp sgt <4 x i32> undef, %conv2.i86 ; <<4 x i1>> [#uses=1]
  %sext.i88 = sext <4 x i1> %cmp.i87 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %conv3.i89 = bitcast <4 x i32> %sext.i88 to <2 x i64> ; <<2 x i64>> [#uses=1]
  %and.i = and <2 x i64> %conv3.i89, %conv3.i98   ; <<2 x i64>> [#uses=1]
  %or.i = or <2 x i64> zeroinitializer, %and.i    ; <<2 x i64>> [#uses=1]
  %conv2.i43 = bitcast <2 x i64> %or.i to <4 x i32> ; <<4 x i32>> [#uses=1]
  %sub.i = sub <4 x i32> zeroinitializer, %conv2.i43 ; <<4 x i32>> [#uses=1]
  %conv3.i44 = bitcast <4 x i32> %sub.i to <2 x i64> ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %conv3.i44
}


; PR4908
define void @test2(<1 x i16>* nocapture %b, i32* nocapture %c) nounwind ssp {
entry:
  %arrayidx = getelementptr inbounds <1 x i16>* %b, i64 undef ; <<1 x i16>*>
  %tmp2 = load <1 x i16>* %arrayidx               ; <<1 x i16>> [#uses=1]
  %tmp6 = bitcast <1 x i16> %tmp2 to i16          ; <i16> [#uses=1]
  %tmp7 = zext i16 %tmp6 to i32                   ; <i32> [#uses=1]
  %ins = or i32 0, %tmp7                          ; <i32> [#uses=1]
  %arrayidx20 = getelementptr inbounds i32* %c, i64 undef ; <i32*> [#uses=1]
  store i32 %ins, i32* %arrayidx20
  ret void
}

; PR5262
@tmp2 = global i64 0                              ; <i64*> [#uses=1]

declare void @use(i64) nounwind

define void @foo(i1) nounwind align 2 {
; <label>:1
  br i1 %0, label %2, label %3

; <label>:2                                       ; preds = %1
  br label %3

; <label>:3                                       ; preds = %2, %1
  %4 = phi i8 [ 1, %2 ], [ 0, %1 ]                ; <i8> [#uses=1]
  %5 = icmp eq i8 %4, 0                           ; <i1> [#uses=1]
  %6 = load i64* @tmp2, align 8                   ; <i64> [#uses=1]
  %7 = select i1 %5, i64 0, i64 %6                ; <i64> [#uses=1]
  br label %8

; <label>:8                                       ; preds = %3
  call void @use(i64 %7)
  ret void
}

%t0 = type { i32, i32 }
%t1 = type { i32, i32, i32, i32, i32* }

declare %t0* @bar2(i64)

define void @bar3(i1, i1) nounwind align 2 {
; <label>:2
  br i1 %1, label %10, label %3

; <label>:3                                       ; preds = %2
  %4 = getelementptr inbounds %t0* null, i64 0, i32 1 ; <i32*> [#uses=0]
  %5 = getelementptr inbounds %t1* null, i64 0, i32 4 ; <i32**> [#uses=1]
  %6 = load i32** %5, align 8                     ; <i32*> [#uses=1]
  %7 = icmp ne i32* %6, null                      ; <i1> [#uses=1]
  %8 = zext i1 %7 to i32                          ; <i32> [#uses=1]
  %9 = add i32 %8, 0                              ; <i32> [#uses=1]
  br label %10

; <label>:10                                      ; preds = %3, %2
  %11 = phi i32 [ %9, %3 ], [ 0, %2 ]             ; <i32> [#uses=1]
  br i1 %1, label %12, label %13

; <label>:12                                      ; preds = %10
  br label %13

; <label>:13                                      ; preds = %12, %10
  %14 = zext i32 %11 to i64                       ; <i64> [#uses=1]
  %15 = tail call %t0* @bar2(i64 %14) nounwind      ; <%0*> [#uses=0]
  ret void
}




; PR5262
; Make sure the PHI node gets put in a place where all of its operands dominate
; it.
define i64 @test4(i1 %c, i64* %P) nounwind align 2 {
BB0:
  br i1 %c, label %BB1, label %BB2

BB1:
  br label %BB2

BB2:
  %v5_ = phi i1 [ true, %BB0], [false, %BB1]
  %v6 = load i64* %P
  br label %l8

l8:
  br label %l10
  
l10:
  %v11 = select i1 %v5_, i64 0, i64 %v6
  ret i64 %v11
}

; PR5471
define arm_apcscc i32 @test5a() {
       ret i32 0
}

define arm_apcscc void @test5() {
       store i1 true, i1* undef
       %1 = invoke i32 @test5a() to label %exit unwind label %exit
exit:
       ret void
}


; PR5673

@test6g = external global i32*  

define arm_aapcs_vfpcc i32 @test6(i32 %argc, i8** %argv) nounwind {
entry:
  store i32* getelementptr (i32* bitcast (i32 (i32, i8**)* @test6 to i32*), i32 -2048), i32** @test6g, align 4
  unreachable
}


; PR5827

%class.RuleBasedBreakIterator = type { i64 ()* }
%class.UStack = type { i8** }

define i32 @_ZN22RuleBasedBreakIterator15checkDictionaryEi(%class.RuleBasedBreakIterator* %this, i32 %x) align 2 {
entry:
  %breaks = alloca %class.UStack, align 4         ; <%class.UStack*> [#uses=3]
  call void @_ZN6UStackC1Ei(%class.UStack* %breaks, i32 0)
  %tobool = icmp ne i32 %x, 0                     ; <i1> [#uses=1]
  br i1 %tobool, label %cond.end, label %cond.false

terminate.handler:                                ; preds = %ehcleanup
  %exc = call i8* @llvm.eh.exception()            ; <i8*> [#uses=1]
  %0 = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exc, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i32 1) ; <i32> [#uses=0]
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable

ehcleanup:                                        ; preds = %cond.false
  %exc1 = call i8* @llvm.eh.exception()           ; <i8*> [#uses=2]
  %1 = call i32 (i8*, i8*, ...)* @llvm.eh.selector(i8* %exc1, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null) ; <i32> [#uses=0]
  invoke void @_ZN6UStackD1Ev(%class.UStack* %breaks)
          to label %cont unwind label %terminate.handler

cont:                                             ; preds = %ehcleanup
  call void @_Unwind_Resume_or_Rethrow(i8* %exc1)
  unreachable

cond.false:                                       ; preds = %entry
  %tmp4 = getelementptr inbounds %class.RuleBasedBreakIterator* %this, i32 0, i32 0 ; <i64 ()**> [#uses=1]
  %tmp5 = load i64 ()** %tmp4                     ; <i64 ()*> [#uses=1]
  %call = invoke i64 %tmp5()
          to label %cond.end unwind label %ehcleanup ; <i64> [#uses=1]

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i64 [ 0, %entry ], [ %call, %cond.false ] ; <i64> [#uses=1]
  %conv = trunc i64 %cond to i32                  ; <i32> [#uses=1]
  call void @_ZN6UStackD1Ev(%class.UStack* %breaks)
  ret i32 %conv
}

declare void @_ZN6UStackC1Ei(%class.UStack*, i32)

declare void @_ZN6UStackD1Ev(%class.UStack*)

declare i32 @__gxx_personality_v0(...)

declare i8* @llvm.eh.exception() nounwind readonly

declare i32 @llvm.eh.selector(i8*, i8*, ...) nounwind

declare void @_ZSt9terminatev()

declare void @_Unwind_Resume_or_Rethrow(i8*)



; rdar://7590304
define i8* @test10(i8* %self, i8* %tmp3) {
entry:
  store i1 true, i1* undef
  store i1 true, i1* undef
  invoke arm_apcscc void @test10a()
          to label %invoke.cont unwind label %try.handler ; <i8*> [#uses=0]

invoke.cont:                                      ; preds = %entry
  unreachable

try.handler:                                      ; preds = %entry
  ret i8* %self
}

define void @test10a() {
  ret void
}


; PR6193
define i32 @test11(i32 %aMaskWidth, i8 %aStride) nounwind {
entry:
  %conv41 = sext i8 %aStride to i32
  %neg = xor i32 %conv41, -1
  %and42 = and i32 %aMaskWidth, %neg
  %and47 = and i32 130, %conv41
  %or = or i32 %and42, %and47
  ret i32 %or
}
