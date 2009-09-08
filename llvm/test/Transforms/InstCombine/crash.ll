; RUN: opt < %s -instcombine | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
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
