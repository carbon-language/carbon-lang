; RUN: llc < %s -march=x86-64 -disable-mmx | grep punpcklwd | count 2

define void @foo() nounwind {
  %cti69 = trunc <8 x i32> undef to <8 x i16>     ; <<8 x i16>> [#uses=1]
  store <8 x i16> %cti69, <8 x i16>* undef
  ret void
}

define void @bar() nounwind {
  %cti44 = trunc <4 x i32> undef to <4 x i16>     ; <<4 x i16>> [#uses=1]
  store <4 x i16> %cti44, <4 x i16>* undef
  ret void
}
