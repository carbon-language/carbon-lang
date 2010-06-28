; RUN: llc < %s -march=arm -mattr=+neon

define void @main() nounwind {
entry:
  store <2 x i64> undef, <2 x i64>* undef, align 16
  %0 = load <16 x i8>* undef, align 16            ; <<16 x i8>> [#uses=1]
  %1 = or <16 x i8> zeroinitializer, %0           ; <<16 x i8>> [#uses=1]
  store <16 x i8> %1, <16 x i8>* undef, align 16
  ret void
}
