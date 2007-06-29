; RUN: llvm-as < %s | llc -march=arm

define void @foo(<8 x float>* %f, <8 x float>* %g, <4 x i64>* %y)
{
  %h = load <8 x float>* %f
  %i = mul <8 x float> %h, <float 1.1, float 3.3, float 4.4, float 5.4, float 0.5, float 0.6, float 0.7, float 0.8>
  %m = bitcast <8 x float> %i to <4 x i64>
  %z = load <4 x i64>* %y
  %n = mul <4 x i64> %z, %m
  %p = bitcast <4 x i64> %n to <8 x float>
  store <8 x float> %p, <8 x float>* %g
  ret void
}
