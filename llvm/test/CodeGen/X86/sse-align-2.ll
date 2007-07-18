; RUN: llvm-as < %s | llc -march=x86-64 | grep movup | wc -l | grep 2

define <4 x float> @foo(<4 x float>* %p, <4 x float> %x)
{
  %t = load <4 x float>* %p, align 4
  %z = mul <4 x float> %t, %x
  ret <4 x float> %z
}
define <2 x double> @bar(<2 x double>* %p, <2 x double> %x)
{
  %t = load <2 x double>* %p, align 8
  %z = mul <2 x double> %t, %x
  ret <2 x double> %z
}
