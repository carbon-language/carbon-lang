; RUN: llvm-as < %s | llc -march=x86-64 | grep movap | wc -l | grep 2

define <4 x float> @foo(<4 x float>* %p)
{
  %t = load <4 x float>* %p
  ret <4 x float> %t
}
define <2 x double> @bar(<2 x double>* %p)
{
  %t = load <2 x double>* %p
  ret <2 x double> %t
}
