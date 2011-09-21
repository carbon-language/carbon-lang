; RUN: llc < %s -march=x86-64 -mcpu=corei7 -promote-elements -mattr=+sse41

; Make sure we are not crashing on this code.

define void @load_4_i8(<4 x i8>* %k, <4 x i8>* %y, <4 x double>* %A1, <4 x double>* %A0)  {
   %A = load <4 x i8>* %k
   %B = load <4 x i8>* %y
   %C = load <4 x double>* %A0
   %D= load <4 x double>* %A1
   %M = icmp uge <4 x i8> %A, %B
   %T = select <4 x i1> %M, <4 x double> %C, <4 x double> %D
   store <4 x double> %T, <4 x double>* undef
   ret void
}

