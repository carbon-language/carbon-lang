; RUN: llvm-upgrade < %s | llvm-as | llc | grep no_dead_strip

target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8.7.2"
%x = weak global int 0          ; <int*> [#uses=1]
%llvm.used = appending global [1 x sbyte*] [ sbyte* cast (int* %x to sbyte*) ]
