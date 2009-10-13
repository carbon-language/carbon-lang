// Check bitfields.
// RUN: %llvmgcc -S -O0 -g %s -o - | llvm-as | \
// RUN: llc --disable-fp-elim -o 2009-02-17-BitField-dbg.s
// RUN: %compile_c 2009-02-17-BitField-dbg.s -o 2009-02-17-BitField-dbg.o
// RUN: echo {ptype mystruct} > %t2
// RUN: gdb -q -batch -n -x %t2 2009-02-17-BitField-dbg.o | \
// RUN:   tee 2009-02-17-BitField-dbg.out | grep "int a : 4"
// XFAIL: *

struct {
  int  a:4;
  int  b:2;
} mystruct;

