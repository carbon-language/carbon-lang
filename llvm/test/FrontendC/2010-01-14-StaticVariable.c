// This is a regression test on debug info to make sure that llvm emitted
// debug info does not crash gdb.
// RUN: %llvmgcc -S -O0 -g %s -o - | \
// RUN:    llc -disable-cfi --disable-fp-elim -o %t.s -O0 -relocation-model=pic
// RUN: %compile_c %t.s -o %t.o
// RUN: echo {quit\n} > %t.in 
// RUN: gdb -q -batch -n -x %t.in %t.o > /dev/null

int foo() {
	static int i = 42;
        return i;
}
