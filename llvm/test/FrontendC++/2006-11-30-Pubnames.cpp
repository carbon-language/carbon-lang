// This is a regression test on debug info to make sure that we can access 
// qualified global names.
// RUN: %llvmgcc -S -O0 -g %s -o - | \
// RUN:   llc --disable-fp-elim -o %t.s -O0
// RUN: %compile_c %t.s -o %t.o
// RUN: %link %t.o -o %t.exe
// RUN: %llvmdsymutil %t.exe 
// RUN: echo {break main\nrun\np Pubnames::pubname} > %t.in
// RUN: gdb -q -batch -n -x %t.in %t.exe | tee %t.out | grep {\$1 = 10}
//
// XFAIL: alpha,arm,powerpc-apple-darwin
// FIXME: This doesn't work for PPC Darwin because we turned off debugging on
// that platform.

struct Pubnames {
  static int pubname;
};

int Pubnames::pubname = 10;

int main (int argc, char** argv) {
  Pubnames p;
  return 0;
}
