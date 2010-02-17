// RUN: %llvmgcc -S -O0 -g %s -o - | \
// RUN:    llc --disable-fp-elim -o %t.s -O0 -relocation-model=pic
// RUN: %compile_c %t.s -o %t.o
// RUN: %link %t.o -o %t.exe
// RUN: echo {break 24\nrun\np loc\n} > %t.in 
// RN: gdb -q -batch -n -x %t.in %t.exe | tee %t.out | \
// RN:   grep {$1 = 1}

int g1 = 1;
int g2 = 2;

int  __attribute__((always_inline)) bar() {
  return g2 - g1; 
}
void foobar() {}

void foo(int s) {
  unsigned loc = 0;
  if (s) {
    loc = 1;
    foobar();
  } else {
    loc = bar();
    foobar();
  }
}

int main() {
	foo(0);
}
