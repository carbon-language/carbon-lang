// This regression test checks byval arguments' debug info.
// Radar 8367011
// RUN: %llvmgcc -S -O0 -g %s -o - | \
// RUN:    llc --disable-fp-elim -o %t.s -O0 -relocation-model=pic
// RUN: %compile_c %t.s -o %t.o
// RUN: %link %t.o -o %t.exe
// RUN: echo {break get\nrun\np missing_arg.b} > %t.in 
// RUN: gdb -q -batch -n -x %t.in %t.exe | tee %t.out | \
// RUN:   grep {1 = 4242}

// XTARGET: x86_64-apple-darwin

class EVT {
public:
  int a;
  int b;
  int c;
};

class VAL {
public:
  int x;
  int y;
};
void foo(EVT e);
EVT bar();

void get(int *i, unsigned dl, VAL v, VAL *p, unsigned n, EVT missing_arg) {
//CHECK: .ascii "missing_arg"
  EVT e = bar();
  if (dl == n)
    foo(missing_arg);
}


EVT bar() {
	EVT e;
	return e;
}

void foo(EVT e) {}

int main(){
	VAL v;
	EVT ma;
	ma.a = 1;
	ma.b = 4242;
	ma.c = 3;
	int i = 42;	
	get (&i, 1, v, &v, 2, ma);
	return 0;
}

