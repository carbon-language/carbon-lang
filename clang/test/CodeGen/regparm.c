// RUN: clang-cc -triple i386-unknown-unknown %s -emit-llvm -o - | grep inreg | count 2

#define FASTCALL __attribute__((regparm(2)))

typedef struct {
  int aaa;
  double bbbb;
  int ccc[200];
} foo;

static void FASTCALL
reduced(char b, double c, foo* d, double e, int f)
{
}

int
main(void) {
	reduced(0, 0.0, 0, 0.0, 0);
}
