// RUN: clang-cc -fnext-runtime --emit-llvm -o %t %s

typedef struct {
  int x;
  int y;
  int z[10];
} MyPoint;

void f0(id a) {
  int i;
  MyPoint pt = { 1, 2};

  [a print0];
  [a print1: 10];
  [a print2: 10 and: "hello" and: 2.2];
  [a takeStruct: pt ];
  
  void *s = @selector(print0);
  for (i=0; i<2; ++i)
    [a performSelector:s];
}
