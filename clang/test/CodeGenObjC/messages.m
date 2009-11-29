// RUN: clang-cc -emit-llvm -o %t %s
// RUN: grep "objc_msgSend" %t | count 6
// RUN: clang-cc -fgnu-runtime -emit-llvm -o %t %s
// RUN: grep "objc_msg_lookup" %t | count 6
// RUN: clang-cc -fgnu-runtime -fobjc-nonfragile-abi -emit-llvm -o %t %s
// RUN: grep "objc_msg_lookup_sender" %t | count 6

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
