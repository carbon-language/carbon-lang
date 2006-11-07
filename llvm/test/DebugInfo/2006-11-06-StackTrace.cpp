// This is a regression test on debug info to make sure that we can get a
// meaningful stack trace from a C++ program.
// RUN: %llvmgcc -S -O0 -g %s -o Output/%s.ll
// RUN: llvm-as < Output/%s.ll | llc -o Output/%s.s
// RUN: gcc -g Output/%s.s -o Output/%s.exe
// RUN: ( echo "break DeepStack::deepest"; echo "run 17" ; echo "where" ) | gdb Output/%s.exe | grep '#0  DeepStack::deepest (this=.*,x=33)'

#include <stdlib.h>

class DeepStack {
  int seedVal;
public:
  DeepStack(int seed) : seedVal(seed) {}

  int shallowest( int x ) { return shallower(x + 1); }
  int shallower ( int x ) { return shallow(x + 2); }
  int shallow   ( int x ) { return deep(x + 3); }
  int deep      ( int x ) { return deeper(x + 4); }
  int deeper    ( int x ) { return deepest(x + 6); }
  int deepest   ( int x ) { return x + 7; }

  int runit() { return shallowest(seedVal); }
};

int main ( int argc, char** argv) {

  DeepStack DS9( (argc > 1 ? atoi(argv[1]) : 0) );
  return DS9.runit();
}
