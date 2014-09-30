#ifndef FIRSTHEADER
#define FIRSTHEADER

#include "SecondHeader.h" // Just a class which gets in the lazy deserialization chain

#include "stl_map.h"
#include "vector"
struct A {
   typedef std::map<int, int*>::iterator el;
};

struct B {
  ~B() {}
  std::vector<int> fvec; // Cannot replace with simple mockup
};

#endif
