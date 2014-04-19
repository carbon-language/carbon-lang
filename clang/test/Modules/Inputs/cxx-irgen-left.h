#include "cxx-irgen-top.h"

S<int> s;

inline int instantiate_min() {
  return min(1, 2);
}

inline int instantiate_CtorInit(CtorInit<int> i = CtorInit<int>()) {
  return i.a;
}
