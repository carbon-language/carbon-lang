
// FIXME: Eliminate this file.

#include "llvm/CodeGen/ValueSet.h"
#include "llvm/Value.h"
#include <iostream>

std::ostream &operator<<(std::ostream &O, RAV V) { // func to print a Value 
  const Value &v = V.V;
  if (v.hasName())
    return O << (void*)&v << "(" << v.getName() << ") ";
  else if (isa<Constant>(v))
    return O << (void*)&v << "(" << v << ") ";
  else
    return O << (void*)&v << " ";
}

void printSet(const ValueSet &S) {
  for (ValueSet::const_iterator I = S.begin(), E = S.end(); I != E; ++I)
    std::cerr << RAV(*I);
}

