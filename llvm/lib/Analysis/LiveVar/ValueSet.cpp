


#include "llvm/Analysis/LiveVar/ValueSet.h"
#include "llvm/ConstantVals.h"
#include <iostream>

std::ostream &operator<<(std::ostream &O, RAV V) { // func to print a Value 
  const Value *v = V.V;
  if (v->hasName())
    return O << v << "(" << v->getName() << ") ";
  else if (Constant *C = dyn_cast<Constant>(v))
    return O << v << "(" << C->getStrValue() << ") ";
  else
    return O << v  << " ";
}

void printSet(const ValueSet &S) {
  for (ValueSet::const_iterator I = S.begin(), E = S.end(); I != E; ++I)
    std::cerr << RAV(*I);
}

