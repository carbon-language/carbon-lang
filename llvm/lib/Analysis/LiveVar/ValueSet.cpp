


#include "llvm/Analysis/LiveVar/ValueSet.h"
#include "llvm/ConstantVals.h"
#include <iostream>

ostream &operator<<(ostream &O, RAV V) { // func to print a Value 
  const Value *v = V.V;
  if (v->hasName())
    return O << v << "(" << v->getName() << ") ";
  else if (Constant *C = dyn_cast<Constant>(v))
    return O << v << "(" << C->getStrValue() << ") ";
  else
    return O << v  << " ";
}

bool ValueSet::setUnion( const ValueSet *S) {   
  bool Changed = false;

  for (const_iterator SI = S->begin(), SE = S->end(); SI != SE; ++SI)
    if (insert(*SI).second)
      Changed = true;

  return Changed;
}

void ValueSet::setDifference(const ValueSet *S1, const ValueSet *S2) {
  for (const_iterator SI = S1->begin(), SE = S1->end() ; SI != SE; ++SI)
    if (S2->find(*SI) == S2->end())       // if the element is not in set2
      insert(*SI);
}

void ValueSet::setSubtract(const ValueSet *S) { 
  for (const_iterator SI = S->begin() ; SI != S->end(); ++SI)  
    erase(*SI);
}

void ValueSet::printSet() const {
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    std::cerr << RAV(*I);
}
