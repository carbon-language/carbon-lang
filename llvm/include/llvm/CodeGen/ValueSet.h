#ifndef VALUE_SET_H
#define VALUE_SET_H

#include <set>
class Value;

// RAV - Used to print values in a form used by the register allocator.  
//
struct RAV {  // Register Allocator Value
  const Value *V;
  RAV(const Value *v) : V(v) {}
};
ostream &operator<<(ostream &out, RAV Val);


typedef std::set<const Value*> ValueSet;
void printSet(const ValueSet &S);


// set_union(A, B) - Compute A := A u B, return whether A changed.
//
template <class E>
bool set_union(std::set<E> &S1, const std::set<E> &S2) {   
  bool Changed = false;

  for (std::set<E>::const_iterator SI = S2.begin(), SE = S2.end();
       SI != SE; ++SI)
    if (S1.insert(*SI).second)
      Changed = true;

  return Changed;
}

// set_difference(A, B) - Return A - B
//
template <class E>
std::set<E> set_difference(const std::set<E> &S1, const std::set<E> &S2) {
  std::set<E> Result;
  for (std::set<E>::const_iterator SI = S1.begin(), SE = S1.end();
       SI != SE; ++SI)
    if (S2.find(*SI) == S2.end())       // if the element is not in set2
      Result.insert(*SI);
  return Result;
}

// set_subtract(A, B) - Compute A := A - B
//
template <class E>
void set_subtract(std::set<E> &S1, const std::set<E> &S2) { 
  for (std::set<E>::const_iterator SI = S2.begin() ; SI != S2.end(); ++SI)  
    S1.erase(*SI);
}

#endif
