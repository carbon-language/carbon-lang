//
// VALUESET.H - This header is OBSOLETE, do not use it for new code.
//
// FIXME: Eliminate this file.

#ifndef VALUE_SET_H
#define VALUE_SET_H

#include <set>
class Value;

// RAV - Used to print values in a form used by the register allocator.  
//
struct RAV {  // Register Allocator Value
  const Value &V;
  RAV(const Value *v) : V(*v) {}
  RAV(const Value &v) : V(v) {}
};
std::ostream &operator<<(std::ostream &out, RAV Val);


typedef std::set<const Value*> ValueSet;
void printSet(const ValueSet &S);

#endif
