/* Title:   ValueSet.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: Contains a mathematical set of Values. LiveVarSet is derived from
            this. Contains both class and method definitions.
*/

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


//------------------- Class Definition for ValueSet --------------------------

struct ValueSet : public std::set<const Value*> {
  bool setUnion( const ValueSet *const set1);     // for performing set union
  void setSubtract( const ValueSet *const set1);  // for performing set diff
 
  void setDifference( const ValueSet *const set1, const ValueSet *const set2); 
 
  void printSet() const;                // for printing a live variable set
};

#endif
