/* Title:   ValueSet.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: Contains a mathematical set of Values. LiveVarSet is derived from
            this. Contains both class and method definitions.
*/

#ifndef VALUE_SET_H
#define VALUE_SET_H

class Value;
#include "Support/HashExtras.h"
#include <ext/hash_set>

//------------------- Class Definition for ValueSet --------------------------

void printValue( const Value *const v);  // func to print a Value 



class ValueSet : public std::hash_set<const Value *> {
 public:
  inline void add(const Value *const  val) 
    { assert( val ); insert(val);}      // for adding a live variable to set

  inline void remove(const Value *const  val) 
    { assert( val ); erase(val); }      // for removing a live var from set

  bool setUnion( const ValueSet *const set1);     // for performing set union
  void setSubtract( const ValueSet *const set1);  // for performing set diff

 
  void setDifference( const ValueSet *const set1, const ValueSet *const set2); 
 
  void printSet() const;                // for printing a live variable set
};






#endif
