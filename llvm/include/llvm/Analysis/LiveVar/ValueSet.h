/* Title:   ValueSet.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Jun 30, 01
   Purpose: Contains a mathematical set of Values. LiveVarSet is derived from
            this. Contains both class and method definitions.
*/

#ifndef VALUE_SET_H
#define VALUE_SET_H

class Value;
#include <set>

//------------------- Class Definition for ValueSet --------------------------

void printValue( const Value *v);  // func to print a Value 

struct ValueSet : public std::set<const Value*> {
  bool setUnion( const ValueSet *const set1);     // for performing set union
  void setSubtract( const ValueSet *const set1);  // for performing set diff
 
  void setDifference( const ValueSet *const set1, const ValueSet *const set2); 
 
  void printSet() const;                // for printing a live variable set
};

#endif
