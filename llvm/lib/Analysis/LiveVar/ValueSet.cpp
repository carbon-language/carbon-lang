
#include "llvm/Analysis/LiveVar/ValueSet.h"
#include "llvm/ConstantVals.h"
#include <iostream>
using std::cerr;
using std::endl;
using std::pair;
using std::hash_set;

void printValue( const Value *const v)  // func to print a Value 
{
  if (v->hasName())
    cerr << v << "(" << ((*v).getName()) << ") ";
  else if (Constant *C = dyn_cast<Constant>(v))
    cerr << v << "(" << C->getStrValue() << ") ";
  else
    cerr << v  << " ";
}


//---------------- Method implementations --------------------------
                                             // for performing two set unions
bool ValueSet::setUnion( const ValueSet *const set1) {   
  const_iterator set1it;
  pair<iterator, bool> result;
  bool changed = false;

  for( set1it = set1->begin() ; set1it != set1->end(); ++set1it) {  
                                             // for all all elements in set1
    result = insert( *set1it );              // insert to this set
      if( result.second == true) changed = true;
  }

  return changed;
}


                                             // for performing set difference
void ValueSet::setDifference( const ValueSet *const set1, 
			      const ValueSet *const set2) { 

  const_iterator set1it, set2it;
  for( set1it = set1->begin() ; set1it != set1->end(); ++set1it) {  
                                             // for all elements in set1
    iterator set2it = set2->find( *set1it ); // find wether the elem is in set2
    if( set2it == set2->end() )              // if the element is not in set2
      insert( *set1it );                     // insert to this set
  }
}


                                        // for performing set subtraction
void ValueSet::setSubtract( const ValueSet *const set1) { 
  const_iterator set1it;
  for( set1it = set1->begin() ; set1it != set1->end(); ++set1it)  
                                        // for all elements in set1
    erase( *set1it );                   // erase that element from this set
}




void ValueSet::printSet()  const {     // for printing a live variable set
  for_each(begin(), end(), printValue);
}
