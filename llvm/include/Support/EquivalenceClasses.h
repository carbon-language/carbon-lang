//===-- Support/EquivalenceClasses.h ----------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Generic implementation of equivalence classes and implementation of
// union-find algorithms A not-so-fancy implementation: 2 level tree i.e root
// and one more level Overhead of a union = size of the equivalence class being
// attached Overhead of a find = 1.
// 
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_EQUIVALENCECLASSES_H
#define SUPPORT_EQUIVALENCECLASSES_H

#include <map>
#include <set>
#include <vector>

namespace llvm {

template <class ElemTy>
class EquivalenceClasses {
  // Maps each element to the element that is the leader of its 
  // equivalence class.
  std::map<ElemTy, ElemTy> Elem2LeaderMap;
  
  // Maintains the set of leaders
  std::set<ElemTy> LeaderSet;

  // Caches the equivalence class for each leader
  std::map<ElemTy, std::set<ElemTy> > LeaderToEqClassMap;

  // Make Element2 the leader of the union of classes Element1 and Element2
  // Element1 and Element2 are presumed to be leaders of their respective
  // equivalence classes.
  void attach(ElemTy Element1, ElemTy Element2) {
    for (typename std::map<ElemTy, ElemTy>::iterator ElemI = 
	   Elem2LeaderMap.begin(), ElemE = Elem2LeaderMap.end(); 
	 ElemI != ElemE; ++ElemI) {
      if (ElemI->second == Element1)
	Elem2LeaderMap[ElemI->first] = Element2;
    }
  }

public:
  // If an element has not yet in any class, make it a separate new class.
  // Return the leader of the class containing the element.
  ElemTy addElement (ElemTy NewElement) {
    typename std::map<ElemTy, ElemTy>::iterator ElemI = 
      Elem2LeaderMap.find(NewElement);
    if (ElemI == Elem2LeaderMap.end()) {
      Elem2LeaderMap[NewElement] = NewElement;
      LeaderSet.insert(NewElement);
      return NewElement;
    }
    else
      return ElemI->second;
  }
  
  ElemTy findClass(ElemTy Element) const {
    typename std::map<ElemTy, ElemTy>::const_iterator I =
      Elem2LeaderMap.find(Element);
    return (I == Elem2LeaderMap.end())? (ElemTy) 0 : I->second;
  }

  /// Attach the set with Element1 to the set with Element2 adding Element1 and
  /// Element2 to the set of equivalence classes if they are not there already.
  /// Implication: Make Element1 the element in the smaller set.
  /// Take Leader[Element1] out of the set of leaders.
  void unionSetsWith(ElemTy Element1, ElemTy Element2) {
    // If either Element1 or Element2 does not already exist, include it
    const ElemTy& leader1 = addElement(Element1);
    const ElemTy& leader2 = addElement(Element2);
    assert(leader1 != (ElemTy) 0 && leader2 != (ElemTy) 0);
    if (leader1 != leader2) {
      attach(leader1, leader2);
      LeaderSet.erase(leader1);
    }
  }
  
  // Returns a vector containing all the elements in the equivalence class
  // including Element1
  const std::set<ElemTy> & getEqClass(ElemTy Element1) {
    assert(Elem2LeaderMap.find(Element1) != Elem2LeaderMap.end());
    const ElemTy classLeader = Elem2LeaderMap[Element1];
    
    std::set<ElemTy> & EqClass = LeaderToEqClassMap[classLeader];
    
    // If the EqClass vector is empty, it has not been computed yet: do it now
    if (EqClass.empty()) {
      for (typename std::map<ElemTy, ElemTy>::iterator
             ElemI = Elem2LeaderMap.begin(), ElemE = Elem2LeaderMap.end(); 
           ElemI != ElemE; ++ElemI)
        if (ElemI->second == classLeader)
          EqClass.insert(ElemI->first);
      assert(! EqClass.empty());        // must at least include the leader
    }
    
    return EqClass;
  }

        std::set<ElemTy>& getLeaderSet()       { return LeaderSet; }
  const std::set<ElemTy>& getLeaderSet() const { return LeaderSet; }

        std::map<ElemTy, ElemTy>& getLeaderMap()       { return Elem2LeaderMap;}
  const std::map<ElemTy, ElemTy>& getLeaderMap() const { return Elem2LeaderMap;}
};

} // End llvm namespace

#endif
