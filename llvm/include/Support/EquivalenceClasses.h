//===-- Support/EquivalenceClasses.h -------------------------*- C++ -*--=//
// 
// Generic implementation of equivalence classes and implementation of 
// union-find algorithms
// A not-so-fancy implementation: 2 level tree i.e root and one more level
// Overhead of a union = size of the equivalence class being attached
// Overhead of a find = 1.
// 
//===------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_EQUIVALENCE_CLASSES_H
#define LLVM_SUPPORT_EQUIVALENCE_CLASSES_H

#include <map>
#include <set>
#include <vector>
using std::map;
using std::set;
using std::vector;

template <class ElemTy>
class EquivalenceClasses {
  // Maps each element to the element that is the leader of its 
  // equivalence class.
  map<ElemTy, ElemTy> Elem2ECLeaderMap;
  
  // Make Element2 the leader of the union of classes Element1 and Element2
  // Element1 and Element2 are presumed to be leaders of their respective
  // equivalence classes.
  void attach(ElemTy Element1, ElemTy Element2) {
    for (typename map<ElemTy, ElemTy>::iterator ElemI = 
	   Elem2ECLeaderMap.begin(), ElemE = Elem2ECLeaderMap.end(); 
	 ElemI != ElemE; ++ElemI) {
      if (ElemI->second == Element1)
	Elem2ECLeaderMap[ElemI->first] = Element2;
    }
  }

public:
  
  void addElement (ElemTy NewElement) {
    if (Elem2ECLeaderMap.find(NewElement) == Elem2ECLeaderMap.end())
      Elem2ECLeaderMap[NewElement] = NewElement;
  }
  
  ElemTy findClass(ElemTy Element) {
    if (Elem2ECLeaderMap.find(Element) == Elem2ECLeaderMap.end())
      return 0;
    else 
      return Elem2ECLeaderMap[Element];
  }

  /// Attach the set with Element1 to the set with Element2 adding Element1 and
  /// Element2 to the set of equivalence classes if they are not there already.
  /// Implication: Make Element1 the element in the smaller set.
  void unionElements(ElemTy Element1, ElemTy Element2) {
    // If either Element1 or Element2 does not already exist, include it
    if (Elem2ECLeaderMap.find(Element1) == Elem2ECLeaderMap.end())
      Elem2ECLeaderMap[Element1] = Element1;
    if (Elem2ECLeaderMap.find(Element2) == Elem2ECLeaderMap.end())
      Elem2ECLeaderMap[Element2] = Element2;

    attach(Elem2ECLeaderMap[Element1], Elem2ECLeaderMap[Element2]);
  }
  
  // Returns a vector containing all the elements in the equivalent class
  // including Element1
  vector<ElemTy> getEqClass(ElemTy Element1) {
    vector<ElemTy> EqClass;
    
    if (Elem2ECLeaderMap.find(EqClass) == Elem2ECLeaderMap.end())
      return EqClass;
    
    ElemTy classLeader = Elem2ECLeaderMap[Element1];

    for (typename map<ElemTy, ElemTy>::iterator ElemI = 
	   Elem2ECLeaderMap.begin(), ElemE = Elem2ECLeaderMap.end(); 
	 ElemI != ElemE; ++ElemI) {
      if (ElemI->second == classLeader)
	EqClass.push_back(ElemI->first);
    }
    
    return EqClass;
    
  }

  map<ElemTy, ElemTy> getLeaderMap() {
    return Elem2ECLeaderMap ;
  }
  
};

#endif
