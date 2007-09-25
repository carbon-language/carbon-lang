//===--- PersistentMap.h - Peristent Map Data Structure ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines PersistentMap, a template class that implements a
// persistent map.  A persistent map is an immutable data structure that
// records a mapping from key to values.  "Insertions" and "deletions" 
// to a persistent map result in the construction of a new map.
//
// FIXME: There are efficient ways to implement such data structures.
//  For example, the persistent red-black tree implementation described
//  in Okasaki's book "Purely Functional Data Structures."  The current
//  implementation is inefficient but simple, and will be improved as needed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PERSISTENT_MAP
#define LLVM_CLANG_ANALYSIS_PERSISTENT_MAP

#include "clang/Analysis/Support/IntrusiveSPtr.h"
#include <vector>
#include <functional>

namespace clang {
  
template <typename KeyTy, typename ValTy,
          typename EqualKey = std::equal_to<KeyTy>,
          typename EqualVal = std::equal_to<ValTy> >
class PersistentMap : public RefCounted {
public:
  // Smart-pointer typedef.  All instances of PersistentMap should be
  // access via smart pointers.
  typedef IntrusiveSPtr<PersistentMap> Ptr;
  
  // Typedefs for iterators.
  typedef std::vector< std::pair<KeyTy,ValTy> >              KeyValuesTy;
  typedef typename KeyValuesTy::const_iterator               iterator;
  typedef typename KeyValuesTy::const_iterator               const_iterator;
  
  // STL-like interface for traversal/lookup.
  iterator begin() const { return KeyValues.begin(); }
  iterator end() const { return KeyValues.end(); }
  bool empty() const { return size() == 0; }
  
  iterator find(const KeyTy& K) const {
    EqualKey eq;    
    for (iterator I = begin(), E = end(); I!=E; ++I) { if (eq(*I,K)) return I; }    
    return end();
  }    

  // More user-friendly.
  bool contains(const KeyTy& K) const { return find(K) == end(); }
  unsigned size() const { return KeyValues.size(); }

  // Creation of an empty map.
  static Ptr create() { return Ptr(new PersistentMap()); }

  // Addition/Removal of elements to create new maps.
  Ptr add(const KeyTy& K, const ValTy& V) const {
    EqualKey eq_key;
    EqualKey eq_val;
    
    unsigned i = 0;
    for (; i < KeyValues.size(); ++i) {
      if (eq_key(KeyValues[i].first,K))
        if (eq_val(KeyValues[i].second,V)) { return Ptr(this); }
        else break;
    }

    PersistentMap* M = new PersistentMap(*this);
    
    if (i != KeyValues.size()) // Overwrite the old value.
      M->KeyValues[i].second = V;
    else // Key-Value not in the map.  Add it.
      M->KeyValues.push_back(std::make_pair<KeyTy,ValTy>(K,V));
    
    return Ptr(M);
  }
  
  Ptr remove(const KeyTy& K) const {
    unsigned i = 0;
    EqualKey eq_key;
      
    for(; i < KeyValues.size(); ++i)
      if (eq_key(KeyValues[i].first,K)) break;
    
    if (i == KeyValues.size())
      return Ptr(this);

    PersistentMap* M = new PersistentMap(*this);
    M->KeyValues[i] = M->KeyValues.back();
    M->KeyValues.pop_back();

    return Ptr(M);
  }
  
  // Do two maps have the same key-value pairs?
  bool operator==(PersistentMap& M) {
    if (&M == this) { return true; }
    
    EqualVal eq;
    
    for (iterator I = M.begin(), E = M.end(); I!=E; ++I) {
      iterator X = find(I->first);
      
      if (X == end() || !eq(I->second,X->second))
        return false;
    }
    
    for (iterator I = begin(), E = end(); I!=E; ++I) {
      iterator X = M.find(I->first);
      
      if (X == M.end() || !eq(I->second,X->second))
        return false;
    }
    
    return true;
  }
  
protected:
  PersistentMap() {}
  
  // Used by "add" and "remove".
  PersistentMap(const PersistentMap& M) : RefCounted(),KeyValues(M.KeyValues) {}
  
  virtual ~PersistentMap() {}
  
  KeyValuesTy KeyValues;
};
  
} // end namespace clang

#endif
