//===--- ImmutableMap.h - Immutable (functional) map interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ImmutableMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IMMAP_H
#define LLVM_ADT_IMMAP_H

#include "llvm/ADT/ImmutableSet.h"

namespace llvm {

/// ImutKeyValueInfo -Traits class used by ImmutableMap.  While both the first and
/// second elements in a pair are used to generate profile information,
/// only the first element (the key) is used by isEqual and isLess.
template <typename T, typename S>
struct ImutKeyValueInfo {
  typedef const std::pair<T,S> value_type;
  typedef const value_type& value_type_ref;
  typedef const T   key_type;
  typedef const T&  key_type_ref;
  typedef const S   data_type;
  typedef const S&  data_type_ref;
  
  static inline key_type_ref KeyOfValue(value_type_ref V) {
    return V.first;
  }
  
  static inline bool isEqual(key_type_ref L, key_type_ref R) {
    return ImutContainerInfo<T>::isEqual(L,R);
  }
  
  static inline bool isLess(key_type_ref L, key_type_ref R) {
    return ImutContainerInfo<T>::isLess(L,R);
  }
  
  static inline void Profile(FoldingSetNodeID& ID, value_type_ref V) {
    ImutContainerInfo<T>::Profile(ID, V.first);
    ImutContainerInfo<S>::Profile(ID, V.second);
  }
};  

  
template <typename KeyT, typename ValT, 
          typename ValInfo = ImutKeyValueInfo<KeyT,ValT> >
class ImmutableMap {
  typedef typename ValInfo::value_type      value_type;
  typedef typename ValInfo::value_type_ref  value_type_ref;
  typedef typename ValInfo::key_type        key_type;
  typedef typename ValInfo::key_type_ref    key_type_ref;
  typedef typename ValInfo::data_type       data_type;
  typedef typename ValInfo::data_type_ref   data_type_ref;
  
private:  
  typedef ImutAVLTree<ValInfo> TreeTy;
  TreeTy* Root;
  
  ImmutableMap(TreeTy* R) : Root(R) {}
  
public:
  
  class Factory {
    typename TreeTy::Factory F;
    
  public:
    Factory() {}
    
    ImmutableMap GetEmptyMap() { return ImmutableMap(F.GetEmptyTree()); }
    
    ImmutableMap Add(ImmutableMap Old, key_type_ref K, data_type_ref D) {
      return ImmutableMap(F.Add(Old.Root,std::make_pair<key_type,data_type>(K,D)));
    }
    
    ImmutableMap Remove(ImmutableMap Old, key_type_ref K) {
      return ImmutableMap(F.Remove(Old.Root,K));
    }        
    
  private:
    Factory(const Factory& RHS) {};
    void operator=(const Factory& RHS) {};    
  };
  
  friend class Factory;  
  
  bool contains(key_type_ref K) const {
    return Root ? Root->contains(K) : false;
  }
  
  data_type* find(key_type_ref K) const {
    if (Root) {
      TreeTy* T = Root->find(K);
      if (T) return &T->getValue().second;
    }
    
    return NULL;
  }
  
  bool operator==(ImmutableMap RHS) const {
    return Root && RHS.Root ? Root->isEqual(*RHS.Root) : Root == RHS.Root;
  }
  
  bool operator!=(ImmutableMap RHS) const {
    return Root && RHS.Root ? Root->isNotEqual(*RHS.Root) : Root != RHS.Root;
  }
  
  bool isEmpty() const { return !Root; }

  //===--------------------------------------------------===//    
  // Foreach - A limited form of map iteration.
  //===--------------------------------------------------===//

private:
  template <typename Callback>
  struct CBWrapper {
    Callback C;
    void operator()(value_type_ref V) { C(V.first,V.second); }    
  };  
  
  template <typename Callback>
  struct CBWrapperRef {
    Callback &C;
    CBWrapperRef(Callback& c) : C(c) {}
    
    void operator()(value_type_ref V) { C(V.first,V.second); }    
  };
  
public:  
  template <typename Callback>
  void foreach(Callback& C) { 
    if (Root) { 
      CBWrapperRef<Callback> CB(C);
      Root->foreach(CB);
    }
  }
  
  template <typename Callback>
  void foreach() { 
    if (Root) {
      CBWrapper<Callback> CB;
      Root->foreach(CB);
    }
  }
  
  //===--------------------------------------------------===//    
  // For testing.
  //===--------------------------------------------------===//  
  
  void verify() const { if (Root) Root->verify(); }
  unsigned getHeight() const { return Root ? Root->getHeight() : 0; }
  
};
  
} // end namespace llvm

#endif
