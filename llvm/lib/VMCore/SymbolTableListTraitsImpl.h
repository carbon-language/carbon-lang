//===-- llvm/SymbolTableListTraitsImpl.h - Implementation ------*- C++ -*--===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the stickier parts of the SymbolTableListTraits class,
// and is explicitly instantiated where needed to avoid defining all this code
// in a widely used header.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMBOLTABLELISTTRAITS_IMPL_H
#define LLVM_SYMBOLTABLELISTTRAITS_IMPL_H

#include "llvm/SymbolTableListTraits.h"
#include "llvm/SymbolTable.h"

namespace llvm {

template<typename ValueSubClass, typename ItemParentClass,typename SymTabClass,
         typename SubClass>
void SymbolTableListTraits<ValueSubClass,ItemParentClass,SymTabClass,SubClass>
::setParent(SymTabClass *STO) {
  iplist<ValueSubClass> &List = SubClass::getList(ItemParent);

  // Remove all of the items from the old symtab..
  if (SymTabObject && !List.empty()) {
    SymbolTable &SymTab = SymTabObject->getSymbolTable();
    for (typename iplist<ValueSubClass>::iterator I = List.begin();
         I != List.end(); ++I)
      if (I->hasName()) SymTab.remove(I);
  }

  SymTabObject = STO;

  // Add all of the items to the new symtab...
  if (SymTabObject && !List.empty()) {
    SymbolTable &SymTab = SymTabObject->getSymbolTable();
    for (typename iplist<ValueSubClass>::iterator I = List.begin();
         I != List.end(); ++I)
      if (I->hasName()) SymTab.insert(I);
  }
}

template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass>
void SymbolTableListTraits<ValueSubClass,ItemParentClass,SymTabClass,SubClass>
::addNodeToList(ValueSubClass *V) {
  assert(V->getParent() == 0 && "Value already in a container!!");
  V->setParent(ItemParent);
  if (V->hasName() && SymTabObject)
    SymTabObject->getSymbolTable().insert(V);
}

template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass>
void SymbolTableListTraits<ValueSubClass,ItemParentClass,SymTabClass,SubClass>
::removeNodeFromList(ValueSubClass *V) {
  V->setParent(0);
  if (V->hasName() && SymTabObject)
    SymTabObject->getSymbolTable().remove(V);
}

template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass>
void SymbolTableListTraits<ValueSubClass,ItemParentClass,SymTabClass,SubClass>
::transferNodesFromList(iplist<ValueSubClass, ilist_traits<ValueSubClass> > &L2,
                        ilist_iterator<ValueSubClass> first,
                        ilist_iterator<ValueSubClass> last) {
  // We only have to do work here if transferring instructions between BBs
  ItemParentClass *NewIP = ItemParent, *OldIP = L2.ItemParent;
  if (NewIP == OldIP) return;  // No work to do at all...

  // We only have to update symbol table entries if we are transferring the
  // instructions to a different symtab object...
  SymTabClass *NewSTO = SymTabObject, *OldSTO = L2.SymTabObject;
  if (NewSTO != OldSTO) {
    for (; first != last; ++first) {
      ValueSubClass &V = *first;
      bool HasName = V.hasName();
      if (OldSTO && HasName)
        OldSTO->getSymbolTable().remove(&V);
      V.setParent(NewIP);
      if (NewSTO && HasName)
        NewSTO->getSymbolTable().insert(&V);
    }
  } else {
    // Just transferring between blocks in the same function, simply update the
    // parent fields in the instructions...
    for (; first != last; ++first)
      first->setParent(NewIP);
  }
}

} // End llvm namespace

#endif
