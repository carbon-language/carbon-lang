//===-- llvm/SymbolTableListTraits.h - Traits for iplist --------*- C++ -*-===//
//
// This file defines a generic class that is used to implement the automatic
// symbol table manipulation that occurs when you put (for example) a named
// instruction into a basic block.
//
// The way that this is implemented is by using a special traits class with the
// intrusive list that makes up the list of instructions in a basic block.  When
// a new element is added to the list of instructions, the traits class is
// notified, allowing the symbol table to be updated.
//
// This generic class implements the traits class.  It must be generic so that
// it can work for all uses it, which include lists of instructions, basic
// blocks, arguments, functions, global variables, etc...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMBOLTABLELISTTRAITS_H
#define LLVM_SYMBOLTABLELISTTRAITS_H

template<typename NodeTy> class ilist_iterator;
template<typename NodeTy, typename Traits> class iplist;
template<typename Ty> struct ilist_traits;

// ValueSubClass  - The type of objects that I hold
// ItemParentType - I call setParent() on all of my "ValueSubclass" items, and
//                  this is the value that I pass in.
// SymTabType     - This is the class type, whose symtab I insert my
//                  ValueSubClass items into.  Most of the time it is
//                  ItemParentType, but Instructions have item parents of BB's
//                  but symtabtype's of a Function
//
template<typename ValueSubClass, typename ItemParentClass, typename SymTabClass,
         typename SubClass=ilist_traits<ValueSubClass> >
class SymbolTableListTraits {
  SymTabClass     *SymTabObject;
  ItemParentClass *ItemParent;
public:
  SymbolTableListTraits() : SymTabObject(0), ItemParent(0) {}

        SymTabClass *getParent()       { return SymTabObject; }
  const SymTabClass *getParent() const { return SymTabObject; }

  static ValueSubClass *getPrev(ValueSubClass *V) { return V->getPrev(); }
  static ValueSubClass *getNext(ValueSubClass *V) { return V->getNext(); }
  static const ValueSubClass *getPrev(const ValueSubClass *V) {
    return V->getPrev();
  }
  static const ValueSubClass *getNext(const ValueSubClass *V) {
    return V->getNext();
  }

  static void setPrev(ValueSubClass *V, ValueSubClass *P) { V->setPrev(P); }
  static void setNext(ValueSubClass *V, ValueSubClass *N) { V->setNext(N); }

  void addNodeToList(ValueSubClass *V);
  void removeNodeFromList(ValueSubClass *V);
  void transferNodesFromList(iplist<ValueSubClass, 
                             ilist_traits<ValueSubClass> > &L2,
                             ilist_iterator<ValueSubClass> first,
                             ilist_iterator<ValueSubClass> last);

//private:
  void setItemParent(ItemParentClass *IP) { ItemParent = IP; }//This is private!
  void setParent(SymTabClass *Parent);  // This is private!
};

#endif
