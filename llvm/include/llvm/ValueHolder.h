//===-- llvm/ValueHolder.h - Class to hold multiple values -------*- C++ -*--=//
//
// This defines a class that is used as a fancy Definition container.  It is 
// special because it helps keep the symbol table of the container method up to
// date with the goings on inside of it.
//
// This is used to represent things like the instructions of a basic block and
// the arguments to a method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUEHOLDER_H
#define LLVM_VALUEHOLDER_H

#include <vector>
class SymTabValue;

// ItemParentType ItemParent - I call setParent() on all of my 
// "ValueSubclass" items, and this is the value that I pass in.
//
template<class ValueSubclass, class ItemParentType> 
class ValueHolder {
  // TODO: Should I use a deque instead of a vector?
  vector<ValueSubclass*> ValueList;

  ItemParentType *ItemParent;
  SymTabValue *Parent;

  ValueHolder(const ValueHolder &V);   // DO NOT IMPLEMENT
public:
  inline ValueHolder(ItemParentType *IP, SymTabValue *parent = 0) { 
    assert(IP && "Item parent may not be null!");
    ItemParent = IP;
    Parent = 0;
    setParent(parent); 
  }

  inline ~ValueHolder() {
    // The caller should have called delete_all first...
    assert(empty() && "ValueHolder contains definitions!");
    assert(Parent == 0 && "Should have been unlinked from method!");
  }

  inline const SymTabValue *getParent() const { return Parent; }
  inline SymTabValue *getParent() { return Parent; }
  void setParent(SymTabValue *Parent);  // Defined in ValueHolderImpl.h

  inline unsigned size() const { return ValueList.size(); }
  inline bool empty()    const { return ValueList.empty(); }
  inline const ValueSubclass *front() const { return ValueList.front(); }
  inline       ValueSubclass *front()       { return ValueList.front(); }
  inline const ValueSubclass *back()  const { return ValueList.back(); }
  inline       ValueSubclass *back()        { return ValueList.back(); }

  //===--------------------------------------------------------------------===//
  // sub-Definition iterator code
  //===--------------------------------------------------------------------===//
  // 
  typedef vector<ValueSubclass*>::iterator       iterator;
  typedef vector<ValueSubclass*>::const_iterator const_iterator;

  inline iterator       begin()       { return ValueList.begin(); }
  inline const_iterator begin() const { return ValueList.begin(); }
  inline iterator       end()         { return ValueList.end();   }
  inline const_iterator end()   const { return ValueList.end();   }

  void delete_all() {            // Delete all removes and deletes all elements
    // TODO: REMOVE FROM END OF VECTOR!!!
    while (begin() != end()) {
      iterator I = begin();
      delete remove(I);          // Delete all instructions...
    }
  }

  // ValueHolder::remove(iterator &) this removes the element at the location 
  // specified by the iterator, and leaves the iterator pointing to the element 
  // that used to follow the element deleted.
  //
  ValueSubclass *remove(iterator &DI);  // Defined in ValueHolderImpl.h
  void     remove(ValueSubclass *D);    // Defined in ValueHolderImpl.h

  inline void push_front(ValueSubclass *Inst); // Defined in ValueHolderImpl.h
  inline void push_back(ValueSubclass *Inst);  // Defined in ValueHolderImpl.h
};

#endif
