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

// ValueSubClass  - The type of objects that I hold
// ItemParentType - I call setParent() on all of my "ValueSubclass" items, and
//                  this is the value that I pass in.
// SymTabType     - This is the class type (which should be derived from
//                  SymTabValue), whose symtab I insert my ValueSubClass items
//                  into.  Most of the time it is ItemParentType, but
//                  Instructions have item parents of bb's but symtabtype's of
//                  a Method
//
template<class ValueSubclass, class ItemParentType, class SymTabType> 
class ValueHolder {
  vector<ValueSubclass*> ValueList;

  ItemParentType *ItemParent;
  SymTabType *Parent;

  ValueHolder(const ValueHolder &V);   // DO NOT IMPLEMENT
public:
  inline ValueHolder(ItemParentType *IP, SymTabType *parent = 0) { 
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

  inline const SymTabType *getParent() const { return Parent; }
  inline SymTabType *getParent() { return Parent; }
  void setParent(SymTabType *Parent);  // Defined in ValueHolderImpl.h

  inline             unsigned size()  const { return ValueList.size();  }
  inline                 bool empty() const { return ValueList.empty(); }
  inline const ValueSubclass *front() const { return ValueList.front(); }
  inline       ValueSubclass *front()       { return ValueList.front(); }
  inline const ValueSubclass *back()  const { return ValueList.back();  }
  inline       ValueSubclass *back()        { return ValueList.back();  }
  inline const ValueSubclass *operator[](unsigned i) const {
    return ValueList[i];
  }
  inline       ValueSubclass *operator[](unsigned i) {
    return ValueList[i];
  }

  //===--------------------------------------------------------------------===//
  // sub-Definition iterator code
  //===--------------------------------------------------------------------===//
  // 
  typedef vector<ValueSubclass*>::iterator               iterator;
  typedef vector<ValueSubclass*>::const_iterator   const_iterator;
  typedef reverse_iterator<const_iterator> const_reverse_iterator;
  typedef reverse_iterator<iterator>             reverse_iterator;

  inline iterator                begin()       { return ValueList.begin(); }
  inline const_iterator          begin() const { return ValueList.begin(); }
  inline iterator                end  ()       { return ValueList.end();   }
  inline const_iterator          end  () const { return ValueList.end();   }

  inline reverse_iterator       rbegin()       { return ValueList.rbegin(); }
  inline const_reverse_iterator rbegin() const { return ValueList.rbegin(); }
  inline reverse_iterator       rend  ()       { return ValueList.rend();   }
  inline const_reverse_iterator rend  () const { return ValueList.rend();   }
  
  void delete_all() {            // Delete all removes and deletes all elements
    while (!empty()) {
      iterator it = end();
      delete remove(--it);          // Delete all instructions...
    }
  }

  // ValueHolder::remove(iterator &) this removes the element at the location 
  // specified by the iterator, and leaves the iterator pointing to the element 
  // that used to follow the element deleted.
  //
  ValueSubclass *remove(iterator &DI);         // Defined in ValueHolderImpl.h
  ValueSubclass *remove(const iterator &DI);   // Defined in ValueHolderImpl.h
  void           remove(ValueSubclass *D);     // Defined in ValueHolderImpl.h
  ValueSubclass *pop_back();                   // Defined in ValueHolderImpl.h

  void push_front(ValueSubclass *Inst);        // Defined in ValueHolderImpl.h
  void push_back(ValueSubclass *Inst);         // Defined in ValueHolderImpl.h

  // ValueHolder::insert - This method inserts the specified value *BEFORE* the 
  // indicated iterator position, and returns an interator to the newly inserted
  // value.
  //
  iterator insert(iterator Pos, ValueSubclass *Inst);
};

#endif
