//===-- llvm/Module.h - C++ class to represent a VM module -------*- C++ -*--=//
//
// This file contains the declarations for the Module class that is used to 
// maintain all the information related to a VM module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULE_H
#define LLVM_MODULE_H

#include "llvm/Value.h"
#include "llvm/SymTabValue.h"
class Method;

class Module : public Value, public SymTabValue {
public:
  typedef ValueHolder<Method, Module, Module> MethodListType;

  // Method iterators...
  typedef MethodListType::iterator iterator;
  typedef MethodListType::const_iterator const_iterator;
  typedef reverse_iterator<const_iterator> const_reverse_iterator;
  typedef reverse_iterator<iterator>             reverse_iterator;
private:
  MethodListType MethodList;     // The Methods

public:
  Module();
  ~Module();

  // reduceApply - Apply the specified function to all of the methods in this 
  // module.  The result values are or'd together and the result is returned.
  //
  bool reduceApply(bool (*Func)(Method*));
  bool reduceApply(bool (*Func)(const Method*)) const;


  // Get the underlying elements of the Module...
  inline const MethodListType &getMethodList() const  { return MethodList; }
  inline       MethodListType &getMethodList()        { return MethodList; }

  //===--------------------------------------------------------------------===//
  // Module iterator forwarding functions
  //
  inline iterator                begin()       { return MethodList.begin(); }
  inline const_iterator          begin() const { return MethodList.begin(); }
  inline iterator                end  ()       { return MethodList.end();   }
  inline const_iterator          end  () const { return MethodList.end();   }

  inline reverse_iterator       rbegin()       { return MethodList.rbegin(); }
  inline const_reverse_iterator rbegin() const { return MethodList.rbegin(); }
  inline reverse_iterator       rend  ()       { return MethodList.rend();   }
  inline const_reverse_iterator rend  () const { return MethodList.rend();   }

  inline unsigned                 size() const { return MethodList.size(); }
  inline bool                    empty() const { return MethodList.empty(); }
  inline const Method           *front() const { return MethodList.front(); }
  inline       Method           *front()       { return MethodList.front(); }
  inline const Method            *back() const { return MethodList.back(); }
  inline       Method            *back()       { return MethodList.back(); }


  // dropAllReferences() - This function causes all the subinstructions to "let
  // go" of all references that they are maintaining.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator 
  // delete.
  //
  void dropAllReferences();
};

#endif
