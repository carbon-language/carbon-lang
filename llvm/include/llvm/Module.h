//===-- llvm/Module.h - C++ class to represent a VM module -------*- C++ -*--=//
//
// This file contains the declarations for the Module class that is used to 
// maintain all the information related to a VM module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULE_H
#define LLVM_MODULE_H

#include "llvm/SymTabValue.h"
class Method;

class Module : public SymTabValue {
public:
  typedef ValueHolder<Method, Module> MethodListType;
private:
  MethodListType MethodList;     // The Methods

public:
  Module();
  ~Module();

  inline const MethodListType &getMethodList() const  { return MethodList; }
  inline       MethodListType &getMethodList()        { return MethodList; }

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
