//===-- llvm/Module.h - C++ class to represent a VM module -------*- C++ -*--=//
//
// This file contains the declarations for the Module class that is used to 
// maintain all the information related to a VM module.
//
// A module also maintains a GlobalValRefMap object that is used to hold all
// constant references to global variables in the module.  When a global
// variable is destroyed, it should have no entries in the GlobalValueRefMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULE_H
#define LLVM_MODULE_H

#include "llvm/Value.h"
#include "llvm/SymTabValue.h"
#include "llvm/ValueHolder.h"
class GlobalVariable;
class GlobalValueRefMap;   // Used by ConstantVals.cpp
class ConstantPointerRef;
class FunctionType;

class Module : public Value, public SymTabValue {
public:
  typedef ValueHolder<GlobalVariable, Module, Module> GlobalListType;
  typedef ValueHolder<Function, Module, Module> FunctionListType;

  // Global Variable iterators...
  typedef GlobalListType::iterator                             giterator;
  typedef GlobalListType::const_iterator                 const_giterator;
  typedef std::reverse_iterator<giterator>             reverse_giterator;
  typedef std::reverse_iterator<const_giterator> const_reverse_giterator;

  // Function iterators...
  typedef FunctionListType::iterator                            iterator;
  typedef FunctionListType::const_iterator                const_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

private:
  GlobalListType GlobalList;     // The Global Variables
  FunctionListType FunctionList;     // The Functions

  GlobalValueRefMap *GVRefMap;

  // Accessor for the underlying GlobalValRefMap... only through the
  // ConstantPointerRef class...
  friend class ConstantPointerRef;
  void mutateConstantPointerRef(GlobalValue *OldGV, GlobalValue *NewGV);
  ConstantPointerRef *getConstantPointerRef(GlobalValue *GV);

public:
  Module();
  ~Module();

  // getOrInsertFunction - Look up the specified function in the module symbol
  // table.  If it does not exist, add a prototype for the function and return
  // it.
  Function *getOrInsertFunction(const std::string &Name, const FunctionType *T);

  // getFunction - Look up the specified function in the module symbol table.
  // If it does not exist, return null.
  //
  Function *getFunction(const std::string &Name, const FunctionType *Ty);

  // Get the underlying elements of the Module...
  inline const GlobalListType &getGlobalList() const  { return GlobalList; }
  inline       GlobalListType &getGlobalList()        { return GlobalList; }
  inline const FunctionListType &getFunctionList() const { return FunctionList;}
  inline       FunctionListType &getFunctionList()       { return FunctionList;}

  //===--------------------------------------------------------------------===//
  // Module iterator forwarding functions
  //
  inline giterator                gbegin()       { return GlobalList.begin(); }
  inline const_giterator          gbegin() const { return GlobalList.begin(); }
  inline giterator                gend  ()       { return GlobalList.end();   }
  inline const_giterator          gend  () const { return GlobalList.end();   }

  inline reverse_giterator       grbegin()       { return GlobalList.rbegin(); }
  inline const_reverse_giterator grbegin() const { return GlobalList.rbegin(); }
  inline reverse_giterator       grend  ()       { return GlobalList.rend();   }
  inline const_reverse_giterator grend  () const { return GlobalList.rend();   }

  inline unsigned                  gsize() const { return GlobalList.size(); }
  inline bool                     gempty() const { return GlobalList.empty(); }
  inline const GlobalVariable    *gfront() const { return GlobalList.front(); }
  inline       GlobalVariable    *gfront()       { return GlobalList.front(); }
  inline const GlobalVariable     *gback() const { return GlobalList.back(); }
  inline       GlobalVariable     *gback()       { return GlobalList.back(); }



  inline iterator                begin()       { return FunctionList.begin(); }
  inline const_iterator          begin() const { return FunctionList.begin(); }
  inline iterator                end  ()       { return FunctionList.end();   }
  inline const_iterator          end  () const { return FunctionList.end();   }

  inline reverse_iterator       rbegin()       { return FunctionList.rbegin(); }
  inline const_reverse_iterator rbegin() const { return FunctionList.rbegin(); }
  inline reverse_iterator       rend  ()       { return FunctionList.rend();   }
  inline const_reverse_iterator rend  () const { return FunctionList.rend();   }

  inline unsigned                 size() const { return FunctionList.size(); }
  inline bool                    empty() const { return FunctionList.empty(); }
  inline const Function         *front() const { return FunctionList.front(); }
  inline       Function         *front()       { return FunctionList.front(); }
  inline const Function          *back() const { return FunctionList.back(); }
  inline       Function          *back()       { return FunctionList.back(); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Module *T) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::ModuleVal;
  }

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
