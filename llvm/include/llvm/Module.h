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
class Method;
class GlobalVariable;
class GlobalValueRefMap;   // Used by ConstPoolVals.cpp
class ConstPoolPointerRef;

class Module : public Value, public SymTabValue {
public:
  typedef ValueHolder<GlobalVariable, Module, Module> GlobalListType;
  typedef ValueHolder<Method, Module, Module> MethodListType;

  // Global Variable iterators...
  typedef GlobalListType::iterator                        giterator;
  typedef GlobalListType::const_iterator            const_giterator;
  typedef reverse_iterator<giterator>             reverse_giterator;
  typedef reverse_iterator<const_giterator> const_reverse_giterator;

  // Method iterators...
  typedef MethodListType::iterator                       iterator;
  typedef MethodListType::const_iterator           const_iterator;
  typedef reverse_iterator<const_iterator> const_reverse_iterator;
  typedef reverse_iterator<iterator>             reverse_iterator;

private:
  GlobalListType GlobalList;     // The Global Variables
  MethodListType MethodList;     // The Methods

  GlobalValueRefMap *GVRefMap;

  // Accessor for the underlying GlobalValRefMap... only through the
  // ConstPoolPointerRef class...
  friend class ConstPoolPointerRef;
  void mutateConstPoolPointerRef(GlobalValue *OldGV, GlobalValue *NewGV);
  ConstPoolPointerRef *getConstPoolPointerRef(GlobalValue *GV);

public:
  Module();
  ~Module();

  // reduceApply - Apply the specified function to all of the methods in this 
  // module.  The result values are or'd together and the result is returned.
  //
  bool reduceApply(bool (*Func)(GlobalVariable*));
  bool reduceApply(bool (*Func)(const GlobalVariable*)) const;
  bool reduceApply(bool (*Func)(Method*));
  bool reduceApply(bool (*Func)(const Method*)) const;

  // Get the underlying elements of the Module...
  inline const GlobalListType &getGlobalList() const  { return GlobalList; }
  inline       GlobalListType &getGlobalList()        { return GlobalList; }
  inline const MethodListType &getMethodList() const  { return MethodList; }
  inline       MethodListType &getMethodList()        { return MethodList; }

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
