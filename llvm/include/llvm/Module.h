//===-- llvm/Module.h - C++ class to represent a VM module ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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

#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
class GlobalVariable;
class GlobalValueRefMap;   // Used by ConstantVals.cpp
class ConstantPointerRef;
class FunctionType;
class SymbolTable;

template<> struct ilist_traits<Function>
  : public SymbolTableListTraits<Function, Module, Module> {
  // createNode is used to create a node that marks the end of the list...
  static Function *createNode();
  static iplist<Function> &getList(Module *M);
};
template<> struct ilist_traits<GlobalVariable>
  : public SymbolTableListTraits<GlobalVariable, Module, Module> {
  // createNode is used to create a node that marks the end of the list...
  static GlobalVariable *createNode();
  static iplist<GlobalVariable> &getList(Module *M);
};

struct Module : public Annotable {
  typedef iplist<GlobalVariable> GlobalListType;
  typedef iplist<Function> FunctionListType;

  // Global Variable iterators...
  typedef GlobalListType::iterator                             giterator;
  typedef GlobalListType::const_iterator                 const_giterator;
  typedef std::reverse_iterator<giterator>             reverse_giterator;
  typedef std::reverse_iterator<const_giterator> const_reverse_giterator;

  // Function iterators...
  typedef FunctionListType::iterator                          iterator;
  typedef FunctionListType::const_iterator              const_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  enum Endianness  { AnyEndianness, LittleEndian, BigEndian };
  enum PointerSize { AnyPointerSize, Pointer32, Pointer64 };

private:
  GlobalListType GlobalList;     // The Global Variables in the module
  FunctionListType FunctionList; // The Functions in the module
  GlobalValueRefMap *GVRefMap;   // Keep track of GlobalValueRef's
  SymbolTable *SymTab;           // Symbol Table for the module
  std::string ModuleID;    // Human readable identifier for the module

  // These flags are probably not the right long-term way to handle this kind of
  // target information, but it is sufficient for now.
  Endianness  Endian;       // True if target is little endian
  PointerSize PtrSize;    // True if target has 32-bit pointers (false = 64-bit)

  // Accessor for the underlying GVRefMap... only through the Constant class...
  friend class Constant;
  friend class ConstantPointerRef;
  void mutateConstantPointerRef(GlobalValue *OldGV, GlobalValue *NewGV);
  ConstantPointerRef *getConstantPointerRef(GlobalValue *GV);
  void destroyConstantPointerRef(ConstantPointerRef *CPR);

public:
  Module(const std::string &ModuleID);
  ~Module();

  const std::string &getModuleIdentifier() const { return ModuleID; }

  /// Target endian information...
  Endianness getEndianness() const { return Endian; }
  void setEndianness(Endianness E) { Endian = E; }

  /// Target Pointer Size information...
  PointerSize getPointerSize() const { return PtrSize; }
  void setPointerSize(PointerSize PS) { PtrSize = PS; }

  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  If it does not exist, add a prototype for the function and return
  /// it.
  Function *getOrInsertFunction(const std::string &Name, const FunctionType *T);

  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  If it does not exist, add a prototype for the function and return
  /// it.  This version of the method takes a null terminated list of function
  /// arguments, which makes it easier for clients to use.
  Function *getOrInsertFunction(const std::string &Name, const Type *RetTy,...);

  /// getFunction - Look up the specified function in the module symbol table.
  /// If it does not exist, return null.
  ///
  Function *getFunction(const std::string &Name, const FunctionType *Ty);

  /// getMainFunction - This function looks up main efficiently.  This is such a
  /// common case, that it is a method in Module.  If main cannot be found, a
  /// null pointer is returned.
  ///
  Function *getMainFunction();

  /// getNamedFunction - Return the first function in the module with the
  /// specified name, of arbitrary type.  This method returns null if a function
  /// with the specified name is not found.
  ///
  Function *getNamedFunction(const std::string &Name);

  /// addTypeName - Insert an entry in the symbol table mapping Str to Type.  If
  /// there is already an entry for this name, true is returned and the symbol
  /// table is not modified.
  ///
  bool addTypeName(const std::string &Name, const Type *Ty);

  /// getTypeName - If there is at least one entry in the symbol table for the
  /// specified type, return it.
  ///
  std::string getTypeName(const Type *Ty);

  /// Get the underlying elements of the Module...
  inline const GlobalListType &getGlobalList() const  { return GlobalList; }
  inline       GlobalListType &getGlobalList()        { return GlobalList; }
  inline const FunctionListType &getFunctionList() const { return FunctionList;}
  inline       FunctionListType &getFunctionList()       { return FunctionList;}


  //===--------------------------------------------------------------------===//
  // Symbol table support functions...
  
  /// getSymbolTable() - Get access to the symbol table for the module, where
  /// global variables and functions are identified.
  ///
  inline       SymbolTable &getSymbolTable()       { return *SymTab; }
  inline const SymbolTable &getSymbolTable() const { return *SymTab; }


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
  inline const GlobalVariable    &gfront() const { return GlobalList.front(); }
  inline       GlobalVariable    &gfront()       { return GlobalList.front(); }
  inline const GlobalVariable     &gback() const { return GlobalList.back(); }
  inline       GlobalVariable     &gback()       { return GlobalList.back(); }



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
  inline const Function         &front() const { return FunctionList.front(); }
  inline       Function         &front()       { return FunctionList.front(); }
  inline const Function          &back() const { return FunctionList.back(); }
  inline       Function          &back()       { return FunctionList.back(); }

  void print(std::ostream &OS) const;
  void dump() const;

  /// dropAllReferences() - This function causes all the subinstructions to "let
  /// go" of all references that they are maintaining.  This allows one to
  /// 'delete' a whole class at a time, even though there may be circular
  /// references... first all references are dropped, and all use counts go to
  /// zero.  Then everything is delete'd for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator 
  /// delete.
  ///
  void dropAllReferences();
};

inline std::ostream &operator<<(std::ostream &O, const Module *M) {
  M->print(O);
  return O;
}

inline std::ostream &operator<<(std::ostream &O, const Module &M) {
  M.print(O);
  return O;
}

#endif
