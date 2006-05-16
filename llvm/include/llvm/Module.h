//===-- llvm/Module.h - C++ class to represent a VM module ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file This file contains the declarations for the Module class. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULE_H
#define LLVM_MODULE_H

#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class GlobalVariable;
class GlobalValueRefMap;   // Used by ConstantVals.cpp
class FunctionType;
class SymbolTable;

template<> struct ilist_traits<Function>
  : public SymbolTableListTraits<Function, Module, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  static Function *createSentinel();
  static void destroySentinel(Function *F) { delete F; }
  static iplist<Function> &getList(Module *M);
};
template<> struct ilist_traits<GlobalVariable>
  : public SymbolTableListTraits<GlobalVariable, Module, Module> {
  // createSentinel is used to create a node that marks the end of the list.
  static GlobalVariable *createSentinel();
  static void destroySentinel(GlobalVariable *GV) { delete GV; }
  static iplist<GlobalVariable> &getList(Module *M);
};

/// A Module instance is used to store all the information related to an
/// LLVM module. Modules are the top level container of all other LLVM 
/// Intermediate Representation (IR) objects. Each module directly contains a
/// list of globals variables, a list of functions, a list of libraries (or 
/// other modules) this module depends on, a symbol table, and various data
/// about the target's characteristics.
///
/// A module maintains a GlobalValRefMap object that is used to hold all
/// constant references to global variables in the module.  When a global
/// variable is destroyed, it should have no entries in the GlobalValueRefMap.
/// @brief The main container class for the LLVM Intermediate Representation.
class Module {
/// @name Types And Enumerations
/// @{
public:
  /// The type for the list of global variables.
  typedef iplist<GlobalVariable> GlobalListType;
  /// The type for the list of functions.
  typedef iplist<Function> FunctionListType;

  /// The type for the list of dependent libraries.
  typedef SetVector<std::string> LibraryListType;

  /// The Global Variable iterator.
  typedef GlobalListType::iterator                     global_iterator;
  /// The Global Variable constant iterator.
  typedef GlobalListType::const_iterator         const_global_iterator;

  /// The Function iterators.
  typedef FunctionListType::iterator                          iterator;
  /// The Function constant iterator
  typedef FunctionListType::const_iterator              const_iterator;

  /// The Library list iterator.
  typedef LibraryListType::const_iterator lib_iterator;

  /// An enumeration for describing the endianess of the target machine.
  enum Endianness  { AnyEndianness, LittleEndian, BigEndian };

  /// An enumeration for describing the size of a pointer on the target machine.
  enum PointerSize { AnyPointerSize, Pointer32, Pointer64 };

/// @}
/// @name Member Variables
/// @{
private:
  GlobalListType GlobalList;     ///< The Global Variables in the module
  FunctionListType FunctionList; ///< The Functions in the module
  LibraryListType LibraryList;   ///< The Libraries needed by the module
  std::string GlobalScopeAsm;    ///< Inline Asm at global scope.
  SymbolTable *SymTab;           ///< Symbol Table for the module
  std::string ModuleID;          ///< Human readable identifier for the module
  std::string TargetTriple;      ///< Platform target triple Module compiled on
  Endianness  Endian;            ///< Endianness assumed in the module
  PointerSize PtrSize;           ///< Pointer size assumed in the module

  friend class Constant;

/// @}
/// @name Constructors
/// @{
public:
  /// The Module constructor. Note that there is no default constructor. You
  /// must provide a name for the module upon construction.
  Module(const std::string &ModuleID);
  /// The module destructor. This will dropAllReferences.
  ~Module();

/// @}
/// @name Module Level Accessors
/// @{
public:
  /// Get the module identifier which is, essentially, the name of the module.
  /// @returns the module identifier as a string
  const std::string &getModuleIdentifier() const { return ModuleID; }

  /// Get the target triple which is a string describing the target host.
  /// @returns a string containing the target triple.
  const std::string &getTargetTriple() const { return TargetTriple; }

  /// Get the target endian information.
  /// @returns Endianess - an enumeration for the endianess of the target
  Endianness getEndianness() const { return Endian; }

  /// Get the target pointer size.
  /// @returns PointerSize - an enumeration for the size of the target's pointer
  PointerSize getPointerSize() const { return PtrSize; }

  /// Get any module-scope inline assembly blocks.
  /// @returns a string containing the module-scope inline assembly blocks.
  const std::string &getModuleInlineAsm() const { return GlobalScopeAsm; }
/// @}
/// @name Module Level Mutators
/// @{
public:

  /// Set the module identifier.
  void setModuleIdentifier(const std::string &ID) { ModuleID = ID; }

  /// Set the target triple.
  void setTargetTriple(const std::string &T) { TargetTriple = T; }

  /// Set the target endian information.
  void setEndianness(Endianness E) { Endian = E; }

  /// Set the target pointer size.
  void setPointerSize(PointerSize PS) { PtrSize = PS; }

  /// Set the module-scope inline assembly blocks.
  void setModuleInlineAsm(const std::string &Asm) { GlobalScopeAsm = Asm; }
  
/// @}
/// @name Function Accessors
/// @{
public:
  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  If it does not exist, add a prototype for the function and return
  /// it.
  Function *getOrInsertFunction(const std::string &Name, const FunctionType *T);

  /// getOrInsertFunction - Look up the specified function in the module symbol
  /// table.  If it does not exist, add a prototype for the function and return
  /// it.  This version of the method takes a null terminated list of function
  /// arguments, which makes it easier for clients to use.
  Function *getOrInsertFunction(const std::string &Name, const Type *RetTy,...)
    END_WITH_NULL;

  /// getFunction - Look up the specified function in the module symbol table.
  /// If it does not exist, return null.
  Function *getFunction(const std::string &Name, const FunctionType *Ty);

  /// getMainFunction - This function looks up main efficiently.  This is such a
  /// common case, that it is a method in Module.  If main cannot be found, a
  /// null pointer is returned.
  Function *getMainFunction();

  /// getNamedFunction - Return the first function in the module with the
  /// specified name, of arbitrary type.  This method returns null if a function
  /// with the specified name is not found.
  Function *getNamedFunction(const std::string &Name);

/// @}
/// @name Global Variable Accessors 
/// @{
public:
  /// getGlobalVariable - Look up the specified global variable in the module
  /// symbol table.  If it does not exist, return null.  The type argument
  /// should be the underlying type of the global, i.e., it should not have
  /// the top-level PointerType, which represents the address of the global.
  /// If AllowInternal is set to true, this function will return types that
  /// have InternalLinkage. By default, these types are not returned.
  GlobalVariable *getGlobalVariable(const std::string &Name, const Type *Ty,
                                    bool AllowInternal = false);

  /// getNamedGlobal - Return the first global variable in the module with the
  /// specified name, of arbitrary type.  This method returns null if a global
  /// with the specified name is not found.
  GlobalVariable *getNamedGlobal(const std::string &Name);
  
/// @}
/// @name Type Accessors
/// @{
public:
  /// addTypeName - Insert an entry in the symbol table mapping Str to Type.  If
  /// there is already an entry for this name, true is returned and the symbol
  /// table is not modified.
  bool addTypeName(const std::string &Name, const Type *Ty);

  /// getTypeName - If there is at least one entry in the symbol table for the
  /// specified type, return it.
  std::string getTypeName(const Type *Ty) const;

  /// getTypeByName - Return the type with the specified name in this module, or
  /// null if there is none by that name.
  const Type *getTypeByName(const std::string &Name) const;

/// @}
/// @name Direct access to the globals list, functions list, and symbol table
/// @{
public:
  /// Get the Module's list of global variables (constant).
  const GlobalListType   &getGlobalList() const       { return GlobalList; }
  /// Get the Module's list of global variables.
  GlobalListType         &getGlobalList()             { return GlobalList; }
  /// Get the Module's list of functions (constant).
  const FunctionListType &getFunctionList() const     { return FunctionList; }
  /// Get the Module's list of functions.
  FunctionListType       &getFunctionList()           { return FunctionList; }
  /// Get the symbol table of global variable and function identifiers
  const SymbolTable      &getSymbolTable() const      { return *SymTab; }
  /// Get the Module's symbol table of global variable and function identifiers.
  SymbolTable            &getSymbolTable()            { return *SymTab; }

/// @}
/// @name Global Variable Iteration
/// @{
public:
  /// Get an iterator to the first global variable
  global_iterator       global_begin()       { return GlobalList.begin(); }
  /// Get a constant iterator to the first global variable
  const_global_iterator global_begin() const { return GlobalList.begin(); }
  /// Get an iterator to the last global variable
  global_iterator       global_end  ()       { return GlobalList.end(); }
  /// Get a constant iterator to the last global variable
  const_global_iterator global_end  () const { return GlobalList.end(); }
  /// Determine if the list of globals is empty.
  bool                  global_empty() const { return GlobalList.empty(); }

/// @}
/// @name Function Iteration
/// @{
public:
  /// Get an iterator to the first function.
  iterator                begin()       { return FunctionList.begin(); }
  /// Get a constant iterator to the first function.
  const_iterator          begin() const { return FunctionList.begin(); }
  /// Get an iterator to the last function.
  iterator                end  ()       { return FunctionList.end();   }
  /// Get a constant iterator to the last function.
  const_iterator          end  () const { return FunctionList.end();   }
  /// Determine how many functions are in the Module's list of functions.
  size_t                   size() const { return FunctionList.size(); }
  /// Determine if the list of functions is empty.
  bool                    empty() const { return FunctionList.empty(); }

/// @}
/// @name Dependent Library Iteration 
/// @{
public:
  /// @brief Get a constant iterator to beginning of dependent library list.
  inline lib_iterator lib_begin() const { return LibraryList.begin(); }
  /// @brief Get a constant iterator to end of dependent library list.
  inline lib_iterator lib_end() const { return LibraryList.end(); }
  /// @brief Returns the number of items in the list of libraries.
  inline size_t lib_size() const { return LibraryList.size(); }
  /// @brief Add a library to the list of dependent libraries
  inline void addLibrary(const std::string& Lib){ LibraryList.insert(Lib); }
  /// @brief Remove a library from the list of dependent libraries
  inline void removeLibrary(const std::string& Lib) { LibraryList.remove(Lib); }
  /// @brief Get all the libraries
  inline const LibraryListType& getLibraries() const { return LibraryList; }

/// @}
/// @name Utility functions for printing and dumping Module objects
/// @{
public:
  /// Print the module to an output stream
  void print(std::ostream &OS) const { print(OS, 0); }
  /// Print the module to an output stream with AssemblyAnnotationWriter.
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;
  /// Dump the module to std::cerr (for debugging).
  void dump() const;
  /// This function causes all the subinstructions to "let go" of all references
  /// that they are maintaining.  This allows one to 'delete' a whole class at 
  /// a time, even though there may be circular references... first all 
  /// references are dropped, and all use counts go to zero.  Then everything 
  /// is delete'd for real.  Note that no operations are valid on an object 
  /// that has "dropped all references", except operator delete.
  void dropAllReferences();
/// @}
};

/// An iostream inserter for modules.
inline std::ostream &operator<<(std::ostream &O, const Module &M) {
  M.print(O);
  return O;
}

} // End llvm namespace

#endif
