//===-------- llvm/GlobalAlias.h - GlobalAlias class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the GlobalAlias class, which
// represents a single function or variable alias in the IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GLOBAL_ALIAS_H
#define LLVM_GLOBAL_ALIAS_H

#include "llvm/GlobalValue.h"

namespace llvm {

class Module;
class Constant;
class PointerType;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class GlobalAlias : public GlobalValue {
  friend class SymbolTableListTraits<GlobalAlias, Module>;
  void operator=(const GlobalAlias &);     // Do not implement
  GlobalAlias(const GlobalAlias &);     // Do not implement

  void setParent(Module *parent);

  GlobalAlias *Prev, *Next;
  void setNext(GlobalAlias *N) { Next = N; }
  void setPrev(GlobalAlias *N) { Prev = N; }

  // getNext/Prev - Return the next or previous alias in the list.
        GlobalAlias *getNext()       { return Next; }
  const GlobalAlias *getNext() const { return Next; }
        GlobalAlias *getPrev()       { return Prev; }
  const GlobalAlias *getPrev() const { return Prev; }

  Use Aliasee;
public:
  /// GlobalAlias ctor - If a parent module is specified, the alias is
  /// automatically inserted into the end of the specified module's alias list.
  GlobalAlias(const Type *Ty, LinkageTypes Linkage, const std::string &Name = "",
              Constant* Aliasee = 0, Module *Parent = 0);

  /// isDeclaration - Is this global variable lacking an initializer?  If so, 
  /// the global variable is defined in some other translation unit, and is thus
  /// only a declaration here.
  virtual bool isDeclaration() const;

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent();

  virtual void print(std::ostream &OS) const;
  void print(std::ostream *OS) const { if (OS) print(*OS); }

  /// set/getAliasee - These methods retrive and set alias target.
  void setAliasee(Constant* GV);
  const Constant* getAliasee() const {
    return cast_or_null<Constant>(getOperand(0));
  }
  Constant* getAliasee() {
    return cast_or_null<Constant>(getOperand(0));
  }
  /// getAliasedGlobal() - Aliasee can be either global or bitcast of
  /// global. This method retrives the global for both aliasee flavours.
  const GlobalValue* getAliasedGlobal() const;
    
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const GlobalAlias *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalAliasVal;
  }
};

} // End llvm namespace

#endif
