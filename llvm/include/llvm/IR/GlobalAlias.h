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

#ifndef LLVM_IR_GLOBALALIAS_H
#define LLVM_IR_GLOBALALIAS_H

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/OperandTraits.h"

namespace llvm {

class Module;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class GlobalAlias : public GlobalValue, public ilist_node<GlobalAlias> {
  friend class SymbolTableListTraits<GlobalAlias, Module>;
  void operator=(const GlobalAlias &) LLVM_DELETED_FUNCTION;
  GlobalAlias(const GlobalAlias &) LLVM_DELETED_FUNCTION;

  void setParent(Module *parent);

public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }
  /// If a parent module is specified, the alias is automatically inserted into
  /// the end of the specified module's alias list.
  GlobalAlias(Type *Ty, LinkageTypes Linkage, const Twine &Name = "",
              GlobalObject *Aliasee = nullptr, Module *Parent = nullptr,
              unsigned AddressSpace = 0);

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent() override;

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent() override;

  /// set/getAliasee - These methods retrive and set alias target.
  void setAliasee(GlobalObject *GO);
  const GlobalObject *getAliasee() const {
    return const_cast<GlobalAlias *>(this)->getAliasee();
  }

  GlobalObject *getAliasee() {
    return cast_or_null<GlobalObject>(getOperand(0));
  }

  GlobalObject *getAliasedGlobal() {
    return getAliasee();
  }

  const GlobalObject *getAliasedGlobal() const {
    return const_cast<GlobalAlias *>(this)->getAliasedGlobal();
  }

  static bool isValidLinkage(LinkageTypes L) {
    return isExternalLinkage(L) || isLocalLinkage(L) ||
      isWeakLinkage(L) || isLinkOnceLinkage(L);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalAliasVal;
  }
};

template <>
struct OperandTraits<GlobalAlias> :
  public FixedNumOperandTraits<GlobalAlias, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GlobalAlias, Constant)

} // End llvm namespace

#endif
