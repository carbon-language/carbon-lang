//===-- llvm/Argument.h - Definition of the Argument class ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Argument class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ARGUMENT_H
#define LLVM_IR_ARGUMENT_H

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Value.h"

namespace llvm {

template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

/// \brief LLVM Argument representation
///
/// This class represents an incoming formal argument to a Function. A formal
/// argument, since it is ``formal'', does not contain an actual value but
/// instead represents the type, argument number, and attributes of an argument
/// for a specific function. When used in the body of said function, the
/// argument of course represents the value of the actual argument that the
/// function was called with.
class Argument : public Value, public ilist_node<Argument> {
  virtual void anchor();
  Function *Parent;

  friend class SymbolTableListTraits<Argument, Function>;
  void setParent(Function *parent);

public:
  /// \brief Constructor.
  ///
  /// If \p F is specified, the argument is inserted at the end of the argument
  /// list for \p F.
  explicit Argument(Type *Ty, const Twine &Name = "", Function *F = 0);

  inline const Function *getParent() const { return Parent; }
  inline       Function *getParent()       { return Parent; }

  /// \brief Return the index of this formal argument in its containing
  /// function.
  ///
  /// For example in "void foo(int a, float b)" a is 0 and b is 1.
  unsigned getArgNo() const;

  /// \brief Return true if this argument has the byval attribute on it in its
  /// containing function.
  bool hasByValAttr() const;

  /// \brief If this is a byval argument, return its alignment.
  unsigned getParamAlignment() const;

  /// \brief Return true if this argument has the nest attribute on it in its
  /// containing function.
  bool hasNestAttr() const;

  /// \brief Return true if this argument has the noalias attribute on it in its
  /// containing function.
  bool hasNoAliasAttr() const;

  /// \brief Return true if this argument has the nocapture attribute on it in
  /// its containing function.
  bool hasNoCaptureAttr() const;

  /// \brief Return true if this argument has the sret attribute on it in its
  /// containing function.
  bool hasStructRetAttr() const;

  /// \brief Return true if this argument has the returned attribute on it in
  /// its containing function.
  bool hasReturnedAttr() const;

  /// \brief Return true if this argument has the readonly or readnone attribute
  /// on it in its containing function.
  bool onlyReadsMemory() const;


  /// \brief Add a Attribute to an argument.
  void addAttr(AttributeSet AS);

  /// \brief Remove a Attribute from an argument.
  void removeAttr(AttributeSet AS);

  /// \brief Method for support type inquiry through isa, cast, and
  /// dyn_cast.
  static inline bool classof(const Value *V) {
    return V->getValueID() == ArgumentVal;
  }
};

} // End llvm namespace

#endif
