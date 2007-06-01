//===--- Attr.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Attribute class interfaces
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ATTR_H
#define LLVM_CLANG_AST_ATTR_H

#include "clang/Basic/SourceLocation.h"
#include "clang/AST/Type.h"

namespace llvm {
namespace clang {
class IdentifierInfo;
class Expr;

/// Attr - Represents GCC's __attribute__ declaration. There are
/// 4 forms of this construct...they are:
///
/// 1: __attribute__(( const )). ParmName/Args/NumArgs will all be unused.
/// 2: __attribute__(( mode(byte) )). ParmName used, Args/NumArgs unused.
/// 3: __attribute__(( format(printf, 1, 2) )). ParmName/Args/NumArgs all used.
/// 4: __attribute__(( aligned(16) )). ParmName is unused, Args/Num used.
///
class Attr {
  IdentifierInfo *AttrName;
  SourceLocation AttrLoc;
  IdentifierInfo *ParmName;
  Expr **Args;
  unsigned NumArgs;
  Attr *Next;
public:
  Attr(SourceLocation L, IdentifierInfo *AttrName, 
       IdentifierInfo *ParmName, Expr **args, unsigned numargs);
  ~Attr() {
    delete [] Args;
  }
  
  IdentifierInfo *getAttributeName() const { return AttrName; }
  IdentifierInfo *getParameterName() const { return ParmName; }
  
  Attr *getNext() const { return Next; }
  void setNext(Attr *N) { Next = N; }

  /// getNumArgs - Return the number of actual arguments to this attribute.
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
};

}  // end namespace clang
}  // end namespace llvm

#endif
