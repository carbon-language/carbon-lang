//===--- AttributeList.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ATTRLIST_H
#define LLVM_CLANG_ATTRLIST_H

#include "clang/Parse/Action.h"
#include <cassert>

namespace clang {

/// AttributeList - Represents GCC's __attribute__ declaration. There are
/// 4 forms of this construct...they are:
///
/// 1: __attribute__(( const )). ParmName/Args/NumArgs will all be unused.
/// 2: __attribute__(( mode(byte) )). ParmName used, Args/NumArgs unused.
/// 3: __attribute__(( format(printf, 1, 2) )). ParmName/Args/NumArgs all used.
/// 4: __attribute__(( aligned(16) )). ParmName is unused, Args/Num used.
///
class AttributeList {
  IdentifierInfo *AttrName;
  SourceLocation AttrLoc;
  IdentifierInfo *ParmName;
  SourceLocation ParmLoc;
  Action::ExprTy **Args;
  unsigned NumArgs;
  AttributeList *Next;
public:
  AttributeList(IdentifierInfo *AttrName, SourceLocation AttrLoc,
                IdentifierInfo *ParmName, SourceLocation ParmLoc,
                Action::ExprTy **args, unsigned numargs, AttributeList *Next);
  ~AttributeList() {
    if (Args) {
      // FIXME: before we delete the vector, we need to make sure the Expr's 
      // have been deleted. Since Action::ExprTy is "void", we are dependent
      // on the actions module for actually freeing the memory. The specific
      // hooks are ParseDeclarator, ParseTypeName, ParseParamDeclaratorType, 
      // ParseField, ParseTag. Once these routines have freed the expression, 
      // they should zero out the Args slot (to indicate the memory has been 
      // freed). If any element of the vector is non-null, we should assert.
      delete [] Args;
    }
    if (Next)
      delete Next;
  }
  
  IdentifierInfo *getAttributeName() const { return AttrName; }
  SourceLocation getAttributeLoc() const { return AttrLoc; }
  IdentifierInfo *getParameterName() const { return ParmName; }
  
  AttributeList *getNext() const { return Next; }
  void setNext(AttributeList *N) { Next = N; }
  
  void addAttributeList(AttributeList *alist) {
    assert((alist != 0) && "addAttributeList(): alist is null");
    AttributeList *next = this, *prev;
    do {
      prev = next;
      next = next->getNext();
    } while (next);
    prev->setNext(alist);
  }

  /// getNumArgs - Return the number of actual arguments to this attribute.
  unsigned getNumArgs() const { return NumArgs; }
  
  /// getArg - Return the specified argument.
  Action::ExprTy *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return Args[Arg];
  }
};

}  // end namespace clang

#endif
