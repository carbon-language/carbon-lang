//===--- ObjCMethodList.h - A singly linked list of methods -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ObjCMethodList, a singly-linked list of methods.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_OBJCMETHODLIST_H
#define LLVM_CLANG_SEMA_OBJCMETHODLIST_H

#include "llvm/ADT/PointerIntPair.h"

namespace clang {

class ObjCMethodDecl;

/// ObjCMethodList - a linked list of methods with different signatures.
struct ObjCMethodList {
  ObjCMethodDecl *Method;
  /// \brief count of methods with same signature.
  unsigned Count;
  /// \brief The next list object and 2 bits for extra info.
  llvm::PointerIntPair<ObjCMethodList *, 2> NextAndExtraBits;

  ObjCMethodList() : Method(nullptr), Count(0) { }
  ObjCMethodList(ObjCMethodDecl *M, unsigned count, ObjCMethodList *C)
    : Method(M), Count(count), NextAndExtraBits(C, 0) { }

  ObjCMethodList *getNext() const { return NextAndExtraBits.getPointer(); }
  unsigned getBits() const { return NextAndExtraBits.getInt(); }
  void setNext(ObjCMethodList *L) { NextAndExtraBits.setPointer(L); }
  void setBits(unsigned B) { NextAndExtraBits.setInt(B); }
};

}

#endif
