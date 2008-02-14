//===--- Attr.h - Classes for representing expressions ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Attr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPR_H
#define LLVM_CLANG_AST_EXPR_H

namespace clang {

/// Attr - This represents one attribute.
class Attr {
public:
  enum Kind {
    AddressSpace,
    Aligned,
    OCUVectorType,
    Packed,
    VectorSize
  };
    
private:
  Attr *next;
  
  Kind AttrKind;
  
protected:
  Attr(Kind AK) : AttrKind(AK) {}
  virtual ~Attr() {
    if (Next)
      delete Next;
  }
  
public:
  Kind getKind() const { return AttrKind; }

  Attr *getNext() const { return Next; }
  void setNext(Attr *N) { Next = N; }
  
  void addAttr(Attr *attr) {
    assert((attr != 0) && "addAttr(): attr is null");
    Attr *next = this, *prev;
    do {
      prev = next;
      next = next->getNext();
    } while (next);
    prev->setNext(attr);
  }
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *) { return true; }
};

class PackedAttr : public Attr {
public:
  PackedAttr() : Attr(Packed) {}
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Attr *A) {
    return A->getKind() == Packed;
  }
  static bool classof(const PackedAttr *A) { return true; }
};
  
}  // end namespace clang

#endif
