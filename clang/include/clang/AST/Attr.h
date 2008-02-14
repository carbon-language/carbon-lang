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

#ifndef LLVM_CLANG_AST_ATTR_H
#define LLVM_CLANG_AST_ATTR_H

namespace clang {

/// Attr - This represents one attribute.
class Attr {
public:
  enum Kind {
    Aligned,
    Packed
  };
    
private:
  Attr *Next;
  Kind AttrKind;
  
protected:
  Attr(Kind AK) : AttrKind(AK) {}
  virtual ~Attr() {
    delete Next;
  }
  
public:
  Kind getKind() const { return AttrKind; }

  Attr *getNext() { return Next; }
  const Attr *getNext() const { return Next; }
  void setNext(Attr *next) { Next = next; }
  
  void addAttr(Attr *attr) {
    assert((attr != 0) && "addAttr(): attr is null");
    
    // FIXME: This doesn't preserve the order in any way.
    attr->Next = Next;
    Next = attr;
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
