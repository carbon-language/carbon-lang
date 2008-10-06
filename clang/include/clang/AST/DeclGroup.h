//===--- DeclGroup.h - Classes for representing groups of Decls -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclGroup, DeclGroupRef, and OwningDeclGroup classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLGROUP_H
#define LLVM_CLANG_AST_DECLGROUP_H

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace clang {
  
class ASTContext;
class Decl;
class DeclGroup;
class DeclGroupIterator;

class DeclGroup {
  // FIXME: Include a TypeSpecifier object.
  unsigned NumDecls;
  
private:
  DeclGroup(unsigned numdecls, Decl** decls);

public:
  static DeclGroup* Create(ASTContext& C, unsigned numdecls, Decl** decls);
  void Destroy(ASTContext& C);

  unsigned size() const { return NumDecls; }
  Decl*& operator[](unsigned i) { 
    assert (i < NumDecls && "Out-of-bounds access.");
    return *((Decl**) (this+1));
  }
};
  
    
class DeclGroupRef {
protected:
  enum Kind { DeclKind=0x0, DeclGroupKind=0x1, Mask=0x1 };  
  Decl* D;

  Kind getKind() const {
    return (Kind) (reinterpret_cast<uintptr_t>(D) & Mask);
  }  
  
public:    
  DeclGroupRef() : D(0) {}
  
  explicit DeclGroupRef(Decl* d) : D(d) {}
  explicit DeclGroupRef(DeclGroup* dg)
    : D((Decl*) (reinterpret_cast<uintptr_t>(D) | DeclGroupKind)) {}
  
  typedef Decl** iterator;

  iterator begin() {
    if (getKind() == DeclKind) return D ? &D : 0;
    DeclGroup& G = *((DeclGroup*) (reinterpret_cast<uintptr_t>(D) & ~Mask));
    return &G[0];
  }

  iterator end() {
    if (getKind() == DeclKind) return D ? &D + 1 : 0;
    DeclGroup& G = *((DeclGroup*) (reinterpret_cast<uintptr_t>(D) & ~Mask));
    return &G[0] + G.size();
  }  
};
  
class DeclGroupOwningRef : public DeclGroupRef {
public:
  ~DeclGroupOwningRef();  
  void Destroy(ASTContext& C);
  
  explicit DeclGroupOwningRef(DeclGroupOwningRef& R)
    : DeclGroupRef(R) { R.D = 0; }

  DeclGroupOwningRef& operator=(DeclGroupOwningRef& R) {
    D = R.D;
    R.D = 0;
    return *this;
  }
};

} // end clang namespace
#endif
