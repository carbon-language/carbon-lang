//===-- DeclContextInternals.h - DeclContext Representation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the data structures used in the implementation
//  of DeclContext.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DECLCONTEXTINTERNALS_H
#define LLVM_CLANG_AST_DECLCONTEXTINTERNALS_H

#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace clang {

/// StoredDeclsList - This is an array of decls optimized a common case of only
/// containing one entry.
struct StoredDeclsList {
  /// VectorTy - When in vector form, this is what the Data pointer points to.
  typedef llvm::SmallVector<NamedDecl*, 4> VectorTy;

  /// Data - Union of NamedDecl*/VectorTy*.
  llvm::PointerUnion<NamedDecl*, VectorTy*> Data;
public:
  StoredDeclsList() {}
  StoredDeclsList(const StoredDeclsList &RHS) : Data(RHS.Data) {
    if (isVector())
      Data = new VectorTy(*Data.get<VectorTy*>());
  }
  
  ~StoredDeclsList() {
    // If this is a vector-form, free the vector.
    if (isVector())
      delete Data.get<VectorTy*>();
  }
  
  StoredDeclsList &operator=(const StoredDeclsList &RHS) {
    if (isVector())
      delete Data.get<VectorTy*>();
    Data = RHS.Data;
    if (isVector())
      Data = new VectorTy(*Data.get<VectorTy*>());
    return *this;
  }
  
  bool isVector() const { return Data.is<VectorTy*>(); }
  bool isInline() const { return Data.is<NamedDecl*>(); }
  bool isNull() const { return Data.isNull(); }
  
  void setOnlyValue(NamedDecl *ND) {
    assert(isInline() && "Not inline");
    Data = ND;
  }

  /// getLookupResult - Return an array of all the decls that this list
  /// represents.
  DeclContext::lookup_result getLookupResult() {
    // If we have a single inline unit, return it.
    if (isInline()) {
      assert(!isNull() && "Empty list isn't allowed");
      
      // Data is a raw pointer to a NamedDecl*, return it.
      void *Ptr = &Data;
      return DeclContext::lookup_result((NamedDecl**)Ptr, (NamedDecl**)Ptr+1);
    }
    
    // Otherwise, we have a range result.
    VectorTy &V = *Data.get<VectorTy*>();
    return DeclContext::lookup_result(&V[0], &V[0]+V.size());
  }
  
  /// HandleRedeclaration - If this is a redeclaration of an existing decl,
  /// replace the old one with D and return true.  Otherwise return false.
  bool HandleRedeclaration(NamedDecl *D) {
    // Most decls only have one entry in their list, special case it.
    if (isInline()) {
      if (!D->declarationReplaces(Data.get<NamedDecl*>()))
        return false;
      setOnlyValue(D);
      return true;
    }
    
    // Determine if this declaration is actually a redeclaration.
    VectorTy &Vec = *Data.get<VectorTy*>();
    VectorTy::iterator RDI
      = std::find_if(Vec.begin(), Vec.end(),
                     std::bind1st(std::mem_fun(&NamedDecl::declarationReplaces),
                                  D));
    if (RDI == Vec.end())
      return false;
    *RDI = D;
    return true;
  }
  
  /// AddSubsequentDecl - This is called on the second and later decl when it is
  /// not a redeclaration to merge it into the appropriate place in our list.
  /// 
  void AddSubsequentDecl(NamedDecl *D) {
    // If this is the second decl added to the list, convert this to vector
    // form.
    if (isInline()) {
      NamedDecl *OldD = Data.get<NamedDecl*>();
      VectorTy *VT = new VectorTy();
      VT->push_back(OldD);
      Data = VT;
    }
    
    VectorTy &Vec = *Data.get<VectorTy*>();
    if (isa<UsingDirectiveDecl>(D) ||
        D->getIdentifierNamespace() == Decl::IDNS_Tag)
      Vec.push_back(D);
    else if (Vec.back()->getIdentifierNamespace() == Decl::IDNS_Tag) {
      NamedDecl *TagD = Vec.back();
      Vec.back() = D;
      Vec.push_back(TagD);
    } else
      Vec.push_back(D);
  }
};

typedef llvm::DenseMap<DeclarationName, StoredDeclsList> StoredDeclsMap;


} // end namespace clang

#endif 
