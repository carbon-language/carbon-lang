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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

namespace clang {

/// StoredDeclsList - This is an array of decls optimized a common case of only
/// containing one entry.
struct StoredDeclsList {
  /// The kind of data encoded in this list.
  enum DataKind {
    /// \brief The data is a NamedDecl*.
    DK_Decl = 0,
    /// \brief The data is a declaration ID (an unsigned value),
    /// shifted left by 2 bits.
    DK_DeclID = 1,
    /// \brief The data is a pointer to a vector (of type VectorTy)
    /// that contains declarations.
    DK_Decl_Vector = 2,
    /// \brief The data is a pointer to a vector (of type VectorTy)
    /// that contains declaration ID.
    DK_ID_Vector = 3
  };

  /// VectorTy - When in vector form, this is what the Data pointer points to.
  typedef llvm::SmallVector<uintptr_t, 4> VectorTy;

  /// \brief The stored data, which will be either a declaration ID, a
  /// pointer to a NamedDecl, or a pointer to a vector.
  uintptr_t Data;

public:
  StoredDeclsList() : Data(0) {}

  StoredDeclsList(const StoredDeclsList &RHS) : Data(RHS.Data) {
    if (VectorTy *RHSVec = RHS.getAsVector()) {
      VectorTy *New = new VectorTy(*RHSVec);
      Data = reinterpret_cast<uintptr_t>(New) | (Data & 0x03);
    }
  }

  ~StoredDeclsList() {
    // If this is a vector-form, free the vector.
    if (VectorTy *Vector = getAsVector())
      delete Vector;
  }

  StoredDeclsList &operator=(const StoredDeclsList &RHS) {
    if (VectorTy *Vector = getAsVector())
      delete Vector;
    Data = RHS.Data;
    if (VectorTy *RHSVec = RHS.getAsVector()) {
      VectorTy *New = new VectorTy(*RHSVec);
      Data = reinterpret_cast<uintptr_t>(New) | (Data & 0x03);
    }
    return *this;
  }

  bool isNull() const { return (Data & ~0x03) == 0; }

  NamedDecl *getAsDecl() const {
    if ((Data & 0x03) != DK_Decl)
      return 0;

    return reinterpret_cast<NamedDecl *>(Data & ~0x03);
  }

  VectorTy *getAsVector() const {
    if ((Data & 0x03) != DK_ID_Vector && (Data & 0x03) != DK_Decl_Vector)
      return 0;

    return reinterpret_cast<VectorTy *>(Data & ~0x03);
  }

  void setOnlyValue(NamedDecl *ND) {
    assert(!getAsVector() && "Not inline");
    Data = reinterpret_cast<uintptr_t>(ND);
  }

  void setFromDeclIDs(const llvm::SmallVectorImpl<unsigned> &Vec) {
    if (Vec.size() > 1) {
      VectorTy *Vector = getAsVector();
      if (!Vector) {
        Vector = new VectorTy;
        Data = reinterpret_cast<uintptr_t>(Vector) | DK_ID_Vector;
      }

      Vector->resize(Vec.size());
      std::copy(Vec.begin(), Vec.end(), Vector->begin());
      return;
    }

    if (VectorTy *Vector = getAsVector())
      delete Vector;

    if (Vec.empty())
      Data = 0;
    else
      Data = (Vec[0] << 2) | DK_DeclID;
  }

  /// \brief Force the stored declarations list to contain actual
  /// declarations.
  ///
  /// This routine will resolve any declaration IDs for declarations
  /// that may not yet have been loaded from external storage.
  void materializeDecls(ASTContext &Context);

  bool hasDeclarationIDs() const {
    DataKind DK = (DataKind)(Data & 0x03);
    return DK == DK_DeclID || DK == DK_ID_Vector;
  }

  /// getLookupResult - Return an array of all the decls that this list
  /// represents.
  DeclContext::lookup_result getLookupResult(ASTContext &Context) {
    if (isNull())
      return DeclContext::lookup_result(0, 0);

    if (hasDeclarationIDs())
      materializeDecls(Context);

    // If we have a single NamedDecl, return it.
    if (getAsDecl()) {
      assert(!isNull() && "Empty list isn't allowed");

      // Data is a raw pointer to a NamedDecl*, return it.
      void *Ptr = &Data;
      return DeclContext::lookup_result((NamedDecl**)Ptr, (NamedDecl**)Ptr+1);
    }

    assert(getAsVector() && "Must have a vector at this point");
    VectorTy &Vector = *getAsVector();

    // Otherwise, we have a range result.
    return DeclContext::lookup_result((NamedDecl **)&Vector[0],
                                      (NamedDecl **)&Vector[0]+Vector.size());
  }

  /// HandleRedeclaration - If this is a redeclaration of an existing decl,
  /// replace the old one with D and return true.  Otherwise return false.
  bool HandleRedeclaration(ASTContext &Context, NamedDecl *D) {
    if (hasDeclarationIDs())
      materializeDecls(Context);

    // Most decls only have one entry in their list, special case it.
    if (NamedDecl *OldD = getAsDecl()) {
      if (!D->declarationReplaces(OldD))
        return false;
      setOnlyValue(D);
      return true;
    }

    // Determine if this declaration is actually a redeclaration.
    VectorTy &Vec = *getAsVector();
    for (VectorTy::iterator OD = Vec.begin(), ODEnd = Vec.end();
         OD != ODEnd; ++OD) {
      NamedDecl *OldD = reinterpret_cast<NamedDecl *>(*OD);
      if (D->declarationReplaces(OldD)) {
        *OD = reinterpret_cast<uintptr_t>(D);
        return true;
      }
    }

    return false;
  }

  /// AddSubsequentDecl - This is called on the second and later decl when it is
  /// not a redeclaration to merge it into the appropriate place in our list.
  ///
  void AddSubsequentDecl(NamedDecl *D) {
    assert(!hasDeclarationIDs() && "Must materialize before adding decls");

    // If this is the second decl added to the list, convert this to vector
    // form.
    if (NamedDecl *OldD = getAsDecl()) {
      VectorTy *VT = new VectorTy();
      VT->push_back(reinterpret_cast<uintptr_t>(OldD));
      Data = reinterpret_cast<uintptr_t>(VT) | DK_Decl_Vector;
    }

    VectorTy &Vec = *getAsVector();
    if (isa<UsingDirectiveDecl>(D) ||
        D->getIdentifierNamespace() == Decl::IDNS_Tag)
      Vec.push_back(reinterpret_cast<uintptr_t>(D));
    else if (reinterpret_cast<NamedDecl *>(Vec.back())
               ->getIdentifierNamespace() == Decl::IDNS_Tag) {
      uintptr_t TagD = Vec.back();
      Vec.back() = reinterpret_cast<uintptr_t>(D);
      Vec.push_back(TagD);
    } else
      Vec.push_back(reinterpret_cast<uintptr_t>(D));
  }
};

typedef llvm::DenseMap<DeclarationName, StoredDeclsList> StoredDeclsMap;


} // end namespace clang

#endif
