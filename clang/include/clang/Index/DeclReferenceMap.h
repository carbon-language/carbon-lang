//===--- DeclReferenceMap.h - Map Decls to their references -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  DeclReferenceMap creates a mapping from Decls to the ASTLocations that
//  reference them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_DECLREFERENCEMAP_H
#define LLVM_CLANG_INDEX_DECLREFERENCEMAP_H

#include "clang/Index/ASTLocation.h"
#include "clang/Index/STLExtras.h"
#include <map>

namespace clang {
  class ASTContext;
  class NamedDecl;

namespace idx {

/// \brief Maps NamedDecls with the ASTLocations that reference them.
///
/// References are mapped and retrieved using the canonical decls.
class DeclReferenceMap {
public:
  explicit DeclReferenceMap(ASTContext &Ctx);

  typedef std::multimap<NamedDecl*, ASTLocation> MapTy;
  typedef pair_value_iterator<MapTy::iterator> astlocation_iterator;

  astlocation_iterator refs_begin(NamedDecl *D) const;
  astlocation_iterator refs_end(NamedDecl *D) const;
  bool refs_empty(NamedDecl *D) const;

private:
  mutable MapTy Map;
};

} // end idx namespace

} // end clang namespace

#endif
