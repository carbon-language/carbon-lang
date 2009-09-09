//===--- SelectorMap.h - Maps selectors to methods and messages -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  SelectorMap creates a mapping from selectors to ObjC method declarations
//  and ObjC message expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_SELECTORMAP_H
#define LLVM_CLANG_INDEX_SELECTORMAP_H

#include "clang/Index/ASTLocation.h"
#include "clang/Index/STLExtras.h"
#include "clang/Basic/IdentifierTable.h"
#include <map>

namespace clang {
  class ASTContext;
  class ObjCMethodDecl;

namespace idx {

/// \brief Maps NamedDecls with the ASTLocations that reference them.
///
/// References are mapped and retrieved using the canonical decls.
class SelectorMap {
public:
  explicit SelectorMap(ASTContext &Ctx);

  typedef std::multimap<Selector, ObjCMethodDecl *> SelMethMapTy;
  typedef std::multimap<Selector, ASTLocation> SelRefMapTy;

  typedef pair_value_iterator<SelMethMapTy::iterator> method_iterator;
  typedef pair_value_iterator<SelRefMapTy::iterator> astlocation_iterator;

  method_iterator methods_begin(Selector Sel) const;
  method_iterator methods_end(Selector Sel) const;

  astlocation_iterator refs_begin(Selector Sel) const;
  astlocation_iterator refs_end(Selector Sel) const;

private:
  mutable SelMethMapTy SelMethMap;
  mutable SelRefMapTy SelRefMap;
};

} // end idx namespace

} // end clang namespace

#endif
