//===- SelectorMap.cpp - Maps selectors to methods and messages -*- C++ -*-===//
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

#include "clang/Index/SelectorMap.h"
#include "ASTVisitor.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace idx;

namespace {

class VISIBILITY_HIDDEN SelMapper : public ASTVisitor<SelMapper> {
  SelectorMap::SelMethMapTy &SelMethMap;
  SelectorMap::SelRefMapTy &SelRefMap;

public:
  SelMapper(SelectorMap::SelMethMapTy &MethMap,
            SelectorMap::SelRefMapTy &RefMap)
    : SelMethMap(MethMap), SelRefMap(RefMap) { }

  void VisitObjCMethodDecl(ObjCMethodDecl *D);
  void VisitObjCMessageExpr(ObjCMessageExpr *Node);
  void VisitObjCSelectorExpr(ObjCSelectorExpr *Node);
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SelMapper Implementation
//===----------------------------------------------------------------------===//

void SelMapper::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  if (D->getCanonicalDecl() == D)
    SelMethMap.insert(std::make_pair(D->getSelector(), D));
  Base::VisitObjCMethodDecl(D);
}

void SelMapper::VisitObjCMessageExpr(ObjCMessageExpr *Node) {
  ASTLocation ASTLoc(CurrentDecl, Node);
  SelRefMap.insert(std::make_pair(Node->getSelector(), ASTLoc));
}

void SelMapper::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  ASTLocation ASTLoc(CurrentDecl, Node);
  SelRefMap.insert(std::make_pair(Node->getSelector(), ASTLoc));
}

//===----------------------------------------------------------------------===//
// SelectorMap Implementation
//===----------------------------------------------------------------------===//

SelectorMap::SelectorMap(ASTContext &Ctx) {
  SelMapper(SelMethMap, SelRefMap).Visit(Ctx.getTranslationUnitDecl());
}

SelectorMap::method_iterator
SelectorMap::methods_begin(Selector Sel) const {
  return method_iterator(SelMethMap.lower_bound(Sel));
}

SelectorMap::method_iterator
SelectorMap::methods_end(Selector Sel) const {
  return method_iterator(SelMethMap.upper_bound(Sel));
}

SelectorMap::astlocation_iterator
SelectorMap::refs_begin(Selector Sel) const {
  return astlocation_iterator(SelRefMap.lower_bound(Sel));
}

SelectorMap::astlocation_iterator
SelectorMap::refs_end(Selector Sel) const {
  return astlocation_iterator(SelRefMap.upper_bound(Sel));
}
