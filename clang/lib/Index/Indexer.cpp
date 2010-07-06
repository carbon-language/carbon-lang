//===--- Indexer.cpp - IndexProvider implementation -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  IndexProvider implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/Indexer.h"
#include "clang/Index/Program.h"
#include "clang/Index/Handlers.h"
#include "clang/Index/TranslationUnit.h"
#include "ASTVisitor.h"
#include "clang/AST/DeclBase.h"
using namespace clang;
using namespace idx;

namespace {

class EntityIndexer : public EntityHandler {
  TranslationUnit *TU;
  Indexer::MapTy &Map;
  Indexer::DefMapTy &DefMap;

public:
  EntityIndexer(TranslationUnit *tu, Indexer::MapTy &map, 
                Indexer::DefMapTy &defmap) 
    : TU(tu), Map(map), DefMap(defmap) { }

  virtual void Handle(Entity Ent) {
    if (Ent.isInternalToTU())
      return;
    Map[Ent].insert(TU);

    Decl *D = Ent.getDecl(TU->getASTContext());
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
      if (FD->isThisDeclarationADefinition())
        DefMap[Ent] = std::make_pair(FD, TU);
  }
};

class SelectorIndexer : public ASTVisitor<SelectorIndexer> {
  Program &Prog;
  TranslationUnit *TU;
  Indexer::SelMapTy &Map;

public:
  SelectorIndexer(Program &prog, TranslationUnit *tu, Indexer::SelMapTy &map)
    : Prog(prog), TU(tu), Map(map) { }

  void VisitObjCMethodDecl(ObjCMethodDecl *D) {
    Map[GlobalSelector::get(D->getSelector(), Prog)].insert(TU);
    Base::VisitObjCMethodDecl(D);
  }

  void VisitObjCMessageExpr(ObjCMessageExpr *Node) {
    Map[GlobalSelector::get(Node->getSelector(), Prog)].insert(TU);
    Base::VisitObjCMessageExpr(Node);
  }
};

} // anonymous namespace

void Indexer::IndexAST(TranslationUnit *TU) {
  assert(TU && "Passed null TranslationUnit");
  ASTContext &Ctx = TU->getASTContext();
  CtxTUMap[&Ctx] = TU;
  EntityIndexer Idx(TU, Map, DefMap);
  Prog.FindEntities(Ctx, Idx);

  SelectorIndexer SelIdx(Prog, TU, SelMap);
  SelIdx.Visit(Ctx.getTranslationUnitDecl());
}

void Indexer::GetTranslationUnitsFor(Entity Ent,
                                     TranslationUnitHandler &Handler) {
  assert(Ent.isValid() && "Expected valid Entity");

  if (Ent.isInternalToTU()) {
    Decl *D = Ent.getInternalDecl();
    CtxTUMapTy::iterator I = CtxTUMap.find(&D->getASTContext());
    if (I != CtxTUMap.end())
      Handler.Handle(I->second);
    return;
  }

  MapTy::iterator I = Map.find(Ent);
  if (I == Map.end())
    return;

  TUSetTy &Set = I->second;
  for (TUSetTy::iterator I = Set.begin(), E = Set.end(); I != E; ++I)
    Handler.Handle(*I);
}

void Indexer::GetTranslationUnitsFor(GlobalSelector Sel,
                                    TranslationUnitHandler &Handler) {
  assert(Sel.isValid() && "Expected valid GlobalSelector");

  SelMapTy::iterator I = SelMap.find(Sel);
  if (I == SelMap.end())
    return;

  TUSetTy &Set = I->second;
  for (TUSetTy::iterator I = Set.begin(), E = Set.end(); I != E; ++I)
    Handler.Handle(*I);
}

std::pair<FunctionDecl *, TranslationUnit *> 
Indexer::getDefinitionFor(Entity Ent) {
  DefMapTy::iterator I = DefMap.find(Ent);
  if (I == DefMap.end())
    return std::make_pair((FunctionDecl *)0, (TranslationUnit *)0);
  else
    return I->second;
}
