//=-- AggExprVisitor.cpp - evaluating expressions of C++ class type -*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines AggExprVisitor class, which contains lots of boiler
// plate code for evaluating expressions of C++ class type.
//
//===----------------------------------------------------------------------===//

#include "clang/Checker/PathSensitive/GRExprEngine.h"
#include "clang/AST/StmtVisitor.h"

using namespace clang;

namespace {
class AggExprVisitor : public StmtVisitor<AggExprVisitor> {
  SVal DestPtr;
  ExplodedNode *Pred;
  ExplodedNodeSet &DstSet;
  GRExprEngine &Eng;

public:
  AggExprVisitor(SVal dest, ExplodedNode *N, ExplodedNodeSet &dst, 
                 GRExprEngine &eng)
    : DestPtr(dest), Pred(N), DstSet(dst), Eng(eng) {}

  void VisitCastExpr(CastExpr *E);
  void VisitCXXConstructExpr(CXXConstructExpr *E);
};
}

void AggExprVisitor::VisitCastExpr(CastExpr *E) {
  switch (E->getCastKind()) {
  default: 
    assert(0 && "Unhandled cast kind");
  case CK_NoOp:
  case CK_ConstructorConversion:
    Visit(E->getSubExpr());
    break;
  }
}

void AggExprVisitor::VisitCXXConstructExpr(CXXConstructExpr *E) {
  Eng.VisitCXXConstructExpr(E, DestPtr, Pred, DstSet);
}

void GRExprEngine::VisitAggExpr(const Expr *E, SVal Dest, ExplodedNode *Pred,
                                ExplodedNodeSet &Dst) {
  AggExprVisitor(Dest, Pred, Dst, *this).Visit(const_cast<Expr *>(E));
}
