//== Environment.cpp - Map from Expr* to Locations/Values -------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the Environment and EnvironmentManager classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Environment.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;

RVal Environment::GetRVal(Expr* E, BasicValueFactory& BasicVals) const {
  
  for (;;) {
    
    switch (E->getStmtClass()) {
        
      case Stmt::AddrLabelExprClass:        
        return LVal::MakeVal(cast<AddrLabelExpr>(E));
        
        // ParenExprs are no-ops.
        
      case Stmt::ParenExprClass:        
        E = cast<ParenExpr>(E)->getSubExpr();
        continue;
        
      case Stmt::CharacterLiteralClass: {
        CharacterLiteral* C = cast<CharacterLiteral>(E);
        return NonLVal::MakeVal(BasicVals, C->getValue(), C->getType());
      }
        
      case Stmt::IntegerLiteralClass: {
        return NonLVal::MakeVal(BasicVals, cast<IntegerLiteral>(E));
      }
        
      case Stmt::StringLiteralClass:
        return LVal::MakeVal(cast<StringLiteral>(E));
        
        // Casts where the source and target type are the same
        // are no-ops.  We blast through these to get the descendant
        // subexpression that has a value.
        
      case Stmt::ImplicitCastExprClass: {
        ImplicitCastExpr* C = cast<ImplicitCastExpr>(E);
        QualType CT = C->getType();
        
        if (CT->isVoidType())
          return UnknownVal();
        
        QualType ST = C->getSubExpr()->getType();
        
        break;
      }
        
      case Stmt::CastExprClass: {
        CastExpr* C = cast<CastExpr>(E);
        QualType CT = C->getType();
        QualType ST = C->getSubExpr()->getType();
        
        if (CT->isVoidType())
          return UnknownVal();
        
        break;
      }
        
        // Handle all other Expr* using a lookup.
        
      default:
        break;
    };
    
    break;
  }
  
  return LookupExpr(E);
}

RVal Environment::GetBlkExprRVal(Expr* E, BasicValueFactory& BasicVals) const {
  
  E = E->IgnoreParens();
  
  switch (E->getStmtClass()) {
    case Stmt::CharacterLiteralClass: {
      CharacterLiteral* C = cast<CharacterLiteral>(E);
      return NonLVal::MakeVal(BasicVals, C->getValue(), C->getType());
    }
      
    case Stmt::IntegerLiteralClass: {
      return NonLVal::MakeVal(BasicVals, cast<IntegerLiteral>(E));
    }
      
    default:
      return LookupBlkExpr(E);
  }
}

Environment EnvironmentManager::SetRVal(const Environment& Env, Expr* E, RVal V,
                                        bool isBlkExpr, bool Invalidate) {  
  assert (E);
  
  if (V.isUnknown()) {    
    if (Invalidate)
      return isBlkExpr ? RemoveBlkExpr(Env, E) : RemoveSubExpr(Env, E);
    else
      return Env;
  }

  return isBlkExpr ? AddBlkExpr(Env, E, V) : AddSubExpr(Env, E, V);
}
