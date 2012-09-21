//== BodyFarm.cpp  - Factory for conjuring up fake bodies ----------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BodyFarm is a factory for creating faux implementations for functions/methods
// for analysis purposes.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSwitch.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "BodyFarm.h"

using namespace clang;

typedef Stmt *(*FunctionFarmer)(ASTContext &C, const FunctionDecl *D);


/// Create a fake body for dispatch_sync.
static Stmt *create_dispatch_sync(ASTContext &C, const FunctionDecl *D) {
  // Check if we have at least two parameters.
  if (D->param_size() != 2)
    return 0;
  
  // Check if the second parameter is a block.
  const ParmVarDecl *PV = D->getParamDecl(1);
  QualType Ty = PV->getType();
  const BlockPointerType *BPT = Ty->getAs<BlockPointerType>();
  if (!BPT)
    return 0;
  
  // Check if the block pointer type takes no arguments and
  // returns void.
  const FunctionProtoType *FT =
    BPT->getPointeeType()->getAs<FunctionProtoType>();
  if (!FT || !FT->getResultType()->isVoidType()  ||
      FT->getNumArgs() != 0)
    return 0;

  // Everything checks out.  Create a fake body that just calls the block.
  // This is basically just an AST dump of:
  //
  // void dispatch_sync(dispatch_queue_t queue, void (^block)(void)) {
  //   block();
  // }
  //
  DeclRefExpr *DR = DeclRefExpr::CreateEmpty(C, false, false, false, false);
  DR->setDecl(const_cast<ParmVarDecl*>(PV));
  DR->setValueKind(VK_LValue);
  ImplicitCastExpr *ICE = ImplicitCastExpr::Create(C, Ty, CK_LValueToRValue,
                                                   DR, 0, VK_RValue);
  CallExpr *CE = new (C) CallExpr(C, ICE, ArrayRef<Expr*>(), C.VoidTy,
                                  VK_RValue, SourceLocation());
  return CE;
}

Stmt *BodyFarm::getBody(const FunctionDecl *D) {
  D = D->getCanonicalDecl();
  
  llvm::Optional<Stmt *> &Val = Bodies[D];
  if (Val.hasValue())
    return Val.getValue();
  
  Val = 0;
  
  if (D->getIdentifier() == 0)
    return 0;

  StringRef Name = D->getName();
  if (Name.empty())
    return 0;
  
  FunctionFarmer FF =
    llvm::StringSwitch<FunctionFarmer>(Name)
      .Case("dispatch_sync", create_dispatch_sync)
      .Default(NULL);
  
  if (FF) {
    Val = FF(C, D);
  }
  
  return Val.getValue();
}

