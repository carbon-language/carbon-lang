//===--- CodeGenFunction.cpp - Emit LLVM Code from ASTs for a Function ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
using namespace llvm;
using namespace clang;
using namespace CodeGen;

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm) 
  : CGM(cgm), Target(CGM.getContext().Target) {}


llvm::BasicBlock *CodeGenFunction::getBasicBlockForLabel(const LabelStmt *S) {
  BasicBlock *&BB = LabelMap[S];
  if (BB) return BB;
  
  // Create, but don't insert, the new block.
  return BB = new BasicBlock(S->getName());
}


/// ConvertType - Convert the specified type to its LLVM form.
const llvm::Type *CodeGenFunction::ConvertType(QualType T, SourceLocation Loc) {
  // FIXME: Cache these, move the CodeGenModule, expand, etc.
  const clang::Type &Ty = *T.getCanonicalType();
  
  switch (Ty.getTypeClass()) {
  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty).getKind()) {
    case BuiltinType::Void:
      // LLVM void type can only be used as the result of a function call.  Just
      // map to the same as char.
    case BuiltinType::Char:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
      return IntegerType::get(Target.getCharWidth(Loc));

    case BuiltinType::Bool:
      return IntegerType::get(Target.getBoolWidth(Loc));
      
    case BuiltinType::Short:
    case BuiltinType::UShort:
      return IntegerType::get(Target.getShortWidth(Loc));
      
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return IntegerType::get(Target.getIntWidth(Loc));

    case BuiltinType::Long:
    case BuiltinType::ULong:
      return IntegerType::get(Target.getLongWidth(Loc));

    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
      return IntegerType::get(Target.getLongLongWidth(Loc));
      
    case BuiltinType::Float:      return llvm::Type::FloatTy;
    case BuiltinType::Double:     return llvm::Type::DoubleTy;
    case BuiltinType::LongDouble:
    case BuiltinType::FloatComplex:
    case BuiltinType::DoubleComplex:
    case BuiltinType::LongDoubleComplex:
      ;
    }
    break;
  }
  case Type::Pointer:
  case Type::Reference:
  case Type::Array:
    break;
  case Type::FunctionNoProto:
  case Type::FunctionProto: {
    const FunctionType &FP = cast<FunctionType>(Ty);
    const llvm::Type *ResultType;
    
    if (FP.getResultType()->isVoidType())
      ResultType = llvm::Type::VoidTy;    // Result of function uses llvm void.
    else
      ResultType = ConvertType(FP.getResultType(), Loc);
    
    // FIXME: Convert argument types.
    
    return llvm::FunctionType::get(ResultType,
                                   std::vector<const llvm::Type*>(),
                                   false,
                                   0);
  }
  case Type::TypeName:
  case Type::Tagged:
    break;
  }
  
  // FIXME: implement.
  return OpaqueType::get();
}


void CodeGenFunction::GenerateCode(const FunctionDecl *FD) {
  const llvm::Type *Ty = ConvertType(FD->getType(), FD->getLocation());
  
  CurFn = new Function(cast<llvm::FunctionType>(Ty),
                       Function::ExternalLinkage,
                       FD->getName(), &CGM.getModule());
  
  BasicBlock *EntryBB = new BasicBlock("entry", CurFn);
  
  // TODO: Walk the decls, creating allocas etc.
  
  Builder.SetInsertPoint(EntryBB);
  
  EmitStmt(FD->getBody());
}


//===----------------------------------------------------------------------===//
//                              Statement Emission
//===----------------------------------------------------------------------===//

void CodeGenFunction::EmitStmt(const Stmt *S) {
  assert(S && "Null statement?");
  
  switch (S->getStmtClass()) {
  default:
    // Must be an expression in a stmt context.  Emit the value and ignore the
    // result.
    if (const Expr *E = dyn_cast<Expr>(S)) {
      EmitExpr(E);
    } else {
      printf("Unimplemented stmt!\n");
      S->dump();
    }
    break;
  case Stmt::NullStmtClass: break;
  case Stmt::CompoundStmtClass: EmitCompoundStmt(cast<CompoundStmt>(*S)); break;
  case Stmt::LabelStmtClass:    EmitLabelStmt(cast<LabelStmt>(*S));       break;
  case Stmt::GotoStmtClass:     EmitGotoStmt(cast<GotoStmt>(*S));         break;
  }
}

void CodeGenFunction::EmitCompoundStmt(const CompoundStmt &S) {
  // FIXME: handle vla's etc.
  
  for (CompoundStmt::const_body_iterator I = S.body_begin(), E = S.body_end();
       I != E; ++I)
    EmitStmt(*I);
}

void CodeGenFunction::EmitBlock(BasicBlock *BB) {
  // Emit a branch from this block to the next one if this was a real block.  If
  // this was just a fall-through block after a terminator, don't emit it.
  if (!Builder.GetInsertBlock()->empty() ||
      Builder.GetInsertBlock()->getValueName()) {
    Builder.CreateBr(BB);
  } else {
    // TODO: cache and reuse these.
    Builder.GetInsertBlock()->eraseFromParent();
  }
  CurFn->getBasicBlockList().push_back(BB);
  Builder.SetInsertPoint(BB);
}

void CodeGenFunction::EmitLabelStmt(const LabelStmt &S) {
  llvm::BasicBlock *NextBB = getBasicBlockForLabel(&S);
  
  EmitBlock(NextBB);
}

void CodeGenFunction::EmitGotoStmt(const GotoStmt &S) {
  Builder.CreateBr(getBasicBlockForLabel(S.getLabel()));
  
  Builder.SetInsertPoint(new BasicBlock("", CurFn));
}



//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

ExprResult CodeGenFunction::EmitExpr(const Expr *E) {
  assert(E && "Null expression?");
  
  switch (E->getStmtClass()) {
  default:
    printf("Unimplemented expr!\n");
    E->dump();
    return ExprResult::get(UndefValue::get(llvm::Type::Int32Ty));
  case Stmt::IntegerLiteralClass:
    return EmitIntegerLiteral(cast<IntegerLiteral>(E)); 
  }
  
}

ExprResult CodeGenFunction::EmitIntegerLiteral(const IntegerLiteral *E) {
  return ExprResult::get(ConstantInt::get(E->getValue()));
}


