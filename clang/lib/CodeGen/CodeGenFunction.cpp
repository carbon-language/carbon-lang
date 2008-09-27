//===--- CodeGenFunction.cpp - Emit LLVM Code from ASTs for a Function ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGDebugInfo.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CFG.h"
using namespace clang;
using namespace CodeGen;

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm) 
  : CGM(cgm), Target(CGM.getContext().Target), SwitchInsn(NULL), 
    CaseRangeBlock(NULL) {
    LLVMIntTy = ConvertType(getContext().IntTy);
    LLVMPointerWidth = Target.getPointerWidth(0);
}

ASTContext &CodeGenFunction::getContext() const {
  return CGM.getContext();
}


llvm::BasicBlock *CodeGenFunction::getBasicBlockForLabel(const LabelStmt *S) {
  llvm::BasicBlock *&BB = LabelMap[S];
  if (BB) return BB;
  
  // Create, but don't insert, the new block.
  return BB = llvm::BasicBlock::Create(S->getName());
}

llvm::Constant *
CodeGenFunction::GetAddrOfStaticLocalVar(const VarDecl *BVD) {
  return cast<llvm::Constant>(LocalDeclMap[BVD]);
}

llvm::Value *CodeGenFunction::GetAddrOfLocalVar(const VarDecl *VD)
{
  return LocalDeclMap[VD];
}

const llvm::Type *CodeGenFunction::ConvertType(QualType T) {
  return CGM.getTypes().ConvertType(T);
}

bool CodeGenFunction::isObjCPointerType(QualType T) {
  // All Objective-C types are pointers.
  return T->isObjCInterfaceType() ||
    T->isObjCQualifiedInterfaceType() || T->isObjCQualifiedIdType();
}

bool CodeGenFunction::hasAggregateLLVMType(QualType T) {
  return !isObjCPointerType(T) &&!T->isRealType() && !T->isPointerLikeType() &&
    !T->isVoidType() && !T->isVectorType() && !T->isFunctionType();
}

void CodeGenFunction::FinishFunction(SourceLocation EndLoc) {
  // Finish emission of indirect switches.
  EmitIndirectSwitches();

  // Emit debug descriptor for function end.
  if (CGDebugInfo *DI = CGM.getDebugInfo()) {
    if (EndLoc.isValid()) {
      DI->setLocation(EndLoc);
    }
    DI->EmitRegionEnd(CurFn, Builder);
  }
 
  assert(BreakContinueStack.empty() &&
         "mismatched push/pop in break/continue stack!");

  // Emit function epilog (to return). This has the nice side effect
  // of also automatically handling code that falls off the end.
  EmitBlock(ReturnBlock);
  EmitFunctionEpilog(FnRetTy, ReturnValue);

  // Remove the AllocaInsertPt instruction, which is just a convenience for us.
  AllocaInsertPt->eraseFromParent();
  AllocaInsertPt = 0;
  
  // Verify that the function is well formed.
  if (verifyFunction(*CurFn, llvm::PrintMessageAction)) {
    CurFn->dump();
    assert(0 && "Function failed verification!");
  }
}

void CodeGenFunction::StartFunction(const Decl *D, QualType RetTy, 
                                    llvm::Function *Fn,
                                    const FunctionArgList &Args) {
  CurFuncDecl = D;
  FnRetTy = RetTy;
  CurFn = Fn;
  assert(CurFn->isDeclaration() && "Function already has body?");

  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", CurFn);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::Int32Ty, "allocapt",
                                         EntryBB);

  ReturnBlock = llvm::BasicBlock::Create("return");
  ReturnValue = 0;
  if (!RetTy->isVoidType())
    ReturnValue = CreateTempAlloca(ConvertType(RetTy), "retval");
    
  Builder.SetInsertPoint(EntryBB);
  
  // Emit subprogram debug descriptor.
  // FIXME: The cast here is a huge hack.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    if (CGDebugInfo *DI = CGM.getDebugInfo()) {
      CompoundStmt* body = dyn_cast<CompoundStmt>(FD->getBody());
      if (body && body->getLBracLoc().isValid()) {
        DI->setLocation(body->getLBracLoc());
      }
      DI->EmitFunctionStart(FD, CurFn, Builder);
    }
  }

  EmitFunctionProlog(CurFn, FnRetTy, Args);
}

void CodeGenFunction::GenerateCode(const FunctionDecl *FD,
                                   llvm::Function *Fn) {
  FunctionArgList Args;
  if (FD->getNumParams()) {
    const FunctionTypeProto* FProto = FD->getType()->getAsFunctionTypeProto();
    assert(FProto && "Function def must have prototype!");

    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
      Args.push_back(std::make_pair(FD->getParamDecl(i), 
                                    FProto->getArgType(i)));
  }

  StartFunction(FD, FD->getResultType(), Fn, Args);

  EmitStmt(FD->getBody());
  
  const CompoundStmt *S = dyn_cast<CompoundStmt>(FD->getBody());
  if (S) {
    FinishFunction(S->getRBracLoc());
  } else {
    FinishFunction();
  }
}

/// isDummyBlock - Return true if BB is an empty basic block
/// with no predecessors.
bool CodeGenFunction::isDummyBlock(const llvm::BasicBlock *BB) {
  if (BB->empty() && pred_begin(BB) == pred_end(BB) && !BB->hasName()) 
    return true;
  return false;
}

/// StartBlock - Start new block named N. If insert block is a dummy block
/// then reuse it.
void CodeGenFunction::StartBlock(const char *N) {
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (!isDummyBlock(BB))
    EmitBlock(llvm::BasicBlock::Create(N));
  else
    BB->setName(N);
}

/// getCGRecordLayout - Return record layout info.
const CGRecordLayout *CodeGenFunction::getCGRecordLayout(CodeGenTypes &CGT,
                                                         QualType Ty) {
  const RecordType *RTy = Ty->getAsRecordType();
  assert (RTy && "Unexpected type. RecordType expected here.");

  return CGT.getCGRecordLayout(RTy->getDecl());
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenFunction::ErrorUnsupported(const Stmt *S, const char *Type,
                                       bool OmitOnError) {
  CGM.ErrorUnsupported(S, Type, OmitOnError);
}

unsigned CodeGenFunction::GetIDForAddrOfLabel(const LabelStmt *L) {
  // Use LabelIDs.size() as the new ID if one hasn't been assigned.
  return LabelIDs.insert(std::make_pair(L, LabelIDs.size())).first->second;
}

void CodeGenFunction::EmitMemSetToZero(llvm::Value *DestPtr, QualType Ty)
{
  const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");

  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo = getContext().getTypeInfo(Ty);

  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(LLVMPointerWidth);

  Builder.CreateCall4(CGM.getMemSetFn(), DestPtr,
                      llvm::ConstantInt::getNullValue(llvm::Type::Int8Ty),
                      // TypeInfo.first describes size in bits.
                      llvm::ConstantInt::get(IntPtr, TypeInfo.first/8),
                      llvm::ConstantInt::get(llvm::Type::Int32Ty, 
                                             TypeInfo.second/8));
}

void CodeGenFunction::EmitIndirectSwitches() {
  llvm::BasicBlock *Default;
  
  if (IndirectSwitches.empty())
    return;
  
  if (!LabelIDs.empty()) {
    Default = getBasicBlockForLabel(LabelIDs.begin()->first);
  } else {
    // No possible targets for indirect goto, just emit an infinite
    // loop.
    Default = llvm::BasicBlock::Create("indirectgoto.loop", CurFn);
    llvm::BranchInst::Create(Default, Default);
  }

  for (std::vector<llvm::SwitchInst*>::iterator i = IndirectSwitches.begin(),
         e = IndirectSwitches.end(); i != e; ++i) {
    llvm::SwitchInst *I = *i;
    
    I->setSuccessor(0, Default);
    for (std::map<const LabelStmt*,unsigned>::iterator LI = LabelIDs.begin(), 
           LE = LabelIDs.end(); LI != LE; ++LI) {
      I->addCase(llvm::ConstantInt::get(llvm::Type::Int32Ty,
                                        LI->second), 
                 getBasicBlockForLabel(LI->first));
    }
  }         
}
