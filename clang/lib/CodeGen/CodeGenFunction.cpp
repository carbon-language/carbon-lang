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
  CGDebugInfo *DI = CGM.getDebugInfo(); 
  if (DI) {
    if (EndLoc.isValid()) {
      DI->setLocation(EndLoc);
    }
    DI->EmitRegionEnd(CurFn, Builder);
  }
 
  // Emit a return for code that falls off the end. If insert point
  // is a dummy block with no predecessors then remove the block itself.
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (isDummyBlock(BB))
    BB->eraseFromParent();
  else {
    // FIXME: if this is C++ main, this should return 0.
    if (CurFn->getReturnType() == llvm::Type::VoidTy)
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(llvm::UndefValue::get(CurFn->getReturnType()));
  }
  assert(BreakContinueStack.empty() &&
         "mismatched push/pop in break/continue stack!");
  
  // Remove the AllocaInsertPt instruction, which is just a convenience for us.
  AllocaInsertPt->eraseFromParent();
  AllocaInsertPt = 0;
  
  // Verify that the function is well formed.
  assert(!verifyFunction(*CurFn) && "Generated function is not well formed.");
}

void CodeGenFunction::GenerateCode(const FunctionDecl *FD,
                                   llvm::Function *Fn) {
  CurFuncDecl = FD;
  FnRetTy = FD->getResultType();
  CurFn = Fn;
  assert(CurFn->isDeclaration() && "Function already has body?");

  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", CurFn);
  
  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::Int32Ty, "allocapt",
                                         EntryBB);
  
  Builder.SetInsertPoint(EntryBB);
  
  // Emit subprogram debug descriptor.
  CGDebugInfo *DI = CGM.getDebugInfo();
  if (DI) {
    CompoundStmt* body = dyn_cast<CompoundStmt>(FD->getBody());
    if (body && body->getLBracLoc().isValid()) {
      DI->setLocation(body->getLBracLoc());
    }
    DI->EmitFunctionStart(FD, CurFn, Builder);
  }

  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = CurFn->arg_begin();
  
  // Name the struct return argument.
  if (hasAggregateLLVMType(FD->getResultType())) {
    AI->setName("agg.result");
    ++AI;
  }

  if (FD->getNumParams()) {
    const FunctionTypeProto* FProto = FD->getType()->getAsFunctionTypeProto();
    assert(FProto && "Function def must have prototype!");
    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i, ++AI) {
      assert(AI != CurFn->arg_end() && "Argument mismatch!");
      const ParmVarDecl* CurParam = FD->getParamDecl(i);
      llvm::Value* V = AI;
      if (!getContext().typesAreCompatible(FProto->getArgType(i),
                                           CurParam->getType())) {
        // This must be a promotion, for something like
        // "void a(x) short x; {..."
        V = EmitScalarConversion(V, FProto->getArgType(i),
                                 CurParam->getType());
      }
      EmitParmDecl(*CurParam, V);
    }
  }

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
