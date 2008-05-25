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
#include "clang/AST/AST.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CFG.h"
using namespace clang;
using namespace CodeGen;

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm) 
  : CGM(cgm), Target(CGM.getContext().Target), SwitchInsn(NULL), 
    CaseRangeBlock(NULL) {}

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

bool CodeGenFunction::hasAggregateLLVMType(QualType T) {
  return !T->isRealType() && !T->isPointerLikeType() &&
         !T->isVoidType() && !T->isVectorType() && !T->isFunctionType();
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.  
// FIXME: This method contains a lot of code copied and pasted from
// GenerateCode.  This should be factored out.
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  llvm::SmallVector<const llvm::Type *, 16> ParamTypes;
  for (unsigned i=0 ; i<OMD->param_size() ; i++) {
    ParamTypes.push_back(ConvertType(OMD->getParamDecl(i)->getType()));
  }
  CurFn =CGM.getObjCRuntime()->MethodPreamble(ConvertType(OMD->getResultType()),
                      llvm::PointerType::getUnqual(llvm::Type::Int32Ty),
                                              ParamTypes.begin(),
                                              OMD->param_size(),
                                              OMD->isVariadic());
  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", CurFn);
  
  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::Int32Ty, "allocapt",
                                         EntryBB);

  FnRetTy = OMD->getResultType();

  Builder.SetInsertPoint(EntryBB);
  
  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = CurFn->arg_begin();
  
  // Name the struct return argument.
  // FIXME: Probably should be in the runtime, or it will trample the other
  // hidden arguments.
  if (hasAggregateLLVMType(OMD->getResultType())) {
    AI->setName("agg.result");
    ++AI;
  }

  // Add implicit parameters to the decl map.
  // TODO: Add something to AST to let the runtime specify the names and types
  // of these.
  llvm::Value *&DMEntry = LocalDeclMap[&(*OMD->getSelfDecl())];
  const llvm::Type *SelfTy = AI->getType();
  llvm::Value *DeclPtr = new llvm::AllocaInst(SelfTy, 0, "self.addr",
                                   AllocaInsertPt);

  // Store the initial value into the alloca.
  Builder.CreateStore(AI, DeclPtr);
  DMEntry = DeclPtr;
  ++AI; ++AI;


  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*OMD->getParamDecl(i), AI);
  }
  
  // Emit the function body.
  EmitStmt(OMD->getBody());
  
  // Emit a return for code that falls off the end. If insert point
  // is a dummy block with no predecessors then remove the block itself.
  llvm::BasicBlock *BB = Builder.GetInsertBlock();
  if (isDummyBlock(BB))
    BB->eraseFromParent();
  else {
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
  assert(!verifyFunction(*CurFn) && "Generated method is not well formed.");
}

llvm::Value *CodeGenFunction::LoadObjCSelf(void)
{
  if(const ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(CurFuncDecl)) {
    llvm::Value *SelfPtr = LocalDeclMap[&(*OMD->getSelfDecl())];
    return Builder.CreateLoad(SelfPtr, "self");
  }
  return NULL;
}

void CodeGenFunction::GenerateCode(const FunctionDecl *FD) {
  LLVMIntTy = ConvertType(getContext().IntTy);
  LLVMPointerWidth = static_cast<unsigned>(
    getContext().getTypeSize(getContext().getPointerType(getContext().VoidTy)));
  
  CurFuncDecl = FD;
  FnRetTy = FD->getType()->getAsFunctionType()->getResultType();

  CurFn = cast<llvm::Function>(CGM.GetAddrOfFunctionDecl(FD, true));
  assert(CurFn->isDeclaration() && "Function already has body?");
  
  // TODO: Set up linkage and many other things.  Note, this is a simple 
  // approximation of what we really want.
  if (FD->getStorageClass() == FunctionDecl::Static)
    CurFn->setLinkage(llvm::Function::InternalLinkage);
  else if (FD->getAttr<DLLImportAttr>())
    CurFn->setLinkage(llvm::Function::DLLImportLinkage);
  else if (FD->getAttr<DLLExportAttr>())
    CurFn->setLinkage(llvm::Function::DLLExportLinkage);
  else if (FD->getAttr<WeakAttr>() || FD->isInline())
    CurFn->setLinkage(llvm::Function::WeakLinkage);

  if (FD->getAttr<FastCallAttr>())
    CurFn->setCallingConv(llvm::CallingConv::Fast);

  if (const VisibilityAttr *attr = FD->getAttr<VisibilityAttr>())
    CodeGenModule::setVisibility(CurFn, attr->getVisibility());
  // FIXME: else handle -fvisibility


  unsigned FuncAttrs = 0;
  if (FD->getAttr<NoThrowAttr>())
    FuncAttrs |= llvm::ParamAttr::NoUnwind;
  if (FD->getAttr<NoReturnAttr>())
    FuncAttrs |= llvm::ParamAttr::NoReturn;
  
  if (FuncAttrs) {
    llvm::ParamAttrsWithIndex PAWI = 
      llvm::ParamAttrsWithIndex::get(0, FuncAttrs);
    CurFn->setParamAttrs(llvm::PAListPtr::get(&PAWI, 1));
  }

  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", CurFn);
  
  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::Int32Ty, "allocapt",
                                         EntryBB);
  
  Builder.SetInsertPoint(EntryBB);

  CGDebugInfo *DI = CGM.getDebugInfo();
  if (DI) {
    CompoundStmt* body = cast<CompoundStmt>(CurFuncDecl->getBody());
    if (body->getLBracLoc().isValid()) {
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
  
  for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*FD->getParamDecl(i), AI);
  }
  
  // Emit the function body.
  EmitStmt(FD->getBody());
  
  if (DI) {
    CompoundStmt* body = cast<CompoundStmt>(CurFuncDecl->getBody());
    if (body->getRBracLoc().isValid()) {
      DI->setLocation(body->getRBracLoc());
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

/// isDummyBlock - Return true if BB is an empty basic block
/// with no predecessors.
bool CodeGenFunction::isDummyBlock(const llvm::BasicBlock *BB) {
  if (BB->empty() && pred_begin(BB) == pred_end(BB)) 
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

/// WarnUnsupported - Print out a warning that codegen doesn't support the
/// specified stmt yet.
void CodeGenFunction::WarnUnsupported(const Stmt *S, const char *Type) {
  CGM.WarnUnsupported(S, Type);
}

