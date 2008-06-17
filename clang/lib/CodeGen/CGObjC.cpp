//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Objective-C code as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/Constant.h"
#include "llvm/Function.h"

using namespace clang;
using namespace CodeGen;

llvm::Value *CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral *E){
  std::string S(E->getString()->getStrData(), E->getString()->getByteLength());
  return CGM.GetAddrOfConstantCFString(S);
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.  
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {

  llvm::SmallVector<const llvm::Type *, 16> ParamTypes;
  for (unsigned i=0 ; i<OMD->param_size() ; i++) {
    const llvm::Type *Ty = ConvertType(OMD->getParamDecl(i)->getType());
    if (Ty->isFirstClassType())
      ParamTypes.push_back(Ty);
    else
      ParamTypes.push_back(llvm::PointerType::getUnqual(Ty));
  }
  std::string CategoryName = "";
  if (ObjCCategoryImplDecl *OCD =
      dyn_cast<ObjCCategoryImplDecl>(OMD->getMethodContext())) {
    CategoryName = OCD->getName();
  }
  const llvm::Type *ReturnTy = CGM.getTypes().ConvertReturnType(OMD->getResultType());
  CurFn = CGM.getObjCRuntime()->MethodPreamble(
                                              OMD->getClassInterface()->getName(),
                                              CategoryName,
                                              OMD->getSelector().getName(),
                                              ReturnTy,
                                              llvm::PointerType::getUnqual(
                                              llvm::Type::Int32Ty),
                                              ParamTypes.begin(),
                                              OMD->param_size(),
                                              !OMD->isInstance(),
                                              OMD->isVariadic());
  llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create("entry", CurFn);
  
  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(llvm::Type::Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, llvm::Type::Int32Ty, "allocapt",
                                         EntryBB);

  FnRetTy = OMD->getResultType();
  CurFuncDecl = OMD;

  Builder.SetInsertPoint(EntryBB);
  
  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = CurFn->arg_begin();
  
  if (hasAggregateLLVMType(OMD->getResultType())) {
    ++AI;
  }
  // Add implicit parameters to the decl map.
  // TODO: Add something to AST to let the runtime specify the names and types
  // of these.

  llvm::Value *&SelfEntry = LocalDeclMap[OMD->getSelfDecl()];
  const llvm::Type *IPTy = AI->getType();
  llvm::Value *DeclPtr = new llvm::AllocaInst(IPTy, 0, AI->getName() +
      ".addr", AllocaInsertPt);
  // Store the initial value into the alloca.
  Builder.CreateStore(AI, DeclPtr);
  SelfEntry = DeclPtr;
  ++AI;
  llvm::Value *&CmdEntry = LocalDeclMap[OMD->getCmdDecl()];
  IPTy = AI->getType();
  DeclPtr = new llvm::AllocaInst(IPTy, 0, AI->getName() +
      ".addr", AllocaInsertPt);
  // Store the initial value into the alloca.
  Builder.CreateStore(AI, DeclPtr);
  CmdEntry = DeclPtr;

  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*OMD->getParamDecl(i), AI);
  }
  
  GenerateFunction(OMD->getBody());
}

llvm::Value *CodeGenFunction::LoadObjCSelf(void)
{
  if (const ObjCMethodDecl *OMD = dyn_cast<ObjCMethodDecl>(CurFuncDecl)) {
    ValueDecl *Decl = OMD->getSelfDecl();
    llvm::Value *SelfPtr = LocalDeclMap[&(*(Decl))];
    return Builder.CreateLoad(SelfPtr, "self");
  }
  return NULL;
}

CGObjCRuntime::~CGObjCRuntime() {}
