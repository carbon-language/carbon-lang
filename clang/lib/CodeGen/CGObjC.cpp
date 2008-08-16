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
#include "clang/AST/DeclObjC.h"

using namespace clang;
using namespace CodeGen;

/// Emits an instance of NSConstantString representing the object.
llvm::Value *CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral *E) {
  std::string String(E->getString()->getStrData(), E->getString()->getByteLength());
  llvm::Constant *C = CGM.getObjCRuntime().GenerateConstantString(String);
  return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
}

/// Emit a selector.
llvm::Value *CodeGenFunction::EmitObjCSelectorExpr(const ObjCSelectorExpr *E) {
  // Untyped selector.
  // Note that this implementation allows for non-constant strings to be passed
  // as arguments to @selector().  Currently, the only thing preventing this
  // behaviour is the type checking in the front end.
  return CGM.getObjCRuntime().GetSelector(Builder, E->getSelector());
}



llvm::Value *CodeGenFunction::EmitObjCMessageExpr(const ObjCMessageExpr *E) {
  // Only the lookup mechanism and first two arguments of the method
  // implementation vary between runtimes.  We can get the receiver and
  // arguments in generic code.
  
  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  const Expr *ReceiverExpr = E->getReceiver();
  bool isSuperMessage = false;
  // Find the receiver
  llvm::Value *Receiver;
  if (!ReceiverExpr) {
    const ObjCInterfaceDecl *OID = E->getClassInfo().first;

    // Very special case, super send in class method. The receiver is
    // self (the class object) and the send uses super semantics.
    if (!OID) {
      assert(!strcmp(E->getClassName()->getName(), "super") &&
             "Unexpected missing class interface in message send.");
      OID = E->getMethodDecl()->getClassInterface();
      isSuperMessage = true;
    }

    Receiver = Runtime.GetClass(Builder, OID);   
  } else if (const PredefinedExpr *PDE =
               dyn_cast<PredefinedExpr>(E->getReceiver())) {
    assert(PDE->getIdentType() == PredefinedExpr::ObjCSuper);
    isSuperMessage = true;
    Receiver = LoadObjCSelf();
  } else {
    Receiver = EmitScalarExpr(E->getReceiver());
  }

  // Process the arguments
  unsigned ArgC = E->getNumArgs();
  llvm::SmallVector<llvm::Value*, 16> Args;
  for (unsigned i = 0; i != ArgC; ++i) {
    const Expr *ArgExpr = E->getArg(i);
    QualType ArgTy = ArgExpr->getType();
    if (!hasAggregateLLVMType(ArgTy)) {
      // Scalar argument is passed by-value.
      Args.push_back(EmitScalarExpr(ArgExpr));
    } else if (ArgTy->isAnyComplexType()) {
      // Make a temporary alloca to pass the argument.
      llvm::Value *DestMem = CreateTempAlloca(ConvertType(ArgTy));
      EmitComplexExprIntoAddr(ArgExpr, DestMem, false);
      Args.push_back(DestMem);
    } else {
      llvm::Value *DestMem = CreateTempAlloca(ConvertType(ArgTy));
      EmitAggExpr(ArgExpr, DestMem, false);
      Args.push_back(DestMem);
    }
  }

  if (isSuperMessage) {
    // super is only valid in an Objective-C method
    const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
    return Runtime.GenerateMessageSendSuper(Builder, ConvertType(E->getType()),
                                             OMD->getClassInterface()->getSuperClass(),
                                             Receiver, E->getSelector(),
                                             &Args[0], Args.size());
  }
  return Runtime.GenerateMessageSend(Builder, ConvertType(E->getType()),
                                      Receiver, E->getSelector(),
                                      &Args[0], Args.size());
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.  
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  CurFn = CGM.getObjCRuntime().GenerateMethod(OMD);
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
  
  // Name the struct return argument.
  if (hasAggregateLLVMType(OMD->getResultType())) {
    AI->setName("agg.result");
    ++AI;
  }

  // Add implicit parameters to the decl map.
  EmitParmDecl(*OMD->getSelfDecl(), AI); 
  ++AI;

  EmitParmDecl(*OMD->getCmdDecl(), AI); 
  ++AI;

  for (unsigned i = 0, e = OMD->getNumParams(); i != e; ++i, ++AI) {
    assert(AI != CurFn->arg_end() && "Argument mismatch!");
    EmitParmDecl(*OMD->getParamDecl(i), AI);
  }
  assert(AI == CurFn->arg_end() && "Argument mismatch");

  GenerateFunction(OMD->getBody());
}

llvm::Value *CodeGenFunction::LoadObjCSelf(void) {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  return Builder.CreateLoad(LocalDeclMap[OMD->getSelfDecl()], "self");
}

CGObjCRuntime::~CGObjCRuntime() {}
