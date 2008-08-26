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
  // FIXME: This bitcast should just be made an invariant on the Runtime.
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

llvm::Value *CodeGenFunction::EmitObjCProtocolExpr(const ObjCProtocolExpr *E) {
  // FIXME: This should pass the Decl not the name.
  return CGM.getObjCRuntime().GenerateProtocolRef(Builder, E->getProtocol());
}


RValue CodeGenFunction::EmitObjCMessageExpr(const ObjCMessageExpr *E) {
  // Only the lookup mechanism and first two arguments of the method
  // implementation vary between runtimes.  We can get the receiver and
  // arguments in generic code.
  
  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  const Expr *ReceiverExpr = E->getReceiver();
  bool isSuperMessage = false;
  bool isClassMessage = false;
  // Find the receiver
  llvm::Value *Receiver;
  if (!ReceiverExpr) {
    const ObjCInterfaceDecl *OID = E->getClassInfo().first;

    // Very special case, super send in class method. The receiver is
    // self (the class object) and the send uses super semantics.
    if (!OID) {
      assert(!strcmp(E->getClassName()->getName(), "super") &&
             "Unexpected missing class interface in message send.");
      isSuperMessage = true;
      Receiver = LoadObjCSelf();
    } else {
      Receiver = Runtime.GetClass(Builder, OID);
    }
    
    isClassMessage = true;
  } else if (const PredefinedExpr *PDE =
               dyn_cast<PredefinedExpr>(E->getReceiver())) {
    assert(PDE->getIdentType() == PredefinedExpr::ObjCSuper);
    isSuperMessage = true;
    Receiver = LoadObjCSelf();
  } else {
    Receiver = EmitScalarExpr(E->getReceiver());
  }

  if (isSuperMessage) {
    // super is only valid in an Objective-C method
    const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
    return Runtime.GenerateMessageSendSuper(*this, E,
                                            OMD->getClassInterface(),
                                            Receiver,
                                            isClassMessage);
  }
  return Runtime.GenerateMessageSend(*this, E, Receiver, isClassMessage);
}

/// StartObjCMethod - Begin emission of an ObjCMethod. This generates
/// the LLVM function and sets the other context used by
/// CodeGenFunction.

// FIXME: This should really be merged with GenerateCode.
void CodeGenFunction::StartObjCMethod(const ObjCMethodDecl *OMD) {
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
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.  
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  StartObjCMethod(OMD);
  EmitStmt(OMD->getBody());

  const CompoundStmt *S = dyn_cast<CompoundStmt>(OMD->getBody());
  if (S) {
    FinishFunction(S->getRBracLoc());
  } else {
    FinishFunction();
  }
}

// FIXME: I wasn't sure about the synthesis approach. If we end up
// generating an AST for the whole body we can just fall back to
// having a GenerateFunction which takes the body Stmt.

/// GenerateObjCGetter - Generate an Objective-C property getter
/// function. The given Decl must be either an ObjCCategoryImplDecl
/// or an ObjCImplementationDecl.
void CodeGenFunction::GenerateObjCGetter(const ObjCPropertyImplDecl *PID) {
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCMethodDecl *OMD = PD->getGetterMethodDecl();
  assert(OMD && "Invalid call to generate getter (empty method)");
  // FIXME: This is rather murky, we create this here since they will
  // not have been created by Sema for us.
  OMD->createImplicitParams(getContext());
  StartObjCMethod(OMD);

  // FIXME: What about nonatomic?
  SourceLocation Loc = PD->getLocation();
  ValueDecl *Self = OMD->getSelfDecl();
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  DeclRefExpr Base(Self, Self->getType(), Loc);
  ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base,
                          true, true);
  ReturnStmt Return(Loc, &IvarRef);
  EmitStmt(&Return);

  FinishFunction();
}

/// GenerateObjCSetter - Generate an Objective-C property setter
/// function. The given Decl must be either an ObjCCategoryImplDecl
/// or an ObjCImplementationDecl.
void CodeGenFunction::GenerateObjCSetter(const ObjCPropertyImplDecl *PID) {
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCMethodDecl *OMD = PD->getSetterMethodDecl();
  assert(OMD && "Invalid call to generate setter (empty method)");
  // FIXME: This is rather murky, we create this here since they will
  // not have been created by Sema for us.  
  OMD->createImplicitParams(getContext());
  StartObjCMethod(OMD);
  
  switch (PD->getSetterKind()) {
  case ObjCPropertyDecl::Assign: break;
  case ObjCPropertyDecl::Copy:
      CGM.ErrorUnsupported(PID, "Obj-C setter with 'copy'");
      break;
  case ObjCPropertyDecl::Retain:
      CGM.ErrorUnsupported(PID, "Obj-C setter with 'retain'");
      break;
  }

  // FIXME: What about nonatomic?
  SourceLocation Loc = PD->getLocation();
  ValueDecl *Self = OMD->getSelfDecl();
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  DeclRefExpr Base(Self, Self->getType(), Loc);
  ParmVarDecl *ArgDecl = OMD->getParamDecl(0);
  DeclRefExpr Arg(ArgDecl, ArgDecl->getType(), Loc);
  ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base,
                          true, true);
  BinaryOperator Assign(&IvarRef, &Arg, BinaryOperator::Assign,
                        Ivar->getType(), Loc);
  EmitStmt(&Assign);

  FinishFunction();
}

llvm::Value *CodeGenFunction::LoadObjCSelf(void) {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  return Builder.CreateLoad(LocalDeclMap[OMD->getSelfDecl()], "self");
}

CGObjCRuntime::~CGObjCRuntime() {}
