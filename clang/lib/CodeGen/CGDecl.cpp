//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Type.h"
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDecl(const Decl &D) {
  switch (D.getKind()) {
  default: assert(0 && "Unknown decl kind!");
  case Decl::ParmVar:
    assert(0 && "Parmdecls should not be in declstmts!");
  case Decl::Function:  // void X();
  case Decl::Record:    // struct/union/class X;
  case Decl::Enum:      // enum X;
  case Decl::EnumConstant: // enum ? { X = ? } 
  case Decl::CXXRecord: // struct/union/class X; [C++]
    // None of these decls require codegen support.
    return;
    
  case Decl::Var: {
    const VarDecl &VD = cast<VarDecl>(D);
    assert(VD.isBlockVarDecl() && 
           "Should not see file-scope variables inside a function!");
    return EmitBlockVarDecl(VD);
  }
        
  case Decl::Typedef: {   // typedef int X;
    const TypedefDecl &TD = cast<TypedefDecl>(D);
    QualType Ty = TD.getUnderlyingType();
    
    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  }
  }
}

/// EmitBlockVarDecl - This method handles emission of any variable declaration
/// inside a function, including static vars etc.
void CodeGenFunction::EmitBlockVarDecl(const VarDecl &D) {
  switch (D.getStorageClass()) {
  case VarDecl::Static:
    return EmitStaticBlockVarDecl(D);
  case VarDecl::Extern:
    // Don't emit it now, allow it to be emitted lazily on its first use.
    return;
  default:
    assert((D.getStorageClass() == VarDecl::None ||
            D.getStorageClass() == VarDecl::Auto ||
            D.getStorageClass() == VarDecl::Register) &&
           "Unknown storage class");
    return EmitLocalBlockVarDecl(D);
  }
}

llvm::GlobalValue *
CodeGenFunction::GenerateStaticBlockVarDecl(const VarDecl &D,
                                            bool NoInit,
                                            const char *Separator,
                                            llvm::GlobalValue
					    	::LinkageTypes Linkage) {
  QualType Ty = D.getType();
  assert(Ty->isConstantSizeType() && "VLAs can't be static");
  
  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(Ty);
  llvm::Constant *Init = 0;
  if ((D.getInit() == 0) || NoInit) {
    Init = llvm::Constant::getNullValue(LTy);
  } else {
    Init = CGM.EmitConstantExpr(D.getInit(), this);

    // If constant emission failed, then this should be a C++ static
    // initializer.
    if (!Init) {
      if (!getContext().getLangOptions().CPlusPlus) {
        CGM.ErrorUnsupported(D.getInit(), "constant l-value expression");
        Init = llvm::Constant::getNullValue(LTy);
      } else {
        return GenerateStaticCXXBlockVarDecl(D);
      }
    }
  }

  assert(Init && "Unable to create initialiser for static decl");

  std::string ContextName;
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(CurFuncDecl))
    ContextName = CGM.getMangledName(FD)->getName();
  else if (isa<ObjCMethodDecl>(CurFuncDecl))
    ContextName = std::string(CurFn->getNameStart(), 
                              CurFn->getNameStart() + CurFn->getNameLen());
  else
    assert(0 && "Unknown context for block var decl");

  llvm::GlobalValue *GV =
    new llvm::GlobalVariable(Init->getType(), Ty.isConstant(getContext()),
                             Linkage,
                             Init, ContextName + Separator +D.getNameAsString(),
                             &CGM.getModule(), 0, Ty.getAddressSpace());

  return GV;
}

void CodeGenFunction::EmitStaticBlockVarDecl(const VarDecl &D) { 

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  
  llvm::GlobalValue *GV;
  GV = GenerateStaticBlockVarDecl(D, false, ".",
                                  llvm::GlobalValue::InternalLinkage);

  // FIXME: Merge attribute handling.
  if (const AnnotateAttr *AA = D.getAttr<AnnotateAttr>()) {
    SourceManager &SM = CGM.getContext().getSourceManager();
    llvm::Constant *Ann =
      CGM.EmitAnnotateAttr(GV, AA, 
                           SM.getInstantiationLineNumber(D.getLocation()));
    CGM.AddAnnotation(Ann);
  }

  if (const SectionAttr *SA = D.getAttr<SectionAttr>())
    GV->setSection(SA->getName());
  
  if (D.getAttr<UsedAttr>())
    CGM.AddUsedGlobal(GV);

  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(D.getType());
  const llvm::Type *LPtrTy =
    llvm::PointerType::get(LTy, D.getType().getAddressSpace());
  DMEntry = llvm::ConstantExpr::getBitCast(GV, LPtrTy);

  // Emit global variable debug descriptor for static vars.
  CGDebugInfo *DI = getDebugInfo();
  if(DI) {
    DI->setLocation(D.getLocation());
    DI->EmitGlobalVariable(static_cast<llvm::GlobalVariable *>(GV), &D);
  }
}
  
/// EmitLocalBlockVarDecl - Emit code and set up an entry in LocalDeclMap for a
/// variable declaration with auto, register, or no storage class specifier.
/// These turn into simple stack objects, or GlobalValues depending on target.
void CodeGenFunction::EmitLocalBlockVarDecl(const VarDecl &D) {
  QualType Ty = D.getType();

  llvm::Value *DeclPtr;
  if (Ty->isConstantSizeType()) {
    if (!Target.useGlobalsForAutomaticVariables()) {
      // A normal fixed sized variable becomes an alloca in the entry block.
      const llvm::Type *LTy = ConvertType(Ty);
      llvm::AllocaInst *Alloc =
        CreateTempAlloca(LTy, CGM.getMangledName(&D)->getName());
      unsigned align = getContext().getTypeAlign(Ty);
      if (const AlignedAttr* AA = D.getAttr<AlignedAttr>())
        align = std::max(align, AA->getAlignment());
      Alloc->setAlignment(align >> 3);
      DeclPtr = Alloc;
    } else {
      // Targets that don't support recursion emit locals as globals.
      const char *Class =
        D.getStorageClass() == VarDecl::Register ? ".reg." : ".auto.";
      DeclPtr = GenerateStaticBlockVarDecl(D, true, Class, 
                                           llvm::GlobalValue
                                           ::InternalLinkage);
    }
    
    if (Ty->isVariablyModifiedType())
      EmitVLASize(Ty);
  } else {
    if (!DidCallStackSave) {
      // Save the stack.
      const llvm::Type *LTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
      llvm::Value *Stack = CreateTempAlloca(LTy, "saved_stack");
      
      llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::stacksave);
      llvm::Value *V = Builder.CreateCall(F);
      
      Builder.CreateStore(V, Stack);

      DidCallStackSave = true;
      
      {
        // Push a cleanup block and restore the stack there.
        CleanupScope scope(*this);
      
        V = Builder.CreateLoad(Stack, "tmp");
        llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::stackrestore);
        Builder.CreateCall(F, V);
      }
    }
    
    // Get the element type.
    const llvm::Type *LElemTy = ConvertType(Ty);    
    const llvm::Type *LElemPtrTy =
      llvm::PointerType::get(LElemTy, D.getType().getAddressSpace());

    llvm::Value *VLASize = EmitVLASize(Ty);

    // Downcast the VLA size expression
    VLASize = Builder.CreateIntCast(VLASize, llvm::Type::Int32Ty, false, "tmp");
    
    // Allocate memory for the array.
    llvm::Value *VLA = Builder.CreateAlloca(llvm::Type::Int8Ty, VLASize, "vla");
    DeclPtr = Builder.CreateBitCast(VLA, LElemPtrTy, "tmp");
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for local var declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D.getLocation());
    DI->EmitDeclareOfAutoVariable(&D, DeclPtr, Builder);
  }

  // If this local has an initializer, emit it now.
  if (const Expr *Init = D.getInit()) {
    if (!hasAggregateLLVMType(Init->getType())) {
      llvm::Value *V = EmitScalarExpr(Init);
      Builder.CreateStore(V, DeclPtr, D.getType().isVolatileQualified());
    } else if (Init->getType()->isAnyComplexType()) {
      EmitComplexExprIntoAddr(Init, DeclPtr, D.getType().isVolatileQualified());
    } else {
      EmitAggExpr(Init, DeclPtr, D.getType().isVolatileQualified());
    }
  }

  // Handle the cleanup attribute
  if (const CleanupAttr *CA = D.getAttr<CleanupAttr>()) {
    const FunctionDecl *FD = CA->getFunctionDecl();
    
    llvm::Constant* F = CGM.GetAddrOfFunction(FD);
    assert(F && "Could not find function!");
  
    CleanupScope scope(*this);

    CallArgList Args;
    Args.push_back(std::make_pair(RValue::get(DeclPtr), 
                                  getContext().getPointerType(D.getType())));
      
    EmitCall(CGM.getTypes().getFunctionInfo(FD), F, Args);
  }
}

/// Emit an alloca (or GlobalValue depending on target) 
/// for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const VarDecl &D, llvm::Value *Arg) {
  // FIXME: Why isn't ImplicitParamDecl a ParmVarDecl?
  assert((isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D)) &&
         "Invalid argument to EmitParmDecl");
  QualType Ty = D.getType();
  
  llvm::Value *DeclPtr;
  if (!Ty->isConstantSizeType()) {
    // Variable sized values always are passed by-reference.
    DeclPtr = Arg;
  } else if (Target.useGlobalsForAutomaticVariables()) {
    // Targets that don't have stack use global address space for parameters.
    // Specify external linkage for such globals so that llvm optimizer do
    // not assume there values initialized as zero.
    DeclPtr = GenerateStaticBlockVarDecl(D, true, ".auto.",
                                         llvm::GlobalValue::ExternalLinkage);
  } else {
    // A fixed sized single-value variable becomes an alloca in the entry block.
    const llvm::Type *LTy = ConvertType(Ty);
    if (LTy->isSingleValueType()) {
      // TODO: Alignment
      std::string Name = D.getNameAsString();
      Name += ".addr";
      DeclPtr = CreateTempAlloca(LTy, Name.c_str());
      
      // Store the initial value into the alloca.
      Builder.CreateStore(Arg, DeclPtr,Ty.isVolatileQualified());
    } else {
      // Otherwise, if this is an aggregate, just use the input pointer.
      DeclPtr = Arg;
    }
    Arg->setName(D.getNameAsString());
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for param declaration.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D.getLocation());
    DI->EmitDeclareOfArgVariable(&D, DeclPtr, Builder);
  }
}

