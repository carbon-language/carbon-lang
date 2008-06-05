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
#include "clang/AST/AST.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Type.h"
#include "llvm/Support/Dwarf.h"
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDecl(const Decl &D) {
  switch (D.getKind()) {
  default: assert(0 && "Unknown decl kind!");
  case Decl::ParmVar:
    assert(0 && "Parmdecls should not be in declstmts!");
  case Decl::Typedef:   // typedef int X;
  case Decl::Function:  // void X();
  case Decl::Struct:    // struct X;
  case Decl::Union:     // union X;
  case Decl::Class:     // class X;
  case Decl::Enum:      // enum X;
    // None of these decls require codegen support.
    return;
    
  case Decl::Var:
    if (cast<VarDecl>(D).isBlockVarDecl())
      return EmitBlockVarDecl(cast<VarDecl>(D));
    assert(0 && "Should not see file-scope variables inside a function!");
  
  case Decl::EnumConstant:
    return EmitEnumConstantDecl(cast<EnumConstantDecl>(D));
  }
}

void CodeGenFunction::EmitEnumConstantDecl(const EnumConstantDecl &D) {
  assert(0 && "FIXME: Enum constant decls not implemented yet!");  
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
                                            const char *Separator) {
  QualType Ty = D.getType();
  assert(Ty->isConstantSizeType() && "VLAs can't be static");
  
  const llvm::Type *LTy = CGM.getTypes().ConvertTypeForMem(Ty);
  llvm::Constant *Init = 0;
  if ((D.getInit() == 0) || NoInit) {
    Init = llvm::Constant::getNullValue(LTy);
  } else {
    Init = CGM.EmitConstantExpr(D.getInit(), this);
  }

  assert(Init && "Unable to create initialiser for static decl");

  std::string ContextName;
  if (const FunctionDecl * FD = dyn_cast<FunctionDecl>(CurFuncDecl))
    ContextName = FD->getName();
  else
    assert(0 && "Unknown context for block var decl"); // FIXME Handle objc.

  llvm::GlobalValue *GV = 
    new llvm::GlobalVariable(LTy, false, llvm::GlobalValue::InternalLinkage,
                             Init, ContextName + Separator + D.getName(),
                             &CGM.getModule(), 0, Ty.getAddressSpace());

  return GV;
}

void CodeGenFunction::EmitStaticBlockVarDecl(const VarDecl &D) { 

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  
  llvm::GlobalValue *GV = GenerateStaticBlockVarDecl(D, false, ".");

  if (const AnnotateAttr *AA = D.getAttr<AnnotateAttr>()) {
    SourceManager &SM = CGM.getContext().getSourceManager();
    llvm::Constant *Ann =
      CGM.EmitAnnotateAttr(GV, AA, SM.getLogicalLineNumber(D.getLocation()));
    CGM.AddAnnotation(Ann);
  }

  DMEntry = GV;

  // Emit global variable debug descriptor for static vars.
  CGDebugInfo *DI = CGM.getDebugInfo();
  if(DI) {
    if(D.getLocation().isValid())
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
      llvm::AllocaInst * Alloc = CreateTempAlloca(LTy, D.getName());
      unsigned align = getContext().getTypeAlign(Ty);
      if (const AlignedAttr* AA = D.getAttr<AlignedAttr>())
        align = std::max(align, AA->getAlignment());
      Alloc->setAlignment(align >> 3);
      DeclPtr = Alloc;
    } else {
      // Targets that don't support recursion emit locals as globals.
      const char *Class =
        D.getStorageClass() == VarDecl::Register ? ".reg." : ".auto.";
      DeclPtr = GenerateStaticBlockVarDecl(D, true, Class);
    }
  } else {
    // TODO: Create a dynamic alloca.
    assert(0 && "FIXME: Local VLAs not implemented yet");
  }
  
  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for local var declaration.
  CGDebugInfo *DI = CGM.getDebugInfo();
  if(DI) {
    if(D.getLocation().isValid())
      DI->setLocation(D.getLocation());
    DI->EmitDeclare(&D, llvm::dwarf::DW_TAG_auto_variable,
                    DeclPtr, Builder);
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
}

/// Emit an alloca (or GlobalValue depending on target) 
/// for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const ParmVarDecl &D, llvm::Value *Arg) {
  QualType Ty = D.getType();
  
  llvm::Value *DeclPtr;
  if (!Ty->isConstantSizeType()) {
    // Variable sized values always are passed by-reference.
    DeclPtr = Arg;
  } else if (Target.useGlobalsForAutomaticVariables()) {
    DeclPtr = GenerateStaticBlockVarDecl(D, true, ".arg.");
  } else {
    // A fixed sized single-value variable becomes an alloca in the entry block.
    const llvm::Type *LTy = ConvertType(Ty);
    if (LTy->isSingleValueType()) {
      // TODO: Alignment
      DeclPtr = new llvm::AllocaInst(LTy, 0, std::string(D.getName())+".addr",
                                     AllocaInsertPt);
      
      // Store the initial value into the alloca.
      Builder.CreateStore(Arg, DeclPtr);
    } else {
      // Otherwise, if this is an aggregate, just use the input pointer.
      DeclPtr = Arg;
    }
    Arg->setName(D.getName());
  }

  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;

  // Emit debug info for param declaration.
  CGDebugInfo *DI = CGM.getDebugInfo();
  if(DI) {
    if(D.getLocation().isValid())
      DI->setLocation(D.getLocation());
    DI->EmitDeclare(&D, llvm::dwarf::DW_TAG_arg_variable,
                    DeclPtr, Builder);
  }

}

