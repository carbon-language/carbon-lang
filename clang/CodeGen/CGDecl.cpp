//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/AST.h"
#include "llvm/Type.h"
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDecl(const Decl &D) {

  switch (D.getKind()) {
  default: assert(0 && "Unknown decl kind!");
  case Decl::FileVariable:
    assert(0 && "Should not see file-scope variables inside a function!");
  case Decl::ParmVariable:
    assert(0 && "Parmdecls should not be in declstmts!");
  case Decl::Typedef:   // typedef int X;
  case Decl::Function:  // void X();
  case Decl::Struct:    // struct X;
  case Decl::Union:     // union X;
  case Decl::Class:     // class X;
  case Decl::Enum:      // enum X;
    // None of these decls require codegen support.
    return;
    
  case Decl::BlockVariable:
    return EmitBlockVarDecl(cast<BlockVarDecl>(D));
  case Decl::EnumConstant:
    return EmitEnumConstantDecl(cast<EnumConstantDecl>(D));
  }
}

void CodeGenFunction::EmitEnumConstantDecl(const EnumConstantDecl &D) {
  assert(0 && "FIXME: Enum constant decls not implemented yet!");  
}

/// EmitBlockVarDecl - This method handles emission of any variable declaration
/// inside a function, including static vars etc.
void CodeGenFunction::EmitBlockVarDecl(const BlockVarDecl &D) {
  switch (D.getStorageClass()) {
  case VarDecl::Static:
    assert(0 && "FIXME: local static vars not implemented yet");
  case VarDecl::Extern:
    assert(0 && "FIXME: should call up to codegenmodule");
  default:
    assert((D.getStorageClass() == VarDecl::None ||
            D.getStorageClass() == VarDecl::Auto ||
            D.getStorageClass() == VarDecl::Register) &&
           "Unknown storage class");
    return EmitLocalBlockVarDecl(D);
  }
}

/// EmitLocalBlockVarDecl - Emit code and set up an entry in LocalDeclMap for a
/// variable declaration with auto, register, or no storage class specifier.
/// These turn into simple stack objects.
void CodeGenFunction::EmitLocalBlockVarDecl(const BlockVarDecl &D) {
  QualType Ty = D.getCanonicalType();

  llvm::Value *DeclPtr;
  if (Ty->isConstantSizeType()) {
    // A normal fixed sized variable becomes an alloca in the entry block.
    const llvm::Type *LTy = ConvertType(Ty);
    // TODO: Alignment
    DeclPtr = CreateTempAlloca(LTy, D.getName());
  } else {
    // TODO: Create a dynamic alloca.
    assert(0 && "FIXME: Local VLAs not implemented yet");
  }
  
  llvm::Value *&DMEntry = LocalDeclMap[&D];
  assert(DMEntry == 0 && "Decl already exists in localdeclmap!");
  DMEntry = DeclPtr;
  
  // FIXME: Evaluate initializer.
}

/// Emit an alloca for the specified parameter and set up LocalDeclMap.
void CodeGenFunction::EmitParmDecl(const ParmVarDecl &D, llvm::Value *Arg) {
  QualType Ty = D.getCanonicalType();
  
  llvm::Value *DeclPtr;
  if (!Ty->isConstantSizeType()) {
    // Variable sized values always are passed by-reference.
    DeclPtr = Arg;
  } else {
    // A fixed sized first class variable becomes an alloca in the entry block.
    const llvm::Type *LTy = ConvertType(Ty);
    if (LTy->isFirstClassType()) {
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
}

