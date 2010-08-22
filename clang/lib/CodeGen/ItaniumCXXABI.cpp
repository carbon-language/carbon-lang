//===------- ItaniumCXXABI.cpp - Emit LLVM Code from ASTs for a Module ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targetting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  http://www.codesourcery.com/public/cxx-abi/abi.html
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0041c/IHI0041C_cppabi.pdf
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "Mangle.h"
#include <clang/AST/Type.h>
#include <llvm/Value.h>

using namespace clang;
using namespace CodeGen;

namespace {
class ItaniumCXXABI : public CodeGen::CGCXXABI {
protected:
  CodeGenModule &CGM;
  CodeGen::MangleContext MangleCtx;
public:
  ItaniumCXXABI(CodeGen::CodeGenModule &CGM) :
    CGM(CGM), MangleCtx(CGM.getContext(), CGM.getDiags()) { }

  CodeGen::MangleContext &getMangleContext() {
    return MangleCtx;
  }

  llvm::Value *EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                               llvm::Value *&This,
                                               llvm::Value *MemFnPtr,
                                               const MemberPointerType *MPT);
};

class ARMCXXABI : public ItaniumCXXABI {
public:
  ARMCXXABI(CodeGen::CodeGenModule &CGM) : ItaniumCXXABI(CGM) {}
};
}

CodeGen::CGCXXABI *CodeGen::CreateItaniumCXXABI(CodeGenModule &CGM) {
  return new ItaniumCXXABI(CGM);
}

CodeGen::CGCXXABI *CodeGen::CreateARMCXXABI(CodeGenModule &CGM) {
  return new ARMCXXABI(CGM);
}

llvm::Value *
ItaniumCXXABI::EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                               llvm::Value *&This,
                                               llvm::Value *MemFnPtr,
                                               const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;

  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());

  const llvm::FunctionType *FTy = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(RD, FPT),
                                   FPT->isVariadic());

  llvm::BasicBlock *FnVirtual = CGF.createBasicBlock("fn.virtual");
  llvm::BasicBlock *FnNonVirtual = CGF.createBasicBlock("fn.nonvirtual");
  llvm::BasicBlock *FnEnd = CGF.createBasicBlock("fn.end");

  // Load the adjustment, which is in the second field.
  llvm::Value *Adj = Builder.CreateStructGEP(MemFnPtr, 1);
  Adj = Builder.CreateLoad(Adj, "mem.fn.adj");

  // Apply the adjustment and cast back to the original struct type
  // for consistency.
  llvm::Value *Ptr = Builder.CreateBitCast(This, Builder.getInt8PtrTy(), "ptr");
  Ptr = Builder.CreateInBoundsGEP(Ptr, Adj, "adj");
  This = Builder.CreateBitCast(Ptr, This->getType(), "this");
  
  // Load the function pointer.
  llvm::Value *FnPtr = Builder.CreateStructGEP(MemFnPtr, 0, "mem.fn.ptr");
  llvm::Value *FnAsInt = Builder.CreateLoad(FnPtr, "fn");
  
  // If the LSB in the function pointer is 1, the function pointer points to
  // a virtual function.
  llvm::Constant *iptr_1 = llvm::ConstantInt::get(FnAsInt->getType(), 1);
  llvm::Value *IsVirtual = Builder.CreateAnd(FnAsInt, iptr_1);
  IsVirtual = Builder.CreateIsNotNull(IsVirtual);
  Builder.CreateCondBr(IsVirtual, FnVirtual, FnNonVirtual);

  // In the virtual path, the adjustment left 'This' pointing to the
  // vtable of the correct base subobject.  The "function pointer" is an
  // offset within the vtable (+1 for the virtual flag).
  CGF.EmitBlock(FnVirtual);

  // Cast the adjusted this to a pointer to vtable pointer and load.
  const llvm::Type *VTableTy = Builder.getInt8PtrTy();
  llvm::Value *VTable = Builder.CreateBitCast(This, VTableTy->getPointerTo());
  VTable = Builder.CreateLoad(VTable);

  // Apply the offset.
  llvm::Value *VTableOffset = Builder.CreateSub(FnAsInt, iptr_1);
  VTable = Builder.CreateGEP(VTable, VTableOffset, "fn");

  // Load the virtual function to call.
  VTable = Builder.CreateBitCast(VTable, FTy->getPointerTo()->getPointerTo());
  llvm::Value *VirtualFn = Builder.CreateLoad(VTable, "virtualfn");
  CGF.EmitBranch(FnEnd);

  // In the non-virtual path, the function pointer is actually a
  // function pointer.
  CGF.EmitBlock(FnNonVirtual);
  llvm::Value *NonVirtualFn =
    Builder.CreateIntToPtr(FnAsInt, FTy->getPointerTo());
  
  // We're done.
  CGF.EmitBlock(FnEnd);
  llvm::PHINode *Callee = Builder.CreatePHI(FTy->getPointerTo());
  Callee->reserveOperandSpace(2);
  Callee->addIncoming(VirtualFn, FnVirtual);
  Callee->addIncoming(NonVirtualFn, FnNonVirtual);
  return Callee;
}
