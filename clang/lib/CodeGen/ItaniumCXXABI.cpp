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
  bool IsARM;
public:
  ItaniumCXXABI(CodeGen::CodeGenModule &CGM, bool IsARM = false) :
    CGM(CGM), MangleCtx(CGM.getContext(), CGM.getDiags()), IsARM(IsARM) { }

  CodeGen::MangleContext &getMangleContext() {
    return MangleCtx;
  }

  bool RequiresNonZeroInitializer(QualType T);
  bool RequiresNonZeroInitializer(const CXXRecordDecl *D);

  llvm::Value *EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                               llvm::Value *&This,
                                               llvm::Value *MemFnPtr,
                                               const MemberPointerType *MPT);

  void EmitMemberFunctionPointerConversion(CodeGenFunction &CGF,
                                           const CastExpr *E,
                                           llvm::Value *Src,
                                           llvm::Value *Dest,
                                           bool VolatileDest);

  llvm::Constant *EmitMemberFunctionPointerConversion(llvm::Constant *C,
                                                      const CastExpr *E);

  void EmitNullMemberFunctionPointer(CodeGenFunction &CGF,
                                     const MemberPointerType *MPT,
                                     llvm::Value *Dest,
                                     bool VolatileDest);

  llvm::Constant *EmitNullMemberFunctionPointer(const MemberPointerType *MPT);

};

class ARMCXXABI : public ItaniumCXXABI {
public:
  ARMCXXABI(CodeGen::CodeGenModule &CGM) : ItaniumCXXABI(CGM, /*ARM*/ true) {}
};
}

CodeGen::CGCXXABI *CodeGen::CreateItaniumCXXABI(CodeGenModule &CGM) {
  return new ItaniumCXXABI(CGM);
}

CodeGen::CGCXXABI *CodeGen::CreateARMCXXABI(CodeGenModule &CGM) {
  return new ARMCXXABI(CGM);
}

/// In the Itanium and ARM ABIs, method pointers have the form:
///   struct { ptrdiff_t ptr; ptrdiff_t adj; } memptr;
///
/// In the Itanium ABI:
///  - method pointers are virtual if (memptr.ptr & 1) is nonzero
///  - the this-adjustment is (memptr.adj)
///  - the virtual offset is (memptr.ptr - 1)
///
/// In the ARM ABI:
///  - method pointers are virtual if (memptr.adj & 1) is nonzero
///  - the this-adjustment is (memptr.adj >> 1)
///  - the virtual offset is (memptr.ptr)
/// ARM uses 'adj' for the virtual flag because Thumb functions
/// may be only single-byte aligned.
///
/// If the member is virtual, the adjusted 'this' pointer points
/// to a vtable pointer from which the virtual offset is applied.
///
/// If the member is non-virtual, memptr.ptr is the address of
/// the function to call.
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

  const llvm::IntegerType *ptrdiff = CGF.IntPtrTy;
  llvm::Constant *ptrdiff_1 = llvm::ConstantInt::get(ptrdiff, 1);

  llvm::BasicBlock *FnVirtual = CGF.createBasicBlock("memptr.virtual");
  llvm::BasicBlock *FnNonVirtual = CGF.createBasicBlock("memptr.nonvirtual");
  llvm::BasicBlock *FnEnd = CGF.createBasicBlock("memptr.end");

  // Load memptr.adj, which is in the second field.
  llvm::Value *RawAdj = Builder.CreateStructGEP(MemFnPtr, 1);
  RawAdj = Builder.CreateLoad(RawAdj, "memptr.adj");

  // Compute the true adjustment.
  llvm::Value *Adj = RawAdj;
  if (IsARM)
    Adj = Builder.CreateAShr(Adj, ptrdiff_1, "memptr.adj.shifted");

  // Apply the adjustment and cast back to the original struct type
  // for consistency.
  llvm::Value *Ptr = Builder.CreateBitCast(This, Builder.getInt8PtrTy());
  Ptr = Builder.CreateInBoundsGEP(Ptr, Adj);
  This = Builder.CreateBitCast(Ptr, This->getType(), "this.adjusted");
  
  // Load the function pointer.
  llvm::Value *FnPtr = Builder.CreateStructGEP(MemFnPtr, 0);
  llvm::Value *FnAsInt = Builder.CreateLoad(FnPtr, "memptr.ptr");
  
  // If the LSB in the function pointer is 1, the function pointer points to
  // a virtual function.
  llvm::Value *IsVirtual;
  if (IsARM)
    IsVirtual = Builder.CreateAnd(RawAdj, ptrdiff_1);
  else
    IsVirtual = Builder.CreateAnd(FnAsInt, ptrdiff_1);
  IsVirtual = Builder.CreateIsNotNull(IsVirtual, "memptr.isvirtual");
  Builder.CreateCondBr(IsVirtual, FnVirtual, FnNonVirtual);

  // In the virtual path, the adjustment left 'This' pointing to the
  // vtable of the correct base subobject.  The "function pointer" is an
  // offset within the vtable (+1 for the virtual flag on non-ARM).
  CGF.EmitBlock(FnVirtual);

  // Cast the adjusted this to a pointer to vtable pointer and load.
  const llvm::Type *VTableTy = Builder.getInt8PtrTy();
  llvm::Value *VTable = Builder.CreateBitCast(This, VTableTy->getPointerTo());
  VTable = Builder.CreateLoad(VTable, "memptr.vtable");

  // Apply the offset.
  llvm::Value *VTableOffset = FnAsInt;
  if (!IsARM) VTableOffset = Builder.CreateSub(VTableOffset, ptrdiff_1);
  VTable = Builder.CreateGEP(VTable, VTableOffset);

  // Load the virtual function to call.
  VTable = Builder.CreateBitCast(VTable, FTy->getPointerTo()->getPointerTo());
  llvm::Value *VirtualFn = Builder.CreateLoad(VTable, "memptr.virtualfn");
  CGF.EmitBranch(FnEnd);

  // In the non-virtual path, the function pointer is actually a
  // function pointer.
  CGF.EmitBlock(FnNonVirtual);
  llvm::Value *NonVirtualFn =
    Builder.CreateIntToPtr(FnAsInt, FTy->getPointerTo(), "memptr.nonvirtualfn");
  
  // We're done.
  CGF.EmitBlock(FnEnd);
  llvm::PHINode *Callee = Builder.CreatePHI(FTy->getPointerTo());
  Callee->reserveOperandSpace(2);
  Callee->addIncoming(VirtualFn, FnVirtual);
  Callee->addIncoming(NonVirtualFn, FnNonVirtual);
  return Callee;
}

/// Perform a derived-to-base or base-to-derived member pointer conversion.
void ItaniumCXXABI::EmitMemberFunctionPointerConversion(CodeGenFunction &CGF,
                                                        const CastExpr *E,
                                                        llvm::Value *Src,
                                                        llvm::Value *Dest,
                                                        bool VolatileDest) {
  assert(E->getCastKind() == CastExpr::CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CastExpr::CK_BaseToDerivedMemberPointer);

  CGBuilderTy &Builder = CGF.Builder;

  const MemberPointerType *SrcTy =
    E->getSubExpr()->getType()->getAs<MemberPointerType>();
  const MemberPointerType *DestTy = E->getType()->getAs<MemberPointerType>();

  const CXXRecordDecl *SrcDecl = SrcTy->getClass()->getAsCXXRecordDecl();
  const CXXRecordDecl *DestDecl = DestTy->getClass()->getAsCXXRecordDecl();

  llvm::Value *SrcPtr = Builder.CreateStructGEP(Src, 0, "src.ptr");
  SrcPtr = Builder.CreateLoad(SrcPtr);
    
  llvm::Value *SrcAdj = Builder.CreateStructGEP(Src, 1, "src.adj");
  SrcAdj = Builder.CreateLoad(SrcAdj);
    
  llvm::Value *DstPtr = Builder.CreateStructGEP(Dest, 0, "dst.ptr");
  Builder.CreateStore(SrcPtr, DstPtr, VolatileDest);
    
  llvm::Value *DstAdj = Builder.CreateStructGEP(Dest, 1, "dst.adj");

  bool DerivedToBase =
    E->getCastKind() == CastExpr::CK_DerivedToBaseMemberPointer;

  const CXXRecordDecl *BaseDecl, *DerivedDecl;
  if (DerivedToBase)
    DerivedDecl = SrcDecl, BaseDecl = DestDecl;
  else
    BaseDecl = SrcDecl, DerivedDecl = DestDecl;

  if (llvm::Constant *Adj = 
        CGF.CGM.GetNonVirtualBaseClassOffset(DerivedDecl,
                                             E->path_begin(),
                                             E->path_end())) {
    if (DerivedToBase)
      SrcAdj = Builder.CreateSub(SrcAdj, Adj, "adj");
    else
      SrcAdj = Builder.CreateAdd(SrcAdj, Adj, "adj");
  }
    
  Builder.CreateStore(SrcAdj, DstAdj, VolatileDest);
}

llvm::Constant *
ItaniumCXXABI::EmitMemberFunctionPointerConversion(llvm::Constant *C,
                                                   const CastExpr *E) {
  const MemberPointerType *SrcTy = 
    E->getSubExpr()->getType()->getAs<MemberPointerType>();
  const MemberPointerType *DestTy = 
    E->getType()->getAs<MemberPointerType>();

  bool DerivedToBase =
    E->getCastKind() == CastExpr::CK_DerivedToBaseMemberPointer;

  const CXXRecordDecl *DerivedDecl;
  if (DerivedToBase)
    DerivedDecl = SrcTy->getClass()->getAsCXXRecordDecl();
  else
    DerivedDecl = DestTy->getClass()->getAsCXXRecordDecl();

  // Calculate the offset to the base class.
  llvm::Constant *Offset = 
    CGM.GetNonVirtualBaseClassOffset(DerivedDecl,
                                     E->path_begin(),
                                     E->path_end());
  // If there's no offset, we're done.
  if (!Offset) return C;

  llvm::ConstantStruct *CS = cast<llvm::ConstantStruct>(C);

  llvm::Constant *Values[2] = {
    CS->getOperand(0),
    llvm::ConstantExpr::getAdd(CS->getOperand(1), Offset)
  };
  return llvm::ConstantStruct::get(CGM.getLLVMContext(), Values, 2,
                                   /*Packed=*/false);
}        


void ItaniumCXXABI::EmitNullMemberFunctionPointer(CodeGenFunction &CGF,
                                                  const MemberPointerType *MPT,
                                                  llvm::Value *Dest,
                                                  bool VolatileDest) {
  // Should this be "unabstracted" and implemented in terms of the
  // Constant version?

  CGBuilderTy &Builder = CGF.Builder;

  const llvm::IntegerType *PtrDiffTy = CGF.IntPtrTy;
  llvm::Value *Zero = llvm::ConstantInt::get(PtrDiffTy, 0);

  llvm::Value *Ptr = Builder.CreateStructGEP(Dest, 0, "ptr");
  Builder.CreateStore(Zero, Ptr, VolatileDest);
    
  llvm::Value *Adj = Builder.CreateStructGEP(Dest, 1, "adj");
  Builder.CreateStore(Zero, Adj, VolatileDest);
}

llvm::Constant *
ItaniumCXXABI::EmitNullMemberFunctionPointer(const MemberPointerType *MPT) {
  return CGM.EmitNullConstant(QualType(MPT, 0));
}

bool ItaniumCXXABI::RequiresNonZeroInitializer(QualType T) {
  return CGM.getTypes().ContainsPointerToDataMember(T);
}

bool ItaniumCXXABI::RequiresNonZeroInitializer(const CXXRecordDecl *D) {
  return CGM.getTypes().ContainsPointerToDataMember(D);
}
