//===------- ItaniumCXXABI.cpp - Emit LLVM Code from ASTs for a Module ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Itanium C++ ABI.  The class
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
#include "CGRecordLayout.h"
#include "CGVTables.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/Type.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"

using namespace clang;
using namespace CodeGen;

namespace {
class ItaniumCXXABI : public CodeGen::CGCXXABI {
protected:
  bool IsARM;

public:
  ItaniumCXXABI(CodeGen::CodeGenModule &CGM, bool IsARM = false) :
    CGCXXABI(CGM), IsARM(IsARM) { }

  bool isReturnTypeIndirect(const CXXRecordDecl *RD) const {
    // Structures with either a non-trivial destructor or a non-trivial
    // copy constructor are always indirect.
    return !RD->hasTrivialDestructor() || RD->hasNonTrivialCopyConstructor();
  }

  RecordArgABI getRecordArgABI(const CXXRecordDecl *RD) const {
    // Structures with either a non-trivial destructor or a non-trivial
    // copy constructor are always indirect.
    if (!RD->hasTrivialDestructor() || RD->hasNonTrivialCopyConstructor())
      return RAA_Indirect;
    return RAA_Default;
  }

  bool isZeroInitializable(const MemberPointerType *MPT);

  llvm::Type *ConvertMemberPointerType(const MemberPointerType *MPT);

  llvm::Value *EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                               llvm::Value *&This,
                                               llvm::Value *MemFnPtr,
                                               const MemberPointerType *MPT);

  llvm::Value *EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                            llvm::Value *Base,
                                            llvm::Value *MemPtr,
                                            const MemberPointerType *MPT);

  llvm::Value *EmitMemberPointerConversion(CodeGenFunction &CGF,
                                           const CastExpr *E,
                                           llvm::Value *Src);
  llvm::Constant *EmitMemberPointerConversion(const CastExpr *E,
                                              llvm::Constant *Src);

  llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT);

  llvm::Constant *EmitMemberPointer(const CXXMethodDecl *MD);
  llvm::Constant *EmitMemberDataPointer(const MemberPointerType *MPT,
                                        CharUnits offset);
  llvm::Constant *EmitMemberPointer(const APValue &MP, QualType MPT);
  llvm::Constant *BuildMemberPointer(const CXXMethodDecl *MD,
                                     CharUnits ThisAdjustment);

  llvm::Value *EmitMemberPointerComparison(CodeGenFunction &CGF,
                                           llvm::Value *L,
                                           llvm::Value *R,
                                           const MemberPointerType *MPT,
                                           bool Inequality);

  llvm::Value *EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                          llvm::Value *Addr,
                                          const MemberPointerType *MPT);

  llvm::Value *adjustToCompleteObject(CodeGenFunction &CGF,
                                      llvm::Value *ptr,
                                      QualType type);

  llvm::Value *GetVirtualBaseClassOffset(CodeGenFunction &CGF,
                                         llvm::Value *This,
                                         const CXXRecordDecl *ClassDecl,
                                         const CXXRecordDecl *BaseClassDecl);

  void BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                 CXXCtorType T,
                                 CanQualType &ResTy,
                                 SmallVectorImpl<CanQualType> &ArgTys);

  void BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                CXXDtorType T,
                                CanQualType &ResTy,
                                SmallVectorImpl<CanQualType> &ArgTys);

  void BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                   QualType &ResTy,
                                   FunctionArgList &Params);

  void EmitInstanceFunctionProlog(CodeGenFunction &CGF);

  RValue EmitConstructorCall(CodeGenFunction &CGF,
                             const CXXConstructorDecl *D,
                             CXXCtorType Type,
                             bool ForVirtualBase, bool Delegating,
                             ReturnValueSlot ReturnValue,
                             llvm::Value *This,
                             CallExpr::const_arg_iterator ArgBeg,
                             CallExpr::const_arg_iterator ArgEnd);

  RValue EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                   const CXXDestructorDecl *Dtor,
                                   CXXDtorType DtorType,
                                   SourceLocation CallLoc,
                                   ReturnValueSlot ReturnValue,
                                   llvm::Value *This);

  void EmitVirtualInheritanceTables(llvm::GlobalVariable::LinkageTypes Linkage,
                                    const CXXRecordDecl *RD);

  StringRef GetPureVirtualCallName() { return "__cxa_pure_virtual"; }
  StringRef GetDeletedVirtualCallName() { return "__cxa_deleted_virtual"; }

  CharUnits getArrayCookieSizeImpl(QualType elementType);
  llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                     llvm::Value *NewPtr,
                                     llvm::Value *NumElements,
                                     const CXXNewExpr *expr,
                                     QualType ElementType);
  llvm::Value *readArrayCookieImpl(CodeGenFunction &CGF,
                                   llvm::Value *allocPtr,
                                   CharUnits cookieSize);

  void EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                       llvm::GlobalVariable *DeclPtr, bool PerformInit);
  void registerGlobalDtor(CodeGenFunction &CGF, const VarDecl &D,
                          llvm::Constant *dtor, llvm::Constant *addr);

  llvm::Function *getOrCreateThreadLocalWrapper(const VarDecl *VD,
                                                llvm::GlobalVariable *Var);
  void EmitThreadLocalInitFuncs(
      llvm::ArrayRef<std::pair<const VarDecl *, llvm::GlobalVariable *> > Decls,
      llvm::Function *InitFunc);
  LValue EmitThreadLocalDeclRefExpr(CodeGenFunction &CGF,
                                    const DeclRefExpr *DRE);
};

class ARMCXXABI : public ItaniumCXXABI {
public:
  ARMCXXABI(CodeGen::CodeGenModule &CGM) : ItaniumCXXABI(CGM, /*ARM*/ true) {}

  bool HasThisReturn(GlobalDecl GD) const {
    return (isa<CXXConstructorDecl>(GD.getDecl()) || (
              isa<CXXDestructorDecl>(GD.getDecl()) &&
              GD.getDtorType() != Dtor_Deleting));
  }

  void EmitReturnFromThunk(CodeGenFunction &CGF, RValue RV, QualType ResTy);

  CharUnits getArrayCookieSizeImpl(QualType elementType);
  llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                     llvm::Value *NewPtr,
                                     llvm::Value *NumElements,
                                     const CXXNewExpr *expr,
                                     QualType ElementType);
  llvm::Value *readArrayCookieImpl(CodeGenFunction &CGF, llvm::Value *allocPtr,
                                   CharUnits cookieSize);
};
}

CodeGen::CGCXXABI *CodeGen::CreateItaniumCXXABI(CodeGenModule &CGM) {
  switch (CGM.getTarget().getCXXABI().getKind()) {
  // For IR-generation purposes, there's no significant difference
  // between the ARM and iOS ABIs.
  case TargetCXXABI::GenericARM:
  case TargetCXXABI::iOS:
    return new ARMCXXABI(CGM);

  // Note that AArch64 uses the generic ItaniumCXXABI class since it doesn't
  // include the other 32-bit ARM oddities: constructor/destructor return values
  // and array cookies.
  case TargetCXXABI::GenericAArch64:
    return  new ItaniumCXXABI(CGM, /*IsARM = */ true);

  case TargetCXXABI::GenericItanium:
    return new ItaniumCXXABI(CGM);

  case TargetCXXABI::Microsoft:
    llvm_unreachable("Microsoft ABI is not Itanium-based");
  }
  llvm_unreachable("bad ABI kind");
}

llvm::Type *
ItaniumCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  if (MPT->isMemberDataPointer())
    return CGM.PtrDiffTy;
  return llvm::StructType::get(CGM.PtrDiffTy, CGM.PtrDiffTy, NULL);
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

  llvm::FunctionType *FTy = 
    CGM.getTypes().GetFunctionType(
      CGM.getTypes().arrangeCXXMethodType(RD, FPT));

  llvm::Constant *ptrdiff_1 = llvm::ConstantInt::get(CGM.PtrDiffTy, 1);

  llvm::BasicBlock *FnVirtual = CGF.createBasicBlock("memptr.virtual");
  llvm::BasicBlock *FnNonVirtual = CGF.createBasicBlock("memptr.nonvirtual");
  llvm::BasicBlock *FnEnd = CGF.createBasicBlock("memptr.end");

  // Extract memptr.adj, which is in the second field.
  llvm::Value *RawAdj = Builder.CreateExtractValue(MemFnPtr, 1, "memptr.adj");

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
  llvm::Value *FnAsInt = Builder.CreateExtractValue(MemFnPtr, 0, "memptr.ptr");
  
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
  llvm::Type *VTableTy = Builder.getInt8PtrTy();
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
  llvm::PHINode *Callee = Builder.CreatePHI(FTy->getPointerTo(), 2);
  Callee->addIncoming(VirtualFn, FnVirtual);
  Callee->addIncoming(NonVirtualFn, FnNonVirtual);
  return Callee;
}

/// Compute an l-value by applying the given pointer-to-member to a
/// base object.
llvm::Value *ItaniumCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                         llvm::Value *Base,
                                                         llvm::Value *MemPtr,
                                           const MemberPointerType *MPT) {
  assert(MemPtr->getType() == CGM.PtrDiffTy);

  CGBuilderTy &Builder = CGF.Builder;

  unsigned AS = Base->getType()->getPointerAddressSpace();

  // Cast to char*.
  Base = Builder.CreateBitCast(Base, Builder.getInt8Ty()->getPointerTo(AS));

  // Apply the offset, which we assume is non-null.
  llvm::Value *Addr = Builder.CreateInBoundsGEP(Base, MemPtr, "memptr.offset");

  // Cast the address to the appropriate pointer type, adopting the
  // address space of the base pointer.
  llvm::Type *PType
    = CGF.ConvertTypeForMem(MPT->getPointeeType())->getPointerTo(AS);
  return Builder.CreateBitCast(Addr, PType);
}

/// Perform a bitcast, derived-to-base, or base-to-derived member pointer
/// conversion.
///
/// Bitcast conversions are always a no-op under Itanium.
///
/// Obligatory offset/adjustment diagram:
///         <-- offset -->          <-- adjustment -->
///   |--------------------------|----------------------|--------------------|
///   ^Derived address point     ^Base address point    ^Member address point
///
/// So when converting a base member pointer to a derived member pointer,
/// we add the offset to the adjustment because the address point has
/// decreased;  and conversely, when converting a derived MP to a base MP
/// we subtract the offset from the adjustment because the address point
/// has increased.
///
/// The standard forbids (at compile time) conversion to and from
/// virtual bases, which is why we don't have to consider them here.
///
/// The standard forbids (at run time) casting a derived MP to a base
/// MP when the derived MP does not point to a member of the base.
/// This is why -1 is a reasonable choice for null data member
/// pointers.
llvm::Value *
ItaniumCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                           const CastExpr *E,
                                           llvm::Value *src) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer ||
         E->getCastKind() == CK_ReinterpretMemberPointer);

  // Under Itanium, reinterprets don't require any additional processing.
  if (E->getCastKind() == CK_ReinterpretMemberPointer) return src;

  // Use constant emission if we can.
  if (isa<llvm::Constant>(src))
    return EmitMemberPointerConversion(E, cast<llvm::Constant>(src));

  llvm::Constant *adj = getMemberPointerAdjustment(E);
  if (!adj) return src;

  CGBuilderTy &Builder = CGF.Builder;
  bool isDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);

  const MemberPointerType *destTy =
    E->getType()->castAs<MemberPointerType>();

  // For member data pointers, this is just a matter of adding the
  // offset if the source is non-null.
  if (destTy->isMemberDataPointer()) {
    llvm::Value *dst;
    if (isDerivedToBase)
      dst = Builder.CreateNSWSub(src, adj, "adj");
    else
      dst = Builder.CreateNSWAdd(src, adj, "adj");

    // Null check.
    llvm::Value *null = llvm::Constant::getAllOnesValue(src->getType());
    llvm::Value *isNull = Builder.CreateICmpEQ(src, null, "memptr.isnull");
    return Builder.CreateSelect(isNull, src, dst);
  }

  // The this-adjustment is left-shifted by 1 on ARM.
  if (IsARM) {
    uint64_t offset = cast<llvm::ConstantInt>(adj)->getZExtValue();
    offset <<= 1;
    adj = llvm::ConstantInt::get(adj->getType(), offset);
  }

  llvm::Value *srcAdj = Builder.CreateExtractValue(src, 1, "src.adj");
  llvm::Value *dstAdj;
  if (isDerivedToBase)
    dstAdj = Builder.CreateNSWSub(srcAdj, adj, "adj");
  else
    dstAdj = Builder.CreateNSWAdd(srcAdj, adj, "adj");

  return Builder.CreateInsertValue(src, dstAdj, 1);
}

llvm::Constant *
ItaniumCXXABI::EmitMemberPointerConversion(const CastExpr *E,
                                           llvm::Constant *src) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer ||
         E->getCastKind() == CK_ReinterpretMemberPointer);

  // Under Itanium, reinterprets don't require any additional processing.
  if (E->getCastKind() == CK_ReinterpretMemberPointer) return src;

  // If the adjustment is trivial, we don't need to do anything.
  llvm::Constant *adj = getMemberPointerAdjustment(E);
  if (!adj) return src;

  bool isDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);

  const MemberPointerType *destTy =
    E->getType()->castAs<MemberPointerType>();

  // For member data pointers, this is just a matter of adding the
  // offset if the source is non-null.
  if (destTy->isMemberDataPointer()) {
    // null maps to null.
    if (src->isAllOnesValue()) return src;

    if (isDerivedToBase)
      return llvm::ConstantExpr::getNSWSub(src, adj);
    else
      return llvm::ConstantExpr::getNSWAdd(src, adj);
  }

  // The this-adjustment is left-shifted by 1 on ARM.
  if (IsARM) {
    uint64_t offset = cast<llvm::ConstantInt>(adj)->getZExtValue();
    offset <<= 1;
    adj = llvm::ConstantInt::get(adj->getType(), offset);
  }

  llvm::Constant *srcAdj = llvm::ConstantExpr::getExtractValue(src, 1);
  llvm::Constant *dstAdj;
  if (isDerivedToBase)
    dstAdj = llvm::ConstantExpr::getNSWSub(srcAdj, adj);
  else
    dstAdj = llvm::ConstantExpr::getNSWAdd(srcAdj, adj);

  return llvm::ConstantExpr::getInsertValue(src, dstAdj, 1);
}

llvm::Constant *
ItaniumCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  if (MPT->isMemberDataPointer()) 
    return llvm::ConstantInt::get(CGM.PtrDiffTy, -1ULL, /*isSigned=*/true);

  llvm::Constant *Zero = llvm::ConstantInt::get(CGM.PtrDiffTy, 0);
  llvm::Constant *Values[2] = { Zero, Zero };
  return llvm::ConstantStruct::getAnon(Values);
}

llvm::Constant *
ItaniumCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                     CharUnits offset) {
  // Itanium C++ ABI 2.3:
  //   A pointer to data member is an offset from the base address of
  //   the class object containing it, represented as a ptrdiff_t
  return llvm::ConstantInt::get(CGM.PtrDiffTy, offset.getQuantity());
}

llvm::Constant *ItaniumCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return BuildMemberPointer(MD, CharUnits::Zero());
}

llvm::Constant *ItaniumCXXABI::BuildMemberPointer(const CXXMethodDecl *MD,
                                                  CharUnits ThisAdjustment) {
  assert(MD->isInstance() && "Member function must not be static!");
  MD = MD->getCanonicalDecl();

  CodeGenTypes &Types = CGM.getTypes();

  // Get the function pointer (or index if this is a virtual function).
  llvm::Constant *MemPtr[2];
  if (MD->isVirtual()) {
    uint64_t Index = CGM.getVTableContext().getMethodVTableIndex(MD);

    const ASTContext &Context = getContext();
    CharUnits PointerWidth =
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
    uint64_t VTableOffset = (Index * PointerWidth.getQuantity());

    if (IsARM) {
      // ARM C++ ABI 3.2.1:
      //   This ABI specifies that adj contains twice the this
      //   adjustment, plus 1 if the member function is virtual. The
      //   least significant bit of adj then makes exactly the same
      //   discrimination as the least significant bit of ptr does for
      //   Itanium.
      MemPtr[0] = llvm::ConstantInt::get(CGM.PtrDiffTy, VTableOffset);
      MemPtr[1] = llvm::ConstantInt::get(CGM.PtrDiffTy,
                                         2 * ThisAdjustment.getQuantity() + 1);
    } else {
      // Itanium C++ ABI 2.3:
      //   For a virtual function, [the pointer field] is 1 plus the
      //   virtual table offset (in bytes) of the function,
      //   represented as a ptrdiff_t.
      MemPtr[0] = llvm::ConstantInt::get(CGM.PtrDiffTy, VTableOffset + 1);
      MemPtr[1] = llvm::ConstantInt::get(CGM.PtrDiffTy,
                                         ThisAdjustment.getQuantity());
    }
  } else {
    const FunctionProtoType *FPT = MD->getType()->castAs<FunctionProtoType>();
    llvm::Type *Ty;
    // Check whether the function has a computable LLVM signature.
    if (Types.isFuncTypeConvertible(FPT)) {
      // The function has a computable LLVM signature; use the correct type.
      Ty = Types.GetFunctionType(Types.arrangeCXXMethodDeclaration(MD));
    } else {
      // Use an arbitrary non-function type to tell GetAddrOfFunction that the
      // function type is incomplete.
      Ty = CGM.PtrDiffTy;
    }
    llvm::Constant *addr = CGM.GetAddrOfFunction(MD, Ty);

    MemPtr[0] = llvm::ConstantExpr::getPtrToInt(addr, CGM.PtrDiffTy);
    MemPtr[1] = llvm::ConstantInt::get(CGM.PtrDiffTy, (IsARM ? 2 : 1) *
                                       ThisAdjustment.getQuantity());
  }
  
  return llvm::ConstantStruct::getAnon(MemPtr);
}

llvm::Constant *ItaniumCXXABI::EmitMemberPointer(const APValue &MP,
                                                 QualType MPType) {
  const MemberPointerType *MPT = MPType->castAs<MemberPointerType>();
  const ValueDecl *MPD = MP.getMemberPointerDecl();
  if (!MPD)
    return EmitNullMemberPointer(MPT);

  CharUnits ThisAdjustment = getMemberPointerPathAdjustment(MP);

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MPD))
    return BuildMemberPointer(MD, ThisAdjustment);

  CharUnits FieldOffset =
    getContext().toCharUnitsFromBits(getContext().getFieldOffset(MPD));
  return EmitMemberDataPointer(MPT, ThisAdjustment + FieldOffset);
}

/// The comparison algorithm is pretty easy: the member pointers are
/// the same if they're either bitwise identical *or* both null.
///
/// ARM is different here only because null-ness is more complicated.
llvm::Value *
ItaniumCXXABI::EmitMemberPointerComparison(CodeGenFunction &CGF,
                                           llvm::Value *L,
                                           llvm::Value *R,
                                           const MemberPointerType *MPT,
                                           bool Inequality) {
  CGBuilderTy &Builder = CGF.Builder;

  llvm::ICmpInst::Predicate Eq;
  llvm::Instruction::BinaryOps And, Or;
  if (Inequality) {
    Eq = llvm::ICmpInst::ICMP_NE;
    And = llvm::Instruction::Or;
    Or = llvm::Instruction::And;
  } else {
    Eq = llvm::ICmpInst::ICMP_EQ;
    And = llvm::Instruction::And;
    Or = llvm::Instruction::Or;
  }

  // Member data pointers are easy because there's a unique null
  // value, so it just comes down to bitwise equality.
  if (MPT->isMemberDataPointer())
    return Builder.CreateICmp(Eq, L, R);

  // For member function pointers, the tautologies are more complex.
  // The Itanium tautology is:
  //   (L == R) <==> (L.ptr == R.ptr && (L.ptr == 0 || L.adj == R.adj))
  // The ARM tautology is:
  //   (L == R) <==> (L.ptr == R.ptr &&
  //                  (L.adj == R.adj ||
  //                   (L.ptr == 0 && ((L.adj|R.adj) & 1) == 0)))
  // The inequality tautologies have exactly the same structure, except
  // applying De Morgan's laws.
  
  llvm::Value *LPtr = Builder.CreateExtractValue(L, 0, "lhs.memptr.ptr");
  llvm::Value *RPtr = Builder.CreateExtractValue(R, 0, "rhs.memptr.ptr");

  // This condition tests whether L.ptr == R.ptr.  This must always be
  // true for equality to hold.
  llvm::Value *PtrEq = Builder.CreateICmp(Eq, LPtr, RPtr, "cmp.ptr");

  // This condition, together with the assumption that L.ptr == R.ptr,
  // tests whether the pointers are both null.  ARM imposes an extra
  // condition.
  llvm::Value *Zero = llvm::Constant::getNullValue(LPtr->getType());
  llvm::Value *EqZero = Builder.CreateICmp(Eq, LPtr, Zero, "cmp.ptr.null");

  // This condition tests whether L.adj == R.adj.  If this isn't
  // true, the pointers are unequal unless they're both null.
  llvm::Value *LAdj = Builder.CreateExtractValue(L, 1, "lhs.memptr.adj");
  llvm::Value *RAdj = Builder.CreateExtractValue(R, 1, "rhs.memptr.adj");
  llvm::Value *AdjEq = Builder.CreateICmp(Eq, LAdj, RAdj, "cmp.adj");

  // Null member function pointers on ARM clear the low bit of Adj,
  // so the zero condition has to check that neither low bit is set.
  if (IsARM) {
    llvm::Value *One = llvm::ConstantInt::get(LPtr->getType(), 1);

    // Compute (l.adj | r.adj) & 1 and test it against zero.
    llvm::Value *OrAdj = Builder.CreateOr(LAdj, RAdj, "or.adj");
    llvm::Value *OrAdjAnd1 = Builder.CreateAnd(OrAdj, One);
    llvm::Value *OrAdjAnd1EqZero = Builder.CreateICmp(Eq, OrAdjAnd1, Zero,
                                                      "cmp.or.adj");
    EqZero = Builder.CreateBinOp(And, EqZero, OrAdjAnd1EqZero);
  }

  // Tie together all our conditions.
  llvm::Value *Result = Builder.CreateBinOp(Or, EqZero, AdjEq);
  Result = Builder.CreateBinOp(And, PtrEq, Result,
                               Inequality ? "memptr.ne" : "memptr.eq");
  return Result;
}

llvm::Value *
ItaniumCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                          llvm::Value *MemPtr,
                                          const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;

  /// For member data pointers, this is just a check against -1.
  if (MPT->isMemberDataPointer()) {
    assert(MemPtr->getType() == CGM.PtrDiffTy);
    llvm::Value *NegativeOne =
      llvm::Constant::getAllOnesValue(MemPtr->getType());
    return Builder.CreateICmpNE(MemPtr, NegativeOne, "memptr.tobool");
  }
  
  // In Itanium, a member function pointer is not null if 'ptr' is not null.
  llvm::Value *Ptr = Builder.CreateExtractValue(MemPtr, 0, "memptr.ptr");

  llvm::Constant *Zero = llvm::ConstantInt::get(Ptr->getType(), 0);
  llvm::Value *Result = Builder.CreateICmpNE(Ptr, Zero, "memptr.tobool");

  // On ARM, a member function pointer is also non-null if the low bit of 'adj'
  // (the virtual bit) is set.
  if (IsARM) {
    llvm::Constant *One = llvm::ConstantInt::get(Ptr->getType(), 1);
    llvm::Value *Adj = Builder.CreateExtractValue(MemPtr, 1, "memptr.adj");
    llvm::Value *VirtualBit = Builder.CreateAnd(Adj, One, "memptr.virtualbit");
    llvm::Value *IsVirtual = Builder.CreateICmpNE(VirtualBit, Zero,
                                                  "memptr.isvirtual");
    Result = Builder.CreateOr(Result, IsVirtual);
  }

  return Result;
}

/// The Itanium ABI requires non-zero initialization only for data
/// member pointers, for which '0' is a valid offset.
bool ItaniumCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  return MPT->getPointeeType()->isFunctionType();
}

/// The Itanium ABI always places an offset to the complete object
/// at entry -2 in the vtable.
llvm::Value *ItaniumCXXABI::adjustToCompleteObject(CodeGenFunction &CGF,
                                                   llvm::Value *ptr,
                                                   QualType type) {
  // Grab the vtable pointer as an intptr_t*.
  llvm::Value *vtable = CGF.GetVTablePtr(ptr, CGF.IntPtrTy->getPointerTo());

  // Track back to entry -2 and pull out the offset there.
  llvm::Value *offsetPtr = 
    CGF.Builder.CreateConstInBoundsGEP1_64(vtable, -2, "complete-offset.ptr");
  llvm::LoadInst *offset = CGF.Builder.CreateLoad(offsetPtr);
  offset->setAlignment(CGF.PointerAlignInBytes);

  // Apply the offset.
  ptr = CGF.Builder.CreateBitCast(ptr, CGF.Int8PtrTy);
  return CGF.Builder.CreateInBoundsGEP(ptr, offset);
}

llvm::Value *
ItaniumCXXABI::GetVirtualBaseClassOffset(CodeGenFunction &CGF,
                                         llvm::Value *This,
                                         const CXXRecordDecl *ClassDecl,
                                         const CXXRecordDecl *BaseClassDecl) {
  llvm::Value *VTablePtr = CGF.GetVTablePtr(This, CGM.Int8PtrTy);
  CharUnits VBaseOffsetOffset =
    CGM.getVTableContext().getVirtualBaseOffsetOffset(ClassDecl, BaseClassDecl);

  llvm::Value *VBaseOffsetPtr =
    CGF.Builder.CreateConstGEP1_64(VTablePtr, VBaseOffsetOffset.getQuantity(),
                                   "vbase.offset.ptr");
  VBaseOffsetPtr = CGF.Builder.CreateBitCast(VBaseOffsetPtr,
                                             CGM.PtrDiffTy->getPointerTo());

  llvm::Value *VBaseOffset =
    CGF.Builder.CreateLoad(VBaseOffsetPtr, "vbase.offset");

  return VBaseOffset;
}

/// The generic ABI passes 'this', plus a VTT if it's initializing a
/// base subobject.
void ItaniumCXXABI::BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                              CXXCtorType Type,
                                              CanQualType &ResTy,
                                SmallVectorImpl<CanQualType> &ArgTys) {
  ASTContext &Context = getContext();

  // 'this' parameter is already there, as well as 'this' return if
  // HasThisReturn(GlobalDecl(Ctor, Type)) is true

  // Check if we need to add a VTT parameter (which has type void **).
  if (Type == Ctor_Base && Ctor->getParent()->getNumVBases() != 0)
    ArgTys.push_back(Context.getPointerType(Context.VoidPtrTy));
}

/// The generic ABI passes 'this', plus a VTT if it's destroying a
/// base subobject.
void ItaniumCXXABI::BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                             CXXDtorType Type,
                                             CanQualType &ResTy,
                                SmallVectorImpl<CanQualType> &ArgTys) {
  ASTContext &Context = getContext();

  // 'this' parameter is already there, as well as 'this' return if
  // HasThisReturn(GlobalDecl(Dtor, Type)) is true

  // Check if we need to add a VTT parameter (which has type void **).
  if (Type == Dtor_Base && Dtor->getParent()->getNumVBases() != 0)
    ArgTys.push_back(Context.getPointerType(Context.VoidPtrTy));
}

void ItaniumCXXABI::BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                                QualType &ResTy,
                                                FunctionArgList &Params) {
  /// Create the 'this' variable.
  BuildThisParam(CGF, Params);

  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  assert(MD->isInstance());

  // Check if we need a VTT parameter as well.
  if (CodeGenVTables::needsVTTParameter(CGF.CurGD)) {
    ASTContext &Context = getContext();

    // FIXME: avoid the fake decl
    QualType T = Context.getPointerType(Context.VoidPtrTy);
    ImplicitParamDecl *VTTDecl
      = ImplicitParamDecl::Create(Context, 0, MD->getLocation(),
                                  &Context.Idents.get("vtt"), T);
    Params.push_back(VTTDecl);
    getVTTDecl(CGF) = VTTDecl;
  }
}

void ItaniumCXXABI::EmitInstanceFunctionProlog(CodeGenFunction &CGF) {
  /// Initialize the 'this' slot.
  EmitThisParam(CGF);

  /// Initialize the 'vtt' slot if needed.
  if (getVTTDecl(CGF)) {
    getVTTValue(CGF)
      = CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(getVTTDecl(CGF)),
                               "vtt");
  }

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to 'this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CGCXXABI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation
  if (HasThisReturn(CGF.CurGD))
    CGF.Builder.CreateStore(getThisValue(CGF), CGF.ReturnValue);
}

RValue ItaniumCXXABI::EmitConstructorCall(CodeGenFunction &CGF,
                                        const CXXConstructorDecl *D,
                                        CXXCtorType Type,
                                        bool ForVirtualBase, bool Delegating,
                                        ReturnValueSlot ReturnValue,
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  llvm::Value *VTT = CGF.GetVTTParameter(GlobalDecl(D, Type), ForVirtualBase,
                                         Delegating);
  QualType VTTTy = getContext().getPointerType(getContext().VoidPtrTy);
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  // FIXME: Provide a source location here.
  return CGF.EmitCXXMemberCall(D, SourceLocation(), Callee, ReturnValue,
                               This, VTT, VTTTy, ArgBeg, ArgEnd);
}

RValue ItaniumCXXABI::EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                                const CXXDestructorDecl *Dtor,
                                                CXXDtorType DtorType,
                                                SourceLocation CallLoc,
                                                ReturnValueSlot ReturnValue,
                                                llvm::Value *This) {
  assert(DtorType == Dtor_Deleting || DtorType == Dtor_Complete);

  const CGFunctionInfo *FInfo
    = &CGM.getTypes().arrangeCXXDestructor(Dtor, DtorType);
  llvm::Type *Ty = CGF.CGM.getTypes().GetFunctionType(*FInfo);
  llvm::Value *Callee = CGF.BuildVirtualCall(Dtor, DtorType, This, Ty);

  return CGF.EmitCXXMemberCall(Dtor, CallLoc, Callee, ReturnValue, This,
                               /*ImplicitParam=*/0, QualType(), 0, 0);
}

void ItaniumCXXABI::EmitVirtualInheritanceTables(
    llvm::GlobalVariable::LinkageTypes Linkage, const CXXRecordDecl *RD) {
  CodeGenVTables &VTables = CGM.getVTables();
  llvm::GlobalVariable *VTT = VTables.GetAddrOfVTT(RD);
  VTables.EmitVTTDefinition(VTT, Linkage, RD);
}

void ARMCXXABI::EmitReturnFromThunk(CodeGenFunction &CGF,
                                    RValue RV, QualType ResultType) {
  if (!isa<CXXDestructorDecl>(CGF.CurGD.getDecl()))
    return ItaniumCXXABI::EmitReturnFromThunk(CGF, RV, ResultType);

  // Destructor thunks in the ARM ABI have indeterminate results.
  llvm::Type *T =
    cast<llvm::PointerType>(CGF.ReturnValue->getType())->getElementType();
  RValue Undef = RValue::get(llvm::UndefValue::get(T));
  return ItaniumCXXABI::EmitReturnFromThunk(CGF, Undef, ResultType);
}

/************************** Array allocation cookies **************************/

CharUnits ItaniumCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // The array cookie is a size_t; pad that up to the element alignment.
  // The cookie is actually right-justified in that space.
  return std::max(CharUnits::fromQuantity(CGM.SizeSizeInBytes),
                  CGM.getContext().getTypeAlignInChars(elementType));
}

llvm::Value *ItaniumCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                                  llvm::Value *NewPtr,
                                                  llvm::Value *NumElements,
                                                  const CXXNewExpr *expr,
                                                  QualType ElementType) {
  assert(requiresArrayCookie(expr));

  unsigned AS = NewPtr->getType()->getPointerAddressSpace();

  ASTContext &Ctx = getContext();
  QualType SizeTy = Ctx.getSizeType();
  CharUnits SizeSize = Ctx.getTypeSizeInChars(SizeTy);

  // The size of the cookie.
  CharUnits CookieSize =
    std::max(SizeSize, Ctx.getTypeAlignInChars(ElementType));
  assert(CookieSize == getArrayCookieSizeImpl(ElementType));

  // Compute an offset to the cookie.
  llvm::Value *CookiePtr = NewPtr;
  CharUnits CookieOffset = CookieSize - SizeSize;
  if (!CookieOffset.isZero())
    CookiePtr = CGF.Builder.CreateConstInBoundsGEP1_64(CookiePtr,
                                                 CookieOffset.getQuantity());

  // Write the number of elements into the appropriate slot.
  llvm::Value *NumElementsPtr
    = CGF.Builder.CreateBitCast(CookiePtr,
                                CGF.ConvertType(SizeTy)->getPointerTo(AS));
  CGF.Builder.CreateStore(NumElements, NumElementsPtr);

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  return CGF.Builder.CreateConstInBoundsGEP1_64(NewPtr,
                                                CookieSize.getQuantity());  
}

llvm::Value *ItaniumCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                                llvm::Value *allocPtr,
                                                CharUnits cookieSize) {
  // The element size is right-justified in the cookie.
  llvm::Value *numElementsPtr = allocPtr;
  CharUnits numElementsOffset =
    cookieSize - CharUnits::fromQuantity(CGF.SizeSizeInBytes);
  if (!numElementsOffset.isZero())
    numElementsPtr =
      CGF.Builder.CreateConstInBoundsGEP1_64(numElementsPtr,
                                             numElementsOffset.getQuantity());

  unsigned AS = allocPtr->getType()->getPointerAddressSpace();
  numElementsPtr = 
    CGF.Builder.CreateBitCast(numElementsPtr, CGF.SizeTy->getPointerTo(AS));
  return CGF.Builder.CreateLoad(numElementsPtr);
}

CharUnits ARMCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // ARM says that the cookie is always:
  //   struct array_cookie {
  //     std::size_t element_size; // element_size != 0
  //     std::size_t element_count;
  //   };
  // But the base ABI doesn't give anything an alignment greater than
  // 8, so we can dismiss this as typical ABI-author blindness to
  // actual language complexity and round up to the element alignment.
  return std::max(CharUnits::fromQuantity(2 * CGM.SizeSizeInBytes),
                  CGM.getContext().getTypeAlignInChars(elementType));
}

llvm::Value *ARMCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                              llvm::Value *newPtr,
                                              llvm::Value *numElements,
                                              const CXXNewExpr *expr,
                                              QualType elementType) {
  assert(requiresArrayCookie(expr));

  // NewPtr is a char*, but we generalize to arbitrary addrspaces.
  unsigned AS = newPtr->getType()->getPointerAddressSpace();

  // The cookie is always at the start of the buffer.
  llvm::Value *cookie = newPtr;

  // The first element is the element size.
  cookie = CGF.Builder.CreateBitCast(cookie, CGF.SizeTy->getPointerTo(AS));
  llvm::Value *elementSize = llvm::ConstantInt::get(CGF.SizeTy,
                 getContext().getTypeSizeInChars(elementType).getQuantity());
  CGF.Builder.CreateStore(elementSize, cookie);

  // The second element is the element count.
  cookie = CGF.Builder.CreateConstInBoundsGEP1_32(cookie, 1);
  CGF.Builder.CreateStore(numElements, cookie);

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  CharUnits cookieSize = ARMCXXABI::getArrayCookieSizeImpl(elementType);
  return CGF.Builder.CreateConstInBoundsGEP1_64(newPtr,
                                                cookieSize.getQuantity());
}

llvm::Value *ARMCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                            llvm::Value *allocPtr,
                                            CharUnits cookieSize) {
  // The number of elements is at offset sizeof(size_t) relative to
  // the allocated pointer.
  llvm::Value *numElementsPtr
    = CGF.Builder.CreateConstInBoundsGEP1_64(allocPtr, CGF.SizeSizeInBytes);

  unsigned AS = allocPtr->getType()->getPointerAddressSpace();
  numElementsPtr = 
    CGF.Builder.CreateBitCast(numElementsPtr, CGF.SizeTy->getPointerTo(AS));
  return CGF.Builder.CreateLoad(numElementsPtr);
}

/*********************** Static local initialization **************************/

static llvm::Constant *getGuardAcquireFn(CodeGenModule &CGM,
                                         llvm::PointerType *GuardPtrTy) {
  // int __cxa_guard_acquire(__guard *guard_object);
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.getTypes().ConvertType(CGM.getContext().IntTy),
                            GuardPtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_guard_acquire",
                                   llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                                 llvm::Attribute::NoUnwind));
}

static llvm::Constant *getGuardReleaseFn(CodeGenModule &CGM,
                                         llvm::PointerType *GuardPtrTy) {
  // void __cxa_guard_release(__guard *guard_object);
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.VoidTy, GuardPtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_guard_release",
                                   llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                                 llvm::Attribute::NoUnwind));
}

static llvm::Constant *getGuardAbortFn(CodeGenModule &CGM,
                                       llvm::PointerType *GuardPtrTy) {
  // void __cxa_guard_abort(__guard *guard_object);
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(CGM.VoidTy, GuardPtrTy, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FTy, "__cxa_guard_abort",
                                   llvm::AttributeSet::get(CGM.getLLVMContext(),
                                              llvm::AttributeSet::FunctionIndex,
                                                 llvm::Attribute::NoUnwind));
}

namespace {
  struct CallGuardAbort : EHScopeStack::Cleanup {
    llvm::GlobalVariable *Guard;
    CallGuardAbort(llvm::GlobalVariable *Guard) : Guard(Guard) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.EmitNounwindRuntimeCall(getGuardAbortFn(CGF.CGM, Guard->getType()),
                                  Guard);
    }
  };
}

/// The ARM code here follows the Itanium code closely enough that we
/// just special-case it at particular places.
void ItaniumCXXABI::EmitGuardedInit(CodeGenFunction &CGF,
                                    const VarDecl &D,
                                    llvm::GlobalVariable *var,
                                    bool shouldPerformInit) {
  CGBuilderTy &Builder = CGF.Builder;

  // We only need to use thread-safe statics for local non-TLS variables;
  // global initialization is always single-threaded.
  bool threadsafe = getContext().getLangOpts().ThreadsafeStatics &&
                    D.isLocalVarDecl() && !D.getTLSKind();

  // If we have a global variable with internal linkage and thread-safe statics
  // are disabled, we can just let the guard variable be of type i8.
  bool useInt8GuardVariable = !threadsafe && var->hasInternalLinkage();

  llvm::IntegerType *guardTy;
  if (useInt8GuardVariable) {
    guardTy = CGF.Int8Ty;
  } else {
    // Guard variables are 64 bits in the generic ABI and size width on ARM
    // (i.e. 32-bit on AArch32, 64-bit on AArch64).
    guardTy = (IsARM ? CGF.SizeTy : CGF.Int64Ty);
  }
  llvm::PointerType *guardPtrTy = guardTy->getPointerTo();

  // Create the guard variable if we don't already have it (as we
  // might if we're double-emitting this function body).
  llvm::GlobalVariable *guard = CGM.getStaticLocalDeclGuardAddress(&D);
  if (!guard) {
    // Mangle the name for the guard.
    SmallString<256> guardName;
    {
      llvm::raw_svector_ostream out(guardName);
      getMangleContext().mangleItaniumGuardVariable(&D, out);
      out.flush();
    }

    // Create the guard variable with a zero-initializer.
    // Just absorb linkage and visibility from the guarded variable.
    guard = new llvm::GlobalVariable(CGM.getModule(), guardTy,
                                     false, var->getLinkage(),
                                     llvm::ConstantInt::get(guardTy, 0),
                                     guardName.str());
    guard->setVisibility(var->getVisibility());
    // If the variable is thread-local, so is its guard variable.
    guard->setThreadLocalMode(var->getThreadLocalMode());

    CGM.setStaticLocalDeclGuardAddress(&D, guard);
  }

  // Test whether the variable has completed initialization.
  llvm::Value *isInitialized;

  // ARM C++ ABI 3.2.3.1:
  //   To support the potential use of initialization guard variables
  //   as semaphores that are the target of ARM SWP and LDREX/STREX
  //   synchronizing instructions we define a static initialization
  //   guard variable to be a 4-byte aligned, 4- byte word with the
  //   following inline access protocol.
  //     #define INITIALIZED 1
  //     if ((obj_guard & INITIALIZED) != INITIALIZED) {
  //       if (__cxa_guard_acquire(&obj_guard))
  //         ...
  //     }
  if (IsARM && !useInt8GuardVariable) {
    llvm::Value *V = Builder.CreateLoad(guard);
    llvm::Value *Test1 = llvm::ConstantInt::get(guardTy, 1);
    V = Builder.CreateAnd(V, Test1);
    isInitialized = Builder.CreateIsNull(V, "guard.uninitialized");

  // Itanium C++ ABI 3.3.2:
  //   The following is pseudo-code showing how these functions can be used:
  //     if (obj_guard.first_byte == 0) {
  //       if ( __cxa_guard_acquire (&obj_guard) ) {
  //         try {
  //           ... initialize the object ...;
  //         } catch (...) {
  //            __cxa_guard_abort (&obj_guard);
  //            throw;
  //         }
  //         ... queue object destructor with __cxa_atexit() ...;
  //         __cxa_guard_release (&obj_guard);
  //       }
  //     }
  } else {
    // Load the first byte of the guard variable.
    llvm::LoadInst *LI = 
      Builder.CreateLoad(Builder.CreateBitCast(guard, CGM.Int8PtrTy));
    LI->setAlignment(1);

    // Itanium ABI:
    //   An implementation supporting thread-safety on multiprocessor
    //   systems must also guarantee that references to the initialized
    //   object do not occur before the load of the initialization flag.
    //
    // In LLVM, we do this by marking the load Acquire.
    if (threadsafe)
      LI->setAtomic(llvm::Acquire);

    isInitialized = Builder.CreateIsNull(LI, "guard.uninitialized");
  }

  llvm::BasicBlock *InitCheckBlock = CGF.createBasicBlock("init.check");
  llvm::BasicBlock *EndBlock = CGF.createBasicBlock("init.end");

  // Check if the first byte of the guard variable is zero.
  Builder.CreateCondBr(isInitialized, InitCheckBlock, EndBlock);

  CGF.EmitBlock(InitCheckBlock);

  // Variables used when coping with thread-safe statics and exceptions.
  if (threadsafe) {    
    // Call __cxa_guard_acquire.
    llvm::Value *V
      = CGF.EmitNounwindRuntimeCall(getGuardAcquireFn(CGM, guardPtrTy), guard);
               
    llvm::BasicBlock *InitBlock = CGF.createBasicBlock("init");
  
    Builder.CreateCondBr(Builder.CreateIsNotNull(V, "tobool"),
                         InitBlock, EndBlock);
  
    // Call __cxa_guard_abort along the exceptional edge.
    CGF.EHStack.pushCleanup<CallGuardAbort>(EHCleanup, guard);
    
    CGF.EmitBlock(InitBlock);
  }

  // Emit the initializer and add a global destructor if appropriate.
  CGF.EmitCXXGlobalVarDeclInit(D, var, shouldPerformInit);

  if (threadsafe) {
    // Pop the guard-abort cleanup if we pushed one.
    CGF.PopCleanupBlock();

    // Call __cxa_guard_release.  This cannot throw.
    CGF.EmitNounwindRuntimeCall(getGuardReleaseFn(CGM, guardPtrTy), guard);
  } else {
    Builder.CreateStore(llvm::ConstantInt::get(guardTy, 1), guard);
  }

  CGF.EmitBlock(EndBlock);
}

/// Register a global destructor using __cxa_atexit.
static void emitGlobalDtorWithCXAAtExit(CodeGenFunction &CGF,
                                        llvm::Constant *dtor,
                                        llvm::Constant *addr,
                                        bool TLS) {
  const char *Name = "__cxa_atexit";
  if (TLS) {
    const llvm::Triple &T = CGF.getTarget().getTriple();
    Name = T.isMacOSX() ?  "_tlv_atexit" : "__cxa_thread_atexit";
  }

  // We're assuming that the destructor function is something we can
  // reasonably call with the default CC.  Go ahead and cast it to the
  // right prototype.
  llvm::Type *dtorTy =
    llvm::FunctionType::get(CGF.VoidTy, CGF.Int8PtrTy, false)->getPointerTo();

  // extern "C" int __cxa_atexit(void (*f)(void *), void *p, void *d);
  llvm::Type *paramTys[] = { dtorTy, CGF.Int8PtrTy, CGF.Int8PtrTy };
  llvm::FunctionType *atexitTy =
    llvm::FunctionType::get(CGF.IntTy, paramTys, false);

  // Fetch the actual function.
  llvm::Constant *atexit = CGF.CGM.CreateRuntimeFunction(atexitTy, Name);
  if (llvm::Function *fn = dyn_cast<llvm::Function>(atexit))
    fn->setDoesNotThrow();

  // Create a variable that binds the atexit to this shared object.
  llvm::Constant *handle =
    CGF.CGM.CreateRuntimeVariable(CGF.Int8Ty, "__dso_handle");

  llvm::Value *args[] = {
    llvm::ConstantExpr::getBitCast(dtor, dtorTy),
    llvm::ConstantExpr::getBitCast(addr, CGF.Int8PtrTy),
    handle
  };
  CGF.EmitNounwindRuntimeCall(atexit, args);
}

/// Register a global destructor as best as we know how.
void ItaniumCXXABI::registerGlobalDtor(CodeGenFunction &CGF,
                                       const VarDecl &D,
                                       llvm::Constant *dtor,
                                       llvm::Constant *addr) {
  // Use __cxa_atexit if available.
  if (CGM.getCodeGenOpts().CXAAtExit)
    return emitGlobalDtorWithCXAAtExit(CGF, dtor, addr, D.getTLSKind());

  if (D.getTLSKind())
    CGM.ErrorUnsupported(&D, "non-trivial TLS destruction");

  // In Apple kexts, we want to add a global destructor entry.
  // FIXME: shouldn't this be guarded by some variable?
  if (CGM.getLangOpts().AppleKext) {
    // Generate a global destructor entry.
    return CGM.AddCXXDtorEntry(dtor, addr);
  }

  CGF.registerGlobalDtorWithAtExit(dtor, addr);
}

/// Get the appropriate linkage for the wrapper function. This is essentially
/// the weak form of the variable's linkage; every translation unit which wneeds
/// the wrapper emits a copy, and we want the linker to merge them.
static llvm::GlobalValue::LinkageTypes getThreadLocalWrapperLinkage(
    llvm::GlobalValue::LinkageTypes VarLinkage) {
  if (llvm::GlobalValue::isLinkerPrivateLinkage(VarLinkage))
    return llvm::GlobalValue::LinkerPrivateWeakLinkage;
  // For internal linkage variables, we don't need an external or weak wrapper.
  if (llvm::GlobalValue::isLocalLinkage(VarLinkage))
    return VarLinkage;
  return llvm::GlobalValue::WeakODRLinkage;
}

llvm::Function *
ItaniumCXXABI::getOrCreateThreadLocalWrapper(const VarDecl *VD,
                                             llvm::GlobalVariable *Var) {
  // Mangle the name for the thread_local wrapper function.
  SmallString<256> WrapperName;
  {
    llvm::raw_svector_ostream Out(WrapperName);
    getMangleContext().mangleItaniumThreadLocalWrapper(VD, Out);
    Out.flush();
  }

  if (llvm::Value *V = Var->getParent()->getNamedValue(WrapperName))
    return cast<llvm::Function>(V);

  llvm::Type *RetTy = Var->getType();
  if (VD->getType()->isReferenceType())
    RetTy = RetTy->getPointerElementType();

  llvm::FunctionType *FnTy = llvm::FunctionType::get(RetTy, false);
  llvm::Function *Wrapper = llvm::Function::Create(
      FnTy, getThreadLocalWrapperLinkage(Var->getLinkage()), WrapperName.str(),
      &CGM.getModule());
  // Always resolve references to the wrapper at link time.
  Wrapper->setVisibility(llvm::GlobalValue::HiddenVisibility);
  return Wrapper;
}

void ItaniumCXXABI::EmitThreadLocalInitFuncs(
    llvm::ArrayRef<std::pair<const VarDecl *, llvm::GlobalVariable *> > Decls,
    llvm::Function *InitFunc) {
  for (unsigned I = 0, N = Decls.size(); I != N; ++I) {
    const VarDecl *VD = Decls[I].first;
    llvm::GlobalVariable *Var = Decls[I].second;

    // Mangle the name for the thread_local initialization function.
    SmallString<256> InitFnName;
    {
      llvm::raw_svector_ostream Out(InitFnName);
      getMangleContext().mangleItaniumThreadLocalInit(VD, Out);
      Out.flush();
    }

    // If we have a definition for the variable, emit the initialization
    // function as an alias to the global Init function (if any). Otherwise,
    // produce a declaration of the initialization function.
    llvm::GlobalValue *Init = 0;
    bool InitIsInitFunc = false;
    if (VD->hasDefinition()) {
      InitIsInitFunc = true;
      if (InitFunc)
        Init =
            new llvm::GlobalAlias(InitFunc->getType(), Var->getLinkage(),
                                  InitFnName.str(), InitFunc, &CGM.getModule());
    } else {
      // Emit a weak global function referring to the initialization function.
      // This function will not exist if the TU defining the thread_local
      // variable in question does not need any dynamic initialization for
      // its thread_local variables.
      llvm::FunctionType *FnTy = llvm::FunctionType::get(CGM.VoidTy, false);
      Init = llvm::Function::Create(
          FnTy, llvm::GlobalVariable::ExternalWeakLinkage, InitFnName.str(),
          &CGM.getModule());
    }

    if (Init)
      Init->setVisibility(Var->getVisibility());

    llvm::Function *Wrapper = getOrCreateThreadLocalWrapper(VD, Var);
    llvm::LLVMContext &Context = CGM.getModule().getContext();
    llvm::BasicBlock *Entry = llvm::BasicBlock::Create(Context, "", Wrapper);
    CGBuilderTy Builder(Entry);
    if (InitIsInitFunc) {
      if (Init)
        Builder.CreateCall(Init);
    } else {
      // Don't know whether we have an init function. Call it if it exists.
      llvm::Value *Have = Builder.CreateIsNotNull(Init);
      llvm::BasicBlock *InitBB = llvm::BasicBlock::Create(Context, "", Wrapper);
      llvm::BasicBlock *ExitBB = llvm::BasicBlock::Create(Context, "", Wrapper);
      Builder.CreateCondBr(Have, InitBB, ExitBB);

      Builder.SetInsertPoint(InitBB);
      Builder.CreateCall(Init);
      Builder.CreateBr(ExitBB);

      Builder.SetInsertPoint(ExitBB);
    }

    // For a reference, the result of the wrapper function is a pointer to
    // the referenced object.
    llvm::Value *Val = Var;
    if (VD->getType()->isReferenceType()) {
      llvm::LoadInst *LI = Builder.CreateLoad(Val);
      LI->setAlignment(CGM.getContext().getDeclAlign(VD).getQuantity());
      Val = LI;
    }

    Builder.CreateRet(Val);
  }
}

LValue ItaniumCXXABI::EmitThreadLocalDeclRefExpr(CodeGenFunction &CGF,
                                                 const DeclRefExpr *DRE) {
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  QualType T = VD->getType();
  llvm::Type *Ty = CGF.getTypes().ConvertTypeForMem(T);
  llvm::Value *Val = CGF.CGM.GetAddrOfGlobalVar(VD, Ty);
  llvm::Function *Wrapper =
      getOrCreateThreadLocalWrapper(VD, cast<llvm::GlobalVariable>(Val));

  Val = CGF.Builder.CreateCall(Wrapper);

  LValue LV;
  if (VD->getType()->isReferenceType())
    LV = CGF.MakeNaturalAlignAddrLValue(Val, T);
  else
    LV = CGF.MakeAddrLValue(Val, DRE->getType(),
                            CGF.getContext().getDeclAlign(VD));
  // FIXME: need setObjCGCLValueClass?
  return LV;
}
