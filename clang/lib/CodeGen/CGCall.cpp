//===----- CGCall.h - Encapsulate calling convention details ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CGCall.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/CodeGen/CodeGenOptions.h"
#include "llvm/Attributes.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Target/TargetData.h"

#include "ABIInfo.h"

using namespace clang;
using namespace CodeGen;

/***/

// FIXME: Use iterator and sidestep silly type array creation.

static unsigned ClangCallConvToLLVMCallConv(CallingConv CC) {
  switch (CC) {
  default: return llvm::CallingConv::C;
  case CC_X86StdCall: return llvm::CallingConv::X86_StdCall;
  case CC_X86FastCall: return llvm::CallingConv::X86_FastCall;
  case CC_X86ThisCall: return llvm::CallingConv::X86_ThisCall;
  }
}

/// Derives the 'this' type for codegen purposes, i.e. ignoring method
/// qualification.
/// FIXME: address space qualification?
static CanQualType GetThisType(ASTContext &Context, const CXXRecordDecl *RD) {
  QualType RecTy = Context.getTagDeclType(RD)->getCanonicalTypeInternal();
  return Context.getPointerType(CanQualType::CreateUnsafe(RecTy));
}

/// Returns the canonical formal type of the given C++ method.
static CanQual<FunctionProtoType> GetFormalType(const CXXMethodDecl *MD) {
  return MD->getType()->getCanonicalTypeUnqualified()
           .getAs<FunctionProtoType>();
}

/// Returns the "extra-canonicalized" return type, which discards
/// qualifiers on the return type.  Codegen doesn't care about them,
/// and it makes ABI code a little easier to be able to assume that
/// all parameter and return types are top-level unqualified.
static CanQualType GetReturnType(QualType RetTy) {
  return RetTy->getCanonicalTypeUnqualified().getUnqualifiedType();
}

const CGFunctionInfo &
CodeGenTypes::getFunctionInfo(CanQual<FunctionNoProtoType> FTNP) {
  return getFunctionInfo(FTNP->getResultType().getUnqualifiedType(),
                         llvm::SmallVector<CanQualType, 16>(),
                         FTNP->getExtInfo());
}

/// \param Args - contains any initial parameters besides those
///   in the formal type
static const CGFunctionInfo &getFunctionInfo(CodeGenTypes &CGT,
                                  llvm::SmallVectorImpl<CanQualType> &ArgTys,
                                             CanQual<FunctionProtoType> FTP) {
  // FIXME: Kill copy.
  for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
    ArgTys.push_back(FTP->getArgType(i));
  CanQualType ResTy = FTP->getResultType().getUnqualifiedType();
  return CGT.getFunctionInfo(ResTy, ArgTys,
                             FTP->getExtInfo());
}

const CGFunctionInfo &
CodeGenTypes::getFunctionInfo(CanQual<FunctionProtoType> FTP) {
  llvm::SmallVector<CanQualType, 16> ArgTys;
  return ::getFunctionInfo(*this, ArgTys, FTP);
}

static CallingConv getCallingConventionForDecl(const Decl *D) {
  // Set the appropriate calling convention for the Function.
  if (D->hasAttr<StdCallAttr>())
    return CC_X86StdCall;

  if (D->hasAttr<FastCallAttr>())
    return CC_X86FastCall;

  if (D->hasAttr<ThisCallAttr>())
    return CC_X86ThisCall;

  return CC_C;
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(const CXXRecordDecl *RD,
                                                 const FunctionProtoType *FTP) {
  llvm::SmallVector<CanQualType, 16> ArgTys;

  // Add the 'this' pointer.
  ArgTys.push_back(GetThisType(Context, RD));

  return ::getFunctionInfo(*this, ArgTys,
              FTP->getCanonicalTypeUnqualified().getAs<FunctionProtoType>());
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(const CXXMethodDecl *MD) {
  llvm::SmallVector<CanQualType, 16> ArgTys;

  // Add the 'this' pointer unless this is a static method.
  if (MD->isInstance())
    ArgTys.push_back(GetThisType(Context, MD->getParent()));

  return ::getFunctionInfo(*this, ArgTys, GetFormalType(MD));
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(const CXXConstructorDecl *D, 
                                                    CXXCtorType Type) {
  llvm::SmallVector<CanQualType, 16> ArgTys;

  // Add the 'this' pointer.
  ArgTys.push_back(GetThisType(Context, D->getParent()));

  // Check if we need to add a VTT parameter (which has type void **).
  if (Type == Ctor_Base && D->getParent()->getNumVBases() != 0)
    ArgTys.push_back(Context.getPointerType(Context.VoidPtrTy));

  return ::getFunctionInfo(*this, ArgTys, GetFormalType(D));
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(const CXXDestructorDecl *D,
                                                    CXXDtorType Type) {
  llvm::SmallVector<CanQualType, 16> ArgTys;
  
  // Add the 'this' pointer.
  ArgTys.push_back(GetThisType(Context, D->getParent()));
  
  // Check if we need to add a VTT parameter (which has type void **).
  if (Type == Dtor_Base && D->getParent()->getNumVBases() != 0)
    ArgTys.push_back(Context.getPointerType(Context.VoidPtrTy));

  return ::getFunctionInfo(*this, ArgTys, GetFormalType(D));
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(const FunctionDecl *FD) {
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD))
    if (MD->isInstance())
      return getFunctionInfo(MD);

  CanQualType FTy = FD->getType()->getCanonicalTypeUnqualified();
  assert(isa<FunctionType>(FTy));
  if (isa<FunctionNoProtoType>(FTy))
    return getFunctionInfo(FTy.getAs<FunctionNoProtoType>());  
  assert(isa<FunctionProtoType>(FTy));
  return getFunctionInfo(FTy.getAs<FunctionProtoType>());
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(const ObjCMethodDecl *MD) {
  llvm::SmallVector<CanQualType, 16> ArgTys;
  ArgTys.push_back(Context.getCanonicalParamType(MD->getSelfDecl()->getType()));
  ArgTys.push_back(Context.getCanonicalParamType(Context.getObjCSelType()));
  // FIXME: Kill copy?
  for (ObjCMethodDecl::param_iterator i = MD->param_begin(),
         e = MD->param_end(); i != e; ++i) {
    ArgTys.push_back(Context.getCanonicalParamType((*i)->getType()));
  }
  return getFunctionInfo(GetReturnType(MD->getResultType()),
                         ArgTys,
                         FunctionType::ExtInfo(
                             /*NoReturn*/ false,
                             /*RegParm*/ 0,
                             getCallingConventionForDecl(MD)));
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(GlobalDecl GD) {
  // FIXME: Do we need to handle ObjCMethodDecl?
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
                                              
  if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD))
    return getFunctionInfo(CD, GD.getCtorType());

  if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(FD))
    return getFunctionInfo(DD, GD.getDtorType());
  
  return getFunctionInfo(FD);
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(QualType ResTy,
                                                    const CallArgList &Args,
                                            const FunctionType::ExtInfo &Info) {
  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> ArgTys;
  for (CallArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i)
    ArgTys.push_back(Context.getCanonicalParamType(i->second));
  return getFunctionInfo(GetReturnType(ResTy), ArgTys, Info);
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(QualType ResTy,
                                                    const FunctionArgList &Args,
                                            const FunctionType::ExtInfo &Info) {
  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> ArgTys;
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i)
    ArgTys.push_back(Context.getCanonicalParamType(i->second));
  return getFunctionInfo(GetReturnType(ResTy), ArgTys, Info);
}

const CGFunctionInfo &CodeGenTypes::getFunctionInfo(CanQualType ResTy,
                           const llvm::SmallVectorImpl<CanQualType> &ArgTys,
                                            const FunctionType::ExtInfo &Info) {
#ifndef NDEBUG
  for (llvm::SmallVectorImpl<CanQualType>::const_iterator
         I = ArgTys.begin(), E = ArgTys.end(); I != E; ++I)
    assert(I->isCanonicalAsParam());
#endif

  unsigned CC = ClangCallConvToLLVMCallConv(Info.getCC());

  // Lookup or create unique function info.
  llvm::FoldingSetNodeID ID;
  CGFunctionInfo::Profile(ID, Info, ResTy,
                          ArgTys.begin(), ArgTys.end());

  void *InsertPos = 0;
  CGFunctionInfo *FI = FunctionInfos.FindNodeOrInsertPos(ID, InsertPos);
  if (FI)
    return *FI;

  // Construct the function info.
  FI = new CGFunctionInfo(CC, Info.getNoReturn(), Info.getRegParm(), ResTy, ArgTys);
  FunctionInfos.InsertNode(FI, InsertPos);

  // Compute ABI information.
  getABIInfo().computeInfo(*FI, getContext(), TheModule.getContext());

  return *FI;
}

CGFunctionInfo::CGFunctionInfo(unsigned _CallingConvention,
                               bool _NoReturn,
                               unsigned _RegParm,
                               CanQualType ResTy,
                               const llvm::SmallVectorImpl<CanQualType> &ArgTys)
  : CallingConvention(_CallingConvention),
    EffectiveCallingConvention(_CallingConvention),
    NoReturn(_NoReturn), RegParm(_RegParm)
{
  NumArgs = ArgTys.size();
  Args = new ArgInfo[1 + NumArgs];
  Args[0].type = ResTy;
  for (unsigned i = 0; i < NumArgs; ++i)
    Args[1 + i].type = ArgTys[i];
}

/***/

void CodeGenTypes::GetExpandedTypes(QualType Ty,
                                    std::vector<const llvm::Type*> &ArgTys) {
  const RecordType *RT = Ty->getAsStructureType();
  assert(RT && "Can only expand structure types.");
  const RecordDecl *RD = RT->getDecl();
  assert(!RD->hasFlexibleArrayMember() &&
         "Cannot expand structure with flexible array.");

  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    const FieldDecl *FD = *i;
    assert(!FD->isBitField() &&
           "Cannot expand structure with bit-field members.");

    QualType FT = FD->getType();
    if (CodeGenFunction::hasAggregateLLVMType(FT)) {
      GetExpandedTypes(FT, ArgTys);
    } else {
      ArgTys.push_back(ConvertType(FT));
    }
  }
}

llvm::Function::arg_iterator
CodeGenFunction::ExpandTypeFromArgs(QualType Ty, LValue LV,
                                    llvm::Function::arg_iterator AI) {
  const RecordType *RT = Ty->getAsStructureType();
  assert(RT && "Can only expand structure types.");

  RecordDecl *RD = RT->getDecl();
  assert(LV.isSimple() &&
         "Unexpected non-simple lvalue during struct expansion.");
  llvm::Value *Addr = LV.getAddress();
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    FieldDecl *FD = *i;
    QualType FT = FD->getType();

    // FIXME: What are the right qualifiers here?
    LValue LV = EmitLValueForField(Addr, FD, 0);
    if (CodeGenFunction::hasAggregateLLVMType(FT)) {
      AI = ExpandTypeFromArgs(FT, LV, AI);
    } else {
      EmitStoreThroughLValue(RValue::get(AI), LV, FT);
      ++AI;
    }
  }

  return AI;
}

void
CodeGenFunction::ExpandTypeToArgs(QualType Ty, RValue RV,
                                  llvm::SmallVector<llvm::Value*, 16> &Args) {
  const RecordType *RT = Ty->getAsStructureType();
  assert(RT && "Can only expand structure types.");

  RecordDecl *RD = RT->getDecl();
  assert(RV.isAggregate() && "Unexpected rvalue during struct expansion");
  llvm::Value *Addr = RV.getAggregateAddr();
  for (RecordDecl::field_iterator i = RD->field_begin(), e = RD->field_end();
         i != e; ++i) {
    FieldDecl *FD = *i;
    QualType FT = FD->getType();

    // FIXME: What are the right qualifiers here?
    LValue LV = EmitLValueForField(Addr, FD, 0);
    if (CodeGenFunction::hasAggregateLLVMType(FT)) {
      ExpandTypeToArgs(FT, RValue::getAggregate(LV.getAddress()), Args);
    } else {
      RValue RV = EmitLoadOfLValue(LV, FT);
      assert(RV.isScalar() &&
             "Unexpected non-scalar rvalue during struct expansion.");
      Args.push_back(RV.getScalarVal());
    }
  }
}

/// CreateCoercedLoad - Create a load from \arg SrcPtr interpreted as
/// a pointer to an object of type \arg Ty.
///
/// This safely handles the case when the src type is smaller than the
/// destination type; in this situation the values of bits which not
/// present in the src are undefined.
static llvm::Value *CreateCoercedLoad(llvm::Value *SrcPtr,
                                      const llvm::Type *Ty,
                                      CodeGenFunction &CGF) {
  const llvm::Type *SrcTy =
    cast<llvm::PointerType>(SrcPtr->getType())->getElementType();
  uint64_t SrcSize = CGF.CGM.getTargetData().getTypeAllocSize(SrcTy);
  uint64_t DstSize = CGF.CGM.getTargetData().getTypeAllocSize(Ty);

  // If load is legal, just bitcast the src pointer.
  if (SrcSize >= DstSize) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    llvm::Value *Casted =
      CGF.Builder.CreateBitCast(SrcPtr, llvm::PointerType::getUnqual(Ty));
    llvm::LoadInst *Load = CGF.Builder.CreateLoad(Casted);
    // FIXME: Use better alignment / avoid requiring aligned load.
    Load->setAlignment(1);
    return Load;
  } else {
    // Otherwise do coercion through memory. This is stupid, but
    // simple.
    llvm::Value *Tmp = CGF.CreateTempAlloca(Ty);
    llvm::Value *Casted =
      CGF.Builder.CreateBitCast(Tmp, llvm::PointerType::getUnqual(SrcTy));
    llvm::StoreInst *Store =
      CGF.Builder.CreateStore(CGF.Builder.CreateLoad(SrcPtr), Casted);
    // FIXME: Use better alignment / avoid requiring aligned store.
    Store->setAlignment(1);
    return CGF.Builder.CreateLoad(Tmp);
  }
}

/// CreateCoercedStore - Create a store to \arg DstPtr from \arg Src,
/// where the source and destination may have different types.
///
/// This safely handles the case when the src type is larger than the
/// destination type; the upper bits of the src will be lost.
static void CreateCoercedStore(llvm::Value *Src,
                               llvm::Value *DstPtr,
                               bool DstIsVolatile,
                               CodeGenFunction &CGF) {
  const llvm::Type *SrcTy = Src->getType();
  const llvm::Type *DstTy =
    cast<llvm::PointerType>(DstPtr->getType())->getElementType();

  uint64_t SrcSize = CGF.CGM.getTargetData().getTypeAllocSize(SrcTy);
  uint64_t DstSize = CGF.CGM.getTargetData().getTypeAllocSize(DstTy);

  // If store is legal, just bitcast the src pointer.
  if (SrcSize <= DstSize) {
    llvm::Value *Casted =
      CGF.Builder.CreateBitCast(DstPtr, llvm::PointerType::getUnqual(SrcTy));
    // FIXME: Use better alignment / avoid requiring aligned store.
    CGF.Builder.CreateStore(Src, Casted, DstIsVolatile)->setAlignment(1);
  } else {
    // Otherwise do coercion through memory. This is stupid, but
    // simple.

    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    llvm::Value *Tmp = CGF.CreateTempAlloca(SrcTy);
    CGF.Builder.CreateStore(Src, Tmp);
    llvm::Value *Casted =
      CGF.Builder.CreateBitCast(Tmp, llvm::PointerType::getUnqual(DstTy));
    llvm::LoadInst *Load = CGF.Builder.CreateLoad(Casted);
    // FIXME: Use better alignment / avoid requiring aligned load.
    Load->setAlignment(1);
    CGF.Builder.CreateStore(Load, DstPtr, DstIsVolatile);
  }
}

/***/

bool CodeGenModule::ReturnTypeUsesSret(const CGFunctionInfo &FI) {
  return FI.getReturnInfo().isIndirect();
}

const llvm::FunctionType *CodeGenTypes::GetFunctionType(GlobalDecl GD) {
  const CGFunctionInfo &FI = getFunctionInfo(GD);
  
  // For definition purposes, don't consider a K&R function variadic.
  bool Variadic = false;
  if (const FunctionProtoType *FPT =
        cast<FunctionDecl>(GD.getDecl())->getType()->getAs<FunctionProtoType>())
    Variadic = FPT->isVariadic();

  return GetFunctionType(FI, Variadic);
}

const llvm::FunctionType *
CodeGenTypes::GetFunctionType(const CGFunctionInfo &FI, bool IsVariadic) {
  std::vector<const llvm::Type*> ArgTys;

  const llvm::Type *ResultType = 0;

  QualType RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();
  switch (RetAI.getKind()) {
  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    ResultType = ConvertType(RetTy);
    break;

  case ABIArgInfo::Indirect: {
    assert(!RetAI.getIndirectAlign() && "Align unused on indirect return.");
    ResultType = llvm::Type::getVoidTy(getLLVMContext());
    const llvm::Type *STy = ConvertType(RetTy);
    ArgTys.push_back(llvm::PointerType::get(STy, RetTy.getAddressSpace()));
    break;
  }

  case ABIArgInfo::Ignore:
    ResultType = llvm::Type::getVoidTy(getLLVMContext());
    break;

  case ABIArgInfo::Coerce:
    ResultType = RetAI.getCoerceToType();
    break;
  }

  for (CGFunctionInfo::const_arg_iterator it = FI.arg_begin(),
         ie = FI.arg_end(); it != ie; ++it) {
    const ABIArgInfo &AI = it->info;

    switch (AI.getKind()) {
    case ABIArgInfo::Ignore:
      break;

    case ABIArgInfo::Coerce:
      ArgTys.push_back(AI.getCoerceToType());
      break;

    case ABIArgInfo::Indirect: {
      // indirect arguments are always on the stack, which is addr space #0.
      const llvm::Type *LTy = ConvertTypeForMem(it->type);
      ArgTys.push_back(llvm::PointerType::getUnqual(LTy));
      break;
    }

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct:
      ArgTys.push_back(ConvertType(it->type));
      break;

    case ABIArgInfo::Expand:
      GetExpandedTypes(it->type, ArgTys);
      break;
    }
  }

  return llvm::FunctionType::get(ResultType, ArgTys, IsVariadic);
}

static bool HasIncompleteReturnTypeOrArgumentTypes(const FunctionProtoType *T) {
  if (const TagType *TT = T->getResultType()->getAs<TagType>()) {
    if (!TT->getDecl()->isDefinition())
      return true;
  }

  for (unsigned i = 0, e = T->getNumArgs(); i != e; ++i) {
    if (const TagType *TT = T->getArgType(i)->getAs<TagType>()) {
      if (!TT->getDecl()->isDefinition())
        return true;
    }
  }

  return false;
}

const llvm::Type *
CodeGenTypes::GetFunctionTypeForVTable(const CXXMethodDecl *MD) {
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  
  if (!HasIncompleteReturnTypeOrArgumentTypes(FPT))
    return GetFunctionType(getFunctionInfo(MD), FPT->isVariadic());

  return llvm::OpaqueType::get(getLLVMContext());
}

void CodeGenModule::ConstructAttributeList(const CGFunctionInfo &FI,
                                           const Decl *TargetDecl,
                                           AttributeListType &PAL, 
                                           unsigned &CallingConv) {
  unsigned FuncAttrs = 0;
  unsigned RetAttrs = 0;

  CallingConv = FI.getEffectiveCallingConvention();

  if (FI.isNoReturn())
    FuncAttrs |= llvm::Attribute::NoReturn;

  // FIXME: handle sseregparm someday...
  if (TargetDecl) {
    if (TargetDecl->hasAttr<NoThrowAttr>())
      FuncAttrs |= llvm::Attribute::NoUnwind;
    if (TargetDecl->hasAttr<NoReturnAttr>())
      FuncAttrs |= llvm::Attribute::NoReturn;
    if (TargetDecl->hasAttr<ConstAttr>())
      FuncAttrs |= llvm::Attribute::ReadNone;
    else if (TargetDecl->hasAttr<PureAttr>())
      FuncAttrs |= llvm::Attribute::ReadOnly;
    if (TargetDecl->hasAttr<MallocAttr>())
      RetAttrs |= llvm::Attribute::NoAlias;
  }

  if (CodeGenOpts.OptimizeSize)
    FuncAttrs |= llvm::Attribute::OptimizeForSize;
  if (CodeGenOpts.DisableRedZone)
    FuncAttrs |= llvm::Attribute::NoRedZone;
  if (CodeGenOpts.NoImplicitFloat)
    FuncAttrs |= llvm::Attribute::NoImplicitFloat;

  QualType RetTy = FI.getReturnType();
  unsigned Index = 1;
  const ABIArgInfo &RetAI = FI.getReturnInfo();
  switch (RetAI.getKind()) {
  case ABIArgInfo::Extend:
   if (RetTy->isSignedIntegerType()) {
     RetAttrs |= llvm::Attribute::SExt;
   } else if (RetTy->isUnsignedIntegerType()) {
     RetAttrs |= llvm::Attribute::ZExt;
   }
   // FALLTHROUGH
  case ABIArgInfo::Direct:
    break;

  case ABIArgInfo::Indirect:
    PAL.push_back(llvm::AttributeWithIndex::get(Index,
                                                llvm::Attribute::StructRet));
    ++Index;
    // sret disables readnone and readonly
    FuncAttrs &= ~(llvm::Attribute::ReadOnly |
                   llvm::Attribute::ReadNone);
    break;

  case ABIArgInfo::Ignore:
  case ABIArgInfo::Coerce:
    break;

  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");
  }

  if (RetAttrs)
    PAL.push_back(llvm::AttributeWithIndex::get(0, RetAttrs));

  // FIXME: we need to honour command line settings also...
  // FIXME: RegParm should be reduced in case of nested functions and/or global
  // register variable.
  signed RegParm = FI.getRegParm();

  unsigned PointerWidth = getContext().Target.getPointerWidth(0);
  for (CGFunctionInfo::const_arg_iterator it = FI.arg_begin(),
         ie = FI.arg_end(); it != ie; ++it) {
    QualType ParamType = it->type;
    const ABIArgInfo &AI = it->info;
    unsigned Attributes = 0;

    // 'restrict' -> 'noalias' is done in EmitFunctionProlog when we
    // have the corresponding parameter variable.  It doesn't make
    // sense to do it here because parameters are so fucked up.

    switch (AI.getKind()) {
    case ABIArgInfo::Coerce:
      break;

    case ABIArgInfo::Indirect:
      if (AI.getIndirectByVal())
        Attributes |= llvm::Attribute::ByVal;

      Attributes |=
        llvm::Attribute::constructAlignmentFromInt(AI.getIndirectAlign());
      // byval disables readnone and readonly.
      FuncAttrs &= ~(llvm::Attribute::ReadOnly |
                     llvm::Attribute::ReadNone);
      break;

    case ABIArgInfo::Extend:
     if (ParamType->isSignedIntegerType()) {
       Attributes |= llvm::Attribute::SExt;
     } else if (ParamType->isUnsignedIntegerType()) {
       Attributes |= llvm::Attribute::ZExt;
     }
     // FALLS THROUGH
    case ABIArgInfo::Direct:
      if (RegParm > 0 &&
          (ParamType->isIntegerType() || ParamType->isPointerType())) {
        RegParm -=
          (Context.getTypeSize(ParamType) + PointerWidth - 1) / PointerWidth;
        if (RegParm >= 0)
          Attributes |= llvm::Attribute::InReg;
      }
      // FIXME: handle sseregparm someday...
      break;

    case ABIArgInfo::Ignore:
      // Skip increment, no matching LLVM parameter.
      continue;

    case ABIArgInfo::Expand: {
      std::vector<const llvm::Type*> Tys;
      // FIXME: This is rather inefficient. Do we ever actually need to do
      // anything here? The result should be just reconstructed on the other
      // side, so extension should be a non-issue.
      getTypes().GetExpandedTypes(ParamType, Tys);
      Index += Tys.size();
      continue;
    }
    }

    if (Attributes)
      PAL.push_back(llvm::AttributeWithIndex::get(Index, Attributes));
    ++Index;
  }
  if (FuncAttrs)
    PAL.push_back(llvm::AttributeWithIndex::get(~0, FuncAttrs));
}

void CodeGenFunction::EmitFunctionProlog(const CGFunctionInfo &FI,
                                         llvm::Function *Fn,
                                         const FunctionArgList &Args) {
  // If this is an implicit-return-zero function, go ahead and
  // initialize the return value.  TODO: it might be nice to have
  // a more general mechanism for this that didn't require synthesized
  // return statements.
  if (const FunctionDecl* FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl)) {
    if (FD->hasImplicitReturnZero()) {
      QualType RetTy = FD->getResultType().getUnqualifiedType();
      const llvm::Type* LLVMTy = CGM.getTypes().ConvertType(RetTy);
      llvm::Constant* Zero = llvm::Constant::getNullValue(LLVMTy);
      Builder.CreateStore(Zero, ReturnValue);
    }
  }

  // FIXME: We no longer need the types from FunctionArgList; lift up and
  // simplify.

  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = Fn->arg_begin();

  // Name the struct return argument.
  if (CGM.ReturnTypeUsesSret(FI)) {
    AI->setName("agg.result");
    ++AI;
  }

  assert(FI.arg_size() == Args.size() &&
         "Mismatch between function signature & arguments.");
  CGFunctionInfo::const_arg_iterator info_it = FI.arg_begin();
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i, ++info_it) {
    const VarDecl *Arg = i->first;
    QualType Ty = info_it->type;
    const ABIArgInfo &ArgI = info_it->info;

    switch (ArgI.getKind()) {
    case ABIArgInfo::Indirect: {
      llvm::Value* V = AI;
      if (hasAggregateLLVMType(Ty)) {
        // Do nothing, aggregates and complex variables are accessed by
        // reference.
      } else {
        // Load scalar value from indirect argument.
        V = EmitLoadOfScalar(V, false, Ty);
        if (!getContext().typesAreCompatible(Ty, Arg->getType())) {
          // This must be a promotion, for something like
          // "void a(x) short x; {..."
          V = EmitScalarConversion(V, Ty, Arg->getType());
        }
      }
      EmitParmDecl(*Arg, V);
      break;
    }

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      assert(AI != Fn->arg_end() && "Argument mismatch!");
      llvm::Value* V = AI;
      if (hasAggregateLLVMType(Ty)) {
        // Create a temporary alloca to hold the argument; the rest of
        // codegen expects to access aggregates & complex values by
        // reference.
        V = CreateMemTemp(Ty);
        Builder.CreateStore(AI, V);
      } else {
        if (Arg->getType().isRestrictQualified())
          AI->addAttr(llvm::Attribute::NoAlias);

        if (!getContext().typesAreCompatible(Ty, Arg->getType())) {
          // This must be a promotion, for something like
          // "void a(x) short x; {..."
          V = EmitScalarConversion(V, Ty, Arg->getType());
        }
      }
      EmitParmDecl(*Arg, V);
      break;
    }

    case ABIArgInfo::Expand: {
      // If this structure was expanded into multiple arguments then
      // we need to create a temporary and reconstruct it from the
      // arguments.
      llvm::Value *Temp = CreateMemTemp(Ty, Arg->getName() + ".addr");
      // FIXME: What are the right qualifiers here?
      llvm::Function::arg_iterator End =
        ExpandTypeFromArgs(Ty, LValue::MakeAddr(Temp, Qualifiers()), AI);
      EmitParmDecl(*Arg, Temp);

      // Name the arguments used in expansion and increment AI.
      unsigned Index = 0;
      for (; AI != End; ++AI, ++Index)
        AI->setName(Arg->getName() + "." + llvm::Twine(Index));
      continue;
    }

    case ABIArgInfo::Ignore:
      // Initialize the local variable appropriately.
      if (hasAggregateLLVMType(Ty)) {
        EmitParmDecl(*Arg, CreateMemTemp(Ty));
      } else {
        EmitParmDecl(*Arg, llvm::UndefValue::get(ConvertType(Arg->getType())));
      }

      // Skip increment, no matching LLVM parameter.
      continue;

    case ABIArgInfo::Coerce: {
      assert(AI != Fn->arg_end() && "Argument mismatch!");
      // FIXME: This is very wasteful; EmitParmDecl is just going to drop the
      // result in a new alloca anyway, so we could just store into that
      // directly if we broke the abstraction down more.
      llvm::Value *V = CreateMemTemp(Ty, "coerce");
      CreateCoercedStore(AI, V, /*DestIsVolatile=*/false, *this);
      // Match to what EmitParmDecl is expecting for this type.
      if (!CodeGenFunction::hasAggregateLLVMType(Ty)) {
        V = EmitLoadOfScalar(V, false, Ty);
        if (!getContext().typesAreCompatible(Ty, Arg->getType())) {
          // This must be a promotion, for something like
          // "void a(x) short x; {..."
          V = EmitScalarConversion(V, Ty, Arg->getType());
        }
      }
      EmitParmDecl(*Arg, V);
      break;
    }
    }

    ++AI;
  }
  assert(AI == Fn->arg_end() && "Argument mismatch!");
}

void CodeGenFunction::EmitFunctionEpilog(const CGFunctionInfo &FI,
                                         llvm::Value *ReturnValue) {
  llvm::Value *RV = 0;

  // Functions with no result always return void.
  if (ReturnValue) {
    QualType RetTy = FI.getReturnType();
    const ABIArgInfo &RetAI = FI.getReturnInfo();

    switch (RetAI.getKind()) {
    case ABIArgInfo::Indirect:
      if (RetTy->isAnyComplexType()) {
        ComplexPairTy RT = LoadComplexFromAddr(ReturnValue, false);
        StoreComplexToAddr(RT, CurFn->arg_begin(), false);
      } else if (CodeGenFunction::hasAggregateLLVMType(RetTy)) {
        // Do nothing; aggregrates get evaluated directly into the destination.
      } else {
        EmitStoreOfScalar(Builder.CreateLoad(ReturnValue), CurFn->arg_begin(),
                          false, RetTy);
      }
      break;

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct:
      // The internal return value temp always will have
      // pointer-to-return-type type.
      RV = Builder.CreateLoad(ReturnValue);
      break;

    case ABIArgInfo::Ignore:
      break;

    case ABIArgInfo::Coerce:
      RV = CreateCoercedLoad(ReturnValue, RetAI.getCoerceToType(), *this);
      break;

    case ABIArgInfo::Expand:
      assert(0 && "Invalid ABI kind for return argument");
    }
  }

  if (RV) {
    Builder.CreateRet(RV);
  } else {
    Builder.CreateRetVoid();
  }
}

RValue CodeGenFunction::EmitDelegateCallArg(const VarDecl *Param) {
  // StartFunction converted the ABI-lowered parameter(s) into a
  // local alloca.  We need to turn that into an r-value suitable
  // for EmitCall.
  llvm::Value *Local = GetAddrOfLocalVar(Param);

  QualType ArgType = Param->getType();
 
  // For the most part, we just need to load the alloca, except:
  // 1) aggregate r-values are actually pointers to temporaries, and
  // 2) references to aggregates are pointers directly to the aggregate.
  // I don't know why references to non-aggregates are different here.
  if (const ReferenceType *RefType = ArgType->getAs<ReferenceType>()) {
    if (hasAggregateLLVMType(RefType->getPointeeType()))
      return RValue::getAggregate(Local);

    // Locals which are references to scalars are represented
    // with allocas holding the pointer.
    return RValue::get(Builder.CreateLoad(Local));
  }

  if (ArgType->isAnyComplexType())
    return RValue::getComplex(LoadComplexFromAddr(Local, /*volatile*/ false));

  if (hasAggregateLLVMType(ArgType))
    return RValue::getAggregate(Local);

  return RValue::get(EmitLoadOfScalar(Local, false, ArgType));
}

RValue CodeGenFunction::EmitCallArg(const Expr *E, QualType ArgType) {
  if (ArgType->isReferenceType())
    return EmitReferenceBindingToExpr(E);

  return EmitAnyExprToTemp(E);
}

RValue CodeGenFunction::EmitCall(const CGFunctionInfo &CallInfo,
                                 llvm::Value *Callee,
                                 ReturnValueSlot ReturnValue,
                                 const CallArgList &CallArgs,
                                 const Decl *TargetDecl,
                                 llvm::Instruction **callOrInvoke) {
  // FIXME: We no longer need the types from CallArgs; lift up and simplify.
  llvm::SmallVector<llvm::Value*, 16> Args;

  // Handle struct-return functions by passing a pointer to the
  // location that we would like to return into.
  QualType RetTy = CallInfo.getReturnType();
  const ABIArgInfo &RetAI = CallInfo.getReturnInfo();


  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  if (CGM.ReturnTypeUsesSret(CallInfo)) {
    llvm::Value *Value = ReturnValue.getValue();
    if (!Value)
      Value = CreateMemTemp(RetTy);
    Args.push_back(Value);
  }

  assert(CallInfo.arg_size() == CallArgs.size() &&
         "Mismatch between function signature & arguments.");
  CGFunctionInfo::const_arg_iterator info_it = CallInfo.arg_begin();
  for (CallArgList::const_iterator I = CallArgs.begin(), E = CallArgs.end();
       I != E; ++I, ++info_it) {
    const ABIArgInfo &ArgInfo = info_it->info;
    RValue RV = I->first;

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Indirect:
      if (RV.isScalar() || RV.isComplex()) {
        // Make a temporary alloca to pass the argument.
        Args.push_back(CreateMemTemp(I->second));
        if (RV.isScalar())
          EmitStoreOfScalar(RV.getScalarVal(), Args.back(), false, I->second);
        else
          StoreComplexToAddr(RV.getComplexVal(), Args.back(), false);
      } else {
        Args.push_back(RV.getAggregateAddr());
      }
      break;

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct:
      if (RV.isScalar()) {
        Args.push_back(RV.getScalarVal());
      } else if (RV.isComplex()) {
        llvm::Value *Tmp = llvm::UndefValue::get(ConvertType(I->second));
        Tmp = Builder.CreateInsertValue(Tmp, RV.getComplexVal().first, 0);
        Tmp = Builder.CreateInsertValue(Tmp, RV.getComplexVal().second, 1);
        Args.push_back(Tmp);
      } else {
        Args.push_back(Builder.CreateLoad(RV.getAggregateAddr()));
      }
      break;

    case ABIArgInfo::Ignore:
      break;

    case ABIArgInfo::Coerce: {
      // FIXME: Avoid the conversion through memory if possible.
      llvm::Value *SrcPtr;
      if (RV.isScalar()) {
        SrcPtr = CreateMemTemp(I->second, "coerce");
        EmitStoreOfScalar(RV.getScalarVal(), SrcPtr, false, I->second);
      } else if (RV.isComplex()) {
        SrcPtr = CreateMemTemp(I->second, "coerce");
        StoreComplexToAddr(RV.getComplexVal(), SrcPtr, false);
      } else
        SrcPtr = RV.getAggregateAddr();
      Args.push_back(CreateCoercedLoad(SrcPtr, ArgInfo.getCoerceToType(),
                                       *this));
      break;
    }

    case ABIArgInfo::Expand:
      ExpandTypeToArgs(I->second, RV, Args);
      break;
    }
  }

  // If the callee is a bitcast of a function to a varargs pointer to function
  // type, check to see if we can remove the bitcast.  This handles some cases
  // with unprototyped functions.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Callee))
    if (llvm::Function *CalleeF = dyn_cast<llvm::Function>(CE->getOperand(0))) {
      const llvm::PointerType *CurPT=cast<llvm::PointerType>(Callee->getType());
      const llvm::FunctionType *CurFT =
        cast<llvm::FunctionType>(CurPT->getElementType());
      const llvm::FunctionType *ActualFT = CalleeF->getFunctionType();

      if (CE->getOpcode() == llvm::Instruction::BitCast &&
          ActualFT->getReturnType() == CurFT->getReturnType() &&
          ActualFT->getNumParams() == CurFT->getNumParams() &&
          ActualFT->getNumParams() == Args.size()) {
        bool ArgsMatch = true;
        for (unsigned i = 0, e = ActualFT->getNumParams(); i != e; ++i)
          if (ActualFT->getParamType(i) != CurFT->getParamType(i)) {
            ArgsMatch = false;
            break;
          }

        // Strip the cast if we can get away with it.  This is a nice cleanup,
        // but also allows us to inline the function at -O0 if it is marked
        // always_inline.
        if (ArgsMatch)
          Callee = CalleeF;
      }
    }


  llvm::BasicBlock *InvokeDest = getInvokeDest();
  unsigned CallingConv;
  CodeGen::AttributeListType AttributeList;
  CGM.ConstructAttributeList(CallInfo, TargetDecl, AttributeList, CallingConv);
  llvm::AttrListPtr Attrs = llvm::AttrListPtr::get(AttributeList.begin(),
                                                   AttributeList.end());

  llvm::CallSite CS;
  if (!InvokeDest || (Attrs.getFnAttributes() & llvm::Attribute::NoUnwind)) {
    CS = Builder.CreateCall(Callee, Args.data(), Args.data()+Args.size());
  } else {
    llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
    CS = Builder.CreateInvoke(Callee, Cont, InvokeDest,
                              Args.data(), Args.data()+Args.size());
    EmitBlock(Cont);
  }
  if (callOrInvoke) {
    *callOrInvoke = CS.getInstruction();
  }

  CS.setAttributes(Attrs);
  CS.setCallingConv(static_cast<llvm::CallingConv::ID>(CallingConv));

  // If the call doesn't return, finish the basic block and clear the
  // insertion point; this allows the rest of IRgen to discard
  // unreachable code.
  if (CS.doesNotReturn()) {
    Builder.CreateUnreachable();
    Builder.ClearInsertionPoint();

    // FIXME: For now, emit a dummy basic block because expr emitters in
    // generally are not ready to handle emitting expressions at unreachable
    // points.
    EnsureInsertPoint();

    // Return a reasonable RValue.
    return GetUndefRValue(RetTy);
  }

  llvm::Instruction *CI = CS.getInstruction();
  if (Builder.isNamePreserving() && !CI->getType()->isVoidTy())
    CI->setName("call");

  switch (RetAI.getKind()) {
  case ABIArgInfo::Indirect:
    if (RetTy->isAnyComplexType())
      return RValue::getComplex(LoadComplexFromAddr(Args[0], false));
    if (CodeGenFunction::hasAggregateLLVMType(RetTy))
      return RValue::getAggregate(Args[0]);
    return RValue::get(EmitLoadOfScalar(Args[0], false, RetTy));

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    if (RetTy->isAnyComplexType()) {
      llvm::Value *Real = Builder.CreateExtractValue(CI, 0);
      llvm::Value *Imag = Builder.CreateExtractValue(CI, 1);
      return RValue::getComplex(std::make_pair(Real, Imag));
    }
    if (CodeGenFunction::hasAggregateLLVMType(RetTy)) {
      llvm::Value *DestPtr = ReturnValue.getValue();
      bool DestIsVolatile = ReturnValue.isVolatile();

      if (!DestPtr) {
        DestPtr = CreateMemTemp(RetTy, "agg.tmp");
        DestIsVolatile = false;
      }
      Builder.CreateStore(CI, DestPtr, DestIsVolatile);
      return RValue::getAggregate(DestPtr);
    }
    return RValue::get(CI);

  case ABIArgInfo::Ignore:
    // If we are ignoring an argument that had a result, make sure to
    // construct the appropriate return value for our caller.
    return GetUndefRValue(RetTy);

  case ABIArgInfo::Coerce: {
    llvm::Value *DestPtr = ReturnValue.getValue();
    bool DestIsVolatile = ReturnValue.isVolatile();
    
    if (!DestPtr) {
      DestPtr = CreateMemTemp(RetTy, "coerce");
      DestIsVolatile = false;
    }
    
    CreateCoercedStore(CI, DestPtr, DestIsVolatile, *this);
    if (RetTy->isAnyComplexType())
      return RValue::getComplex(LoadComplexFromAddr(DestPtr, false));
    if (CodeGenFunction::hasAggregateLLVMType(RetTy))
      return RValue::getAggregate(DestPtr);
    return RValue::get(EmitLoadOfScalar(DestPtr, false, RetTy));
  }

  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");
  }

  assert(0 && "Unhandled ABIArgInfo::Kind");
  return RValue::get(0);
}

/* VarArg handling */

llvm::Value *CodeGenFunction::EmitVAArg(llvm::Value *VAListAddr, QualType Ty) {
  return CGM.getTypes().getABIInfo().EmitVAArg(VAListAddr, Ty, *this);
}
