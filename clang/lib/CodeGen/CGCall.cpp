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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Attributes.h"
using namespace clang;
using namespace CodeGen;

/***/

// FIXME: Use iterator and sidestep silly type array creation.

CGFunctionInfo::CGFunctionInfo(const FunctionTypeNoProto *FTNP)
  : IsVariadic(true)
{
  ArgTypes.push_back(FTNP->getResultType());
}

CGFunctionInfo::CGFunctionInfo(const FunctionTypeProto *FTP)
  : IsVariadic(FTP->isVariadic())
{
  ArgTypes.push_back(FTP->getResultType());
  for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
    ArgTypes.push_back(FTP->getArgType(i));
}

// FIXME: Is there really any reason to have this still?
CGFunctionInfo::CGFunctionInfo(const FunctionDecl *FD)
{
  const FunctionType *FTy = FD->getType()->getAsFunctionType();
  const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(FTy);

  ArgTypes.push_back(FTy->getResultType());
  if (FTP) {
    IsVariadic = FTP->isVariadic();
    for (unsigned i = 0, e = FTP->getNumArgs(); i != e; ++i)
      ArgTypes.push_back(FTP->getArgType(i));
  } else {
    IsVariadic = true;
  }
}

CGFunctionInfo::CGFunctionInfo(const ObjCMethodDecl *MD,
                               const ASTContext &Context)
  : IsVariadic(MD->isVariadic())
{
  ArgTypes.push_back(MD->getResultType());
  ArgTypes.push_back(MD->getSelfDecl()->getType());
  ArgTypes.push_back(Context.getObjCSelType());
  for (ObjCMethodDecl::param_const_iterator i = MD->param_begin(),
         e = MD->param_end(); i != e; ++i)
    ArgTypes.push_back((*i)->getType());
}

ArgTypeIterator CGFunctionInfo::argtypes_begin() const {
  return ArgTypes.begin();
}

ArgTypeIterator CGFunctionInfo::argtypes_end() const {
  return ArgTypes.end();
}

/***/

CGCallInfo::CGCallInfo(QualType _ResultType, const CallArgList &_Args) {
  ArgTypes.push_back(_ResultType);
  for (CallArgList::const_iterator i = _Args.begin(), e = _Args.end(); i!=e; ++i)
    ArgTypes.push_back(i->second);
}

ArgTypeIterator CGCallInfo::argtypes_begin() const {
  return ArgTypes.begin();
}

ArgTypeIterator CGCallInfo::argtypes_end() const {
  return ArgTypes.end();
}

/***/

/// ABIArgInfo - Helper class to encapsulate information about how a
/// specific C type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind {
    Default,
    StructRet, /// Only valid for aggregate return types.

    Coerce,    /// Only valid for aggregate return types, the argument
               /// should be accessed by coercion to a provided type.

    ByVal,     /// Only valid for aggregate argument types. The
               /// structure should be passed "byval" with the
               /// specified alignment (0 indicates default
               /// alignment).

    Expand,    /// Only valid for aggregate argument types. The
               /// structure should be expanded into consecutive
               /// arguments for its constituent fields. Currently
               /// expand is only allowed on structures whose fields
               /// are all scalar types or are themselves expandable
               /// types.

    KindFirst=Default, KindLast=Expand
  };

private:
  Kind TheKind;
  const llvm::Type *TypeData;
  unsigned UIntData;

  ABIArgInfo(Kind K, const llvm::Type *TD=0,
             unsigned UI=0) : TheKind(K),
                              TypeData(TD),
                              UIntData(0) {}
public:
  static ABIArgInfo getDefault() { 
    return ABIArgInfo(Default); 
  }
  static ABIArgInfo getStructRet() { 
    return ABIArgInfo(StructRet); 
  }
  static ABIArgInfo getCoerce(const llvm::Type *T) { 
    assert(T->isSingleValueType() && "Can only coerce to simple types");
    return ABIArgInfo(Coerce, T);
  }
  static ABIArgInfo getByVal(unsigned Alignment) {
    return ABIArgInfo(ByVal, 0, Alignment);
  }
  static ABIArgInfo getExpand() {
    return ABIArgInfo(Expand);
  }

  Kind getKind() const { return TheKind; }
  bool isDefault() const { return TheKind == Default; }
  bool isStructRet() const { return TheKind == StructRet; }
  bool isCoerce() const { return TheKind == Coerce; }
  bool isByVal() const { return TheKind == ByVal; }
  bool isExpand() const { return TheKind == Expand; }

  // Coerce accessors
  const llvm::Type *getCoerceToType() const {
    assert(TheKind == Coerce && "Invalid kind!");
    return TypeData;
  }

  // ByVal accessors
  unsigned getByValAlignment() const {
    assert(TheKind == ByVal && "Invalid kind!");
    return UIntData;
  }
};

/***/

/// isEmptyStruct - Return true iff a structure has no non-empty
/// members. Note that a structure with a flexible array member is not
/// considered empty.
static bool isEmptyStruct(QualType T) {
  const RecordType *RT = T->getAsStructureType();
  if (!RT)
    return 0;
  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return false;
  for (RecordDecl::field_const_iterator i = RD->field_begin(), 
         e = RD->field_end(); i != e; ++i) {
    const FieldDecl *FD = *i;
    if (!isEmptyStruct(FD->getType()))
      return false;
  }
  return true;
}

/// isSingleElementStruct - Determine if a structure is a "single
/// element struct", i.e. it has exactly one non-empty field or
/// exactly one field which is itself a single element
/// struct. Structures with flexible array members are never
/// considered single element structs.
///
/// \return The field declaration for the single non-empty field, if
/// it exists.
static const FieldDecl *isSingleElementStruct(QualType T) {
  const RecordType *RT = T->getAsStructureType();
  if (!RT)
    return 0;

  const RecordDecl *RD = RT->getDecl();
  if (RD->hasFlexibleArrayMember())
    return 0;

  const FieldDecl *Found = 0;
  for (RecordDecl::field_const_iterator i = RD->field_begin(), 
         e = RD->field_end(); i != e; ++i) {
    const FieldDecl *FD = *i;
    QualType FT = FD->getType();

    if (isEmptyStruct(FT)) {
      // Ignore
    } else if (Found) {
      return 0;
    } else if (!CodeGenFunction::hasAggregateLLVMType(FT)) {
      Found = FD;
    } else {
      Found = isSingleElementStruct(FT);
      if (!Found)
        return 0;
    }
  }

  return Found;
}

static bool is32Or64BitBasicType(QualType Ty, ASTContext &Context) {
  if (!Ty->getAsBuiltinType() && !Ty->isPointerType())
    return false;

  uint64_t Size = Context.getTypeSize(Ty);
  return Size == 32 || Size == 64;
}

static bool areAllFields32Or64BitBasicType(const RecordDecl *RD,
                                           ASTContext &Context) {
  for (RecordDecl::field_const_iterator i = RD->field_begin(), 
         e = RD->field_end(); i != e; ++i) {
    const FieldDecl *FD = *i;

    if (!is32Or64BitBasicType(FD->getType(), Context))
      return false;
    
    // If this is a bit-field we need to make sure it is still a
    // 32-bit or 64-bit type.
    if (Expr *BW = FD->getBitWidth()) {
      unsigned Width = BW->getIntegerConstantExprValue(Context).getZExtValue();
      if (Width <= 16)
        return false;
    }
  }
  return true;
}

static ABIArgInfo classifyReturnType(QualType RetTy,
                                     ASTContext &Context) {
  assert(!RetTy->isArrayType() && 
         "Array types cannot be passed directly.");
  if (CodeGenFunction::hasAggregateLLVMType(RetTy)) {
    // Classify "single element" structs as their element type.
    const FieldDecl *SeltFD = isSingleElementStruct(RetTy);
    if (SeltFD) {
      QualType SeltTy = SeltFD->getType()->getDesugaredType();
      if (const BuiltinType *BT = SeltTy->getAsBuiltinType()) {
        // FIXME: This is gross, it would be nice if we could just
        // pass back SeltTy and have clients deal with it. Is it worth
        // supporting coerce to both LLVM and clang Types?
        if (BT->isIntegerType()) {
          uint64_t Size = Context.getTypeSize(SeltTy);
          return ABIArgInfo::getCoerce(llvm::IntegerType::get((unsigned) Size));
        } else if (BT->getKind() == BuiltinType::Float) {
          return ABIArgInfo::getCoerce(llvm::Type::FloatTy);
        } else if (BT->getKind() == BuiltinType::Double) {
          return ABIArgInfo::getCoerce(llvm::Type::DoubleTy);
        }
      } else if (SeltTy->isPointerType()) {
        // FIXME: It would be really nice if this could come out as
        // the proper pointer type.
        llvm::Type *PtrTy = 
          llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
        return ABIArgInfo::getCoerce(PtrTy);
      }
    }

    uint64_t Size = Context.getTypeSize(RetTy);
    if (Size == 8) {
      return ABIArgInfo::getCoerce(llvm::Type::Int8Ty);
    } else if (Size == 16) {
      return ABIArgInfo::getCoerce(llvm::Type::Int16Ty);
    } else if (Size == 32) {
      return ABIArgInfo::getCoerce(llvm::Type::Int32Ty);
    } else if (Size == 64) {
      return ABIArgInfo::getCoerce(llvm::Type::Int64Ty);
    } else {
      return ABIArgInfo::getStructRet();
    }
  } else {
    return ABIArgInfo::getDefault();
  }
}

static ABIArgInfo classifyArgumentType(QualType Ty,
                                       ASTContext &Context) {
  assert(!Ty->isArrayType() && "Array types cannot be passed directly.");
  if (CodeGenFunction::hasAggregateLLVMType(Ty)) {
    // Structures with flexible arrays are always byval.
    if (const RecordType *RT = Ty->getAsStructureType())
      if (RT->getDecl()->hasFlexibleArrayMember())
        return ABIArgInfo::getByVal(0);

    // Expand empty structs (i.e. ignore)
    uint64_t Size = Context.getTypeSize(Ty);
    if (Ty->isStructureType() && Size == 0)
      return ABIArgInfo::getExpand();

    // Expand structs with size <= 128-bits which consist only of
    // basic types (int, long long, float, double, xxx*). This is
    // non-recursive and does not ignore empty fields.
    if (const RecordType *RT = Ty->getAsStructureType()) {
      if (Context.getTypeSize(Ty) <= 4*32 &&
          areAllFields32Or64BitBasicType(RT->getDecl(), Context))
        return ABIArgInfo::getExpand();
    }

    return ABIArgInfo::getByVal(0);
  } else {
    return ABIArgInfo::getDefault();
  }
}

static ABIArgInfo getABIReturnInfo(QualType Ty,
                                   ASTContext &Context) {
  ABIArgInfo Info = classifyReturnType(Ty, Context);
  // Ensure default on aggregate types is StructRet.
  if (Info.isDefault() && CodeGenFunction::hasAggregateLLVMType(Ty))
    return ABIArgInfo::getStructRet();
  return Info;
}

static ABIArgInfo getABIArgumentInfo(QualType Ty,
                                     ASTContext &Context) {
  ABIArgInfo Info = classifyArgumentType(Ty, Context);
  // Ensure default on aggregate types is ByVal.
  if (Info.isDefault() && CodeGenFunction::hasAggregateLLVMType(Ty))
    return ABIArgInfo::getByVal(0);
  return Info;  
}

/***/

void CodeGenTypes::GetExpandedTypes(QualType Ty, 
                                    std::vector<const llvm::Type*> &ArgTys) {
  const RecordType *RT = Ty->getAsStructureType();
  assert(RT && "Can only expand structure types.");
  const RecordDecl *RD = RT->getDecl();
  assert(!RD->hasFlexibleArrayMember() && 
         "Cannot expand structure with flexible array.");
  
  for (RecordDecl::field_const_iterator i = RD->field_begin(), 
         e = RD->field_end(); i != e; ++i) {
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
  for (RecordDecl::field_iterator i = RD->field_begin(), 
         e = RD->field_end(); i != e; ++i) {
    FieldDecl *FD = *i;    
    QualType FT = FD->getType();

    // FIXME: What are the right qualifiers here?
    LValue LV = EmitLValueForField(Addr, FD, false, 0);
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
  for (RecordDecl::field_iterator i = RD->field_begin(), 
         e = RD->field_end(); i != e; ++i) {
    FieldDecl *FD = *i;    
    QualType FT = FD->getType();
    
    // FIXME: What are the right qualifiers here?
    LValue LV = EmitLValueForField(Addr, FD, false, 0);
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

/***/

const llvm::FunctionType *
CodeGenTypes::GetFunctionType(const CGCallInfo &CI, bool IsVariadic) {
  return GetFunctionType(CI.argtypes_begin(), CI.argtypes_end(), IsVariadic);
}

const llvm::FunctionType *
CodeGenTypes::GetFunctionType(const CGFunctionInfo &FI) {
  return GetFunctionType(FI.argtypes_begin(), FI.argtypes_end(), FI.isVariadic());
}

const llvm::FunctionType *
CodeGenTypes::GetFunctionType(ArgTypeIterator begin, ArgTypeIterator end,
                              bool IsVariadic) {
  std::vector<const llvm::Type*> ArgTys;

  const llvm::Type *ResultType = 0;

  QualType RetTy = *begin;
  ABIArgInfo RetAI = getABIReturnInfo(RetTy, getContext());
  switch (RetAI.getKind()) {
  case ABIArgInfo::ByVal:
  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");

  case ABIArgInfo::Default:
    if (RetTy->isVoidType()) {
      ResultType = llvm::Type::VoidTy;
    } else {
      ResultType = ConvertType(RetTy);
    }
    break;

  case ABIArgInfo::StructRet: {
    ResultType = llvm::Type::VoidTy;
    const llvm::Type *STy = ConvertType(RetTy);
    ArgTys.push_back(llvm::PointerType::get(STy, RetTy.getAddressSpace()));
    break;
  }

  case ABIArgInfo::Coerce:
    ResultType = RetAI.getCoerceToType();
    break;
  }
  
  for (++begin; begin != end; ++begin) {
    ABIArgInfo AI = getABIArgumentInfo(*begin, getContext());
    const llvm::Type *Ty = ConvertType(*begin);
    
    switch (AI.getKind()) {
    case ABIArgInfo::Coerce:
    case ABIArgInfo::StructRet:
      assert(0 && "Invalid ABI kind for non-return argument");
    
    case ABIArgInfo::ByVal:
      // byval arguments are always on the stack, which is addr space #0.
      ArgTys.push_back(llvm::PointerType::getUnqual(Ty));
      assert(AI.getByValAlignment() == 0 && "FIXME: alignment unhandled");
      break;
      
    case ABIArgInfo::Default:
      ArgTys.push_back(Ty);
      break;
     
    case ABIArgInfo::Expand:
      GetExpandedTypes(*begin, ArgTys);
      break;
    }
  }

  return llvm::FunctionType::get(ResultType, ArgTys, IsVariadic);
}

bool CodeGenModule::ReturnTypeUsesSret(QualType RetTy) {
  return getABIReturnInfo(RetTy, getContext()).isStructRet();
}

void CodeGenModule::ConstructAttributeList(const Decl *TargetDecl,
                                           ArgTypeIterator begin,
                                           ArgTypeIterator end,
                                           AttributeListType &PAL) {
  unsigned FuncAttrs = 0;
  unsigned RetAttrs = 0;

  if (TargetDecl) {
    if (TargetDecl->getAttr<NoThrowAttr>())
      FuncAttrs |= llvm::Attribute::NoUnwind;
    if (TargetDecl->getAttr<NoReturnAttr>())
      FuncAttrs |= llvm::Attribute::NoReturn;
  }

  QualType RetTy = *begin;
  unsigned Index = 1;
  ABIArgInfo RetAI = getABIReturnInfo(RetTy, getContext());
  switch (RetAI.getKind()) {
  case ABIArgInfo::Default:
    if (RetTy->isPromotableIntegerType()) {
      if (RetTy->isSignedIntegerType()) {
        RetAttrs |= llvm::Attribute::SExt;
      } else if (RetTy->isUnsignedIntegerType()) {
        RetAttrs |= llvm::Attribute::ZExt;
      }
    }
    break;

  case ABIArgInfo::StructRet:
    PAL.push_back(llvm::AttributeWithIndex::get(Index, 
                                                  llvm::Attribute::StructRet|
                                                  llvm::Attribute::NoAlias));
    ++Index;
    break;

  case ABIArgInfo::Coerce:
    break;

  case ABIArgInfo::ByVal:
  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");    
  }

  if (RetAttrs)
    PAL.push_back(llvm::AttributeWithIndex::get(0, RetAttrs));
  for (++begin; begin != end; ++begin) {
    QualType ParamType = *begin;
    unsigned Attributes = 0;
    ABIArgInfo AI = getABIArgumentInfo(ParamType, getContext());
    
    switch (AI.getKind()) {
    case ABIArgInfo::StructRet:
    case ABIArgInfo::Coerce:
      assert(0 && "Invalid ABI kind for non-return argument");
    
    case ABIArgInfo::ByVal:
      Attributes |= llvm::Attribute::ByVal;
      assert(AI.getByValAlignment() == 0 && "FIXME: alignment unhandled");
      break;
      
    case ABIArgInfo::Default:
      if (ParamType->isPromotableIntegerType()) {
        if (ParamType->isSignedIntegerType()) {
          Attributes |= llvm::Attribute::SExt;
        } else if (ParamType->isUnsignedIntegerType()) {
          Attributes |= llvm::Attribute::ZExt;
        }
      }
      break;
     
    case ABIArgInfo::Expand: {
      std::vector<const llvm::Type*> Tys;  
      // FIXME: This is rather inefficient. Do we ever actually need
      // to do anything here? The result should be just reconstructed
      // on the other side, so extension should be a non-issue.
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

void CodeGenFunction::EmitFunctionProlog(llvm::Function *Fn,
                                         QualType RetTy, 
                                         const FunctionArgList &Args) {
  // Emit allocs for param decls.  Give the LLVM Argument nodes names.
  llvm::Function::arg_iterator AI = Fn->arg_begin();
  
  // Name the struct return argument.
  if (CGM.ReturnTypeUsesSret(RetTy)) {
    AI->setName("agg.result");
    ++AI;
  }
     
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i) {
    const VarDecl *Arg = i->first;
    QualType Ty = i->second;
    ABIArgInfo ArgI = getABIArgumentInfo(Ty, getContext());

    switch (ArgI.getKind()) {
    case ABIArgInfo::ByVal: 
    case ABIArgInfo::Default: {
      assert(AI != Fn->arg_end() && "Argument mismatch!");
      llvm::Value* V = AI;
      if (!getContext().typesAreCompatible(Ty, Arg->getType())) {
        // This must be a promotion, for something like
        // "void a(x) short x; {..."
        V = EmitScalarConversion(V, Ty, Arg->getType());
      }
      EmitParmDecl(*Arg, V);
      break;
    }
      
    case ABIArgInfo::Expand: {
      // If this was structure was expand into multiple arguments then
      // we need to create a temporary and reconstruct it from the
      // arguments.
      std::string Name(Arg->getName());
      llvm::Value *Temp = CreateTempAlloca(ConvertType(Ty), 
                                           (Name + ".addr").c_str());
      // FIXME: What are the right qualifiers here?
      llvm::Function::arg_iterator End = 
        ExpandTypeFromArgs(Ty, LValue::MakeAddr(Temp,0), AI);      
      EmitParmDecl(*Arg, Temp);

      // Name the arguments used in expansion and increment AI.
      unsigned Index = 0;
      for (; AI != End; ++AI, ++Index)
        AI->setName(Name + "." + llvm::utostr(Index));
      continue;
    }
      
    case ABIArgInfo::Coerce:
    case ABIArgInfo::StructRet:
      assert(0 && "Invalid ABI kind for non-return argument");        
    }

    ++AI;
  }
  assert(AI == Fn->arg_end() && "Argument mismatch!");
}

void CodeGenFunction::EmitFunctionEpilog(QualType RetTy, 
                                         llvm::Value *ReturnValue) {
  llvm::Value *RV = 0;

  // Functions with no result always return void.
  if (ReturnValue) { 
    ABIArgInfo RetAI = getABIReturnInfo(RetTy, getContext());
    
    switch (RetAI.getKind()) {
    case ABIArgInfo::StructRet:
      EmitAggregateCopy(CurFn->arg_begin(), ReturnValue, RetTy);
      break;

    case ABIArgInfo::Default:
      RV = Builder.CreateLoad(ReturnValue);
      break;

    case ABIArgInfo::Coerce: {
      const llvm::Type *CoerceToPTy = 
        llvm::PointerType::getUnqual(RetAI.getCoerceToType());
      RV = Builder.CreateLoad(Builder.CreateBitCast(ReturnValue, CoerceToPTy));
      break;
    }

    case ABIArgInfo::ByVal:
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

RValue CodeGenFunction::EmitCall(llvm::Value *Callee, 
                                 QualType RetTy, 
                                 const CallArgList &CallArgs) {
  llvm::SmallVector<llvm::Value*, 16> Args;

  // Handle struct-return functions by passing a pointer to the
  // location that we would like to return into.
  ABIArgInfo RetAI = getABIReturnInfo(RetTy, getContext());
  switch (RetAI.getKind()) {
  case ABIArgInfo::StructRet:
    // Create a temporary alloca to hold the result of the call. :(
    Args.push_back(CreateTempAlloca(ConvertType(RetTy)));
    break;
    
  case ABIArgInfo::Default:
  case ABIArgInfo::Coerce:
    break;

  case ABIArgInfo::ByVal:
  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");    
  }
  
  for (CallArgList::const_iterator I = CallArgs.begin(), E = CallArgs.end(); 
       I != E; ++I) {
    ABIArgInfo ArgInfo = getABIArgumentInfo(I->second, getContext());
    RValue RV = I->first;

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::ByVal: // Default is byval
    case ABIArgInfo::Default:      
      if (RV.isScalar()) {
        Args.push_back(RV.getScalarVal());
      } else if (RV.isComplex()) {
        // Make a temporary alloca to pass the argument.
        Args.push_back(CreateTempAlloca(ConvertType(I->second)));
        StoreComplexToAddr(RV.getComplexVal(), Args.back(), false); 
      } else {
        Args.push_back(RV.getAggregateAddr());
      }
      break;
     
    case ABIArgInfo::StructRet:
    case ABIArgInfo::Coerce:
      assert(0 && "Invalid ABI kind for non-return argument");
      break;

    case ABIArgInfo::Expand:
      ExpandTypeToArgs(I->second, RV, Args);
      break;
    }
  }
  
  llvm::CallInst *CI = Builder.CreateCall(Callee,&Args[0],&Args[0]+Args.size());
  CGCallInfo CallInfo(RetTy, CallArgs);

  // FIXME: Provide TargetDecl so nounwind, noreturn, etc, etc get set.
  CodeGen::AttributeListType AttributeList;
  CGM.ConstructAttributeList(0, 
                             CallInfo.argtypes_begin(), CallInfo.argtypes_end(),
                             AttributeList);
  CI->setAttributes(llvm::AttrListPtr::get(AttributeList.begin(), 
                                         AttributeList.size()));  

  if (const llvm::Function *F = dyn_cast<llvm::Function>(Callee))
    CI->setCallingConv(F->getCallingConv());
  if (CI->getType() != llvm::Type::VoidTy)
    CI->setName("call");

  switch (RetAI.getKind()) {
  case ABIArgInfo::StructRet:
    if (RetTy->isAnyComplexType())
      return RValue::getComplex(LoadComplexFromAddr(Args[0], false));
    else 
      // Struct return.
      return RValue::getAggregate(Args[0]);

  case ABIArgInfo::Default:
    return RValue::get(RetTy->isVoidType() ? 0 : CI);

  case ABIArgInfo::Coerce: {
    const llvm::Type *CoerceToPTy = 
      llvm::PointerType::getUnqual(RetAI.getCoerceToType());
    llvm::Value *V = CreateTempAlloca(ConvertType(RetTy), "tmp");
    Builder.CreateStore(CI, Builder.CreateBitCast(V, CoerceToPTy));
    return RValue::getAggregate(V);
  }

  case ABIArgInfo::ByVal:
  case ABIArgInfo::Expand:
    assert(0 && "Invalid ABI kind for return argument");    
  }

  assert(0 && "Unhandled ABIArgInfo::Kind");
  return RValue::get(0);
}
