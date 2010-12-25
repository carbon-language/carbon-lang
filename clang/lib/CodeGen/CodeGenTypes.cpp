//===--- CodeGenTypes.cpp - Type translation for LLVM CodeGen -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> LLVM type lowering.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTypes.h"
#include "CGCall.h"
#include "CGCXXABI.h"
#include "CGRecordLayout.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

CodeGenTypes::CodeGenTypes(ASTContext &Ctx, llvm::Module& M,
                           const llvm::TargetData &TD, const ABIInfo &Info,
                           CGCXXABI &CXXABI)
  : Context(Ctx), Target(Ctx.Target), TheModule(M), TheTargetData(TD),
    TheABIInfo(Info), TheCXXABI(CXXABI) {
}

CodeGenTypes::~CodeGenTypes() {
  for (llvm::DenseMap<const Type *, CGRecordLayout *>::iterator
         I = CGRecordLayouts.begin(), E = CGRecordLayouts.end();
      I != E; ++I)
    delete I->second;

  for (llvm::FoldingSet<CGFunctionInfo>::iterator
       I = FunctionInfos.begin(), E = FunctionInfos.end(); I != E; )
    delete &*I++;
}

/// HandleLateResolvedPointers - For top-level ConvertType calls, this handles
/// pointers that are referenced but have not been converted yet.  This is used
/// to handle cyclic structures properly.
void CodeGenTypes::HandleLateResolvedPointers() {
  assert(!PointersToResolve.empty() && "No pointers to resolve!");
  
  // Any pointers that were converted deferred evaluation of their pointee type,
  // creating an opaque type instead.  This is in order to avoid problems with
  // circular types.  Loop through all these defered pointees, if any, and
  // resolve them now.
  while (!PointersToResolve.empty()) {
    std::pair<QualType, llvm::OpaqueType*> P = PointersToResolve.pop_back_val();
    
    // We can handle bare pointers here because we know that the only pointers
    // to the Opaque type are P.second and from other types.  Refining the
    // opqaue type away will invalidate P.second, but we don't mind :).
    const llvm::Type *NT = ConvertTypeForMemRecursive(P.first);
    P.second->refineAbstractTypeTo(NT);
  }
}


/// ConvertType - Convert the specified type to its LLVM form.
const llvm::Type *CodeGenTypes::ConvertType(QualType T, bool IsRecursive) {
  const llvm::Type *Result = ConvertTypeRecursive(T);
  
  // If this is a top-level call to ConvertType and sub-conversions caused
  // pointers to get lazily built as opaque types, resolve the pointers, which
  // might cause Result to be merged away.
  if (!IsRecursive && !PointersToResolve.empty()) {
    llvm::PATypeHolder ResultHandle = Result;
    HandleLateResolvedPointers();
    Result = ResultHandle;
  }
  return Result;
}

const llvm::Type *CodeGenTypes::ConvertTypeRecursive(QualType T) {
  T = Context.getCanonicalType(T);

  // See if type is already cached.
  llvm::DenseMap<Type *, llvm::PATypeHolder>::iterator
    I = TypeCache.find(T.getTypePtr());
  // If type is found in map and this is not a definition for a opaque
  // place holder type then use it. Otherwise, convert type T.
  if (I != TypeCache.end())
    return I->second.get();

  const llvm::Type *ResultType = ConvertNewType(T);
  TypeCache.insert(std::make_pair(T.getTypePtr(),
                                  llvm::PATypeHolder(ResultType)));
  return ResultType;
}

/// ConvertTypeForMem - Convert type T into a llvm::Type.  This differs from
/// ConvertType in that it is used to convert to the memory representation for
/// a type.  For example, the scalar representation for _Bool is i1, but the
/// memory representation is usually i8 or i32, depending on the target.
const llvm::Type *CodeGenTypes::ConvertTypeForMem(QualType T, bool IsRecursive){
  const llvm::Type *R = ConvertType(T, IsRecursive);

  // If this is a non-bool type, don't map it.
  if (!R->isIntegerTy(1))
    return R;

  // Otherwise, return an integer of the target-specified size.
  return llvm::IntegerType::get(getLLVMContext(),
                                (unsigned)Context.getTypeSize(T));

}

// Code to verify a given function type is complete, i.e. the return type
// and all of the argument types are complete.
const TagType *CodeGenTypes::VerifyFuncTypeComplete(const Type* T) {
  const FunctionType *FT = cast<FunctionType>(T);
  if (const TagType* TT = FT->getResultType()->getAs<TagType>())
    if (!TT->getDecl()->isDefinition())
      return TT;
  if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(T))
    for (unsigned i = 0; i < FPT->getNumArgs(); i++)
      if (const TagType* TT = FPT->getArgType(i)->getAs<TagType>())
        if (!TT->getDecl()->isDefinition())
          return TT;
  return 0;
}

/// UpdateCompletedType - When we find the full definition for a TagDecl,
/// replace the 'opaque' type we previously made for it if applicable.
void CodeGenTypes::UpdateCompletedType(const TagDecl *TD) {
  const Type *Key = Context.getTagDeclType(TD).getTypePtr();
  llvm::DenseMap<const Type*, llvm::PATypeHolder>::iterator TDTI =
    TagDeclTypes.find(Key);
  if (TDTI == TagDeclTypes.end()) return;

  // Remember the opaque LLVM type for this tagdecl.
  llvm::PATypeHolder OpaqueHolder = TDTI->second;
  assert(isa<llvm::OpaqueType>(OpaqueHolder.get()) &&
         "Updating compilation of an already non-opaque type?");

  // Remove it from TagDeclTypes so that it will be regenerated.
  TagDeclTypes.erase(TDTI);

  // Generate the new type.
  const llvm::Type *NT = ConvertTagDeclType(TD);

  // Refine the old opaque type to its new definition.
  cast<llvm::OpaqueType>(OpaqueHolder.get())->refineAbstractTypeTo(NT);

  // Since we just completed a tag type, check to see if any function types
  // were completed along with the tag type.
  // FIXME: This is very inefficient; if we track which function types depend
  // on which tag types, though, it should be reasonably efficient.
  llvm::DenseMap<const Type*, llvm::PATypeHolder>::iterator i;
  for (i = FunctionTypes.begin(); i != FunctionTypes.end(); ++i) {
    if (const TagType* TT = VerifyFuncTypeComplete(i->first)) {
      // This function type still depends on an incomplete tag type; make sure
      // that tag type has an associated opaque type.
      ConvertTagDeclType(TT->getDecl());
    } else {
      // This function no longer depends on an incomplete tag type; create the
      // function type, and refine the opaque type to the new function type.
      llvm::PATypeHolder OpaqueHolder = i->second;
      const llvm::Type *NFT = ConvertNewType(QualType(i->first, 0));
      cast<llvm::OpaqueType>(OpaqueHolder.get())->refineAbstractTypeTo(NFT);
      FunctionTypes.erase(i);
    }
  }
}

static const llvm::Type* getTypeForFormat(llvm::LLVMContext &VMContext,
                                          const llvm::fltSemantics &format) {
  if (&format == &llvm::APFloat::IEEEsingle)
    return llvm::Type::getFloatTy(VMContext);
  if (&format == &llvm::APFloat::IEEEdouble)
    return llvm::Type::getDoubleTy(VMContext);
  if (&format == &llvm::APFloat::IEEEquad)
    return llvm::Type::getFP128Ty(VMContext);
  if (&format == &llvm::APFloat::PPCDoubleDouble)
    return llvm::Type::getPPC_FP128Ty(VMContext);
  if (&format == &llvm::APFloat::x87DoubleExtended)
    return llvm::Type::getX86_FP80Ty(VMContext);
  assert(0 && "Unknown float format!");
  return 0;
}

const llvm::Type *CodeGenTypes::ConvertNewType(QualType T) {
  const clang::Type &Ty = *Context.getCanonicalType(T).getTypePtr();

  switch (Ty.getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Non-canonical or dependent types aren't possible.");
    break;

  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty).getKind()) {
    case BuiltinType::Void:
    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      // LLVM void type can only be used as the result of a function call.  Just
      // map to the same as char.
      return llvm::Type::getInt8Ty(getLLVMContext());

    case BuiltinType::Bool:
      // Note that we always return bool as i1 for use as a scalar type.
      return llvm::Type::getInt1Ty(getLLVMContext());

    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
    case BuiltinType::Char16:
    case BuiltinType::Char32:
      return llvm::IntegerType::get(getLLVMContext(),
        static_cast<unsigned>(Context.getTypeSize(T)));

    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble:
      return getTypeForFormat(getLLVMContext(),
                              Context.getFloatTypeSemantics(T));

    case BuiltinType::NullPtr: {
      // Model std::nullptr_t as i8*
      const llvm::Type *Ty = llvm::Type::getInt8Ty(getLLVMContext());
      return llvm::PointerType::getUnqual(Ty);
    }
        
    case BuiltinType::UInt128:
    case BuiltinType::Int128:
      return llvm::IntegerType::get(getLLVMContext(), 128);
    
    case BuiltinType::Overload:
    case BuiltinType::Dependent:
    case BuiltinType::UndeducedAuto:
      assert(0 && "Unexpected builtin type!");
      break;
    }
    assert(0 && "Unknown builtin type!");
    break;
  }
  case Type::Complex: {
    const llvm::Type *EltTy =
      ConvertTypeRecursive(cast<ComplexType>(Ty).getElementType());
    return llvm::StructType::get(TheModule.getContext(), EltTy, EltTy, NULL);
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType &RTy = cast<ReferenceType>(Ty);
    QualType ETy = RTy.getPointeeType();
    llvm::OpaqueType *PointeeType = llvm::OpaqueType::get(getLLVMContext());
    PointersToResolve.push_back(std::make_pair(ETy, PointeeType));
    return llvm::PointerType::get(PointeeType, ETy.getAddressSpace());
  }
  case Type::Pointer: {
    const PointerType &PTy = cast<PointerType>(Ty);
    QualType ETy = PTy.getPointeeType();
    llvm::OpaqueType *PointeeType = llvm::OpaqueType::get(getLLVMContext());
    PointersToResolve.push_back(std::make_pair(ETy, PointeeType));
    return llvm::PointerType::get(PointeeType, ETy.getAddressSpace());
  }

  case Type::VariableArray: {
    const VariableArrayType &A = cast<VariableArrayType>(Ty);
    assert(A.getIndexTypeCVRQualifiers() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // VLAs resolve to the innermost element type; this matches
    // the return of alloca, and there isn't any obviously better choice.
    return ConvertTypeForMemRecursive(A.getElementType());
  }
  case Type::IncompleteArray: {
    const IncompleteArrayType &A = cast<IncompleteArrayType>(Ty);
    assert(A.getIndexTypeCVRQualifiers() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // int X[] -> [0 x int]
    return llvm::ArrayType::get(ConvertTypeForMemRecursive(A.getElementType()),
                                0);
  }
  case Type::ConstantArray: {
    const ConstantArrayType &A = cast<ConstantArrayType>(Ty);
    const llvm::Type *EltTy = ConvertTypeForMemRecursive(A.getElementType());
    return llvm::ArrayType::get(EltTy, A.getSize().getZExtValue());
  }
  case Type::ExtVector:
  case Type::Vector: {
    const VectorType &VT = cast<VectorType>(Ty);
    return llvm::VectorType::get(ConvertTypeRecursive(VT.getElementType()),
                                 VT.getNumElements());
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto: {
    // First, check whether we can build the full function type.  If the
    // function type depends on an incomplete type (e.g. a struct or enum), we
    // cannot lower the function type.  Instead, turn it into an Opaque pointer
    // and have UpdateCompletedType revisit the function type when/if the opaque
    // argument type is defined.
    if (const TagType *TT = VerifyFuncTypeComplete(&Ty)) {
      // This function's type depends on an incomplete tag type; make sure
      // we have an opaque type corresponding to the tag type.
      ConvertTagDeclType(TT->getDecl());
      // Create an opaque type for this function type, save it, and return it.
      llvm::Type *ResultType = llvm::OpaqueType::get(getLLVMContext());
      FunctionTypes.insert(std::make_pair(&Ty, ResultType));
      return ResultType;
    }
    
    // The function type can be built; call the appropriate routines to
    // build it.
    const CGFunctionInfo *FI;
    bool isVariadic;
    if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(&Ty)) {
      FI = &getFunctionInfo(
                   CanQual<FunctionProtoType>::CreateUnsafe(QualType(FPT, 0)),
                            true /*Recursive*/);
      isVariadic = FPT->isVariadic();
    } else {
      const FunctionNoProtoType *FNPT = cast<FunctionNoProtoType>(&Ty);
      FI = &getFunctionInfo(
                CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(FNPT, 0)),
                            true /*Recursive*/);
      isVariadic = true;
    }

    return GetFunctionType(*FI, isVariadic, true);
  }

  case Type::ObjCObject:
    return ConvertTypeRecursive(cast<ObjCObjectType>(Ty).getBaseType());

  case Type::ObjCInterface: {
    // Objective-C interfaces are always opaque (outside of the
    // runtime, which can do whatever it likes); we never refine
    // these.
    const llvm::Type *&T = InterfaceTypes[cast<ObjCInterfaceType>(&Ty)];
    if (!T)
        T = llvm::OpaqueType::get(getLLVMContext());
    return T;
  }

  case Type::ObjCObjectPointer: {
    // Protocol qualifications do not influence the LLVM type, we just return a
    // pointer to the underlying interface type. We don't need to worry about
    // recursive conversion.
    const llvm::Type *T =
      ConvertTypeRecursive(cast<ObjCObjectPointerType>(Ty).getPointeeType());
    return llvm::PointerType::getUnqual(T);
  }

  case Type::Record:
  case Type::Enum: {
    const TagDecl *TD = cast<TagType>(Ty).getDecl();
    const llvm::Type *Res = ConvertTagDeclType(TD);

    llvm::SmallString<256> TypeName;
    llvm::raw_svector_ostream OS(TypeName);
    OS << TD->getKindName() << '.';

    // Name the codegen type after the typedef name
    // if there is no tag type name available
    if (TD->getIdentifier()) {
      // FIXME: We should not have to check for a null decl context here.
      // Right now we do it because the implicit Obj-C decls don't have one.
      if (TD->getDeclContext())
        OS << TD->getQualifiedNameAsString();
      else
        TD->printName(OS);
    } else if (const TypedefDecl *TDD = TD->getTypedefForAnonDecl()) {
      // FIXME: We should not have to check for a null decl context here.
      // Right now we do it because the implicit Obj-C decls don't have one.
      if (TDD->getDeclContext())
        OS << TDD->getQualifiedNameAsString();
      else
        TDD->printName(OS);
    } else
      OS << "anon";

    TheModule.addTypeName(OS.str(), Res);
    return Res;
  }

  case Type::BlockPointer: {
    const QualType FTy = cast<BlockPointerType>(Ty).getPointeeType();
    llvm::OpaqueType *PointeeType = llvm::OpaqueType::get(getLLVMContext());
    PointersToResolve.push_back(std::make_pair(FTy, PointeeType));
    return llvm::PointerType::get(PointeeType, FTy.getAddressSpace());
  }

  case Type::MemberPointer: {
    return getCXXABI().ConvertMemberPointerType(cast<MemberPointerType>(&Ty));
  }
  }

  // FIXME: implement.
  return llvm::OpaqueType::get(getLLVMContext());
}

/// ConvertTagDeclType - Lay out a tagged decl type like struct or union or
/// enum.
const llvm::Type *CodeGenTypes::ConvertTagDeclType(const TagDecl *TD) {
  // TagDecl's are not necessarily unique, instead use the (clang)
  // type connected to the decl.
  const Type *Key =
    Context.getTagDeclType(TD).getTypePtr();
  llvm::DenseMap<const Type*, llvm::PATypeHolder>::iterator TDTI =
    TagDeclTypes.find(Key);

  // If we've already compiled this tag type, use the previous definition.
  if (TDTI != TagDeclTypes.end())
    return TDTI->second;

  const EnumDecl *ED = dyn_cast<EnumDecl>(TD);

  // If this is still a forward declaration, just define an opaque
  // type to use for this tagged decl.
  // C++0x: If this is a enumeration type with fixed underlying type,
  // consider it complete.
  if (!TD->isDefinition() && !(ED && ED->isFixed())) {
    llvm::Type *ResultType = llvm::OpaqueType::get(getLLVMContext());
    TagDeclTypes.insert(std::make_pair(Key, ResultType));
    return ResultType;
  }

  // Okay, this is a definition of a type.  Compile the implementation now.

  if (ED)  // Don't bother storing enums in TagDeclTypes.
    return ConvertTypeRecursive(ED->getIntegerType());

  // This decl could well be recursive.  In this case, insert an opaque
  // definition of this type, which the recursive uses will get.  We will then
  // refine this opaque version later.

  // Create new OpaqueType now for later use in case this is a recursive
  // type.  This will later be refined to the actual type.
  llvm::PATypeHolder ResultHolder = llvm::OpaqueType::get(getLLVMContext());
  TagDeclTypes.insert(std::make_pair(Key, ResultHolder));

  const RecordDecl *RD = cast<const RecordDecl>(TD);

  // Force conversion of non-virtual base classes recursively.
  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(TD)) {    
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
      if (!i->isVirtual()) {
        const CXXRecordDecl *Base =
          cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
        ConvertTagDeclType(Base);
      }
    }
  }

  // Layout fields.
  CGRecordLayout *Layout = ComputeRecordLayout(RD);

  CGRecordLayouts[Key] = Layout;
  const llvm::Type *ResultType = Layout->getLLVMType();

  // Refine our Opaque type to ResultType.  This can invalidate ResultType, so
  // make sure to read the result out of the holder.
  cast<llvm::OpaqueType>(ResultHolder.get())
    ->refineAbstractTypeTo(ResultType);

  return ResultHolder.get();
}

/// getCGRecordLayout - Return record layout info for the given record decl.
const CGRecordLayout &
CodeGenTypes::getCGRecordLayout(const RecordDecl *RD) {
  const Type *Key = Context.getTagDeclType(RD).getTypePtr();

  const CGRecordLayout *Layout = CGRecordLayouts.lookup(Key);
  if (!Layout) {
    // Compute the type information.
    ConvertTagDeclType(RD);

    // Now try again.
    Layout = CGRecordLayouts.lookup(Key);
  }

  assert(Layout && "Unable to find record layout information for type");
  return *Layout;
}

bool CodeGenTypes::isZeroInitializable(QualType T) {
  // No need to check for member pointers when not compiling C++.
  if (!Context.getLangOptions().CPlusPlus)
    return true;
  
  T = Context.getBaseElementType(T);
  
  // Records are non-zero-initializable if they contain any
  // non-zero-initializable subobjects.
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    return isZeroInitializable(RD);
  }

  // We have to ask the ABI about member pointers.
  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>())
    return getCXXABI().isZeroInitializable(MPT);
  
  // Everything else is okay.
  return true;
}

bool CodeGenTypes::isZeroInitializable(const CXXRecordDecl *RD) {
  return getCGRecordLayout(RD).isZeroInitializable();
}
