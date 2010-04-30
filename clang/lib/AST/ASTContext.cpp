//===--- ASTContext.cpp - Context to hold long-lived AST nodes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "RecordLayoutBuilder.h"

using namespace clang;

enum FloatingRank {
  FloatRank, DoubleRank, LongDoubleRank
};

ASTContext::ASTContext(const LangOptions& LOpts, SourceManager &SM,
                       const TargetInfo &t,
                       IdentifierTable &idents, SelectorTable &sels,
                       Builtin::Context &builtins,
                       bool FreeMem, unsigned size_reserve) :
  GlobalNestedNameSpecifier(0), CFConstantStringTypeDecl(0),
  NSConstantStringTypeDecl(0),
  ObjCFastEnumerationStateTypeDecl(0), FILEDecl(0), jmp_bufDecl(0),
  sigjmp_bufDecl(0), BlockDescriptorType(0), BlockDescriptorExtendedType(0),
  SourceMgr(SM), LangOpts(LOpts), FreeMemory(FreeMem), Target(t),
  Idents(idents), Selectors(sels),
  BuiltinInfo(builtins), ExternalSource(0), PrintingPolicy(LOpts),
  LastSDM(0, 0) {
  ObjCIdRedefinitionType = QualType();
  ObjCClassRedefinitionType = QualType();
  ObjCSelRedefinitionType = QualType();
  if (size_reserve > 0) Types.reserve(size_reserve);
  TUDecl = TranslationUnitDecl::Create(*this);
  InitBuiltinTypes();
}

ASTContext::~ASTContext() {
  // Release the DenseMaps associated with DeclContext objects.
  // FIXME: Is this the ideal solution?
  ReleaseDeclContextMaps();

  // Release all of the memory associated with overridden C++ methods.
  for (llvm::DenseMap<const CXXMethodDecl *, CXXMethodVector>::iterator 
         OM = OverriddenMethods.begin(), OMEnd = OverriddenMethods.end();
       OM != OMEnd; ++OM)
    OM->second.Destroy();
  
  if (FreeMemory) {
    // Deallocate all the types.
    while (!Types.empty()) {
      Types.back()->Destroy(*this);
      Types.pop_back();
    }

    for (llvm::FoldingSet<ExtQuals>::iterator
         I = ExtQualNodes.begin(), E = ExtQualNodes.end(); I != E; ) {
      // Increment in loop to prevent using deallocated memory.
      Deallocate(&*I++);
    }

    for (llvm::DenseMap<const RecordDecl*, const ASTRecordLayout*>::iterator
         I = ASTRecordLayouts.begin(), E = ASTRecordLayouts.end(); I != E; ) {
      // Increment in loop to prevent using deallocated memory.
      if (ASTRecordLayout *R = const_cast<ASTRecordLayout*>((I++)->second))
        R->Destroy(*this);
    }

    for (llvm::DenseMap<const ObjCContainerDecl*,
         const ASTRecordLayout*>::iterator
         I = ObjCLayouts.begin(), E = ObjCLayouts.end(); I != E; ) {
      // Increment in loop to prevent using deallocated memory.
      if (ASTRecordLayout *R = const_cast<ASTRecordLayout*>((I++)->second))
        R->Destroy(*this);
    }
  }

  // Destroy nested-name-specifiers.
  for (llvm::FoldingSet<NestedNameSpecifier>::iterator
         NNS = NestedNameSpecifiers.begin(),
         NNSEnd = NestedNameSpecifiers.end();
       NNS != NNSEnd; ) {
    // Increment in loop to prevent using deallocated memory.
    (*NNS++).Destroy(*this);
  }

  if (GlobalNestedNameSpecifier)
    GlobalNestedNameSpecifier->Destroy(*this);

  TUDecl->Destroy(*this);
}

void
ASTContext::setExternalSource(llvm::OwningPtr<ExternalASTSource> &Source) {
  ExternalSource.reset(Source.take());
}

void ASTContext::PrintStats() const {
  fprintf(stderr, "*** AST Context Stats:\n");
  fprintf(stderr, "  %d types total.\n", (int)Types.size());

  unsigned counts[] = {
#define TYPE(Name, Parent) 0,
#define ABSTRACT_TYPE(Name, Parent)
#include "clang/AST/TypeNodes.def"
    0 // Extra
  };

  for (unsigned i = 0, e = Types.size(); i != e; ++i) {
    Type *T = Types[i];
    counts[(unsigned)T->getTypeClass()]++;
  }

  unsigned Idx = 0;
  unsigned TotalBytes = 0;
#define TYPE(Name, Parent)                                              \
  if (counts[Idx])                                                      \
    fprintf(stderr, "    %d %s types\n", (int)counts[Idx], #Name);      \
  TotalBytes += counts[Idx] * sizeof(Name##Type);                       \
  ++Idx;
#define ABSTRACT_TYPE(Name, Parent)
#include "clang/AST/TypeNodes.def"

  fprintf(stderr, "Total bytes = %d\n", int(TotalBytes));

  if (ExternalSource.get()) {
    fprintf(stderr, "\n");
    ExternalSource->PrintStats();
  }
}


void ASTContext::InitBuiltinType(CanQualType &R, BuiltinType::Kind K) {
  BuiltinType *Ty = new (*this, TypeAlignment) BuiltinType(K);
  R = CanQualType::CreateUnsafe(QualType(Ty, 0));
  Types.push_back(Ty);
}

void ASTContext::InitBuiltinTypes() {
  assert(VoidTy.isNull() && "Context reinitialized?");

  // C99 6.2.5p19.
  InitBuiltinType(VoidTy,              BuiltinType::Void);

  // C99 6.2.5p2.
  InitBuiltinType(BoolTy,              BuiltinType::Bool);
  // C99 6.2.5p3.
  if (LangOpts.CharIsSigned)
    InitBuiltinType(CharTy,            BuiltinType::Char_S);
  else
    InitBuiltinType(CharTy,            BuiltinType::Char_U);
  // C99 6.2.5p4.
  InitBuiltinType(SignedCharTy,        BuiltinType::SChar);
  InitBuiltinType(ShortTy,             BuiltinType::Short);
  InitBuiltinType(IntTy,               BuiltinType::Int);
  InitBuiltinType(LongTy,              BuiltinType::Long);
  InitBuiltinType(LongLongTy,          BuiltinType::LongLong);

  // C99 6.2.5p6.
  InitBuiltinType(UnsignedCharTy,      BuiltinType::UChar);
  InitBuiltinType(UnsignedShortTy,     BuiltinType::UShort);
  InitBuiltinType(UnsignedIntTy,       BuiltinType::UInt);
  InitBuiltinType(UnsignedLongTy,      BuiltinType::ULong);
  InitBuiltinType(UnsignedLongLongTy,  BuiltinType::ULongLong);

  // C99 6.2.5p10.
  InitBuiltinType(FloatTy,             BuiltinType::Float);
  InitBuiltinType(DoubleTy,            BuiltinType::Double);
  InitBuiltinType(LongDoubleTy,        BuiltinType::LongDouble);

  // GNU extension, 128-bit integers.
  InitBuiltinType(Int128Ty,            BuiltinType::Int128);
  InitBuiltinType(UnsignedInt128Ty,    BuiltinType::UInt128);

  if (LangOpts.CPlusPlus) // C++ 3.9.1p5
    InitBuiltinType(WCharTy,           BuiltinType::WChar);
  else // C99
    WCharTy = getFromTargetType(Target.getWCharType());

  if (LangOpts.CPlusPlus) // C++0x 3.9.1p5, extension for C++
    InitBuiltinType(Char16Ty,           BuiltinType::Char16);
  else // C99
    Char16Ty = getFromTargetType(Target.getChar16Type());

  if (LangOpts.CPlusPlus) // C++0x 3.9.1p5, extension for C++
    InitBuiltinType(Char32Ty,           BuiltinType::Char32);
  else // C99
    Char32Ty = getFromTargetType(Target.getChar32Type());

  // Placeholder type for functions.
  InitBuiltinType(OverloadTy,          BuiltinType::Overload);

  // Placeholder type for type-dependent expressions whose type is
  // completely unknown. No code should ever check a type against
  // DependentTy and users should never see it; however, it is here to
  // help diagnose failures to properly check for type-dependent
  // expressions.
  InitBuiltinType(DependentTy,         BuiltinType::Dependent);

  // Placeholder type for C++0x auto declarations whose real type has
  // not yet been deduced.
  InitBuiltinType(UndeducedAutoTy, BuiltinType::UndeducedAuto);

  // C99 6.2.5p11.
  FloatComplexTy      = getComplexType(FloatTy);
  DoubleComplexTy     = getComplexType(DoubleTy);
  LongDoubleComplexTy = getComplexType(LongDoubleTy);

  BuiltinVaListType = QualType();

  // "Builtin" typedefs set by Sema::ActOnTranslationUnitScope().
  ObjCIdTypedefType = QualType();
  ObjCClassTypedefType = QualType();
  ObjCSelTypedefType = QualType();

  // Builtin types for 'id', 'Class', and 'SEL'.
  InitBuiltinType(ObjCBuiltinIdTy, BuiltinType::ObjCId);
  InitBuiltinType(ObjCBuiltinClassTy, BuiltinType::ObjCClass);
  InitBuiltinType(ObjCBuiltinSelTy, BuiltinType::ObjCSel);

  ObjCConstantStringType = QualType();

  // void * type
  VoidPtrTy = getPointerType(VoidTy);

  // nullptr type (C++0x 2.14.7)
  InitBuiltinType(NullPtrTy,           BuiltinType::NullPtr);
}

MemberSpecializationInfo *
ASTContext::getInstantiatedFromStaticDataMember(const VarDecl *Var) {
  assert(Var->isStaticDataMember() && "Not a static data member");
  llvm::DenseMap<const VarDecl *, MemberSpecializationInfo *>::iterator Pos
    = InstantiatedFromStaticDataMember.find(Var);
  if (Pos == InstantiatedFromStaticDataMember.end())
    return 0;

  return Pos->second;
}

void
ASTContext::setInstantiatedFromStaticDataMember(VarDecl *Inst, VarDecl *Tmpl,
                                                TemplateSpecializationKind TSK) {
  assert(Inst->isStaticDataMember() && "Not a static data member");
  assert(Tmpl->isStaticDataMember() && "Not a static data member");
  assert(!InstantiatedFromStaticDataMember[Inst] &&
         "Already noted what static data member was instantiated from");
  InstantiatedFromStaticDataMember[Inst] 
    = new (*this) MemberSpecializationInfo(Tmpl, TSK);
}

NamedDecl *
ASTContext::getInstantiatedFromUsingDecl(UsingDecl *UUD) {
  llvm::DenseMap<UsingDecl *, NamedDecl *>::const_iterator Pos
    = InstantiatedFromUsingDecl.find(UUD);
  if (Pos == InstantiatedFromUsingDecl.end())
    return 0;

  return Pos->second;
}

void
ASTContext::setInstantiatedFromUsingDecl(UsingDecl *Inst, NamedDecl *Pattern) {
  assert((isa<UsingDecl>(Pattern) ||
          isa<UnresolvedUsingValueDecl>(Pattern) ||
          isa<UnresolvedUsingTypenameDecl>(Pattern)) && 
         "pattern decl is not a using decl");
  assert(!InstantiatedFromUsingDecl[Inst] && "pattern already exists");
  InstantiatedFromUsingDecl[Inst] = Pattern;
}

UsingShadowDecl *
ASTContext::getInstantiatedFromUsingShadowDecl(UsingShadowDecl *Inst) {
  llvm::DenseMap<UsingShadowDecl*, UsingShadowDecl*>::const_iterator Pos
    = InstantiatedFromUsingShadowDecl.find(Inst);
  if (Pos == InstantiatedFromUsingShadowDecl.end())
    return 0;

  return Pos->second;
}

void
ASTContext::setInstantiatedFromUsingShadowDecl(UsingShadowDecl *Inst,
                                               UsingShadowDecl *Pattern) {
  assert(!InstantiatedFromUsingShadowDecl[Inst] && "pattern already exists");
  InstantiatedFromUsingShadowDecl[Inst] = Pattern;
}

FieldDecl *ASTContext::getInstantiatedFromUnnamedFieldDecl(FieldDecl *Field) {
  llvm::DenseMap<FieldDecl *, FieldDecl *>::iterator Pos
    = InstantiatedFromUnnamedFieldDecl.find(Field);
  if (Pos == InstantiatedFromUnnamedFieldDecl.end())
    return 0;

  return Pos->second;
}

void ASTContext::setInstantiatedFromUnnamedFieldDecl(FieldDecl *Inst,
                                                     FieldDecl *Tmpl) {
  assert(!Inst->getDeclName() && "Instantiated field decl is not unnamed");
  assert(!Tmpl->getDeclName() && "Template field decl is not unnamed");
  assert(!InstantiatedFromUnnamedFieldDecl[Inst] &&
         "Already noted what unnamed field was instantiated from");

  InstantiatedFromUnnamedFieldDecl[Inst] = Tmpl;
}

ASTContext::overridden_cxx_method_iterator
ASTContext::overridden_methods_begin(const CXXMethodDecl *Method) const {
  llvm::DenseMap<const CXXMethodDecl *, CXXMethodVector>::const_iterator Pos
    = OverriddenMethods.find(Method);
  if (Pos == OverriddenMethods.end())
    return 0;

  return Pos->second.begin();
}

ASTContext::overridden_cxx_method_iterator
ASTContext::overridden_methods_end(const CXXMethodDecl *Method) const {
  llvm::DenseMap<const CXXMethodDecl *, CXXMethodVector>::const_iterator Pos
    = OverriddenMethods.find(Method);
  if (Pos == OverriddenMethods.end())
    return 0;

  return Pos->second.end();
}

void ASTContext::addOverriddenMethod(const CXXMethodDecl *Method, 
                                     const CXXMethodDecl *Overridden) {
  OverriddenMethods[Method].push_back(Overridden);
}

namespace {
  class BeforeInTranslationUnit
    : std::binary_function<SourceRange, SourceRange, bool> {
    SourceManager *SourceMgr;

  public:
    explicit BeforeInTranslationUnit(SourceManager *SM) : SourceMgr(SM) { }

    bool operator()(SourceRange X, SourceRange Y) {
      return SourceMgr->isBeforeInTranslationUnit(X.getBegin(), Y.getBegin());
    }
  };
}

//===----------------------------------------------------------------------===//
//                         Type Sizing and Analysis
//===----------------------------------------------------------------------===//

/// getFloatTypeSemantics - Return the APFloat 'semantics' for the specified
/// scalar floating point type.
const llvm::fltSemantics &ASTContext::getFloatTypeSemantics(QualType T) const {
  const BuiltinType *BT = T->getAs<BuiltinType>();
  assert(BT && "Not a floating point type!");
  switch (BT->getKind()) {
  default: assert(0 && "Not a floating point type!");
  case BuiltinType::Float:      return Target.getFloatFormat();
  case BuiltinType::Double:     return Target.getDoubleFormat();
  case BuiltinType::LongDouble: return Target.getLongDoubleFormat();
  }
}

/// getDeclAlign - Return a conservative estimate of the alignment of the
/// specified decl.  Note that bitfields do not have a valid alignment, so
/// this method will assert on them.
/// If @p RefAsPointee, references are treated like their underlying type
/// (for alignof), else they're treated like pointers (for CodeGen).
CharUnits ASTContext::getDeclAlign(const Decl *D, bool RefAsPointee) {
  unsigned Align = Target.getCharWidth();

  if (const AlignedAttr* AA = D->getAttr<AlignedAttr>())
    Align = std::max(Align, AA->getMaxAlignment());

  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    QualType T = VD->getType();
    if (const ReferenceType* RT = T->getAs<ReferenceType>()) {
      if (RefAsPointee)
        T = RT->getPointeeType();
      else
        T = getPointerType(RT->getPointeeType());
    }
    if (!T->isIncompleteType() && !T->isFunctionType()) {
      // Incomplete or function types default to 1.
      while (isa<VariableArrayType>(T) || isa<IncompleteArrayType>(T))
        T = cast<ArrayType>(T)->getElementType();

      Align = std::max(Align, getPreferredTypeAlign(T.getTypePtr()));
    }
    if (const FieldDecl *FD = dyn_cast<FieldDecl>(VD)) {
      // In the case of a field in a packed struct, we want the minimum
      // of the alignment of the field and the alignment of the struct.
      Align = std::min(Align,
        getPreferredTypeAlign(FD->getParent()->getTypeForDecl()));
    }
  }

  return CharUnits::fromQuantity(Align / Target.getCharWidth());
}

/// getTypeSize - Return the size of the specified type, in bits.  This method
/// does not work on incomplete types.
///
/// FIXME: Pointers into different addr spaces could have different sizes and
/// alignment requirements: getPointerInfo should take an AddrSpace, this
/// should take a QualType, &c.
std::pair<uint64_t, unsigned>
ASTContext::getTypeInfo(const Type *T) {
  uint64_t Width=0;
  unsigned Align=8;
  switch (T->getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Should not see dependent types");
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // GCC extension: alignof(function) = 32 bits
    Width = 0;
    Align = 32;
    break;

  case Type::IncompleteArray:
  case Type::VariableArray:
    Width = 0;
    Align = getTypeAlign(cast<ArrayType>(T)->getElementType());
    break;

  case Type::ConstantArray: {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(T);

    std::pair<uint64_t, unsigned> EltInfo = getTypeInfo(CAT->getElementType());
    Width = EltInfo.first*CAT->getSize().getZExtValue();
    Align = EltInfo.second;
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    const VectorType *VT = cast<VectorType>(T);
    std::pair<uint64_t, unsigned> EltInfo = getTypeInfo(VT->getElementType());
    Width = EltInfo.first*VT->getNumElements();
    Align = Width;
    // If the alignment is not a power of 2, round up to the next power of 2.
    // This happens for non-power-of-2 length vectors.
    if (Align & (Align-1)) {
      Align = llvm::NextPowerOf2(Align);
      Width = llvm::RoundUpToAlignment(Width, Align);
    }
    break;
  }

  case Type::Builtin:
    switch (cast<BuiltinType>(T)->getKind()) {
    default: assert(0 && "Unknown builtin type!");
    case BuiltinType::Void:
      // GCC extension: alignof(void) = 8 bits.
      Width = 0;
      Align = 8;
      break;

    case BuiltinType::Bool:
      Width = Target.getBoolWidth();
      Align = Target.getBoolAlign();
      break;
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
      Width = Target.getCharWidth();
      Align = Target.getCharAlign();
      break;
    case BuiltinType::WChar:
      Width = Target.getWCharWidth();
      Align = Target.getWCharAlign();
      break;
    case BuiltinType::Char16:
      Width = Target.getChar16Width();
      Align = Target.getChar16Align();
      break;
    case BuiltinType::Char32:
      Width = Target.getChar32Width();
      Align = Target.getChar32Align();
      break;
    case BuiltinType::UShort:
    case BuiltinType::Short:
      Width = Target.getShortWidth();
      Align = Target.getShortAlign();
      break;
    case BuiltinType::UInt:
    case BuiltinType::Int:
      Width = Target.getIntWidth();
      Align = Target.getIntAlign();
      break;
    case BuiltinType::ULong:
    case BuiltinType::Long:
      Width = Target.getLongWidth();
      Align = Target.getLongAlign();
      break;
    case BuiltinType::ULongLong:
    case BuiltinType::LongLong:
      Width = Target.getLongLongWidth();
      Align = Target.getLongLongAlign();
      break;
    case BuiltinType::Int128:
    case BuiltinType::UInt128:
      Width = 128;
      Align = 128; // int128_t is 128-bit aligned on all targets.
      break;
    case BuiltinType::Float:
      Width = Target.getFloatWidth();
      Align = Target.getFloatAlign();
      break;
    case BuiltinType::Double:
      Width = Target.getDoubleWidth();
      Align = Target.getDoubleAlign();
      break;
    case BuiltinType::LongDouble:
      Width = Target.getLongDoubleWidth();
      Align = Target.getLongDoubleAlign();
      break;
    case BuiltinType::NullPtr:
      Width = Target.getPointerWidth(0); // C++ 3.9.1p11: sizeof(nullptr_t)
      Align = Target.getPointerAlign(0); //   == sizeof(void*)
      break;
    }
    break;
  case Type::ObjCObjectPointer:
    Width = Target.getPointerWidth(0);
    Align = Target.getPointerAlign(0);
    break;
  case Type::BlockPointer: {
    unsigned AS = cast<BlockPointerType>(T)->getPointeeType().getAddressSpace();
    Width = Target.getPointerWidth(AS);
    Align = Target.getPointerAlign(AS);
    break;
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    // alignof and sizeof should never enter this code path here, so we go
    // the pointer route.
    unsigned AS = cast<ReferenceType>(T)->getPointeeType().getAddressSpace();
    Width = Target.getPointerWidth(AS);
    Align = Target.getPointerAlign(AS);
    break;
  }
  case Type::Pointer: {
    unsigned AS = cast<PointerType>(T)->getPointeeType().getAddressSpace();
    Width = Target.getPointerWidth(AS);
    Align = Target.getPointerAlign(AS);
    break;
  }
  case Type::MemberPointer: {
    QualType Pointee = cast<MemberPointerType>(T)->getPointeeType();
    std::pair<uint64_t, unsigned> PtrDiffInfo =
      getTypeInfo(getPointerDiffType());
    Width = PtrDiffInfo.first;
    if (Pointee->isFunctionType())
      Width *= 2;
    Align = PtrDiffInfo.second;
    break;
  }
  case Type::Complex: {
    // Complex types have the same alignment as their elements, but twice the
    // size.
    std::pair<uint64_t, unsigned> EltInfo =
      getTypeInfo(cast<ComplexType>(T)->getElementType());
    Width = EltInfo.first*2;
    Align = EltInfo.second;
    break;
  }
  case Type::ObjCInterface: {
    const ObjCInterfaceType *ObjCI = cast<ObjCInterfaceType>(T);
    const ASTRecordLayout &Layout = getASTObjCInterfaceLayout(ObjCI->getDecl());
    Width = Layout.getSize();
    Align = Layout.getAlignment();
    break;
  }
  case Type::Record:
  case Type::Enum: {
    const TagType *TT = cast<TagType>(T);

    if (TT->getDecl()->isInvalidDecl()) {
      Width = 1;
      Align = 1;
      break;
    }

    if (const EnumType *ET = dyn_cast<EnumType>(TT))
      return getTypeInfo(ET->getDecl()->getIntegerType());

    const RecordType *RT = cast<RecordType>(TT);
    const ASTRecordLayout &Layout = getASTRecordLayout(RT->getDecl());
    Width = Layout.getSize();
    Align = Layout.getAlignment();
    break;
  }

  case Type::SubstTemplateTypeParm:
    return getTypeInfo(cast<SubstTemplateTypeParmType>(T)->
                       getReplacementType().getTypePtr());

  case Type::Elaborated:
    return getTypeInfo(cast<ElaboratedType>(T)->getUnderlyingType()
                         .getTypePtr());

  case Type::Typedef: {
    const TypedefDecl *Typedef = cast<TypedefType>(T)->getDecl();
    if (const AlignedAttr *Aligned = Typedef->getAttr<AlignedAttr>()) {
      Align = std::max(Aligned->getMaxAlignment(),
                       getTypeAlign(Typedef->getUnderlyingType().getTypePtr()));
      Width = getTypeSize(Typedef->getUnderlyingType().getTypePtr());
    } else
      return getTypeInfo(Typedef->getUnderlyingType().getTypePtr());
    break;
  }

  case Type::TypeOfExpr:
    return getTypeInfo(cast<TypeOfExprType>(T)->getUnderlyingExpr()->getType()
                         .getTypePtr());

  case Type::TypeOf:
    return getTypeInfo(cast<TypeOfType>(T)->getUnderlyingType().getTypePtr());

  case Type::Decltype:
    return getTypeInfo(cast<DecltypeType>(T)->getUnderlyingExpr()->getType()
                        .getTypePtr());

  case Type::QualifiedName:
    return getTypeInfo(cast<QualifiedNameType>(T)->getNamedType().getTypePtr());

  case Type::TemplateSpecialization:
    assert(getCanonicalType(T) != T &&
           "Cannot request the size of a dependent type");
    // FIXME: this is likely to be wrong once we support template
    // aliases, since a template alias could refer to a typedef that
    // has an __aligned__ attribute on it.
    return getTypeInfo(getCanonicalType(T));
  }

  assert(Align && (Align & (Align-1)) == 0 && "Alignment must be power of 2");
  return std::make_pair(Width, Align);
}

/// getTypeSizeInChars - Return the size of the specified type, in characters.
/// This method does not work on incomplete types.
CharUnits ASTContext::getTypeSizeInChars(QualType T) {
  return CharUnits::fromQuantity(getTypeSize(T) / getCharWidth());
}
CharUnits ASTContext::getTypeSizeInChars(const Type *T) {
  return CharUnits::fromQuantity(getTypeSize(T) / getCharWidth());
}

/// getTypeAlignInChars - Return the ABI-specified alignment of a type, in 
/// characters. This method does not work on incomplete types.
CharUnits ASTContext::getTypeAlignInChars(QualType T) {
  return CharUnits::fromQuantity(getTypeAlign(T) / getCharWidth());
}
CharUnits ASTContext::getTypeAlignInChars(const Type *T) {
  return CharUnits::fromQuantity(getTypeAlign(T) / getCharWidth());
}

/// getPreferredTypeAlign - Return the "preferred" alignment of the specified
/// type for the current target in bits.  This can be different than the ABI
/// alignment in cases where it is beneficial for performance to overalign
/// a data type.
unsigned ASTContext::getPreferredTypeAlign(const Type *T) {
  unsigned ABIAlign = getTypeAlign(T);

  // Double and long long should be naturally aligned if possible.
  if (const ComplexType* CT = T->getAs<ComplexType>())
    T = CT->getElementType().getTypePtr();
  if (T->isSpecificBuiltinType(BuiltinType::Double) ||
      T->isSpecificBuiltinType(BuiltinType::LongLong))
    return std::max(ABIAlign, (unsigned)getTypeSize(T));

  return ABIAlign;
}

static void CollectLocalObjCIvars(ASTContext *Ctx,
                                  const ObjCInterfaceDecl *OI,
                                  llvm::SmallVectorImpl<FieldDecl*> &Fields) {
  for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
       E = OI->ivar_end(); I != E; ++I) {
    ObjCIvarDecl *IVDecl = *I;
    if (!IVDecl->isInvalidDecl())
      Fields.push_back(cast<FieldDecl>(IVDecl));
  }
}

void ASTContext::CollectObjCIvars(const ObjCInterfaceDecl *OI,
                             llvm::SmallVectorImpl<FieldDecl*> &Fields) {
  if (const ObjCInterfaceDecl *SuperClass = OI->getSuperClass())
    CollectObjCIvars(SuperClass, Fields);
  CollectLocalObjCIvars(this, OI, Fields);
}

/// ShallowCollectObjCIvars -
/// Collect all ivars, including those synthesized, in the current class.
///
void ASTContext::ShallowCollectObjCIvars(const ObjCInterfaceDecl *OI,
                                 llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars) {
  for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
         E = OI->ivar_end(); I != E; ++I) {
     Ivars.push_back(*I);
  }

  CollectNonClassIvars(OI, Ivars);
}

/// CollectNonClassIvars -
/// This routine collects all other ivars which are not declared in the class.
/// This includes synthesized ivars (via @synthesize) and those in
//  class's @implementation.
///
void ASTContext::CollectNonClassIvars(const ObjCInterfaceDecl *OI,
                                llvm::SmallVectorImpl<ObjCIvarDecl*> &Ivars) {
  // Find ivars declared in class extension.
  if (const ObjCCategoryDecl *CDecl = OI->getClassExtension()) {
    for (ObjCCategoryDecl::ivar_iterator I = CDecl->ivar_begin(),
         E = CDecl->ivar_end(); I != E; ++I) {
      Ivars.push_back(*I);
    }
  }

  // Also add any ivar defined in this class's implementation.  This
  // includes synthesized ivars.
  if (ObjCImplementationDecl *ImplDecl = OI->getImplementation()) {
    for (ObjCImplementationDecl::ivar_iterator I = ImplDecl->ivar_begin(),
         E = ImplDecl->ivar_end(); I != E; ++I)
      Ivars.push_back(*I);
  }
}

/// CollectInheritedProtocols - Collect all protocols in current class and
/// those inherited by it.
void ASTContext::CollectInheritedProtocols(const Decl *CDecl,
                          llvm::SmallPtrSet<ObjCProtocolDecl*, 8> &Protocols) {
  if (const ObjCInterfaceDecl *OI = dyn_cast<ObjCInterfaceDecl>(CDecl)) {
    for (ObjCInterfaceDecl::protocol_iterator P = OI->protocol_begin(),
         PE = OI->protocol_end(); P != PE; ++P) {
      ObjCProtocolDecl *Proto = (*P);
      Protocols.insert(Proto);
      for (ObjCProtocolDecl::protocol_iterator P = Proto->protocol_begin(),
           PE = Proto->protocol_end(); P != PE; ++P) {
        Protocols.insert(*P);
        CollectInheritedProtocols(*P, Protocols);
      }
    }
    
    // Categories of this Interface.
    for (const ObjCCategoryDecl *CDeclChain = OI->getCategoryList(); 
         CDeclChain; CDeclChain = CDeclChain->getNextClassCategory())
      CollectInheritedProtocols(CDeclChain, Protocols);
    if (ObjCInterfaceDecl *SD = OI->getSuperClass())
      while (SD) {
        CollectInheritedProtocols(SD, Protocols);
        SD = SD->getSuperClass();
      }
  } else if (const ObjCCategoryDecl *OC = dyn_cast<ObjCCategoryDecl>(CDecl)) {
    for (ObjCInterfaceDecl::protocol_iterator P = OC->protocol_begin(),
         PE = OC->protocol_end(); P != PE; ++P) {
      ObjCProtocolDecl *Proto = (*P);
      Protocols.insert(Proto);
      for (ObjCProtocolDecl::protocol_iterator P = Proto->protocol_begin(),
           PE = Proto->protocol_end(); P != PE; ++P)
        CollectInheritedProtocols(*P, Protocols);
    }
  } else if (const ObjCProtocolDecl *OP = dyn_cast<ObjCProtocolDecl>(CDecl)) {
    for (ObjCProtocolDecl::protocol_iterator P = OP->protocol_begin(),
         PE = OP->protocol_end(); P != PE; ++P) {
      ObjCProtocolDecl *Proto = (*P);
      Protocols.insert(Proto);
      for (ObjCProtocolDecl::protocol_iterator P = Proto->protocol_begin(),
           PE = Proto->protocol_end(); P != PE; ++P)
        CollectInheritedProtocols(*P, Protocols);
    }
  }
}

unsigned ASTContext::CountNonClassIvars(const ObjCInterfaceDecl *OI) {
  unsigned count = 0;  
  // Count ivars declared in class extension.
  if (const ObjCCategoryDecl *CDecl = OI->getClassExtension())
    count += CDecl->ivar_size();

  // Count ivar defined in this class's implementation.  This
  // includes synthesized ivars.
  if (ObjCImplementationDecl *ImplDecl = OI->getImplementation())
    count += ImplDecl->ivar_size();

  return count;
}

/// \brief Get the implementation of ObjCInterfaceDecl,or NULL if none exists.
ObjCImplementationDecl *ASTContext::getObjCImplementation(ObjCInterfaceDecl *D) {
  llvm::DenseMap<ObjCContainerDecl*, ObjCImplDecl*>::iterator
    I = ObjCImpls.find(D);
  if (I != ObjCImpls.end())
    return cast<ObjCImplementationDecl>(I->second);
  return 0;
}
/// \brief Get the implementation of ObjCCategoryDecl, or NULL if none exists.
ObjCCategoryImplDecl *ASTContext::getObjCImplementation(ObjCCategoryDecl *D) {
  llvm::DenseMap<ObjCContainerDecl*, ObjCImplDecl*>::iterator
    I = ObjCImpls.find(D);
  if (I != ObjCImpls.end())
    return cast<ObjCCategoryImplDecl>(I->second);
  return 0;
}

/// \brief Set the implementation of ObjCInterfaceDecl.
void ASTContext::setObjCImplementation(ObjCInterfaceDecl *IFaceD,
                           ObjCImplementationDecl *ImplD) {
  assert(IFaceD && ImplD && "Passed null params");
  ObjCImpls[IFaceD] = ImplD;
}
/// \brief Set the implementation of ObjCCategoryDecl.
void ASTContext::setObjCImplementation(ObjCCategoryDecl *CatD,
                           ObjCCategoryImplDecl *ImplD) {
  assert(CatD && ImplD && "Passed null params");
  ObjCImpls[CatD] = ImplD;
}

/// \brief Allocate an uninitialized TypeSourceInfo.
///
/// The caller should initialize the memory held by TypeSourceInfo using
/// the TypeLoc wrappers.
///
/// \param T the type that will be the basis for type source info. This type
/// should refer to how the declarator was written in source code, not to
/// what type semantic analysis resolved the declarator to.
TypeSourceInfo *ASTContext::CreateTypeSourceInfo(QualType T,
                                                 unsigned DataSize) {
  if (!DataSize)
    DataSize = TypeLoc::getFullDataSizeForType(T);
  else
    assert(DataSize == TypeLoc::getFullDataSizeForType(T) &&
           "incorrect data size provided to CreateTypeSourceInfo!");

  TypeSourceInfo *TInfo =
    (TypeSourceInfo*)BumpAlloc.Allocate(sizeof(TypeSourceInfo) + DataSize, 8);
  new (TInfo) TypeSourceInfo(T);
  return TInfo;
}

TypeSourceInfo *ASTContext::getTrivialTypeSourceInfo(QualType T,
                                                     SourceLocation L) {
  TypeSourceInfo *DI = CreateTypeSourceInfo(T);
  DI->getTypeLoc().initialize(L);
  return DI;
}

/// getInterfaceLayoutImpl - Get or compute information about the
/// layout of the given interface.
///
/// \param Impl - If given, also include the layout of the interface's
/// implementation. This may differ by including synthesized ivars.
const ASTRecordLayout &
ASTContext::getObjCLayout(const ObjCInterfaceDecl *D,
                          const ObjCImplementationDecl *Impl) {
  assert(!D->isForwardDecl() && "Invalid interface decl!");

  // Look up this layout, if already laid out, return what we have.
  ObjCContainerDecl *Key =
    Impl ? (ObjCContainerDecl*) Impl : (ObjCContainerDecl*) D;
  if (const ASTRecordLayout *Entry = ObjCLayouts[Key])
    return *Entry;

  // Add in synthesized ivar count if laying out an implementation.
  if (Impl) {
    unsigned SynthCount = CountNonClassIvars(D);
    // If there aren't any sythesized ivars then reuse the interface
    // entry. Note we can't cache this because we simply free all
    // entries later; however we shouldn't look up implementations
    // frequently.
    if (SynthCount == 0)
      return getObjCLayout(D, 0);
  }

  const ASTRecordLayout *NewEntry =
    ASTRecordLayoutBuilder::ComputeLayout(*this, D, Impl);
  ObjCLayouts[Key] = NewEntry;

  return *NewEntry;
}

const ASTRecordLayout &
ASTContext::getASTObjCInterfaceLayout(const ObjCInterfaceDecl *D) {
  return getObjCLayout(D, 0);
}

const ASTRecordLayout &
ASTContext::getASTObjCImplementationLayout(const ObjCImplementationDecl *D) {
  return getObjCLayout(D->getClassInterface(), D);
}

/// getASTRecordLayout - Get or compute information about the layout of the
/// specified record (struct/union/class), which indicates its size and field
/// position information.
const ASTRecordLayout &ASTContext::getASTRecordLayout(const RecordDecl *D) {
  D = D->getDefinition();
  assert(D && "Cannot get layout of forward declarations!");

  // Look up this layout, if already laid out, return what we have.
  // Note that we can't save a reference to the entry because this function
  // is recursive.
  const ASTRecordLayout *Entry = ASTRecordLayouts[D];
  if (Entry) return *Entry;

  const ASTRecordLayout *NewEntry =
    ASTRecordLayoutBuilder::ComputeLayout(*this, D);
  ASTRecordLayouts[D] = NewEntry;

  if (getLangOptions().DumpRecordLayouts) {
    llvm::errs() << "\n*** Dumping AST Record Layout\n";
    DumpRecordLayout(D, llvm::errs());
  }

  return *NewEntry;
}

const CXXMethodDecl *ASTContext::getKeyFunction(const CXXRecordDecl *RD) {
  RD = cast<CXXRecordDecl>(RD->getDefinition());
  assert(RD && "Cannot get key function for forward declarations!");
  
  const CXXMethodDecl *&Entry = KeyFunctions[RD];
  if (!Entry) 
    Entry = ASTRecordLayoutBuilder::ComputeKeyFunction(RD);
  else
    assert(Entry == ASTRecordLayoutBuilder::ComputeKeyFunction(RD) &&
           "Key function changed!");
  
  return Entry;
}

//===----------------------------------------------------------------------===//
//                   Type creation/memoization methods
//===----------------------------------------------------------------------===//

QualType ASTContext::getExtQualType(const Type *TypeNode, Qualifiers Quals) {
  unsigned Fast = Quals.getFastQualifiers();
  Quals.removeFastQualifiers();

  // Check if we've already instantiated this type.
  llvm::FoldingSetNodeID ID;
  ExtQuals::Profile(ID, TypeNode, Quals);
  void *InsertPos = 0;
  if (ExtQuals *EQ = ExtQualNodes.FindNodeOrInsertPos(ID, InsertPos)) {
    assert(EQ->getQualifiers() == Quals);
    QualType T = QualType(EQ, Fast);
    return T;
  }

  ExtQuals *New = new (*this, TypeAlignment) ExtQuals(*this, TypeNode, Quals);
  ExtQualNodes.InsertNode(New, InsertPos);
  QualType T = QualType(New, Fast);
  return T;
}

QualType ASTContext::getVolatileType(QualType T) {
  QualType CanT = getCanonicalType(T);
  if (CanT.isVolatileQualified()) return T;

  QualifierCollector Quals;
  const Type *TypeNode = Quals.strip(T);
  Quals.addVolatile();

  return getExtQualType(TypeNode, Quals);
}

QualType ASTContext::getAddrSpaceQualType(QualType T, unsigned AddressSpace) {
  QualType CanT = getCanonicalType(T);
  if (CanT.getAddressSpace() == AddressSpace)
    return T;

  // If we are composing extended qualifiers together, merge together
  // into one ExtQuals node.
  QualifierCollector Quals;
  const Type *TypeNode = Quals.strip(T);

  // If this type already has an address space specified, it cannot get
  // another one.
  assert(!Quals.hasAddressSpace() &&
         "Type cannot be in multiple addr spaces!");
  Quals.addAddressSpace(AddressSpace);

  return getExtQualType(TypeNode, Quals);
}

QualType ASTContext::getObjCGCQualType(QualType T,
                                       Qualifiers::GC GCAttr) {
  QualType CanT = getCanonicalType(T);
  if (CanT.getObjCGCAttr() == GCAttr)
    return T;

  if (T->isPointerType()) {
    QualType Pointee = T->getAs<PointerType>()->getPointeeType();
    if (Pointee->isAnyPointerType()) {
      QualType ResultType = getObjCGCQualType(Pointee, GCAttr);
      return getPointerType(ResultType);
    }
  }

  // If we are composing extended qualifiers together, merge together
  // into one ExtQuals node.
  QualifierCollector Quals;
  const Type *TypeNode = Quals.strip(T);

  // If this type already has an ObjCGC specified, it cannot get
  // another one.
  assert(!Quals.hasObjCGCAttr() &&
         "Type cannot have multiple ObjCGCs!");
  Quals.addObjCGCAttr(GCAttr);

  return getExtQualType(TypeNode, Quals);
}

static QualType getExtFunctionType(ASTContext& Context, QualType T,
                                        const FunctionType::ExtInfo &Info) {
  QualType ResultType;
  if (const PointerType *Pointer = T->getAs<PointerType>()) {
    QualType Pointee = Pointer->getPointeeType();
    ResultType = getExtFunctionType(Context, Pointee, Info);
    if (ResultType == Pointee)
      return T;

    ResultType = Context.getPointerType(ResultType);
  } else if (const BlockPointerType *BlockPointer
                                              = T->getAs<BlockPointerType>()) {
    QualType Pointee = BlockPointer->getPointeeType();
    ResultType = getExtFunctionType(Context, Pointee, Info);
    if (ResultType == Pointee)
      return T;

    ResultType = Context.getBlockPointerType(ResultType);
   } else if (const FunctionType *F = T->getAs<FunctionType>()) {
    if (F->getExtInfo() == Info)
      return T;

    if (const FunctionNoProtoType *FNPT = dyn_cast<FunctionNoProtoType>(F)) {
      ResultType = Context.getFunctionNoProtoType(FNPT->getResultType(),
                                                  Info);
    } else {
      const FunctionProtoType *FPT = cast<FunctionProtoType>(F);
      ResultType
        = Context.getFunctionType(FPT->getResultType(), FPT->arg_type_begin(),
                                  FPT->getNumArgs(), FPT->isVariadic(),
                                  FPT->getTypeQuals(),
                                  FPT->hasExceptionSpec(),
                                  FPT->hasAnyExceptionSpec(),
                                  FPT->getNumExceptions(),
                                  FPT->exception_begin(),
                                  Info);
    }
  } else
    return T;

  return Context.getQualifiedType(ResultType, T.getLocalQualifiers());
}

QualType ASTContext::getNoReturnType(QualType T, bool AddNoReturn) {
  FunctionType::ExtInfo Info = getFunctionExtInfo(T);
  return getExtFunctionType(*this, T,
                                 Info.withNoReturn(AddNoReturn));
}

QualType ASTContext::getCallConvType(QualType T, CallingConv CallConv) {
  FunctionType::ExtInfo Info = getFunctionExtInfo(T);
  return getExtFunctionType(*this, T,
                            Info.withCallingConv(CallConv));
}

QualType ASTContext::getRegParmType(QualType T, unsigned RegParm) {
  FunctionType::ExtInfo Info = getFunctionExtInfo(T);
  return getExtFunctionType(*this, T,
                                 Info.withRegParm(RegParm));
}

/// getComplexType - Return the uniqued reference to the type for a complex
/// number with the specified element type.
QualType ASTContext::getComplexType(QualType T) {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ComplexType::Profile(ID, T);

  void *InsertPos = 0;
  if (ComplexType *CT = ComplexTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(CT, 0);

  // If the pointee type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getComplexType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    ComplexType *NewIP = ComplexTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  ComplexType *New = new (*this, TypeAlignment) ComplexType(T, Canonical);
  Types.push_back(New);
  ComplexTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getPointerType - Return the uniqued reference to the type for a pointer to
/// the specified type.
QualType ASTContext::getPointerType(QualType T) {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  PointerType::Profile(ID, T);

  void *InsertPos = 0;
  if (PointerType *PT = PointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(PT, 0);

  // If the pointee type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getPointerType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    PointerType *NewIP = PointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  PointerType *New = new (*this, TypeAlignment) PointerType(T, Canonical);
  Types.push_back(New);
  PointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getBlockPointerType - Return the uniqued reference to the type for
/// a pointer to the specified block.
QualType ASTContext::getBlockPointerType(QualType T) {
  assert(T->isFunctionType() && "block of function types only");
  // Unique pointers, to guarantee there is only one block of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  BlockPointerType::Profile(ID, T);

  void *InsertPos = 0;
  if (BlockPointerType *PT =
        BlockPointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(PT, 0);

  // If the block pointee type isn't canonical, this won't be a canonical
  // type either so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical()) {
    Canonical = getBlockPointerType(getCanonicalType(T));

    // Get the new insert position for the node we care about.
    BlockPointerType *NewIP =
      BlockPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  BlockPointerType *New
    = new (*this, TypeAlignment) BlockPointerType(T, Canonical);
  Types.push_back(New);
  BlockPointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getLValueReferenceType - Return the uniqued reference to the type for an
/// lvalue reference to the specified type.
QualType ASTContext::getLValueReferenceType(QualType T, bool SpelledAsLValue) {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ReferenceType::Profile(ID, T, SpelledAsLValue);

  void *InsertPos = 0;
  if (LValueReferenceType *RT =
        LValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(RT, 0);

  const ReferenceType *InnerRef = T->getAs<ReferenceType>();

  // If the referencee type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;
  if (!SpelledAsLValue || InnerRef || !T.isCanonical()) {
    QualType PointeeType = (InnerRef ? InnerRef->getPointeeType() : T);
    Canonical = getLValueReferenceType(getCanonicalType(PointeeType));

    // Get the new insert position for the node we care about.
    LValueReferenceType *NewIP =
      LValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  LValueReferenceType *New
    = new (*this, TypeAlignment) LValueReferenceType(T, Canonical,
                                                     SpelledAsLValue);
  Types.push_back(New);
  LValueReferenceTypes.InsertNode(New, InsertPos);

  return QualType(New, 0);
}

/// getRValueReferenceType - Return the uniqued reference to the type for an
/// rvalue reference to the specified type.
QualType ASTContext::getRValueReferenceType(QualType T) {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ReferenceType::Profile(ID, T, false);

  void *InsertPos = 0;
  if (RValueReferenceType *RT =
        RValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(RT, 0);

  const ReferenceType *InnerRef = T->getAs<ReferenceType>();

  // If the referencee type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;
  if (InnerRef || !T.isCanonical()) {
    QualType PointeeType = (InnerRef ? InnerRef->getPointeeType() : T);
    Canonical = getRValueReferenceType(getCanonicalType(PointeeType));

    // Get the new insert position for the node we care about.
    RValueReferenceType *NewIP =
      RValueReferenceTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  RValueReferenceType *New
    = new (*this, TypeAlignment) RValueReferenceType(T, Canonical);
  Types.push_back(New);
  RValueReferenceTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getMemberPointerType - Return the uniqued reference to the type for a
/// member pointer to the specified type, in the specified class.
QualType ASTContext::getMemberPointerType(QualType T, const Type *Cls) {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  MemberPointerType::Profile(ID, T, Cls);

  void *InsertPos = 0;
  if (MemberPointerType *PT =
      MemberPointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(PT, 0);

  // If the pointee or class type isn't canonical, this won't be a canonical
  // type either, so fill in the canonical type field.
  QualType Canonical;
  if (!T.isCanonical() || !Cls->isCanonicalUnqualified()) {
    Canonical = getMemberPointerType(getCanonicalType(T),getCanonicalType(Cls));

    // Get the new insert position for the node we care about.
    MemberPointerType *NewIP =
      MemberPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  MemberPointerType *New
    = new (*this, TypeAlignment) MemberPointerType(T, Cls, Canonical);
  Types.push_back(New);
  MemberPointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getConstantArrayType - Return the unique reference to the type for an
/// array of the specified element type.
QualType ASTContext::getConstantArrayType(QualType EltTy,
                                          const llvm::APInt &ArySizeIn,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned EltTypeQuals) {
  assert((EltTy->isDependentType() ||
          EltTy->isIncompleteType() || EltTy->isConstantSizeType()) &&
         "Constant array of VLAs is illegal!");

  // Convert the array size into a canonical width matching the pointer size for
  // the target.
  llvm::APInt ArySize(ArySizeIn);
  ArySize.zextOrTrunc(Target.getPointerWidth(EltTy.getAddressSpace()));

  llvm::FoldingSetNodeID ID;
  ConstantArrayType::Profile(ID, EltTy, ArySize, ASM, EltTypeQuals);

  void *InsertPos = 0;
  if (ConstantArrayType *ATP =
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(ATP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!EltTy.isCanonical()) {
    Canonical = getConstantArrayType(getCanonicalType(EltTy), ArySize,
                                     ASM, EltTypeQuals);
    // Get the new insert position for the node we care about.
    ConstantArrayType *NewIP =
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  ConstantArrayType *New = new(*this,TypeAlignment)
    ConstantArrayType(EltTy, Canonical, ArySize, ASM, EltTypeQuals);
  ConstantArrayTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getVariableArrayType - Returns a non-unique reference to the type for a
/// variable array of the specified element type.
QualType ASTContext::getVariableArrayType(QualType EltTy,
                                          Expr *NumElts,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned EltTypeQuals,
                                          SourceRange Brackets) {
  // Since we don't unique expressions, it isn't possible to unique VLA's
  // that have an expression provided for their size.

  VariableArrayType *New = new(*this, TypeAlignment)
    VariableArrayType(EltTy, QualType(), NumElts, ASM, EltTypeQuals, Brackets);

  VariableArrayTypes.push_back(New);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getDependentSizedArrayType - Returns a non-unique reference to
/// the type for a dependently-sized array of the specified element
/// type.
QualType ASTContext::getDependentSizedArrayType(QualType EltTy,
                                                Expr *NumElts,
                                                ArrayType::ArraySizeModifier ASM,
                                                unsigned EltTypeQuals,
                                                SourceRange Brackets) {
  assert((!NumElts || NumElts->isTypeDependent() || 
          NumElts->isValueDependent()) &&
         "Size must be type- or value-dependent!");

  void *InsertPos = 0;
  DependentSizedArrayType *Canon = 0;
  llvm::FoldingSetNodeID ID;

  if (NumElts) {
    // Dependently-sized array types that do not have a specified
    // number of elements will have their sizes deduced from an
    // initializer.
    DependentSizedArrayType::Profile(ID, *this, getCanonicalType(EltTy), ASM,
                                     EltTypeQuals, NumElts);

    Canon = DependentSizedArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  DependentSizedArrayType *New;
  if (Canon) {
    // We already have a canonical version of this array type; use it as
    // the canonical type for a newly-built type.
    New = new (*this, TypeAlignment)
      DependentSizedArrayType(*this, EltTy, QualType(Canon, 0),
                              NumElts, ASM, EltTypeQuals, Brackets);
  } else {
    QualType CanonEltTy = getCanonicalType(EltTy);
    if (CanonEltTy == EltTy) {
      New = new (*this, TypeAlignment)
        DependentSizedArrayType(*this, EltTy, QualType(),
                                NumElts, ASM, EltTypeQuals, Brackets);

      if (NumElts) {
        DependentSizedArrayType *CanonCheck
          = DependentSizedArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
        assert(!CanonCheck && "Dependent-sized canonical array type broken");
        (void)CanonCheck;
        DependentSizedArrayTypes.InsertNode(New, InsertPos);
      }
    } else {
      QualType Canon = getDependentSizedArrayType(CanonEltTy, NumElts,
                                                  ASM, EltTypeQuals,
                                                  SourceRange());
      New = new (*this, TypeAlignment)
        DependentSizedArrayType(*this, EltTy, Canon,
                                NumElts, ASM, EltTypeQuals, Brackets);
    }
  }

  Types.push_back(New);
  return QualType(New, 0);
}

QualType ASTContext::getIncompleteArrayType(QualType EltTy,
                                            ArrayType::ArraySizeModifier ASM,
                                            unsigned EltTypeQuals) {
  llvm::FoldingSetNodeID ID;
  IncompleteArrayType::Profile(ID, EltTy, ASM, EltTypeQuals);

  void *InsertPos = 0;
  if (IncompleteArrayType *ATP =
       IncompleteArrayTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(ATP, 0);

  // If the element type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;

  if (!EltTy.isCanonical()) {
    Canonical = getIncompleteArrayType(getCanonicalType(EltTy),
                                       ASM, EltTypeQuals);

    // Get the new insert position for the node we care about.
    IncompleteArrayType *NewIP =
      IncompleteArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  IncompleteArrayType *New = new (*this, TypeAlignment)
    IncompleteArrayType(EltTy, Canonical, ASM, EltTypeQuals);

  IncompleteArrayTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getVectorType - Return the unique reference to a vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType ASTContext::getVectorType(QualType vecType, unsigned NumElts,
                                   bool IsAltiVec, bool IsPixel) {
  BuiltinType *baseType;

  baseType = dyn_cast<BuiltinType>(getCanonicalType(vecType).getTypePtr());
  assert(baseType != 0 && "getVectorType(): Expecting a built-in type");

  // Check if we've already instantiated a vector of this type.
  llvm::FoldingSetNodeID ID;
  VectorType::Profile(ID, vecType, NumElts, Type::Vector,
    IsAltiVec, IsPixel);
  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType.isCanonical() || IsAltiVec || IsPixel) {
    Canonical = getVectorType(getCanonicalType(vecType),
      NumElts, false, false);

    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  VectorType *New = new (*this, TypeAlignment)
    VectorType(vecType, NumElts, Canonical, IsAltiVec, IsPixel);
  VectorTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getExtVectorType - Return the unique reference to an extended vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType ASTContext::getExtVectorType(QualType vecType, unsigned NumElts) {
  BuiltinType *baseType;

  baseType = dyn_cast<BuiltinType>(getCanonicalType(vecType).getTypePtr());
  assert(baseType != 0 && "getExtVectorType(): Expecting a built-in type");

  // Check if we've already instantiated a vector of this type.
  llvm::FoldingSetNodeID ID;
  VectorType::Profile(ID, vecType, NumElts, Type::ExtVector, false, false);
  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType.isCanonical()) {
    Canonical = getExtVectorType(getCanonicalType(vecType), NumElts);

    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  ExtVectorType *New = new (*this, TypeAlignment)
    ExtVectorType(vecType, NumElts, Canonical);
  VectorTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

QualType ASTContext::getDependentSizedExtVectorType(QualType vecType,
                                                    Expr *SizeExpr,
                                                    SourceLocation AttrLoc) {
  llvm::FoldingSetNodeID ID;
  DependentSizedExtVectorType::Profile(ID, *this, getCanonicalType(vecType),
                                       SizeExpr);

  void *InsertPos = 0;
  DependentSizedExtVectorType *Canon
    = DependentSizedExtVectorTypes.FindNodeOrInsertPos(ID, InsertPos);
  DependentSizedExtVectorType *New;
  if (Canon) {
    // We already have a canonical version of this array type; use it as
    // the canonical type for a newly-built type.
    New = new (*this, TypeAlignment)
      DependentSizedExtVectorType(*this, vecType, QualType(Canon, 0),
                                  SizeExpr, AttrLoc);
  } else {
    QualType CanonVecTy = getCanonicalType(vecType);
    if (CanonVecTy == vecType) {
      New = new (*this, TypeAlignment)
        DependentSizedExtVectorType(*this, vecType, QualType(), SizeExpr,
                                    AttrLoc);

      DependentSizedExtVectorType *CanonCheck
        = DependentSizedExtVectorTypes.FindNodeOrInsertPos(ID, InsertPos);
      assert(!CanonCheck && "Dependent-sized ext_vector canonical type broken");
      (void)CanonCheck;
      DependentSizedExtVectorTypes.InsertNode(New, InsertPos);
    } else {
      QualType Canon = getDependentSizedExtVectorType(CanonVecTy, SizeExpr,
                                                      SourceLocation());
      New = new (*this, TypeAlignment) 
        DependentSizedExtVectorType(*this, vecType, Canon, SizeExpr, AttrLoc);
    }
  }

  Types.push_back(New);
  return QualType(New, 0);
}

/// getFunctionNoProtoType - Return a K&R style C function type like 'int()'.
///
QualType ASTContext::getFunctionNoProtoType(QualType ResultTy,
                                            const FunctionType::ExtInfo &Info) {
  const CallingConv CallConv = Info.getCC();
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionNoProtoType::Profile(ID, ResultTy, Info);

  void *InsertPos = 0;
  if (FunctionNoProtoType *FT =
        FunctionNoProtoTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(FT, 0);

  QualType Canonical;
  if (!ResultTy.isCanonical() ||
      getCanonicalCallConv(CallConv) != CallConv) {
    Canonical =
      getFunctionNoProtoType(getCanonicalType(ResultTy),
                     Info.withCallingConv(getCanonicalCallConv(CallConv)));

    // Get the new insert position for the node we care about.
    FunctionNoProtoType *NewIP =
      FunctionNoProtoTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  FunctionNoProtoType *New = new (*this, TypeAlignment)
    FunctionNoProtoType(ResultTy, Canonical, Info);
  Types.push_back(New);
  FunctionNoProtoTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getFunctionType - Return a normal function type with a typed argument
/// list.  isVariadic indicates whether the argument list includes '...'.
QualType ASTContext::getFunctionType(QualType ResultTy,const QualType *ArgArray,
                                     unsigned NumArgs, bool isVariadic,
                                     unsigned TypeQuals, bool hasExceptionSpec,
                                     bool hasAnyExceptionSpec, unsigned NumExs,
                                     const QualType *ExArray,
                                     const FunctionType::ExtInfo &Info) {
  const CallingConv CallConv= Info.getCC();
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionProtoType::Profile(ID, ResultTy, ArgArray, NumArgs, isVariadic,
                             TypeQuals, hasExceptionSpec, hasAnyExceptionSpec,
                             NumExs, ExArray, Info);

  void *InsertPos = 0;
  if (FunctionProtoType *FTP =
        FunctionProtoTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(FTP, 0);

  // Determine whether the type being created is already canonical or not.
  bool isCanonical = !hasExceptionSpec && ResultTy.isCanonical();
  for (unsigned i = 0; i != NumArgs && isCanonical; ++i)
    if (!ArgArray[i].isCanonicalAsParam())
      isCanonical = false;

  // If this type isn't canonical, get the canonical version of it.
  // The exception spec is not part of the canonical type.
  QualType Canonical;
  if (!isCanonical || getCanonicalCallConv(CallConv) != CallConv) {
    llvm::SmallVector<QualType, 16> CanonicalArgs;
    CanonicalArgs.reserve(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      CanonicalArgs.push_back(getCanonicalParamType(ArgArray[i]));

    Canonical = getFunctionType(getCanonicalType(ResultTy),
                                CanonicalArgs.data(), NumArgs,
                                isVariadic, TypeQuals, false,
                                false, 0, 0,
                     Info.withCallingConv(getCanonicalCallConv(CallConv)));

    // Get the new insert position for the node we care about.
    FunctionProtoType *NewIP =
      FunctionProtoTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  // FunctionProtoType objects are allocated with extra bytes after them
  // for two variable size arrays (for parameter and exception types) at the
  // end of them.
  FunctionProtoType *FTP =
    (FunctionProtoType*)Allocate(sizeof(FunctionProtoType) +
                                 NumArgs*sizeof(QualType) +
                                 NumExs*sizeof(QualType), TypeAlignment);
  new (FTP) FunctionProtoType(ResultTy, ArgArray, NumArgs, isVariadic,
                              TypeQuals, hasExceptionSpec, hasAnyExceptionSpec,
                              ExArray, NumExs, Canonical, Info);
  Types.push_back(FTP);
  FunctionProtoTypes.InsertNode(FTP, InsertPos);
  return QualType(FTP, 0);
}

#ifndef NDEBUG
static bool NeedsInjectedClassNameType(const RecordDecl *D) {
  if (!isa<CXXRecordDecl>(D)) return false;
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(D);
  if (isa<ClassTemplatePartialSpecializationDecl>(RD))
    return true;
  if (RD->getDescribedClassTemplate() &&
      !isa<ClassTemplateSpecializationDecl>(RD))
    return true;
  return false;
}
#endif

/// getInjectedClassNameType - Return the unique reference to the
/// injected class name type for the specified templated declaration.
QualType ASTContext::getInjectedClassNameType(CXXRecordDecl *Decl,
                                              QualType TST) {
  assert(NeedsInjectedClassNameType(Decl));
  if (Decl->TypeForDecl) {
    assert(isa<InjectedClassNameType>(Decl->TypeForDecl));
  } else if (CXXRecordDecl *PrevDecl
               = cast_or_null<CXXRecordDecl>(Decl->getPreviousDeclaration())) {
    assert(PrevDecl->TypeForDecl && "previous declaration has no type");
    Decl->TypeForDecl = PrevDecl->TypeForDecl;
    assert(isa<InjectedClassNameType>(Decl->TypeForDecl));
  } else {
    Decl->TypeForDecl =
      new (*this, TypeAlignment) InjectedClassNameType(Decl, TST);
    Types.push_back(Decl->TypeForDecl);
  }
  return QualType(Decl->TypeForDecl, 0);
}

/// getTypeDeclType - Return the unique reference to the type for the
/// specified type declaration.
QualType ASTContext::getTypeDeclTypeSlow(const TypeDecl *Decl) {
  assert(Decl && "Passed null for Decl param");
  assert(!Decl->TypeForDecl && "TypeForDecl present in slow case");

  if (const TypedefDecl *Typedef = dyn_cast<TypedefDecl>(Decl))
    return getTypedefType(Typedef);

  if (const ObjCInterfaceDecl *ObjCInterface
               = dyn_cast<ObjCInterfaceDecl>(Decl))
    return getObjCInterfaceType(ObjCInterface);

  assert(!isa<TemplateTypeParmDecl>(Decl) &&
         "Template type parameter types are always available.");

  if (const RecordDecl *Record = dyn_cast<RecordDecl>(Decl)) {
    assert(!Record->getPreviousDeclaration() &&
           "struct/union has previous declaration");
    assert(!NeedsInjectedClassNameType(Record));
    Decl->TypeForDecl = new (*this, TypeAlignment) RecordType(Record);
  } else if (const EnumDecl *Enum = dyn_cast<EnumDecl>(Decl)) {
    assert(!Enum->getPreviousDeclaration() &&
           "enum has previous declaration");
    Decl->TypeForDecl = new (*this, TypeAlignment) EnumType(Enum);
  } else if (const UnresolvedUsingTypenameDecl *Using =
               dyn_cast<UnresolvedUsingTypenameDecl>(Decl)) {
    Decl->TypeForDecl = new (*this, TypeAlignment) UnresolvedUsingType(Using);
  } else
    llvm_unreachable("TypeDecl without a type?");

  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// getTypedefType - Return the unique reference to the type for the
/// specified typename decl.
QualType ASTContext::getTypedefType(const TypedefDecl *Decl) {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);

  QualType Canonical = getCanonicalType(Decl->getUnderlyingType());
  Decl->TypeForDecl = new(*this, TypeAlignment)
    TypedefType(Type::Typedef, Decl, Canonical);
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// \brief Retrieve a substitution-result type.
QualType
ASTContext::getSubstTemplateTypeParmType(const TemplateTypeParmType *Parm,
                                         QualType Replacement) {
  assert(Replacement.isCanonical()
         && "replacement types must always be canonical");

  llvm::FoldingSetNodeID ID;
  SubstTemplateTypeParmType::Profile(ID, Parm, Replacement);
  void *InsertPos = 0;
  SubstTemplateTypeParmType *SubstParm
    = SubstTemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (!SubstParm) {
    SubstParm = new (*this, TypeAlignment)
      SubstTemplateTypeParmType(Parm, Replacement);
    Types.push_back(SubstParm);
    SubstTemplateTypeParmTypes.InsertNode(SubstParm, InsertPos);
  }

  return QualType(SubstParm, 0);
}

/// \brief Retrieve the template type parameter type for a template
/// parameter or parameter pack with the given depth, index, and (optionally)
/// name.
QualType ASTContext::getTemplateTypeParmType(unsigned Depth, unsigned Index,
                                             bool ParameterPack,
                                             IdentifierInfo *Name) {
  llvm::FoldingSetNodeID ID;
  TemplateTypeParmType::Profile(ID, Depth, Index, ParameterPack, Name);
  void *InsertPos = 0;
  TemplateTypeParmType *TypeParm
    = TemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (TypeParm)
    return QualType(TypeParm, 0);

  if (Name) {
    QualType Canon = getTemplateTypeParmType(Depth, Index, ParameterPack);
    TypeParm = new (*this, TypeAlignment)
      TemplateTypeParmType(Depth, Index, ParameterPack, Name, Canon);

    TemplateTypeParmType *TypeCheck 
      = TemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!TypeCheck && "Template type parameter canonical type broken");
    (void)TypeCheck;
  } else
    TypeParm = new (*this, TypeAlignment)
      TemplateTypeParmType(Depth, Index, ParameterPack);

  Types.push_back(TypeParm);
  TemplateTypeParmTypes.InsertNode(TypeParm, InsertPos);

  return QualType(TypeParm, 0);
}

TypeSourceInfo *
ASTContext::getTemplateSpecializationTypeInfo(TemplateName Name,
                                              SourceLocation NameLoc,
                                        const TemplateArgumentListInfo &Args,
                                              QualType CanonType) {
  QualType TST = getTemplateSpecializationType(Name, Args, CanonType);

  TypeSourceInfo *DI = CreateTypeSourceInfo(TST);
  TemplateSpecializationTypeLoc TL
    = cast<TemplateSpecializationTypeLoc>(DI->getTypeLoc());
  TL.setTemplateNameLoc(NameLoc);
  TL.setLAngleLoc(Args.getLAngleLoc());
  TL.setRAngleLoc(Args.getRAngleLoc());
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    TL.setArgLocInfo(i, Args[i].getLocInfo());
  return DI;
}

QualType
ASTContext::getTemplateSpecializationType(TemplateName Template,
                                          const TemplateArgumentListInfo &Args,
                                          QualType Canon,
                                          bool IsCurrentInstantiation) {
  unsigned NumArgs = Args.size();

  llvm::SmallVector<TemplateArgument, 4> ArgVec;
  ArgVec.reserve(NumArgs);
  for (unsigned i = 0; i != NumArgs; ++i)
    ArgVec.push_back(Args[i].getArgument());

  return getTemplateSpecializationType(Template, ArgVec.data(), NumArgs,
                                       Canon, IsCurrentInstantiation);
}

QualType
ASTContext::getTemplateSpecializationType(TemplateName Template,
                                          const TemplateArgument *Args,
                                          unsigned NumArgs,
                                          QualType Canon,
                                          bool IsCurrentInstantiation) {
  if (!Canon.isNull())
    Canon = getCanonicalType(Canon);
  else {
    assert(!IsCurrentInstantiation &&
           "current-instantiation specializations should always "
           "have a canonical type");

    // Build the canonical template specialization type.
    TemplateName CanonTemplate = getCanonicalTemplateName(Template);
    llvm::SmallVector<TemplateArgument, 4> CanonArgs;
    CanonArgs.reserve(NumArgs);
    for (unsigned I = 0; I != NumArgs; ++I)
      CanonArgs.push_back(getCanonicalTemplateArgument(Args[I]));

    // Determine whether this canonical template specialization type already
    // exists.
    llvm::FoldingSetNodeID ID;
    TemplateSpecializationType::Profile(ID, CanonTemplate, false,
                                        CanonArgs.data(), NumArgs, *this);

    void *InsertPos = 0;
    TemplateSpecializationType *Spec
      = TemplateSpecializationTypes.FindNodeOrInsertPos(ID, InsertPos);

    if (!Spec) {
      // Allocate a new canonical template specialization type.
      void *Mem = Allocate((sizeof(TemplateSpecializationType) +
                            sizeof(TemplateArgument) * NumArgs),
                           TypeAlignment);
      Spec = new (Mem) TemplateSpecializationType(*this, CanonTemplate, false,
                                                  CanonArgs.data(), NumArgs,
                                                  Canon);
      Types.push_back(Spec);
      TemplateSpecializationTypes.InsertNode(Spec, InsertPos);
    }

    if (Canon.isNull())
      Canon = QualType(Spec, 0);
    assert(Canon->isDependentType() &&
           "Non-dependent template-id type must have a canonical type");
  }

  // Allocate the (non-canonical) template specialization type, but don't
  // try to unique it: these types typically have location information that
  // we don't unique and don't want to lose.
  void *Mem = Allocate((sizeof(TemplateSpecializationType) +
                        sizeof(TemplateArgument) * NumArgs),
                       TypeAlignment);
  TemplateSpecializationType *Spec
    = new (Mem) TemplateSpecializationType(*this, Template,
                                           IsCurrentInstantiation,
                                           Args, NumArgs,
                                           Canon);

  Types.push_back(Spec);
  return QualType(Spec, 0);
}

QualType
ASTContext::getQualifiedNameType(NestedNameSpecifier *NNS,
                                 QualType NamedType) {
  llvm::FoldingSetNodeID ID;
  QualifiedNameType::Profile(ID, NNS, NamedType);

  void *InsertPos = 0;
  QualifiedNameType *T
    = QualifiedNameTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  QualType Canon = NamedType;
  if (!Canon.isCanonical()) {
    Canon = getCanonicalType(NamedType);
    QualifiedNameType *CheckT
      = QualifiedNameTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckT && "Qualified name canonical type broken");
    (void)CheckT;
  }

  T = new (*this) QualifiedNameType(NNS, NamedType, Canon);
  Types.push_back(T);
  QualifiedNameTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType ASTContext::getDependentNameType(ElaboratedTypeKeyword Keyword,
                                          NestedNameSpecifier *NNS,
                                          const IdentifierInfo *Name,
                                          QualType Canon) {
  assert(NNS->isDependent() && "nested-name-specifier must be dependent");

  if (Canon.isNull()) {
    NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
    ElaboratedTypeKeyword CanonKeyword = Keyword;
    if (Keyword == ETK_None)
      CanonKeyword = ETK_Typename;
    
    if (CanonNNS != NNS || CanonKeyword != Keyword)
      Canon = getDependentNameType(CanonKeyword, CanonNNS, Name);
  }

  llvm::FoldingSetNodeID ID;
  DependentNameType::Profile(ID, Keyword, NNS, Name);

  void *InsertPos = 0;
  DependentNameType *T
    = DependentNameTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  T = new (*this) DependentNameType(Keyword, NNS, Name, Canon);
  Types.push_back(T);
  DependentNameTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType
ASTContext::getDependentNameType(ElaboratedTypeKeyword Keyword,
                                 NestedNameSpecifier *NNS,
                                 const TemplateSpecializationType *TemplateId,
                                 QualType Canon) {
  assert(NNS->isDependent() && "nested-name-specifier must be dependent");

  llvm::FoldingSetNodeID ID;
  DependentNameType::Profile(ID, Keyword, NNS, TemplateId);

  void *InsertPos = 0;
  DependentNameType *T
    = DependentNameTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  if (Canon.isNull()) {
    NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
    QualType CanonType = getCanonicalType(QualType(TemplateId, 0));
    ElaboratedTypeKeyword CanonKeyword = Keyword;
    if (Keyword == ETK_None)
      CanonKeyword = ETK_Typename;
    if (CanonNNS != NNS || CanonKeyword != Keyword ||
        CanonType != QualType(TemplateId, 0)) {
      const TemplateSpecializationType *CanonTemplateId
        = CanonType->getAs<TemplateSpecializationType>();
      assert(CanonTemplateId &&
             "Canonical type must also be a template specialization type");
      Canon = getDependentNameType(CanonKeyword, CanonNNS, CanonTemplateId);
    }

    DependentNameType *CheckT
      = DependentNameTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckT && "Typename canonical type is broken"); (void)CheckT;
  }

  T = new (*this) DependentNameType(Keyword, NNS, TemplateId, Canon);
  Types.push_back(T);
  DependentNameTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

QualType
ASTContext::getElaboratedType(QualType UnderlyingType,
                              ElaboratedType::TagKind Tag) {
  llvm::FoldingSetNodeID ID;
  ElaboratedType::Profile(ID, UnderlyingType, Tag);

  void *InsertPos = 0;
  ElaboratedType *T = ElaboratedTypes.FindNodeOrInsertPos(ID, InsertPos);
  if (T)
    return QualType(T, 0);

  QualType Canon = UnderlyingType;
  if (!Canon.isCanonical()) {
    Canon = getCanonicalType(Canon);
    ElaboratedType *CheckT = ElaboratedTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckT && "Elaborated canonical type is broken"); (void)CheckT;
  }

  T = new (*this) ElaboratedType(UnderlyingType, Tag, Canon);
  Types.push_back(T);
  ElaboratedTypes.InsertNode(T, InsertPos);
  return QualType(T, 0);
}

/// CmpProtocolNames - Comparison predicate for sorting protocols
/// alphabetically.
static bool CmpProtocolNames(const ObjCProtocolDecl *LHS,
                            const ObjCProtocolDecl *RHS) {
  return LHS->getDeclName() < RHS->getDeclName();
}

static bool areSortedAndUniqued(ObjCProtocolDecl **Protocols,
                                unsigned NumProtocols) {
  if (NumProtocols == 0) return true;

  for (unsigned i = 1; i != NumProtocols; ++i)
    if (!CmpProtocolNames(Protocols[i-1], Protocols[i]))
      return false;
  return true;
}

static void SortAndUniqueProtocols(ObjCProtocolDecl **Protocols,
                                   unsigned &NumProtocols) {
  ObjCProtocolDecl **ProtocolsEnd = Protocols+NumProtocols;

  // Sort protocols, keyed by name.
  std::sort(Protocols, Protocols+NumProtocols, CmpProtocolNames);

  // Remove duplicates.
  ProtocolsEnd = std::unique(Protocols, ProtocolsEnd);
  NumProtocols = ProtocolsEnd-Protocols;
}

/// getObjCObjectPointerType - Return a ObjCObjectPointerType type for
/// the given interface decl and the conforming protocol list.
QualType ASTContext::getObjCObjectPointerType(QualType InterfaceT,
                                              ObjCProtocolDecl **Protocols,
                                              unsigned NumProtocols,
                                              unsigned Quals) {
  llvm::FoldingSetNodeID ID;
  ObjCObjectPointerType::Profile(ID, InterfaceT, Protocols, NumProtocols);
  Qualifiers Qs = Qualifiers::fromCVRMask(Quals);

  void *InsertPos = 0;
  if (ObjCObjectPointerType *QT =
              ObjCObjectPointerTypes.FindNodeOrInsertPos(ID, InsertPos))
    return getQualifiedType(QualType(QT, 0), Qs);

  // Sort the protocol list alphabetically to canonicalize it.
  QualType Canonical;
  if (!InterfaceT.isCanonical() || 
      !areSortedAndUniqued(Protocols, NumProtocols)) {
    if (!areSortedAndUniqued(Protocols, NumProtocols)) {
      llvm::SmallVector<ObjCProtocolDecl*, 8> Sorted(Protocols,
                                                     Protocols + NumProtocols);
      unsigned UniqueCount = NumProtocols;

      SortAndUniqueProtocols(&Sorted[0], UniqueCount);

      Canonical = getObjCObjectPointerType(getCanonicalType(InterfaceT),
                                           &Sorted[0], UniqueCount);
    } else {
      Canonical = getObjCObjectPointerType(getCanonicalType(InterfaceT),
                                           Protocols, NumProtocols);
    }

    // Regenerate InsertPos.
    ObjCObjectPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  // No match.
  unsigned Size = sizeof(ObjCObjectPointerType) 
                + NumProtocols * sizeof(ObjCProtocolDecl *);
  void *Mem = Allocate(Size, TypeAlignment);
  ObjCObjectPointerType *QType = new (Mem) ObjCObjectPointerType(Canonical, 
                                                                 InterfaceT, 
                                                                 Protocols,
                                                                 NumProtocols);

  Types.push_back(QType);
  ObjCObjectPointerTypes.InsertNode(QType, InsertPos);
  return getQualifiedType(QualType(QType, 0), Qs);
}

/// getObjCInterfaceType - Return the unique reference to the type for the
/// specified ObjC interface decl. The list of protocols is optional.
QualType ASTContext::getObjCInterfaceType(const ObjCInterfaceDecl *Decl,
                       ObjCProtocolDecl **Protocols, unsigned NumProtocols) {
  llvm::FoldingSetNodeID ID;
  ObjCInterfaceType::Profile(ID, Decl, Protocols, NumProtocols);

  void *InsertPos = 0;
  if (ObjCInterfaceType *QT =
      ObjCInterfaceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(QT, 0);

  // Sort the protocol list alphabetically to canonicalize it.
  QualType Canonical;
  if (NumProtocols && !areSortedAndUniqued(Protocols, NumProtocols)) {
    llvm::SmallVector<ObjCProtocolDecl*, 8> Sorted(Protocols,
                                                   Protocols + NumProtocols);

    unsigned UniqueCount = NumProtocols;
    SortAndUniqueProtocols(&Sorted[0], UniqueCount);

    Canonical = getObjCInterfaceType(Decl, &Sorted[0], UniqueCount);

    ObjCInterfaceTypes.FindNodeOrInsertPos(ID, InsertPos);
  }

  unsigned Size = sizeof(ObjCInterfaceType) 
    + NumProtocols * sizeof(ObjCProtocolDecl *);
  void *Mem = Allocate(Size, TypeAlignment);
  ObjCInterfaceType *QType = new (Mem) ObjCInterfaceType(Canonical, 
                                        const_cast<ObjCInterfaceDecl*>(Decl),
                                                         Protocols, 
                                                         NumProtocols);

  Types.push_back(QType);
  ObjCInterfaceTypes.InsertNode(QType, InsertPos);
  return QualType(QType, 0);
}

/// getTypeOfExprType - Unlike many "get<Type>" functions, we can't unique
/// TypeOfExprType AST's (since expression's are never shared). For example,
/// multiple declarations that refer to "typeof(x)" all contain different
/// DeclRefExpr's. This doesn't effect the type checker, since it operates
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfExprType(Expr *tofExpr) {
  TypeOfExprType *toe;
  if (tofExpr->isTypeDependent()) {
    llvm::FoldingSetNodeID ID;
    DependentTypeOfExprType::Profile(ID, *this, tofExpr);

    void *InsertPos = 0;
    DependentTypeOfExprType *Canon
      = DependentTypeOfExprTypes.FindNodeOrInsertPos(ID, InsertPos);
    if (Canon) {
      // We already have a "canonical" version of an identical, dependent
      // typeof(expr) type. Use that as our canonical type.
      toe = new (*this, TypeAlignment) TypeOfExprType(tofExpr,
                                          QualType((TypeOfExprType*)Canon, 0));
    }
    else {
      // Build a new, canonical typeof(expr) type.
      Canon
        = new (*this, TypeAlignment) DependentTypeOfExprType(*this, tofExpr);
      DependentTypeOfExprTypes.InsertNode(Canon, InsertPos);
      toe = Canon;
    }
  } else {
    QualType Canonical = getCanonicalType(tofExpr->getType());
    toe = new (*this, TypeAlignment) TypeOfExprType(tofExpr, Canonical);
  }
  Types.push_back(toe);
  return QualType(toe, 0);
}

/// getTypeOfType -  Unlike many "get<Type>" functions, we don't unique
/// TypeOfType AST's. The only motivation to unique these nodes would be
/// memory savings. Since typeof(t) is fairly uncommon, space shouldn't be
/// an issue. This doesn't effect the type checker, since it operates
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfType(QualType tofType) {
  QualType Canonical = getCanonicalType(tofType);
  TypeOfType *tot = new (*this, TypeAlignment) TypeOfType(tofType, Canonical);
  Types.push_back(tot);
  return QualType(tot, 0);
}

/// getDecltypeForExpr - Given an expr, will return the decltype for that
/// expression, according to the rules in C++0x [dcl.type.simple]p4
static QualType getDecltypeForExpr(const Expr *e, ASTContext &Context) {
  if (e->isTypeDependent())
    return Context.DependentTy;

  // If e is an id expression or a class member access, decltype(e) is defined
  // as the type of the entity named by e.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(e)) {
    if (const ValueDecl *VD = dyn_cast<ValueDecl>(DRE->getDecl()))
      return VD->getType();
  }
  if (const MemberExpr *ME = dyn_cast<MemberExpr>(e)) {
    if (const FieldDecl *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
      return FD->getType();
  }
  // If e is a function call or an invocation of an overloaded operator,
  // (parentheses around e are ignored), decltype(e) is defined as the
  // return type of that function.
  if (const CallExpr *CE = dyn_cast<CallExpr>(e->IgnoreParens()))
    return CE->getCallReturnType();

  QualType T = e->getType();

  // Otherwise, where T is the type of e, if e is an lvalue, decltype(e) is
  // defined as T&, otherwise decltype(e) is defined as T.
  if (e->isLvalue(Context) == Expr::LV_Valid)
    T = Context.getLValueReferenceType(T);

  return T;
}

/// getDecltypeType -  Unlike many "get<Type>" functions, we don't unique
/// DecltypeType AST's. The only motivation to unique these nodes would be
/// memory savings. Since decltype(t) is fairly uncommon, space shouldn't be
/// an issue. This doesn't effect the type checker, since it operates
/// on canonical type's (which are always unique).
QualType ASTContext::getDecltypeType(Expr *e) {
  DecltypeType *dt;
  if (e->isTypeDependent()) {
    llvm::FoldingSetNodeID ID;
    DependentDecltypeType::Profile(ID, *this, e);

    void *InsertPos = 0;
    DependentDecltypeType *Canon
      = DependentDecltypeTypes.FindNodeOrInsertPos(ID, InsertPos);
    if (Canon) {
      // We already have a "canonical" version of an equivalent, dependent
      // decltype type. Use that as our canonical type.
      dt = new (*this, TypeAlignment) DecltypeType(e, DependentTy,
                                       QualType((DecltypeType*)Canon, 0));
    }
    else {
      // Build a new, canonical typeof(expr) type.
      Canon = new (*this, TypeAlignment) DependentDecltypeType(*this, e);
      DependentDecltypeTypes.InsertNode(Canon, InsertPos);
      dt = Canon;
    }
  } else {
    QualType T = getDecltypeForExpr(e, *this);
    dt = new (*this, TypeAlignment) DecltypeType(e, T, getCanonicalType(T));
  }
  Types.push_back(dt);
  return QualType(dt, 0);
}

/// getTagDeclType - Return the unique reference to the type for the
/// specified TagDecl (struct/union/class/enum) decl.
QualType ASTContext::getTagDeclType(const TagDecl *Decl) {
  assert (Decl);
  // FIXME: What is the design on getTagDeclType when it requires casting
  // away const?  mutable?
  return getTypeDeclType(const_cast<TagDecl*>(Decl));
}

/// getSizeType - Return the unique type for "size_t" (C99 7.17), the result
/// of the sizeof operator (C99 6.5.3.4p4). The value is target dependent and
/// needs to agree with the definition in <stddef.h>.
CanQualType ASTContext::getSizeType() const {
  return getFromTargetType(Target.getSizeType());
}

/// getSignedWCharType - Return the type of "signed wchar_t".
/// Used when in C++, as a GCC extension.
QualType ASTContext::getSignedWCharType() const {
  // FIXME: derive from "Target" ?
  return WCharTy;
}

/// getUnsignedWCharType - Return the type of "unsigned wchar_t".
/// Used when in C++, as a GCC extension.
QualType ASTContext::getUnsignedWCharType() const {
  // FIXME: derive from "Target" ?
  return UnsignedIntTy;
}

/// getPointerDiffType - Return the unique type for "ptrdiff_t" (ref?)
/// defined in <stddef.h>. Pointer - pointer requires this (C99 6.5.6p9).
QualType ASTContext::getPointerDiffType() const {
  return getFromTargetType(Target.getPtrDiffType(0));
}

//===----------------------------------------------------------------------===//
//                              Type Operators
//===----------------------------------------------------------------------===//

CanQualType ASTContext::getCanonicalParamType(QualType T) {
  // Push qualifiers into arrays, and then discard any remaining
  // qualifiers.
  T = getCanonicalType(T);
  const Type *Ty = T.getTypePtr();

  QualType Result;
  if (isa<ArrayType>(Ty)) {
    Result = getArrayDecayedType(QualType(Ty,0));
  } else if (isa<FunctionType>(Ty)) {
    Result = getPointerType(QualType(Ty, 0));
  } else {
    Result = QualType(Ty, 0);
  }

  return CanQualType::CreateUnsafe(Result);
}

/// getCanonicalType - Return the canonical (structural) type corresponding to
/// the specified potentially non-canonical type.  The non-canonical version
/// of a type may have many "decorated" versions of types.  Decorators can
/// include typedefs, 'typeof' operators, etc. The returned type is guaranteed
/// to be free of any of these, allowing two canonical types to be compared
/// for exact equality with a simple pointer comparison.
CanQualType ASTContext::getCanonicalType(QualType T) {
  QualifierCollector Quals;
  const Type *Ptr = Quals.strip(T);
  QualType CanType = Ptr->getCanonicalTypeInternal();

  // The canonical internal type will be the canonical type *except*
  // that we push type qualifiers down through array types.

  // If there are no new qualifiers to push down, stop here.
  if (!Quals.hasQualifiers())
    return CanQualType::CreateUnsafe(CanType);

  // If the type qualifiers are on an array type, get the canonical
  // type of the array with the qualifiers applied to the element
  // type.
  ArrayType *AT = dyn_cast<ArrayType>(CanType);
  if (!AT)
    return CanQualType::CreateUnsafe(getQualifiedType(CanType, Quals));

  // Get the canonical version of the element with the extra qualifiers on it.
  // This can recursively sink qualifiers through multiple levels of arrays.
  QualType NewEltTy = getQualifiedType(AT->getElementType(), Quals);
  NewEltTy = getCanonicalType(NewEltTy);

  if (ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT))
    return CanQualType::CreateUnsafe(
             getConstantArrayType(NewEltTy, CAT->getSize(),
                                  CAT->getSizeModifier(),
                                  CAT->getIndexTypeCVRQualifiers()));
  if (IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT))
    return CanQualType::CreateUnsafe(
             getIncompleteArrayType(NewEltTy, IAT->getSizeModifier(),
                                    IAT->getIndexTypeCVRQualifiers()));

  if (DependentSizedArrayType *DSAT = dyn_cast<DependentSizedArrayType>(AT))
    return CanQualType::CreateUnsafe(
             getDependentSizedArrayType(NewEltTy,
                                        DSAT->getSizeExpr() ?
                                          DSAT->getSizeExpr()->Retain() : 0,
                                        DSAT->getSizeModifier(),
                                        DSAT->getIndexTypeCVRQualifiers(),
                        DSAT->getBracketsRange())->getCanonicalTypeInternal());

  VariableArrayType *VAT = cast<VariableArrayType>(AT);
  return CanQualType::CreateUnsafe(getVariableArrayType(NewEltTy,
                                                        VAT->getSizeExpr() ?
                                              VAT->getSizeExpr()->Retain() : 0,
                                                        VAT->getSizeModifier(),
                                              VAT->getIndexTypeCVRQualifiers(),
                                                     VAT->getBracketsRange()));
}

QualType ASTContext::getUnqualifiedArrayType(QualType T,
                                             Qualifiers &Quals) {
  Quals = T.getQualifiers();
  if (!isa<ArrayType>(T)) {
    return T.getUnqualifiedType();
  }

  const ArrayType *AT = cast<ArrayType>(T);
  QualType Elt = AT->getElementType();
  QualType UnqualElt = getUnqualifiedArrayType(Elt, Quals);
  if (Elt == UnqualElt)
    return T;

  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(T)) {
    return getConstantArrayType(UnqualElt, CAT->getSize(),
                                CAT->getSizeModifier(), 0);
  }

  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(T)) {
    return getIncompleteArrayType(UnqualElt, IAT->getSizeModifier(), 0);
  }

  const DependentSizedArrayType *DSAT = cast<DependentSizedArrayType>(T);
  return getDependentSizedArrayType(UnqualElt, DSAT->getSizeExpr()->Retain(),
                                    DSAT->getSizeModifier(), 0,
                                    SourceRange());
}

DeclarationName ASTContext::getNameForTemplate(TemplateName Name) {
  if (TemplateDecl *TD = Name.getAsTemplateDecl())
    return TD->getDeclName();
  
  if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    if (DTN->isIdentifier()) {
      return DeclarationNames.getIdentifier(DTN->getIdentifier());
    } else {
      return DeclarationNames.getCXXOperatorName(DTN->getOperator());
    }
  }

  OverloadedTemplateStorage *Storage = Name.getAsOverloadedTemplate();
  assert(Storage);
  return (*Storage->begin())->getDeclName();
}

TemplateName ASTContext::getCanonicalTemplateName(TemplateName Name) {
  // If this template name refers to a template, the canonical
  // template name merely stores the template itself.
  if (TemplateDecl *Template = Name.getAsTemplateDecl())
    return TemplateName(cast<TemplateDecl>(Template->getCanonicalDecl()));

  assert(!Name.getAsOverloadedTemplate());

  DependentTemplateName *DTN = Name.getAsDependentTemplateName();
  assert(DTN && "Non-dependent template names must refer to template decls.");
  return DTN->CanonicalTemplateName;
}

bool ASTContext::hasSameTemplateName(TemplateName X, TemplateName Y) {
  X = getCanonicalTemplateName(X);
  Y = getCanonicalTemplateName(Y);
  return X.getAsVoidPointer() == Y.getAsVoidPointer();
}

TemplateArgument
ASTContext::getCanonicalTemplateArgument(const TemplateArgument &Arg) {
  switch (Arg.getKind()) {
    case TemplateArgument::Null:
      return Arg;

    case TemplateArgument::Expression:
      return Arg;

    case TemplateArgument::Declaration:
      return TemplateArgument(Arg.getAsDecl()->getCanonicalDecl());

    case TemplateArgument::Template:
      return TemplateArgument(getCanonicalTemplateName(Arg.getAsTemplate()));
      
    case TemplateArgument::Integral:
      return TemplateArgument(*Arg.getAsIntegral(),
                              getCanonicalType(Arg.getIntegralType()));

    case TemplateArgument::Type:
      return TemplateArgument(getCanonicalType(Arg.getAsType()));

    case TemplateArgument::Pack: {
      // FIXME: Allocate in ASTContext
      TemplateArgument *CanonArgs = new TemplateArgument[Arg.pack_size()];
      unsigned Idx = 0;
      for (TemplateArgument::pack_iterator A = Arg.pack_begin(),
                                        AEnd = Arg.pack_end();
           A != AEnd; (void)++A, ++Idx)
        CanonArgs[Idx] = getCanonicalTemplateArgument(*A);

      TemplateArgument Result;
      Result.setArgumentPack(CanonArgs, Arg.pack_size(), false);
      return Result;
    }
  }

  // Silence GCC warning
  assert(false && "Unhandled template argument kind");
  return TemplateArgument();
}

NestedNameSpecifier *
ASTContext::getCanonicalNestedNameSpecifier(NestedNameSpecifier *NNS) {
  if (!NNS)
    return 0;

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    // Canonicalize the prefix but keep the identifier the same.
    return NestedNameSpecifier::Create(*this,
                         getCanonicalNestedNameSpecifier(NNS->getPrefix()),
                                       NNS->getAsIdentifier());

  case NestedNameSpecifier::Namespace:
    // A namespace is canonical; build a nested-name-specifier with
    // this namespace and no prefix.
    return NestedNameSpecifier::Create(*this, 0, NNS->getAsNamespace());

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    QualType T = getCanonicalType(QualType(NNS->getAsType(), 0));
    return NestedNameSpecifier::Create(*this, 0,
                 NNS->getKind() == NestedNameSpecifier::TypeSpecWithTemplate,
                                       T.getTypePtr());
  }

  case NestedNameSpecifier::Global:
    // The global specifier is canonical and unique.
    return NNS;
  }

  // Required to silence a GCC warning
  return 0;
}


const ArrayType *ASTContext::getAsArrayType(QualType T) {
  // Handle the non-qualified case efficiently.
  if (!T.hasLocalQualifiers()) {
    // Handle the common positive case fast.
    if (const ArrayType *AT = dyn_cast<ArrayType>(T))
      return AT;
  }

  // Handle the common negative case fast.
  QualType CType = T->getCanonicalTypeInternal();
  if (!isa<ArrayType>(CType))
    return 0;

  // Apply any qualifiers from the array type to the element type.  This
  // implements C99 6.7.3p8: "If the specification of an array type includes
  // any type qualifiers, the element type is so qualified, not the array type."

  // If we get here, we either have type qualifiers on the type, or we have
  // sugar such as a typedef in the way.  If we have type qualifiers on the type
  // we must propagate them down into the element type.

  QualifierCollector Qs;
  const Type *Ty = Qs.strip(T.getDesugaredType());

  // If we have a simple case, just return now.
  const ArrayType *ATy = dyn_cast<ArrayType>(Ty);
  if (ATy == 0 || Qs.empty())
    return ATy;

  // Otherwise, we have an array and we have qualifiers on it.  Push the
  // qualifiers into the array element type and return a new array type.
  // Get the canonical version of the element with the extra qualifiers on it.
  // This can recursively sink qualifiers through multiple levels of arrays.
  QualType NewEltTy = getQualifiedType(ATy->getElementType(), Qs);

  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(ATy))
    return cast<ArrayType>(getConstantArrayType(NewEltTy, CAT->getSize(),
                                                CAT->getSizeModifier(),
                                           CAT->getIndexTypeCVRQualifiers()));
  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(ATy))
    return cast<ArrayType>(getIncompleteArrayType(NewEltTy,
                                                  IAT->getSizeModifier(),
                                           IAT->getIndexTypeCVRQualifiers()));

  if (const DependentSizedArrayType *DSAT
        = dyn_cast<DependentSizedArrayType>(ATy))
    return cast<ArrayType>(
                     getDependentSizedArrayType(NewEltTy,
                                                DSAT->getSizeExpr() ?
                                              DSAT->getSizeExpr()->Retain() : 0,
                                                DSAT->getSizeModifier(),
                                              DSAT->getIndexTypeCVRQualifiers(),
                                                DSAT->getBracketsRange()));

  const VariableArrayType *VAT = cast<VariableArrayType>(ATy);
  return cast<ArrayType>(getVariableArrayType(NewEltTy,
                                              VAT->getSizeExpr() ?
                                              VAT->getSizeExpr()->Retain() : 0,
                                              VAT->getSizeModifier(),
                                              VAT->getIndexTypeCVRQualifiers(),
                                              VAT->getBracketsRange()));
}


/// getArrayDecayedType - Return the properly qualified result of decaying the
/// specified array type to a pointer.  This operation is non-trivial when
/// handling typedefs etc.  The canonical type of "T" must be an array type,
/// this returns a pointer to a properly qualified element of the array.
///
/// See C99 6.7.5.3p7 and C99 6.3.2.1p3.
QualType ASTContext::getArrayDecayedType(QualType Ty) {
  // Get the element type with 'getAsArrayType' so that we don't lose any
  // typedefs in the element type of the array.  This also handles propagation
  // of type qualifiers from the array type into the element type if present
  // (C99 6.7.3p8).
  const ArrayType *PrettyArrayType = getAsArrayType(Ty);
  assert(PrettyArrayType && "Not an array type!");

  QualType PtrTy = getPointerType(PrettyArrayType->getElementType());

  // int x[restrict 4] ->  int *restrict
  return getQualifiedType(PtrTy, PrettyArrayType->getIndexTypeQualifiers());
}

QualType ASTContext::getBaseElementType(QualType QT) {
  QualifierCollector Qs;
  while (const ArrayType *AT = getAsArrayType(QualType(Qs.strip(QT), 0)))
    QT = AT->getElementType();
  return Qs.apply(QT);
}

QualType ASTContext::getBaseElementType(const ArrayType *AT) {
  QualType ElemTy = AT->getElementType();

  if (const ArrayType *AT = getAsArrayType(ElemTy))
    return getBaseElementType(AT);

  return ElemTy;
}

/// getConstantArrayElementCount - Returns number of constant array elements.
uint64_t
ASTContext::getConstantArrayElementCount(const ConstantArrayType *CA)  const {
  uint64_t ElementCount = 1;
  do {
    ElementCount *= CA->getSize().getZExtValue();
    CA = dyn_cast<ConstantArrayType>(CA->getElementType());
  } while (CA);
  return ElementCount;
}

/// getFloatingRank - Return a relative rank for floating point types.
/// This routine will assert if passed a built-in type that isn't a float.
static FloatingRank getFloatingRank(QualType T) {
  if (const ComplexType *CT = T->getAs<ComplexType>())
    return getFloatingRank(CT->getElementType());

  assert(T->getAs<BuiltinType>() && "getFloatingRank(): not a floating type");
  switch (T->getAs<BuiltinType>()->getKind()) {
  default: assert(0 && "getFloatingRank(): not a floating type");
  case BuiltinType::Float:      return FloatRank;
  case BuiltinType::Double:     return DoubleRank;
  case BuiltinType::LongDouble: return LongDoubleRank;
  }
}

/// getFloatingTypeOfSizeWithinDomain - Returns a real floating
/// point or a complex type (based on typeDomain/typeSize).
/// 'typeDomain' is a real floating point or complex type.
/// 'typeSize' is a real floating point or complex type.
QualType ASTContext::getFloatingTypeOfSizeWithinDomain(QualType Size,
                                                       QualType Domain) const {
  FloatingRank EltRank = getFloatingRank(Size);
  if (Domain->isComplexType()) {
    switch (EltRank) {
    default: assert(0 && "getFloatingRank(): illegal value for rank");
    case FloatRank:      return FloatComplexTy;
    case DoubleRank:     return DoubleComplexTy;
    case LongDoubleRank: return LongDoubleComplexTy;
    }
  }

  assert(Domain->isRealFloatingType() && "Unknown domain!");
  switch (EltRank) {
  default: assert(0 && "getFloatingRank(): illegal value for rank");
  case FloatRank:      return FloatTy;
  case DoubleRank:     return DoubleTy;
  case LongDoubleRank: return LongDoubleTy;
  }
}

/// getFloatingTypeOrder - Compare the rank of the two specified floating
/// point types, ignoring the domain of the type (i.e. 'double' ==
/// '_Complex double').  If LHS > RHS, return 1.  If LHS == RHS, return 0. If
/// LHS < RHS, return -1.
int ASTContext::getFloatingTypeOrder(QualType LHS, QualType RHS) {
  FloatingRank LHSR = getFloatingRank(LHS);
  FloatingRank RHSR = getFloatingRank(RHS);

  if (LHSR == RHSR)
    return 0;
  if (LHSR > RHSR)
    return 1;
  return -1;
}

/// getIntegerRank - Return an integer conversion rank (C99 6.3.1.1p1). This
/// routine will assert if passed a built-in type that isn't an integer or enum,
/// or if it is not canonicalized.
unsigned ASTContext::getIntegerRank(Type *T) {
  assert(T->isCanonicalUnqualified() && "T should be canonicalized");
  if (EnumType* ET = dyn_cast<EnumType>(T))
    T = ET->getDecl()->getPromotionType().getTypePtr();

  if (T->isSpecificBuiltinType(BuiltinType::WChar))
    T = getFromTargetType(Target.getWCharType()).getTypePtr();

  if (T->isSpecificBuiltinType(BuiltinType::Char16))
    T = getFromTargetType(Target.getChar16Type()).getTypePtr();

  if (T->isSpecificBuiltinType(BuiltinType::Char32))
    T = getFromTargetType(Target.getChar32Type()).getTypePtr();

  switch (cast<BuiltinType>(T)->getKind()) {
  default: assert(0 && "getIntegerRank(): not a built-in integer");
  case BuiltinType::Bool:
    return 1 + (getIntWidth(BoolTy) << 3);
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
    return 2 + (getIntWidth(CharTy) << 3);
  case BuiltinType::Short:
  case BuiltinType::UShort:
    return 3 + (getIntWidth(ShortTy) << 3);
  case BuiltinType::Int:
  case BuiltinType::UInt:
    return 4 + (getIntWidth(IntTy) << 3);
  case BuiltinType::Long:
  case BuiltinType::ULong:
    return 5 + (getIntWidth(LongTy) << 3);
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    return 6 + (getIntWidth(LongLongTy) << 3);
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return 7 + (getIntWidth(Int128Ty) << 3);
  }
}

/// \brief Whether this is a promotable bitfield reference according
/// to C99 6.3.1.1p2, bullet 2 (and GCC extensions).
///
/// \returns the type this bit-field will promote to, or NULL if no
/// promotion occurs.
QualType ASTContext::isPromotableBitField(Expr *E) {
  FieldDecl *Field = E->getBitField();
  if (!Field)
    return QualType();

  QualType FT = Field->getType();

  llvm::APSInt BitWidthAP = Field->getBitWidth()->EvaluateAsInt(*this);
  uint64_t BitWidth = BitWidthAP.getZExtValue();
  uint64_t IntSize = getTypeSize(IntTy);
  // GCC extension compatibility: if the bit-field size is less than or equal
  // to the size of int, it gets promoted no matter what its type is.
  // For instance, unsigned long bf : 4 gets promoted to signed int.
  if (BitWidth < IntSize)
    return IntTy;

  if (BitWidth == IntSize)
    return FT->isSignedIntegerType() ? IntTy : UnsignedIntTy;

  // Types bigger than int are not subject to promotions, and therefore act
  // like the base type.
  // FIXME: This doesn't quite match what gcc does, but what gcc does here
  // is ridiculous.
  return QualType();
}

/// getPromotedIntegerType - Returns the type that Promotable will
/// promote to: C99 6.3.1.1p2, assuming that Promotable is a promotable
/// integer type.
QualType ASTContext::getPromotedIntegerType(QualType Promotable) {
  assert(!Promotable.isNull());
  assert(Promotable->isPromotableIntegerType());
  if (const EnumType *ET = Promotable->getAs<EnumType>())
    return ET->getDecl()->getPromotionType();
  if (Promotable->isSignedIntegerType())
    return IntTy;
  uint64_t PromotableSize = getTypeSize(Promotable);
  uint64_t IntSize = getTypeSize(IntTy);
  assert(Promotable->isUnsignedIntegerType() && PromotableSize <= IntSize);
  return (PromotableSize != IntSize) ? IntTy : UnsignedIntTy;
}

/// getIntegerTypeOrder - Returns the highest ranked integer type:
/// C99 6.3.1.8p1.  If LHS > RHS, return 1.  If LHS == RHS, return 0. If
/// LHS < RHS, return -1.
int ASTContext::getIntegerTypeOrder(QualType LHS, QualType RHS) {
  Type *LHSC = getCanonicalType(LHS).getTypePtr();
  Type *RHSC = getCanonicalType(RHS).getTypePtr();
  if (LHSC == RHSC) return 0;

  bool LHSUnsigned = LHSC->isUnsignedIntegerType();
  bool RHSUnsigned = RHSC->isUnsignedIntegerType();

  unsigned LHSRank = getIntegerRank(LHSC);
  unsigned RHSRank = getIntegerRank(RHSC);

  if (LHSUnsigned == RHSUnsigned) {  // Both signed or both unsigned.
    if (LHSRank == RHSRank) return 0;
    return LHSRank > RHSRank ? 1 : -1;
  }

  // Otherwise, the LHS is signed and the RHS is unsigned or visa versa.
  if (LHSUnsigned) {
    // If the unsigned [LHS] type is larger, return it.
    if (LHSRank >= RHSRank)
      return 1;

    // If the signed type can represent all values of the unsigned type, it
    // wins.  Because we are dealing with 2's complement and types that are
    // powers of two larger than each other, this is always safe.
    return -1;
  }

  // If the unsigned [RHS] type is larger, return it.
  if (RHSRank >= LHSRank)
    return -1;

  // If the signed type can represent all values of the unsigned type, it
  // wins.  Because we are dealing with 2's complement and types that are
  // powers of two larger than each other, this is always safe.
  return 1;
}

static RecordDecl *
CreateRecordDecl(ASTContext &Ctx, RecordDecl::TagKind TK, DeclContext *DC,
                 SourceLocation L, IdentifierInfo *Id) {
  if (Ctx.getLangOptions().CPlusPlus)
    return CXXRecordDecl::Create(Ctx, TK, DC, L, Id);
  else
    return RecordDecl::Create(Ctx, TK, DC, L, Id);
}
                                    
// getCFConstantStringType - Return the type used for constant CFStrings.
QualType ASTContext::getCFConstantStringType() {
  if (!CFConstantStringTypeDecl) {
    CFConstantStringTypeDecl =
      CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                       &Idents.get("NSConstantString"));
    CFConstantStringTypeDecl->startDefinition();

    QualType FieldTypes[4];

    // const int *isa;
    FieldTypes[0] = getPointerType(IntTy.withConst());
    // int flags;
    FieldTypes[1] = IntTy;
    // const char *str;
    FieldTypes[2] = getPointerType(CharTy.withConst());
    // long length;
    FieldTypes[3] = LongTy;

    // Create fields
    for (unsigned i = 0; i < 4; ++i) {
      FieldDecl *Field = FieldDecl::Create(*this, CFConstantStringTypeDecl,
                                           SourceLocation(), 0,
                                           FieldTypes[i], /*TInfo=*/0,
                                           /*BitWidth=*/0,
                                           /*Mutable=*/false);
      Field->setAccess(AS_public);
      CFConstantStringTypeDecl->addDecl(Field);
    }

    CFConstantStringTypeDecl->completeDefinition();
  }

  return getTagDeclType(CFConstantStringTypeDecl);
}

void ASTContext::setCFConstantStringType(QualType T) {
  const RecordType *Rec = T->getAs<RecordType>();
  assert(Rec && "Invalid CFConstantStringType");
  CFConstantStringTypeDecl = Rec->getDecl();
}

// getNSConstantStringType - Return the type used for constant NSStrings.
QualType ASTContext::getNSConstantStringType() {
  if (!NSConstantStringTypeDecl) {
    NSConstantStringTypeDecl =
    CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                     &Idents.get("__builtin_NSString"));
    NSConstantStringTypeDecl->startDefinition();
    
    QualType FieldTypes[3];
    
    // const int *isa;
    FieldTypes[0] = getPointerType(IntTy.withConst());
    // const char *str;
    FieldTypes[1] = getPointerType(CharTy.withConst());
    // unsigned int length;
    FieldTypes[2] = UnsignedIntTy;
    
    // Create fields
    for (unsigned i = 0; i < 3; ++i) {
      FieldDecl *Field = FieldDecl::Create(*this, NSConstantStringTypeDecl,
                                           SourceLocation(), 0,
                                           FieldTypes[i], /*TInfo=*/0,
                                           /*BitWidth=*/0,
                                           /*Mutable=*/false);
      Field->setAccess(AS_public);
      NSConstantStringTypeDecl->addDecl(Field);
    }
    
    NSConstantStringTypeDecl->completeDefinition();
  }
  
  return getTagDeclType(NSConstantStringTypeDecl);
}

void ASTContext::setNSConstantStringType(QualType T) {
  const RecordType *Rec = T->getAs<RecordType>();
  assert(Rec && "Invalid NSConstantStringType");
  NSConstantStringTypeDecl = Rec->getDecl();
}

QualType ASTContext::getObjCFastEnumerationStateType() {
  if (!ObjCFastEnumerationStateTypeDecl) {
    ObjCFastEnumerationStateTypeDecl =
      CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                       &Idents.get("__objcFastEnumerationState"));
    ObjCFastEnumerationStateTypeDecl->startDefinition();

    QualType FieldTypes[] = {
      UnsignedLongTy,
      getPointerType(ObjCIdTypedefType),
      getPointerType(UnsignedLongTy),
      getConstantArrayType(UnsignedLongTy,
                           llvm::APInt(32, 5), ArrayType::Normal, 0)
    };

    for (size_t i = 0; i < 4; ++i) {
      FieldDecl *Field = FieldDecl::Create(*this,
                                           ObjCFastEnumerationStateTypeDecl,
                                           SourceLocation(), 0,
                                           FieldTypes[i], /*TInfo=*/0,
                                           /*BitWidth=*/0,
                                           /*Mutable=*/false);
      Field->setAccess(AS_public);
      ObjCFastEnumerationStateTypeDecl->addDecl(Field);
    }

    ObjCFastEnumerationStateTypeDecl->completeDefinition();
  }

  return getTagDeclType(ObjCFastEnumerationStateTypeDecl);
}

QualType ASTContext::getBlockDescriptorType() {
  if (BlockDescriptorType)
    return getTagDeclType(BlockDescriptorType);

  RecordDecl *T;
  // FIXME: Needs the FlagAppleBlock bit.
  T = CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                       &Idents.get("__block_descriptor"));
  T->startDefinition();
  
  QualType FieldTypes[] = {
    UnsignedLongTy,
    UnsignedLongTy,
  };

  const char *FieldNames[] = {
    "reserved",
    "Size"
  };

  for (size_t i = 0; i < 2; ++i) {
    FieldDecl *Field = FieldDecl::Create(*this,
                                         T,
                                         SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0,
                                         /*Mutable=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  BlockDescriptorType = T;

  return getTagDeclType(BlockDescriptorType);
}

void ASTContext::setBlockDescriptorType(QualType T) {
  const RecordType *Rec = T->getAs<RecordType>();
  assert(Rec && "Invalid BlockDescriptorType");
  BlockDescriptorType = Rec->getDecl();
}

QualType ASTContext::getBlockDescriptorExtendedType() {
  if (BlockDescriptorExtendedType)
    return getTagDeclType(BlockDescriptorExtendedType);

  RecordDecl *T;
  // FIXME: Needs the FlagAppleBlock bit.
  T = CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                       &Idents.get("__block_descriptor_withcopydispose"));
  T->startDefinition();
  
  QualType FieldTypes[] = {
    UnsignedLongTy,
    UnsignedLongTy,
    getPointerType(VoidPtrTy),
    getPointerType(VoidPtrTy)
  };

  const char *FieldNames[] = {
    "reserved",
    "Size",
    "CopyFuncPtr",
    "DestroyFuncPtr"
  };

  for (size_t i = 0; i < 4; ++i) {
    FieldDecl *Field = FieldDecl::Create(*this,
                                         T,
                                         SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0,
                                         /*Mutable=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  BlockDescriptorExtendedType = T;

  return getTagDeclType(BlockDescriptorExtendedType);
}

void ASTContext::setBlockDescriptorExtendedType(QualType T) {
  const RecordType *Rec = T->getAs<RecordType>();
  assert(Rec && "Invalid BlockDescriptorType");
  BlockDescriptorExtendedType = Rec->getDecl();
}

bool ASTContext::BlockRequiresCopying(QualType Ty) {
  if (Ty->isBlockPointerType())
    return true;
  if (isObjCNSObjectType(Ty))
    return true;
  if (Ty->isObjCObjectPointerType())
    return true;
  return false;
}

QualType ASTContext::BuildByRefType(const char *DeclName, QualType Ty) {
  //  type = struct __Block_byref_1_X {
  //    void *__isa;
  //    struct __Block_byref_1_X *__forwarding;
  //    unsigned int __flags;
  //    unsigned int __size;
  //    void *__copy_helper;		// as needed
  //    void *__destroy_help		// as needed
  //    int X;
  //  } *

  bool HasCopyAndDispose = BlockRequiresCopying(Ty);

  // FIXME: Move up
  static unsigned int UniqueBlockByRefTypeID = 0;
  llvm::SmallString<36> Name;
  llvm::raw_svector_ostream(Name) << "__Block_byref_" <<
                                  ++UniqueBlockByRefTypeID << '_' << DeclName;
  RecordDecl *T;
  T = CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                       &Idents.get(Name.str()));
  T->startDefinition();
  QualType Int32Ty = IntTy;
  assert(getIntWidth(IntTy) == 32 && "non-32bit int not supported");
  QualType FieldTypes[] = {
    getPointerType(VoidPtrTy),
    getPointerType(getTagDeclType(T)),
    Int32Ty,
    Int32Ty,
    getPointerType(VoidPtrTy),
    getPointerType(VoidPtrTy),
    Ty
  };

  const char *FieldNames[] = {
    "__isa",
    "__forwarding",
    "__flags",
    "__size",
    "__copy_helper",
    "__destroy_helper",
    DeclName,
  };

  for (size_t i = 0; i < 7; ++i) {
    if (!HasCopyAndDispose && i >=4 && i <= 5)
      continue;
    FieldDecl *Field = FieldDecl::Create(*this, T, SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0, /*Mutable=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  return getPointerType(getTagDeclType(T));
}


QualType ASTContext::getBlockParmType(
  bool BlockHasCopyDispose,
  llvm::SmallVector<const Expr *, 8> &BlockDeclRefDecls) {
  // FIXME: Move up
  static unsigned int UniqueBlockParmTypeID = 0;
  llvm::SmallString<36> Name;
  llvm::raw_svector_ostream(Name) << "__block_literal_"
                                  << ++UniqueBlockParmTypeID;
  RecordDecl *T;
  T = CreateRecordDecl(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                       &Idents.get(Name.str()));
  T->startDefinition();
  QualType FieldTypes[] = {
    getPointerType(VoidPtrTy),
    IntTy,
    IntTy,
    getPointerType(VoidPtrTy),
    (BlockHasCopyDispose ?
     getPointerType(getBlockDescriptorExtendedType()) :
     getPointerType(getBlockDescriptorType()))
  };

  const char *FieldNames[] = {
    "__isa",
    "__flags",
    "__reserved",
    "__FuncPtr",
    "__descriptor"
  };

  for (size_t i = 0; i < 5; ++i) {
    FieldDecl *Field = FieldDecl::Create(*this, T, SourceLocation(),
                                         &Idents.get(FieldNames[i]),
                                         FieldTypes[i], /*TInfo=*/0,
                                         /*BitWidth=*/0, /*Mutable=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  for (size_t i = 0; i < BlockDeclRefDecls.size(); ++i) {
    const Expr *E = BlockDeclRefDecls[i];
    const BlockDeclRefExpr *BDRE = dyn_cast<BlockDeclRefExpr>(E);
    clang::IdentifierInfo *Name = 0;
    if (BDRE) {
      const ValueDecl *D = BDRE->getDecl();
      Name = &Idents.get(D->getName());
    }
    QualType FieldType = E->getType();

    if (BDRE && BDRE->isByRef())
      FieldType = BuildByRefType(BDRE->getDecl()->getNameAsCString(),
                                 FieldType);

    FieldDecl *Field = FieldDecl::Create(*this, T, SourceLocation(),
                                         Name, FieldType, /*TInfo=*/0,
                                         /*BitWidth=*/0, /*Mutable=*/false);
    Field->setAccess(AS_public);
    T->addDecl(Field);
  }

  T->completeDefinition();

  return getPointerType(getTagDeclType(T));
}

void ASTContext::setObjCFastEnumerationStateType(QualType T) {
  const RecordType *Rec = T->getAs<RecordType>();
  assert(Rec && "Invalid ObjCFAstEnumerationStateType");
  ObjCFastEnumerationStateTypeDecl = Rec->getDecl();
}

// This returns true if a type has been typedefed to BOOL:
// typedef <type> BOOL;
static bool isTypeTypedefedAsBOOL(QualType T) {
  if (const TypedefType *TT = dyn_cast<TypedefType>(T))
    if (IdentifierInfo *II = TT->getDecl()->getIdentifier())
      return II->isStr("BOOL");

  return false;
}

/// getObjCEncodingTypeSize returns size of type for objective-c encoding
/// purpose.
CharUnits ASTContext::getObjCEncodingTypeSize(QualType type) {
  CharUnits sz = getTypeSizeInChars(type);

  // Make all integer and enum types at least as large as an int
  if (sz.isPositive() && type->isIntegralType())
    sz = std::max(sz, getTypeSizeInChars(IntTy));
  // Treat arrays as pointers, since that's how they're passed in.
  else if (type->isArrayType())
    sz = getTypeSizeInChars(VoidPtrTy);
  return sz;
}

static inline 
std::string charUnitsToString(const CharUnits &CU) {
  return llvm::itostr(CU.getQuantity());
}

/// getObjCEncodingForBlockDecl - Return the encoded type for this block
/// declaration.
void ASTContext::getObjCEncodingForBlock(const BlockExpr *Expr, 
                                             std::string& S) {
  const BlockDecl *Decl = Expr->getBlockDecl();
  QualType BlockTy =
      Expr->getType()->getAs<BlockPointerType>()->getPointeeType();
  // Encode result type.
  getObjCEncodingForType(cast<FunctionType>(BlockTy)->getResultType(), S);
  // Compute size of all parameters.
  // Start with computing size of a pointer in number of bytes.
  // FIXME: There might(should) be a better way of doing this computation!
  SourceLocation Loc;
  CharUnits PtrSize = getTypeSizeInChars(VoidPtrTy);
  CharUnits ParmOffset = PtrSize;
  for (BlockDecl::param_const_iterator PI = Decl->param_begin(),
       E = Decl->param_end(); PI != E; ++PI) {
    QualType PType = (*PI)->getType();
    CharUnits sz = getObjCEncodingTypeSize(PType);
    assert (sz.isPositive() && "BlockExpr - Incomplete param type");
    ParmOffset += sz;
  }
  // Size of the argument frame
  S += charUnitsToString(ParmOffset);
  // Block pointer and offset.
  S += "@?0";
  ParmOffset = PtrSize;
  
  // Argument types.
  ParmOffset = PtrSize;
  for (BlockDecl::param_const_iterator PI = Decl->param_begin(), E =
       Decl->param_end(); PI != E; ++PI) {
    ParmVarDecl *PVDecl = *PI;
    QualType PType = PVDecl->getOriginalType(); 
    if (const ArrayType *AT =
          dyn_cast<ArrayType>(PType->getCanonicalTypeInternal())) {
      // Use array's original type only if it has known number of
      // elements.
      if (!isa<ConstantArrayType>(AT))
        PType = PVDecl->getType();
    } else if (PType->isFunctionType())
      PType = PVDecl->getType();
    getObjCEncodingForType(PType, S);
    S += charUnitsToString(ParmOffset);
    ParmOffset += getObjCEncodingTypeSize(PType);
  }
}

/// getObjCEncodingForMethodDecl - Return the encoded type for this method
/// declaration.
void ASTContext::getObjCEncodingForMethodDecl(const ObjCMethodDecl *Decl,
                                              std::string& S) {
  // FIXME: This is not very efficient.
  // Encode type qualifer, 'in', 'inout', etc. for the return type.
  getObjCEncodingForTypeQualifier(Decl->getObjCDeclQualifier(), S);
  // Encode result type.
  getObjCEncodingForType(Decl->getResultType(), S);
  // Compute size of all parameters.
  // Start with computing size of a pointer in number of bytes.
  // FIXME: There might(should) be a better way of doing this computation!
  SourceLocation Loc;
  CharUnits PtrSize = getTypeSizeInChars(VoidPtrTy);
  // The first two arguments (self and _cmd) are pointers; account for
  // their size.
  CharUnits ParmOffset = 2 * PtrSize;
  for (ObjCMethodDecl::param_iterator PI = Decl->param_begin(),
       E = Decl->sel_param_end(); PI != E; ++PI) {
    QualType PType = (*PI)->getType();
    CharUnits sz = getObjCEncodingTypeSize(PType);
    assert (sz.isPositive() && 
        "getObjCEncodingForMethodDecl - Incomplete param type");
    ParmOffset += sz;
  }
  S += charUnitsToString(ParmOffset);
  S += "@0:";
  S += charUnitsToString(PtrSize);

  // Argument types.
  ParmOffset = 2 * PtrSize;
  for (ObjCMethodDecl::param_iterator PI = Decl->param_begin(),
       E = Decl->sel_param_end(); PI != E; ++PI) {
    ParmVarDecl *PVDecl = *PI;
    QualType PType = PVDecl->getOriginalType();
    if (const ArrayType *AT =
          dyn_cast<ArrayType>(PType->getCanonicalTypeInternal())) {
      // Use array's original type only if it has known number of
      // elements.
      if (!isa<ConstantArrayType>(AT))
        PType = PVDecl->getType();
    } else if (PType->isFunctionType())
      PType = PVDecl->getType();
    // Process argument qualifiers for user supplied arguments; such as,
    // 'in', 'inout', etc.
    getObjCEncodingForTypeQualifier(PVDecl->getObjCDeclQualifier(), S);
    getObjCEncodingForType(PType, S);
    S += charUnitsToString(ParmOffset);
    ParmOffset += getObjCEncodingTypeSize(PType);
  }
}

/// getObjCEncodingForPropertyDecl - Return the encoded type for this
/// property declaration. If non-NULL, Container must be either an
/// ObjCCategoryImplDecl or ObjCImplementationDecl; it should only be
/// NULL when getting encodings for protocol properties.
/// Property attributes are stored as a comma-delimited C string. The simple
/// attributes readonly and bycopy are encoded as single characters. The
/// parametrized attributes, getter=name, setter=name, and ivar=name, are
/// encoded as single characters, followed by an identifier. Property types
/// are also encoded as a parametrized attribute. The characters used to encode
/// these attributes are defined by the following enumeration:
/// @code
/// enum PropertyAttributes {
/// kPropertyReadOnly = 'R',   // property is read-only.
/// kPropertyBycopy = 'C',     // property is a copy of the value last assigned
/// kPropertyByref = '&',  // property is a reference to the value last assigned
/// kPropertyDynamic = 'D',    // property is dynamic
/// kPropertyGetter = 'G',     // followed by getter selector name
/// kPropertySetter = 'S',     // followed by setter selector name
/// kPropertyInstanceVariable = 'V'  // followed by instance variable  name
/// kPropertyType = 't'              // followed by old-style type encoding.
/// kPropertyWeak = 'W'              // 'weak' property
/// kPropertyStrong = 'P'            // property GC'able
/// kPropertyNonAtomic = 'N'         // property non-atomic
/// };
/// @endcode
void ASTContext::getObjCEncodingForPropertyDecl(const ObjCPropertyDecl *PD,
                                                const Decl *Container,
                                                std::string& S) {
  // Collect information from the property implementation decl(s).
  bool Dynamic = false;
  ObjCPropertyImplDecl *SynthesizePID = 0;

  // FIXME: Duplicated code due to poor abstraction.
  if (Container) {
    if (const ObjCCategoryImplDecl *CID =
        dyn_cast<ObjCCategoryImplDecl>(Container)) {
      for (ObjCCategoryImplDecl::propimpl_iterator
             i = CID->propimpl_begin(), e = CID->propimpl_end();
           i != e; ++i) {
        ObjCPropertyImplDecl *PID = *i;
        if (PID->getPropertyDecl() == PD) {
          if (PID->getPropertyImplementation()==ObjCPropertyImplDecl::Dynamic) {
            Dynamic = true;
          } else {
            SynthesizePID = PID;
          }
        }
      }
    } else {
      const ObjCImplementationDecl *OID=cast<ObjCImplementationDecl>(Container);
      for (ObjCCategoryImplDecl::propimpl_iterator
             i = OID->propimpl_begin(), e = OID->propimpl_end();
           i != e; ++i) {
        ObjCPropertyImplDecl *PID = *i;
        if (PID->getPropertyDecl() == PD) {
          if (PID->getPropertyImplementation()==ObjCPropertyImplDecl::Dynamic) {
            Dynamic = true;
          } else {
            SynthesizePID = PID;
          }
        }
      }
    }
  }

  // FIXME: This is not very efficient.
  S = "T";

  // Encode result type.
  // GCC has some special rules regarding encoding of properties which
  // closely resembles encoding of ivars.
  getObjCEncodingForTypeImpl(PD->getType(), S, true, true, 0,
                             true /* outermost type */,
                             true /* encoding for property */);

  if (PD->isReadOnly()) {
    S += ",R";
  } else {
    switch (PD->getSetterKind()) {
    case ObjCPropertyDecl::Assign: break;
    case ObjCPropertyDecl::Copy:   S += ",C"; break;
    case ObjCPropertyDecl::Retain: S += ",&"; break;
    }
  }

  // It really isn't clear at all what this means, since properties
  // are "dynamic by default".
  if (Dynamic)
    S += ",D";

  if (PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_nonatomic)
    S += ",N";

  if (PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_getter) {
    S += ",G";
    S += PD->getGetterName().getAsString();
  }

  if (PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_setter) {
    S += ",S";
    S += PD->getSetterName().getAsString();
  }

  if (SynthesizePID) {
    const ObjCIvarDecl *OID = SynthesizePID->getPropertyIvarDecl();
    S += ",V";
    S += OID->getNameAsString();
  }

  // FIXME: OBJCGC: weak & strong
}

/// getLegacyIntegralTypeEncoding -
/// Another legacy compatibility encoding: 32-bit longs are encoded as
/// 'l' or 'L' , but not always.  For typedefs, we need to use
/// 'i' or 'I' instead if encoding a struct field, or a pointer!
///
void ASTContext::getLegacyIntegralTypeEncoding (QualType &PointeeTy) const {
  if (isa<TypedefType>(PointeeTy.getTypePtr())) {
    if (const BuiltinType *BT = PointeeTy->getAs<BuiltinType>()) {
      if (BT->getKind() == BuiltinType::ULong &&
          ((const_cast<ASTContext *>(this))->getIntWidth(PointeeTy) == 32))
        PointeeTy = UnsignedIntTy;
      else
        if (BT->getKind() == BuiltinType::Long &&
            ((const_cast<ASTContext *>(this))->getIntWidth(PointeeTy) == 32))
          PointeeTy = IntTy;
    }
  }
}

void ASTContext::getObjCEncodingForType(QualType T, std::string& S,
                                        const FieldDecl *Field) {
  // We follow the behavior of gcc, expanding structures which are
  // directly pointed to, and expanding embedded structures. Note that
  // these rules are sufficient to prevent recursive encoding of the
  // same type.
  getObjCEncodingForTypeImpl(T, S, true, true, Field,
                             true /* outermost type */);
}

static void EncodeBitField(const ASTContext *Context, std::string& S,
                           const FieldDecl *FD) {
  const Expr *E = FD->getBitWidth();
  assert(E && "bitfield width not there - getObjCEncodingForTypeImpl");
  ASTContext *Ctx = const_cast<ASTContext*>(Context);
  unsigned N = E->EvaluateAsInt(*Ctx).getZExtValue();
  S += 'b';
  S += llvm::utostr(N);
}

// FIXME: Use SmallString for accumulating string.
void ASTContext::getObjCEncodingForTypeImpl(QualType T, std::string& S,
                                            bool ExpandPointedToStructures,
                                            bool ExpandStructures,
                                            const FieldDecl *FD,
                                            bool OutermostType,
                                            bool EncodingProperty) {
  if (const BuiltinType *BT = T->getAs<BuiltinType>()) {
    if (FD && FD->isBitField())
      return EncodeBitField(this, S, FD);
    char encoding;
    switch (BT->getKind()) {
    default: assert(0 && "Unhandled builtin type kind");
    case BuiltinType::Void:       encoding = 'v'; break;
    case BuiltinType::Bool:       encoding = 'B'; break;
    case BuiltinType::Char_U:
    case BuiltinType::UChar:      encoding = 'C'; break;
    case BuiltinType::UShort:     encoding = 'S'; break;
    case BuiltinType::UInt:       encoding = 'I'; break;
    case BuiltinType::ULong:
        encoding =
          (const_cast<ASTContext *>(this))->getIntWidth(T) == 32 ? 'L' : 'Q';
        break;
    case BuiltinType::UInt128:    encoding = 'T'; break;
    case BuiltinType::ULongLong:  encoding = 'Q'; break;
    case BuiltinType::Char_S:
    case BuiltinType::SChar:      encoding = 'c'; break;
    case BuiltinType::Short:      encoding = 's'; break;
    case BuiltinType::Int:        encoding = 'i'; break;
    case BuiltinType::Long:
      encoding =
        (const_cast<ASTContext *>(this))->getIntWidth(T) == 32 ? 'l' : 'q';
      break;
    case BuiltinType::LongLong:   encoding = 'q'; break;
    case BuiltinType::Int128:     encoding = 't'; break;
    case BuiltinType::Float:      encoding = 'f'; break;
    case BuiltinType::Double:     encoding = 'd'; break;
    case BuiltinType::LongDouble: encoding = 'd'; break;
    }

    S += encoding;
    return;
  }

  if (const ComplexType *CT = T->getAs<ComplexType>()) {
    S += 'j';
    getObjCEncodingForTypeImpl(CT->getElementType(), S, false, false, 0, false,
                               false);
    return;
  }
  
  // encoding for pointer or r3eference types.
  QualType PointeeTy;
  if (const PointerType *PT = T->getAs<PointerType>()) {
    if (PT->isObjCSelType()) {
      S += ':';
      return;
    }
    PointeeTy = PT->getPointeeType();
  }
  else if (const ReferenceType *RT = T->getAs<ReferenceType>())
    PointeeTy = RT->getPointeeType();
  if (!PointeeTy.isNull()) {
    bool isReadOnly = false;
    // For historical/compatibility reasons, the read-only qualifier of the
    // pointee gets emitted _before_ the '^'.  The read-only qualifier of
    // the pointer itself gets ignored, _unless_ we are looking at a typedef!
    // Also, do not emit the 'r' for anything but the outermost type!
    if (isa<TypedefType>(T.getTypePtr())) {
      if (OutermostType && T.isConstQualified()) {
        isReadOnly = true;
        S += 'r';
      }
    } else if (OutermostType) {
      QualType P = PointeeTy;
      while (P->getAs<PointerType>())
        P = P->getAs<PointerType>()->getPointeeType();
      if (P.isConstQualified()) {
        isReadOnly = true;
        S += 'r';
      }
    }
    if (isReadOnly) {
      // Another legacy compatibility encoding. Some ObjC qualifier and type
      // combinations need to be rearranged.
      // Rewrite "in const" from "nr" to "rn"
      if (llvm::StringRef(S).endswith("nr"))
        S.replace(S.end()-2, S.end(), "rn");
    }

    if (PointeeTy->isCharType()) {
      // char pointer types should be encoded as '*' unless it is a
      // type that has been typedef'd to 'BOOL'.
      if (!isTypeTypedefedAsBOOL(PointeeTy)) {
        S += '*';
        return;
      }
    } else if (const RecordType *RTy = PointeeTy->getAs<RecordType>()) {
      // GCC binary compat: Need to convert "struct objc_class *" to "#".
      if (RTy->getDecl()->getIdentifier() == &Idents.get("objc_class")) {
        S += '#';
        return;
      }
      // GCC binary compat: Need to convert "struct objc_object *" to "@".
      if (RTy->getDecl()->getIdentifier() == &Idents.get("objc_object")) {
        S += '@';
        return;
      }
      // fall through...
    }
    S += '^';
    getLegacyIntegralTypeEncoding(PointeeTy);

    getObjCEncodingForTypeImpl(PointeeTy, S, false, ExpandPointedToStructures,
                               NULL);
    return;
  }
  
  if (const ArrayType *AT =
      // Ignore type qualifiers etc.
        dyn_cast<ArrayType>(T->getCanonicalTypeInternal())) {
    if (isa<IncompleteArrayType>(AT)) {
      // Incomplete arrays are encoded as a pointer to the array element.
      S += '^';

      getObjCEncodingForTypeImpl(AT->getElementType(), S,
                                 false, ExpandStructures, FD);
    } else {
      S += '[';

      if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT))
        S += llvm::utostr(CAT->getSize().getZExtValue());
      else {
        //Variable length arrays are encoded as a regular array with 0 elements.
        assert(isa<VariableArrayType>(AT) && "Unknown array type!");
        S += '0';
      }

      getObjCEncodingForTypeImpl(AT->getElementType(), S,
                                 false, ExpandStructures, FD);
      S += ']';
    }
    return;
  }

  if (T->getAs<FunctionType>()) {
    S += '?';
    return;
  }

  if (const RecordType *RTy = T->getAs<RecordType>()) {
    RecordDecl *RDecl = RTy->getDecl();
    S += RDecl->isUnion() ? '(' : '{';
    // Anonymous structures print as '?'
    if (const IdentifierInfo *II = RDecl->getIdentifier()) {
      S += II->getName();
    } else {
      S += '?';
    }
    if (ExpandStructures) {
      S += '=';
      for (RecordDecl::field_iterator Field = RDecl->field_begin(),
                                   FieldEnd = RDecl->field_end();
           Field != FieldEnd; ++Field) {
        if (FD) {
          S += '"';
          S += Field->getNameAsString();
          S += '"';
        }

        // Special case bit-fields.
        if (Field->isBitField()) {
          getObjCEncodingForTypeImpl(Field->getType(), S, false, true,
                                     (*Field));
        } else {
          QualType qt = Field->getType();
          getLegacyIntegralTypeEncoding(qt);
          getObjCEncodingForTypeImpl(qt, S, false, true,
                                     FD);
        }
      }
    }
    S += RDecl->isUnion() ? ')' : '}';
    return;
  }

  if (T->isEnumeralType()) {
    if (FD && FD->isBitField())
      EncodeBitField(this, S, FD);
    else
      S += 'i';
    return;
  }

  if (T->isBlockPointerType()) {
    S += "@?"; // Unlike a pointer-to-function, which is "^?".
    return;
  }

  if (const ObjCInterfaceType *OIT = T->getAs<ObjCInterfaceType>()) {
    // @encode(class_name)
    ObjCInterfaceDecl *OI = OIT->getDecl();
    S += '{';
    const IdentifierInfo *II = OI->getIdentifier();
    S += II->getName();
    S += '=';
    llvm::SmallVector<FieldDecl*, 32> RecFields;
    CollectObjCIvars(OI, RecFields);
    for (unsigned i = 0, e = RecFields.size(); i != e; ++i) {
      if (RecFields[i]->isBitField())
        getObjCEncodingForTypeImpl(RecFields[i]->getType(), S, false, true,
                                   RecFields[i]);
      else
        getObjCEncodingForTypeImpl(RecFields[i]->getType(), S, false, true,
                                   FD);
    }
    S += '}';
    return;
  }

  if (const ObjCObjectPointerType *OPT = T->getAs<ObjCObjectPointerType>()) {
    if (OPT->isObjCIdType()) {
      S += '@';
      return;
    }

    if (OPT->isObjCClassType() || OPT->isObjCQualifiedClassType()) {
      // FIXME: Consider if we need to output qualifiers for 'Class<p>'.
      // Since this is a binary compatibility issue, need to consult with runtime
      // folks. Fortunately, this is a *very* obsure construct.
      S += '#';
      return;
    }

    if (OPT->isObjCQualifiedIdType()) {
      getObjCEncodingForTypeImpl(getObjCIdType(), S,
                                 ExpandPointedToStructures,
                                 ExpandStructures, FD);
      if (FD || EncodingProperty) {
        // Note that we do extended encoding of protocol qualifer list
        // Only when doing ivar or property encoding.
        S += '"';
        for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
             E = OPT->qual_end(); I != E; ++I) {
          S += '<';
          S += (*I)->getNameAsString();
          S += '>';
        }
        S += '"';
      }
      return;
    }

    QualType PointeeTy = OPT->getPointeeType();
    if (!EncodingProperty &&
        isa<TypedefType>(PointeeTy.getTypePtr())) {
      // Another historical/compatibility reason.
      // We encode the underlying type which comes out as
      // {...};
      S += '^';
      getObjCEncodingForTypeImpl(PointeeTy, S,
                                 false, ExpandPointedToStructures,
                                 NULL);
      return;
    }

    S += '@';
    if (OPT->getInterfaceDecl() && (FD || EncodingProperty)) {
      S += '"';
      S += OPT->getInterfaceDecl()->getIdentifier()->getName();
      for (ObjCObjectPointerType::qual_iterator I = OPT->qual_begin(),
           E = OPT->qual_end(); I != E; ++I) {
        S += '<';
        S += (*I)->getNameAsString();
        S += '>';
      }
      S += '"';
    }
    return;
  }

  assert(0 && "@encode for type not implemented!");
}

void ASTContext::getObjCEncodingForTypeQualifier(Decl::ObjCDeclQualifier QT,
                                                 std::string& S) const {
  if (QT & Decl::OBJC_TQ_In)
    S += 'n';
  if (QT & Decl::OBJC_TQ_Inout)
    S += 'N';
  if (QT & Decl::OBJC_TQ_Out)
    S += 'o';
  if (QT & Decl::OBJC_TQ_Bycopy)
    S += 'O';
  if (QT & Decl::OBJC_TQ_Byref)
    S += 'R';
  if (QT & Decl::OBJC_TQ_Oneway)
    S += 'V';
}

void ASTContext::setBuiltinVaListType(QualType T) {
  assert(BuiltinVaListType.isNull() && "__builtin_va_list type already set!");

  BuiltinVaListType = T;
}

void ASTContext::setObjCIdType(QualType T) {
  ObjCIdTypedefType = T;
}

void ASTContext::setObjCSelType(QualType T) {
  ObjCSelTypedefType = T;
}

void ASTContext::setObjCProtoType(QualType QT) {
  ObjCProtoType = QT;
}

void ASTContext::setObjCClassType(QualType T) {
  ObjCClassTypedefType = T;
}

void ASTContext::setObjCConstantStringInterface(ObjCInterfaceDecl *Decl) {
  assert(ObjCConstantStringType.isNull() &&
         "'NSConstantString' type already set!");

  ObjCConstantStringType = getObjCInterfaceType(Decl);
}

/// \brief Retrieve the template name that corresponds to a non-empty
/// lookup.
TemplateName ASTContext::getOverloadedTemplateName(UnresolvedSetIterator Begin,
                                                   UnresolvedSetIterator End) {
  unsigned size = End - Begin;
  assert(size > 1 && "set is not overloaded!");

  void *memory = Allocate(sizeof(OverloadedTemplateStorage) +
                          size * sizeof(FunctionTemplateDecl*));
  OverloadedTemplateStorage *OT = new(memory) OverloadedTemplateStorage(size);

  NamedDecl **Storage = OT->getStorage();
  for (UnresolvedSetIterator I = Begin; I != End; ++I) {
    NamedDecl *D = *I;
    assert(isa<FunctionTemplateDecl>(D) ||
           (isa<UsingShadowDecl>(D) &&
            isa<FunctionTemplateDecl>(D->getUnderlyingDecl())));
    *Storage++ = D;
  }

  return TemplateName(OT);
}

/// \brief Retrieve the template name that represents a qualified
/// template name such as \c std::vector.
TemplateName ASTContext::getQualifiedTemplateName(NestedNameSpecifier *NNS,
                                                  bool TemplateKeyword,
                                                  TemplateDecl *Template) {
  // FIXME: Canonicalization?
  llvm::FoldingSetNodeID ID;
  QualifiedTemplateName::Profile(ID, NNS, TemplateKeyword, Template);

  void *InsertPos = 0;
  QualifiedTemplateName *QTN =
    QualifiedTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
  if (!QTN) {
    QTN = new (*this,4) QualifiedTemplateName(NNS, TemplateKeyword, Template);
    QualifiedTemplateNames.InsertNode(QTN, InsertPos);
  }

  return TemplateName(QTN);
}

/// \brief Retrieve the template name that represents a dependent
/// template name such as \c MetaFun::template apply.
TemplateName ASTContext::getDependentTemplateName(NestedNameSpecifier *NNS,
                                                  const IdentifierInfo *Name) {
  assert((!NNS || NNS->isDependent()) &&
         "Nested name specifier must be dependent");

  llvm::FoldingSetNodeID ID;
  DependentTemplateName::Profile(ID, NNS, Name);

  void *InsertPos = 0;
  DependentTemplateName *QTN =
    DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);

  if (QTN)
    return TemplateName(QTN);

  NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
  if (CanonNNS == NNS) {
    QTN = new (*this,4) DependentTemplateName(NNS, Name);
  } else {
    TemplateName Canon = getDependentTemplateName(CanonNNS, Name);
    QTN = new (*this,4) DependentTemplateName(NNS, Name, Canon);
    DependentTemplateName *CheckQTN =
      DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckQTN && "Dependent type name canonicalization broken");
    (void)CheckQTN;
  }

  DependentTemplateNames.InsertNode(QTN, InsertPos);
  return TemplateName(QTN);
}

/// \brief Retrieve the template name that represents a dependent
/// template name such as \c MetaFun::template operator+.
TemplateName 
ASTContext::getDependentTemplateName(NestedNameSpecifier *NNS,
                                     OverloadedOperatorKind Operator) {
  assert((!NNS || NNS->isDependent()) &&
         "Nested name specifier must be dependent");
  
  llvm::FoldingSetNodeID ID;
  DependentTemplateName::Profile(ID, NNS, Operator);
  
  void *InsertPos = 0;
  DependentTemplateName *QTN
    = DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
  
  if (QTN)
    return TemplateName(QTN);
  
  NestedNameSpecifier *CanonNNS = getCanonicalNestedNameSpecifier(NNS);
  if (CanonNNS == NNS) {
    QTN = new (*this,4) DependentTemplateName(NNS, Operator);
  } else {
    TemplateName Canon = getDependentTemplateName(CanonNNS, Operator);
    QTN = new (*this,4) DependentTemplateName(NNS, Operator, Canon);
    
    DependentTemplateName *CheckQTN
      = DependentTemplateNames.FindNodeOrInsertPos(ID, InsertPos);
    assert(!CheckQTN && "Dependent template name canonicalization broken");
    (void)CheckQTN;
  }
  
  DependentTemplateNames.InsertNode(QTN, InsertPos);
  return TemplateName(QTN);
}

/// getFromTargetType - Given one of the integer types provided by
/// TargetInfo, produce the corresponding type. The unsigned @p Type
/// is actually a value of type @c TargetInfo::IntType.
CanQualType ASTContext::getFromTargetType(unsigned Type) const {
  switch (Type) {
  case TargetInfo::NoInt: return CanQualType();
  case TargetInfo::SignedShort: return ShortTy;
  case TargetInfo::UnsignedShort: return UnsignedShortTy;
  case TargetInfo::SignedInt: return IntTy;
  case TargetInfo::UnsignedInt: return UnsignedIntTy;
  case TargetInfo::SignedLong: return LongTy;
  case TargetInfo::UnsignedLong: return UnsignedLongTy;
  case TargetInfo::SignedLongLong: return LongLongTy;
  case TargetInfo::UnsignedLongLong: return UnsignedLongLongTy;
  }

  assert(false && "Unhandled TargetInfo::IntType value");
  return CanQualType();
}

//===----------------------------------------------------------------------===//
//                        Type Predicates.
//===----------------------------------------------------------------------===//

/// isObjCNSObjectType - Return true if this is an NSObject object using
/// NSObject attribute on a c-style pointer type.
/// FIXME - Make it work directly on types.
/// FIXME: Move to Type.
///
bool ASTContext::isObjCNSObjectType(QualType Ty) const {
  if (TypedefType *TDT = dyn_cast<TypedefType>(Ty)) {
    if (TypedefDecl *TD = TDT->getDecl())
      if (TD->getAttr<ObjCNSObjectAttr>())
        return true;
  }
  return false;
}

/// getObjCGCAttr - Returns one of GCNone, Weak or Strong objc's
/// garbage collection attribute.
///
Qualifiers::GC ASTContext::getObjCGCAttrKind(const QualType &Ty) const {
  Qualifiers::GC GCAttrs = Qualifiers::GCNone;
  if (getLangOptions().ObjC1 &&
      getLangOptions().getGCMode() != LangOptions::NonGC) {
    GCAttrs = Ty.getObjCGCAttr();
    // Default behavious under objective-c's gc is for objective-c pointers
    // (or pointers to them) be treated as though they were declared
    // as __strong.
    if (GCAttrs == Qualifiers::GCNone) {
      if (Ty->isObjCObjectPointerType() || Ty->isBlockPointerType())
        GCAttrs = Qualifiers::Strong;
      else if (Ty->isPointerType())
        return getObjCGCAttrKind(Ty->getAs<PointerType>()->getPointeeType());
    }
    // Non-pointers have none gc'able attribute regardless of the attribute
    // set on them.
    else if (!Ty->isAnyPointerType() && !Ty->isBlockPointerType())
      return Qualifiers::GCNone;
  }
  return GCAttrs;
}

//===----------------------------------------------------------------------===//
//                        Type Compatibility Testing
//===----------------------------------------------------------------------===//

/// areCompatVectorTypes - Return true if the two specified vector types are
/// compatible.
static bool areCompatVectorTypes(const VectorType *LHS,
                                 const VectorType *RHS) {
  assert(LHS->isCanonicalUnqualified() && RHS->isCanonicalUnqualified());
  return LHS->getElementType() == RHS->getElementType() &&
         LHS->getNumElements() == RHS->getNumElements();
}

//===----------------------------------------------------------------------===//
// ObjCQualifiedIdTypesAreCompatible - Compatibility testing for qualified id's.
//===----------------------------------------------------------------------===//

/// ProtocolCompatibleWithProtocol - return 'true' if 'lProto' is in the
/// inheritance hierarchy of 'rProto'.
bool ASTContext::ProtocolCompatibleWithProtocol(ObjCProtocolDecl *lProto,
                                                ObjCProtocolDecl *rProto) {
  if (lProto == rProto)
    return true;
  for (ObjCProtocolDecl::protocol_iterator PI = rProto->protocol_begin(),
       E = rProto->protocol_end(); PI != E; ++PI)
    if (ProtocolCompatibleWithProtocol(lProto, *PI))
      return true;
  return false;
}

/// QualifiedIdConformsQualifiedId - compare id<p,...> with id<p1,...>
/// return true if lhs's protocols conform to rhs's protocol; false
/// otherwise.
bool ASTContext::QualifiedIdConformsQualifiedId(QualType lhs, QualType rhs) {
  if (lhs->isObjCQualifiedIdType() && rhs->isObjCQualifiedIdType())
    return ObjCQualifiedIdTypesAreCompatible(lhs, rhs, false);
  return false;
}

/// ObjCQualifiedIdTypesAreCompatible - We know that one of lhs/rhs is an
/// ObjCQualifiedIDType.
bool ASTContext::ObjCQualifiedIdTypesAreCompatible(QualType lhs, QualType rhs,
                                                   bool compare) {
  // Allow id<P..> and an 'id' or void* type in all cases.
  if (lhs->isVoidPointerType() ||
      lhs->isObjCIdType() || lhs->isObjCClassType())
    return true;
  else if (rhs->isVoidPointerType() ||
           rhs->isObjCIdType() || rhs->isObjCClassType())
    return true;

  if (const ObjCObjectPointerType *lhsQID = lhs->getAsObjCQualifiedIdType()) {
    const ObjCObjectPointerType *rhsOPT = rhs->getAs<ObjCObjectPointerType>();

    if (!rhsOPT) return false;

    if (rhsOPT->qual_empty()) {
      // If the RHS is a unqualified interface pointer "NSString*",
      // make sure we check the class hierarchy.
      if (ObjCInterfaceDecl *rhsID = rhsOPT->getInterfaceDecl()) {
        for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
             E = lhsQID->qual_end(); I != E; ++I) {
          // when comparing an id<P> on lhs with a static type on rhs,
          // see if static class implements all of id's protocols, directly or
          // through its super class and categories.
          if (!rhsID->ClassImplementsProtocol(*I, true))
            return false;
        }
      }
      // If there are no qualifiers and no interface, we have an 'id'.
      return true;
    }
    // Both the right and left sides have qualifiers.
    for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
         E = lhsQID->qual_end(); I != E; ++I) {
      ObjCProtocolDecl *lhsProto = *I;
      bool match = false;

      // when comparing an id<P> on lhs with a static type on rhs,
      // see if static class implements all of id's protocols, directly or
      // through its super class and categories.
      for (ObjCObjectPointerType::qual_iterator J = rhsOPT->qual_begin(),
           E = rhsOPT->qual_end(); J != E; ++J) {
        ObjCProtocolDecl *rhsProto = *J;
        if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
            (compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto))) {
          match = true;
          break;
        }
      }
      // If the RHS is a qualified interface pointer "NSString<P>*",
      // make sure we check the class hierarchy.
      if (ObjCInterfaceDecl *rhsID = rhsOPT->getInterfaceDecl()) {
        for (ObjCObjectPointerType::qual_iterator I = lhsQID->qual_begin(),
             E = lhsQID->qual_end(); I != E; ++I) {
          // when comparing an id<P> on lhs with a static type on rhs,
          // see if static class implements all of id's protocols, directly or
          // through its super class and categories.
          if (rhsID->ClassImplementsProtocol(*I, true)) {
            match = true;
            break;
          }
        }
      }
      if (!match)
        return false;
    }

    return true;
  }

  const ObjCObjectPointerType *rhsQID = rhs->getAsObjCQualifiedIdType();
  assert(rhsQID && "One of the LHS/RHS should be id<x>");

  if (const ObjCObjectPointerType *lhsOPT =
        lhs->getAsObjCInterfacePointerType()) {
    if (lhsOPT->qual_empty()) {
      bool match = false;
      if (ObjCInterfaceDecl *lhsID = lhsOPT->getInterfaceDecl()) {
        for (ObjCObjectPointerType::qual_iterator I = rhsQID->qual_begin(),
             E = rhsQID->qual_end(); I != E; ++I) {
          // when comparing an id<P> on lhs with a static type on rhs,
          // see if static class implements all of id's protocols, directly or
          // through its super class and categories.
          if (lhsID->ClassImplementsProtocol(*I, true)) {
            match = true;
            break;
          }
        }
        if (!match)
          return false;
      }
      return true;
    }
    // Both the right and left sides have qualifiers.
    for (ObjCObjectPointerType::qual_iterator I = lhsOPT->qual_begin(),
         E = lhsOPT->qual_end(); I != E; ++I) {
      ObjCProtocolDecl *lhsProto = *I;
      bool match = false;

      // when comparing an id<P> on lhs with a static type on rhs,
      // see if static class implements all of id's protocols, directly or
      // through its super class and categories.
      for (ObjCObjectPointerType::qual_iterator J = rhsQID->qual_begin(),
           E = rhsQID->qual_end(); J != E; ++J) {
        ObjCProtocolDecl *rhsProto = *J;
        if (ProtocolCompatibleWithProtocol(lhsProto, rhsProto) ||
            (compare && ProtocolCompatibleWithProtocol(rhsProto, lhsProto))) {
          match = true;
          break;
        }
      }
      if (!match)
        return false;
    }
    return true;
  }
  return false;
}

/// canAssignObjCInterfaces - Return true if the two interface types are
/// compatible for assignment from RHS to LHS.  This handles validation of any
/// protocol qualifiers on the LHS or RHS.
///
bool ASTContext::canAssignObjCInterfaces(const ObjCObjectPointerType *LHSOPT,
                                         const ObjCObjectPointerType *RHSOPT) {
  // If either type represents the built-in 'id' or 'Class' types, return true.
  if (LHSOPT->isObjCBuiltinType() || RHSOPT->isObjCBuiltinType())
    return true;

  if (LHSOPT->isObjCQualifiedIdType() || RHSOPT->isObjCQualifiedIdType())
    return ObjCQualifiedIdTypesAreCompatible(QualType(LHSOPT,0),
                                             QualType(RHSOPT,0),
                                             false);

  const ObjCInterfaceType* LHS = LHSOPT->getInterfaceType();
  const ObjCInterfaceType* RHS = RHSOPT->getInterfaceType();
  if (LHS && RHS) // We have 2 user-defined types.
    return canAssignObjCInterfaces(LHS, RHS);

  return false;
}

/// canAssignObjCInterfacesInBlockPointer - This routine is specifically written
/// for providing type-safty for objective-c pointers used to pass/return 
/// arguments in block literals. When passed as arguments, passing 'A*' where
/// 'id' is expected is not OK. Passing 'Sub *" where 'Super *" is expected is
/// not OK. For the return type, the opposite is not OK.
bool ASTContext::canAssignObjCInterfacesInBlockPointer(
                                         const ObjCObjectPointerType *LHSOPT,
                                         const ObjCObjectPointerType *RHSOPT) {
  if (RHSOPT->isObjCBuiltinType() || LHSOPT->isObjCIdType())
    return true;
  
  if (LHSOPT->isObjCBuiltinType()) {
    return RHSOPT->isObjCBuiltinType() || RHSOPT->isObjCQualifiedIdType();
  }
  
  if (LHSOPT->isObjCQualifiedIdType() || RHSOPT->isObjCQualifiedIdType())
    return ObjCQualifiedIdTypesAreCompatible(QualType(LHSOPT,0),
                                             QualType(RHSOPT,0),
                                             false);
  
  const ObjCInterfaceType* LHS = LHSOPT->getInterfaceType();
  const ObjCInterfaceType* RHS = RHSOPT->getInterfaceType();
  if (LHS && RHS)  { // We have 2 user-defined types.
    if (LHS != RHS) {
      if (LHS->getDecl()->isSuperClassOf(RHS->getDecl()))
        return false;
      if (RHS->getDecl()->isSuperClassOf(LHS->getDecl()))
        return true;
    }
    else
      return true;
  }
  return false;
}

/// getIntersectionOfProtocols - This routine finds the intersection of set
/// of protocols inherited from two distinct objective-c pointer objects.
/// It is used to build composite qualifier list of the composite type of
/// the conditional expression involving two objective-c pointer objects.
static 
void getIntersectionOfProtocols(ASTContext &Context,
                                const ObjCObjectPointerType *LHSOPT,
                                const ObjCObjectPointerType *RHSOPT,
      llvm::SmallVectorImpl<ObjCProtocolDecl *> &IntersectionOfProtocols) {
  
  const ObjCInterfaceType* LHS = LHSOPT->getInterfaceType();
  const ObjCInterfaceType* RHS = RHSOPT->getInterfaceType();
  
  llvm::SmallPtrSet<ObjCProtocolDecl *, 8> InheritedProtocolSet;
  unsigned LHSNumProtocols = LHS->getNumProtocols();
  if (LHSNumProtocols > 0)
    InheritedProtocolSet.insert(LHS->qual_begin(), LHS->qual_end());
  else {
    llvm::SmallPtrSet<ObjCProtocolDecl *, 8> LHSInheritedProtocols;
    Context.CollectInheritedProtocols(LHS->getDecl(), LHSInheritedProtocols);
    InheritedProtocolSet.insert(LHSInheritedProtocols.begin(), 
                                LHSInheritedProtocols.end());
  }
  
  unsigned RHSNumProtocols = RHS->getNumProtocols();
  if (RHSNumProtocols > 0) {
    ObjCProtocolDecl **RHSProtocols =
      const_cast<ObjCProtocolDecl **>(RHS->qual_begin());
    for (unsigned i = 0; i < RHSNumProtocols; ++i)
      if (InheritedProtocolSet.count(RHSProtocols[i]))
        IntersectionOfProtocols.push_back(RHSProtocols[i]);
  }
  else {
    llvm::SmallPtrSet<ObjCProtocolDecl *, 8> RHSInheritedProtocols;
    Context.CollectInheritedProtocols(RHS->getDecl(), RHSInheritedProtocols);
    for (llvm::SmallPtrSet<ObjCProtocolDecl*,8>::iterator I = 
         RHSInheritedProtocols.begin(),
         E = RHSInheritedProtocols.end(); I != E; ++I) 
      if (InheritedProtocolSet.count((*I)))
        IntersectionOfProtocols.push_back((*I));
  }
}

/// areCommonBaseCompatible - Returns common base class of the two classes if
/// one found. Note that this is O'2 algorithm. But it will be called as the
/// last type comparison in a ?-exp of ObjC pointer types before a 
/// warning is issued. So, its invokation is extremely rare.
QualType ASTContext::areCommonBaseCompatible(
                                          const ObjCObjectPointerType *LHSOPT,
                                          const ObjCObjectPointerType *RHSOPT) {
  const ObjCInterfaceType* LHS = LHSOPT->getInterfaceType();
  const ObjCInterfaceType* RHS = RHSOPT->getInterfaceType();
  if (!LHS || !RHS)
    return QualType();
  
  while (const ObjCInterfaceDecl *LHSIDecl = LHS->getDecl()->getSuperClass()) {
    QualType LHSTy = getObjCInterfaceType(LHSIDecl);
    LHS = LHSTy->getAs<ObjCInterfaceType>();
    if (canAssignObjCInterfaces(LHS, RHS)) {
      llvm::SmallVector<ObjCProtocolDecl *, 8> IntersectionOfProtocols;
      getIntersectionOfProtocols(*this, 
                                 LHSOPT, RHSOPT, IntersectionOfProtocols);
      if (IntersectionOfProtocols.empty())
        LHSTy = getObjCObjectPointerType(LHSTy);
      else
        LHSTy = getObjCObjectPointerType(LHSTy, &IntersectionOfProtocols[0],
                                                IntersectionOfProtocols.size());
      return LHSTy;
    }
  }
    
  return QualType();
}

bool ASTContext::canAssignObjCInterfaces(const ObjCInterfaceType *LHS,
                                         const ObjCInterfaceType *RHS) {
  // Verify that the base decls are compatible: the RHS must be a subclass of
  // the LHS.
  if (!LHS->getDecl()->isSuperClassOf(RHS->getDecl()))
    return false;

  // RHS must have a superset of the protocols in the LHS.  If the LHS is not
  // protocol qualified at all, then we are good.
  if (LHS->getNumProtocols() == 0)
    return true;

  // Okay, we know the LHS has protocol qualifiers.  If the RHS doesn't, then it
  // isn't a superset.
  if (RHS->getNumProtocols() == 0)
    return true;  // FIXME: should return false!

  for (ObjCInterfaceType::qual_iterator LHSPI = LHS->qual_begin(),
                                        LHSPE = LHS->qual_end();
       LHSPI != LHSPE; LHSPI++) {
    bool RHSImplementsProtocol = false;

    // If the RHS doesn't implement the protocol on the left, the types
    // are incompatible.
    for (ObjCInterfaceType::qual_iterator RHSPI = RHS->qual_begin(),
                                          RHSPE = RHS->qual_end();
         RHSPI != RHSPE; RHSPI++) {
      if ((*RHSPI)->lookupProtocolNamed((*LHSPI)->getIdentifier())) {
        RHSImplementsProtocol = true;
        break;
      }
    }
    // FIXME: For better diagnostics, consider passing back the protocol name.
    if (!RHSImplementsProtocol)
      return false;
  }
  // The RHS implements all protocols listed on the LHS.
  return true;
}

bool ASTContext::areComparableObjCPointerTypes(QualType LHS, QualType RHS) {
  // get the "pointed to" types
  const ObjCObjectPointerType *LHSOPT = LHS->getAs<ObjCObjectPointerType>();
  const ObjCObjectPointerType *RHSOPT = RHS->getAs<ObjCObjectPointerType>();

  if (!LHSOPT || !RHSOPT)
    return false;

  return canAssignObjCInterfaces(LHSOPT, RHSOPT) ||
         canAssignObjCInterfaces(RHSOPT, LHSOPT);
}

/// typesAreCompatible - C99 6.7.3p9: For two qualified types to be compatible,
/// both shall have the identically qualified version of a compatible type.
/// C99 6.2.7p1: Two types have compatible types if their types are the
/// same. See 6.7.[2,3,5] for additional rules.
bool ASTContext::typesAreCompatible(QualType LHS, QualType RHS) {
  if (getLangOptions().CPlusPlus)
    return hasSameType(LHS, RHS);
  
  return !mergeTypes(LHS, RHS).isNull();
}

bool ASTContext::typesAreBlockPointerCompatible(QualType LHS, QualType RHS) {
  return !mergeTypes(LHS, RHS, true).isNull();
}

QualType ASTContext::mergeFunctionTypes(QualType lhs, QualType rhs, 
                                        bool OfBlockPointer) {
  const FunctionType *lbase = lhs->getAs<FunctionType>();
  const FunctionType *rbase = rhs->getAs<FunctionType>();
  const FunctionProtoType *lproto = dyn_cast<FunctionProtoType>(lbase);
  const FunctionProtoType *rproto = dyn_cast<FunctionProtoType>(rbase);
  bool allLTypes = true;
  bool allRTypes = true;

  // Check return type
  QualType retType;
  if (OfBlockPointer)
    retType = mergeTypes(rbase->getResultType(), lbase->getResultType(), true);
  else
   retType = mergeTypes(lbase->getResultType(), rbase->getResultType());
  if (retType.isNull()) return QualType();
  if (getCanonicalType(retType) != getCanonicalType(lbase->getResultType()))
    allLTypes = false;
  if (getCanonicalType(retType) != getCanonicalType(rbase->getResultType()))
    allRTypes = false;
  // FIXME: double check this
  // FIXME: should we error if lbase->getRegParmAttr() != 0 &&
  //                           rbase->getRegParmAttr() != 0 &&
  //                           lbase->getRegParmAttr() != rbase->getRegParmAttr()?
  FunctionType::ExtInfo lbaseInfo = lbase->getExtInfo();
  FunctionType::ExtInfo rbaseInfo = rbase->getExtInfo();
  unsigned RegParm = lbaseInfo.getRegParm() == 0 ? rbaseInfo.getRegParm() :
      lbaseInfo.getRegParm();
  bool NoReturn = lbaseInfo.getNoReturn() || rbaseInfo.getNoReturn();
  if (NoReturn != lbaseInfo.getNoReturn() ||
      RegParm != lbaseInfo.getRegParm())
    allLTypes = false;
  if (NoReturn != rbaseInfo.getNoReturn() ||
      RegParm != rbaseInfo.getRegParm())
    allRTypes = false;
  CallingConv lcc = lbaseInfo.getCC();
  CallingConv rcc = rbaseInfo.getCC();
  // Compatible functions must have compatible calling conventions
  if (!isSameCallConv(lcc, rcc))
    return QualType();

  if (lproto && rproto) { // two C99 style function prototypes
    assert(!lproto->hasExceptionSpec() && !rproto->hasExceptionSpec() &&
           "C++ shouldn't be here");
    unsigned lproto_nargs = lproto->getNumArgs();
    unsigned rproto_nargs = rproto->getNumArgs();

    // Compatible functions must have the same number of arguments
    if (lproto_nargs != rproto_nargs)
      return QualType();

    // Variadic and non-variadic functions aren't compatible
    if (lproto->isVariadic() != rproto->isVariadic())
      return QualType();

    if (lproto->getTypeQuals() != rproto->getTypeQuals())
      return QualType();

    // Check argument compatibility
    llvm::SmallVector<QualType, 10> types;
    for (unsigned i = 0; i < lproto_nargs; i++) {
      QualType largtype = lproto->getArgType(i).getUnqualifiedType();
      QualType rargtype = rproto->getArgType(i).getUnqualifiedType();
      QualType argtype = mergeTypes(largtype, rargtype, OfBlockPointer);
      if (argtype.isNull()) return QualType();
      types.push_back(argtype);
      if (getCanonicalType(argtype) != getCanonicalType(largtype))
        allLTypes = false;
      if (getCanonicalType(argtype) != getCanonicalType(rargtype))
        allRTypes = false;
    }
    if (allLTypes) return lhs;
    if (allRTypes) return rhs;
    return getFunctionType(retType, types.begin(), types.size(),
                           lproto->isVariadic(), lproto->getTypeQuals(),
                           false, false, 0, 0,
                           FunctionType::ExtInfo(NoReturn, RegParm, lcc));
  }

  if (lproto) allRTypes = false;
  if (rproto) allLTypes = false;

  const FunctionProtoType *proto = lproto ? lproto : rproto;
  if (proto) {
    assert(!proto->hasExceptionSpec() && "C++ shouldn't be here");
    if (proto->isVariadic()) return QualType();
    // Check that the types are compatible with the types that
    // would result from default argument promotions (C99 6.7.5.3p15).
    // The only types actually affected are promotable integer
    // types and floats, which would be passed as a different
    // type depending on whether the prototype is visible.
    unsigned proto_nargs = proto->getNumArgs();
    for (unsigned i = 0; i < proto_nargs; ++i) {
      QualType argTy = proto->getArgType(i);
      
      // Look at the promotion type of enum types, since that is the type used
      // to pass enum values.
      if (const EnumType *Enum = argTy->getAs<EnumType>())
        argTy = Enum->getDecl()->getPromotionType();
      
      if (argTy->isPromotableIntegerType() ||
          getCanonicalType(argTy).getUnqualifiedType() == FloatTy)
        return QualType();
    }

    if (allLTypes) return lhs;
    if (allRTypes) return rhs;
    return getFunctionType(retType, proto->arg_type_begin(),
                           proto->getNumArgs(), proto->isVariadic(),
                           proto->getTypeQuals(),
                           false, false, 0, 0,
                           FunctionType::ExtInfo(NoReturn, RegParm, lcc));
  }

  if (allLTypes) return lhs;
  if (allRTypes) return rhs;
  FunctionType::ExtInfo Info(NoReturn, RegParm, lcc);
  return getFunctionNoProtoType(retType, Info);
}

QualType ASTContext::mergeTypes(QualType LHS, QualType RHS, 
                                bool OfBlockPointer) {
  // C++ [expr]: If an expression initially has the type "reference to T", the
  // type is adjusted to "T" prior to any further analysis, the expression
  // designates the object or function denoted by the reference, and the
  // expression is an lvalue unless the reference is an rvalue reference and
  // the expression is a function call (possibly inside parentheses).
  assert(!LHS->getAs<ReferenceType>() && "LHS is a reference type?");
  assert(!RHS->getAs<ReferenceType>() && "RHS is a reference type?");
  
  QualType LHSCan = getCanonicalType(LHS),
           RHSCan = getCanonicalType(RHS);

  // If two types are identical, they are compatible.
  if (LHSCan == RHSCan)
    return LHS;

  // If the qualifiers are different, the types aren't compatible... mostly.
  Qualifiers LQuals = LHSCan.getLocalQualifiers();
  Qualifiers RQuals = RHSCan.getLocalQualifiers();
  if (LQuals != RQuals) {
    // If any of these qualifiers are different, we have a type
    // mismatch.
    if (LQuals.getCVRQualifiers() != RQuals.getCVRQualifiers() ||
        LQuals.getAddressSpace() != RQuals.getAddressSpace())
      return QualType();

    // Exactly one GC qualifier difference is allowed: __strong is
    // okay if the other type has no GC qualifier but is an Objective
    // C object pointer (i.e. implicitly strong by default).  We fix
    // this by pretending that the unqualified type was actually
    // qualified __strong.
    Qualifiers::GC GC_L = LQuals.getObjCGCAttr();
    Qualifiers::GC GC_R = RQuals.getObjCGCAttr();
    assert((GC_L != GC_R) && "unequal qualifier sets had only equal elements");

    if (GC_L == Qualifiers::Weak || GC_R == Qualifiers::Weak)
      return QualType();

    if (GC_L == Qualifiers::Strong && RHSCan->isObjCObjectPointerType()) {
      return mergeTypes(LHS, getObjCGCQualType(RHS, Qualifiers::Strong));
    }
    if (GC_R == Qualifiers::Strong && LHSCan->isObjCObjectPointerType()) {
      return mergeTypes(getObjCGCQualType(LHS, Qualifiers::Strong), RHS);
    }
    return QualType();
  }

  // Okay, qualifiers are equal.

  Type::TypeClass LHSClass = LHSCan->getTypeClass();
  Type::TypeClass RHSClass = RHSCan->getTypeClass();

  // We want to consider the two function types to be the same for these
  // comparisons, just force one to the other.
  if (LHSClass == Type::FunctionProto) LHSClass = Type::FunctionNoProto;
  if (RHSClass == Type::FunctionProto) RHSClass = Type::FunctionNoProto;

  // Same as above for arrays
  if (LHSClass == Type::VariableArray || LHSClass == Type::IncompleteArray)
    LHSClass = Type::ConstantArray;
  if (RHSClass == Type::VariableArray || RHSClass == Type::IncompleteArray)
    RHSClass = Type::ConstantArray;

  // Canonicalize ExtVector -> Vector.
  if (LHSClass == Type::ExtVector) LHSClass = Type::Vector;
  if (RHSClass == Type::ExtVector) RHSClass = Type::Vector;

  // If the canonical type classes don't match.
  if (LHSClass != RHSClass) {
    // C99 6.7.2.2p4: Each enumerated type shall be compatible with char,
    // a signed integer type, or an unsigned integer type.
    // Compatibility is based on the underlying type, not the promotion
    // type.
    if (const EnumType* ETy = LHS->getAs<EnumType>()) {
      if (ETy->getDecl()->getIntegerType() == RHSCan.getUnqualifiedType())
        return RHS;
    }
    if (const EnumType* ETy = RHS->getAs<EnumType>()) {
      if (ETy->getDecl()->getIntegerType() == LHSCan.getUnqualifiedType())
        return LHS;
    }

    return QualType();
  }

  // The canonical type classes match.
  switch (LHSClass) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Non-canonical and dependent types shouldn't get here");
    return QualType();

  case Type::LValueReference:
  case Type::RValueReference:
  case Type::MemberPointer:
    assert(false && "C++ should never be in mergeTypes");
    return QualType();

  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::FunctionProto:
  case Type::ExtVector:
    assert(false && "Types are eliminated above");
    return QualType();

  case Type::Pointer:
  {
    // Merge two pointer types, while trying to preserve typedef info
    QualType LHSPointee = LHS->getAs<PointerType>()->getPointeeType();
    QualType RHSPointee = RHS->getAs<PointerType>()->getPointeeType();
    QualType ResultType = mergeTypes(LHSPointee, RHSPointee);
    if (ResultType.isNull()) return QualType();
    if (getCanonicalType(LHSPointee) == getCanonicalType(ResultType))
      return LHS;
    if (getCanonicalType(RHSPointee) == getCanonicalType(ResultType))
      return RHS;
    return getPointerType(ResultType);
  }
  case Type::BlockPointer:
  {
    // Merge two block pointer types, while trying to preserve typedef info
    QualType LHSPointee = LHS->getAs<BlockPointerType>()->getPointeeType();
    QualType RHSPointee = RHS->getAs<BlockPointerType>()->getPointeeType();
    QualType ResultType = mergeTypes(LHSPointee, RHSPointee, OfBlockPointer);
    if (ResultType.isNull()) return QualType();
    if (getCanonicalType(LHSPointee) == getCanonicalType(ResultType))
      return LHS;
    if (getCanonicalType(RHSPointee) == getCanonicalType(ResultType))
      return RHS;
    return getBlockPointerType(ResultType);
  }
  case Type::ConstantArray:
  {
    const ConstantArrayType* LCAT = getAsConstantArrayType(LHS);
    const ConstantArrayType* RCAT = getAsConstantArrayType(RHS);
    if (LCAT && RCAT && RCAT->getSize() != LCAT->getSize())
      return QualType();

    QualType LHSElem = getAsArrayType(LHS)->getElementType();
    QualType RHSElem = getAsArrayType(RHS)->getElementType();
    QualType ResultType = mergeTypes(LHSElem, RHSElem);
    if (ResultType.isNull()) return QualType();
    if (LCAT && getCanonicalType(LHSElem) == getCanonicalType(ResultType))
      return LHS;
    if (RCAT && getCanonicalType(RHSElem) == getCanonicalType(ResultType))
      return RHS;
    if (LCAT) return getConstantArrayType(ResultType, LCAT->getSize(),
                                          ArrayType::ArraySizeModifier(), 0);
    if (RCAT) return getConstantArrayType(ResultType, RCAT->getSize(),
                                          ArrayType::ArraySizeModifier(), 0);
    const VariableArrayType* LVAT = getAsVariableArrayType(LHS);
    const VariableArrayType* RVAT = getAsVariableArrayType(RHS);
    if (LVAT && getCanonicalType(LHSElem) == getCanonicalType(ResultType))
      return LHS;
    if (RVAT && getCanonicalType(RHSElem) == getCanonicalType(ResultType))
      return RHS;
    if (LVAT) {
      // FIXME: This isn't correct! But tricky to implement because
      // the array's size has to be the size of LHS, but the type
      // has to be different.
      return LHS;
    }
    if (RVAT) {
      // FIXME: This isn't correct! But tricky to implement because
      // the array's size has to be the size of RHS, but the type
      // has to be different.
      return RHS;
    }
    if (getCanonicalType(LHSElem) == getCanonicalType(ResultType)) return LHS;
    if (getCanonicalType(RHSElem) == getCanonicalType(ResultType)) return RHS;
    return getIncompleteArrayType(ResultType,
                                  ArrayType::ArraySizeModifier(), 0);
  }
  case Type::FunctionNoProto:
    return mergeFunctionTypes(LHS, RHS, OfBlockPointer);
  case Type::Record:
  case Type::Enum:
    return QualType();
  case Type::Builtin:
    // Only exactly equal builtin types are compatible, which is tested above.
    return QualType();
  case Type::Complex:
    // Distinct complex types are incompatible.
    return QualType();
  case Type::Vector:
    // FIXME: The merged type should be an ExtVector!
    if (areCompatVectorTypes(LHSCan->getAs<VectorType>(),
                             RHSCan->getAs<VectorType>()))
      return LHS;
    return QualType();
  case Type::ObjCInterface: {
    // Check if the interfaces are assignment compatible.
    // FIXME: This should be type compatibility, e.g. whether
    // "LHS x; RHS x;" at global scope is legal.
    const ObjCInterfaceType* LHSIface = LHS->getAs<ObjCInterfaceType>();
    const ObjCInterfaceType* RHSIface = RHS->getAs<ObjCInterfaceType>();
    if (LHSIface && RHSIface &&
        canAssignObjCInterfaces(LHSIface, RHSIface))
      return LHS;

    return QualType();
  }
  case Type::ObjCObjectPointer: {
    if (OfBlockPointer) {
      if (canAssignObjCInterfacesInBlockPointer(
                                          LHS->getAs<ObjCObjectPointerType>(),
                                          RHS->getAs<ObjCObjectPointerType>()))
      return LHS;
      return QualType();
    }
    if (canAssignObjCInterfaces(LHS->getAs<ObjCObjectPointerType>(),
                                RHS->getAs<ObjCObjectPointerType>()))
      return LHS;

    return QualType();
    }
  }

  return QualType();
}

//===----------------------------------------------------------------------===//
//                         Integer Predicates
//===----------------------------------------------------------------------===//

unsigned ASTContext::getIntWidth(QualType T) {
  if (T->isBooleanType())
    return 1;
  if (EnumType *ET = dyn_cast<EnumType>(T))
    T = ET->getDecl()->getIntegerType();
  // For builtin types, just use the standard type sizing method
  return (unsigned)getTypeSize(T);
}

QualType ASTContext::getCorrespondingUnsignedType(QualType T) {
  assert(T->isSignedIntegerType() && "Unexpected type");
  
  // Turn <4 x signed int> -> <4 x unsigned int>
  if (const VectorType *VTy = T->getAs<VectorType>())
    return getVectorType(getCorrespondingUnsignedType(VTy->getElementType()),
             VTy->getNumElements(), VTy->isAltiVec(), VTy->isPixel());

  // For enums, we return the unsigned version of the base type.
  if (const EnumType *ETy = T->getAs<EnumType>())
    T = ETy->getDecl()->getIntegerType();
  
  const BuiltinType *BTy = T->getAs<BuiltinType>();
  assert(BTy && "Unexpected signed integer type");
  switch (BTy->getKind()) {
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
    return UnsignedCharTy;
  case BuiltinType::Short:
    return UnsignedShortTy;
  case BuiltinType::Int:
    return UnsignedIntTy;
  case BuiltinType::Long:
    return UnsignedLongTy;
  case BuiltinType::LongLong:
    return UnsignedLongLongTy;
  case BuiltinType::Int128:
    return UnsignedInt128Ty;
  default:
    assert(0 && "Unexpected signed integer type");
    return QualType();
  }
}

ExternalASTSource::~ExternalASTSource() { }

void ExternalASTSource::PrintStats() { }


//===----------------------------------------------------------------------===//
//                          Builtin Type Computation
//===----------------------------------------------------------------------===//

/// DecodeTypeFromStr - This decodes one type descriptor from Str, advancing the
/// pointer over the consumed characters.  This returns the resultant type.
static QualType DecodeTypeFromStr(const char *&Str, ASTContext &Context,
                                  ASTContext::GetBuiltinTypeError &Error,
                                  bool AllowTypeModifiers = true) {
  // Modifiers.
  int HowLong = 0;
  bool Signed = false, Unsigned = false;

  // Read the modifiers first.
  bool Done = false;
  while (!Done) {
    switch (*Str++) {
    default: Done = true; --Str; break;
    case 'S':
      assert(!Unsigned && "Can't use both 'S' and 'U' modifiers!");
      assert(!Signed && "Can't use 'S' modifier multiple times!");
      Signed = true;
      break;
    case 'U':
      assert(!Signed && "Can't use both 'S' and 'U' modifiers!");
      assert(!Unsigned && "Can't use 'S' modifier multiple times!");
      Unsigned = true;
      break;
    case 'L':
      assert(HowLong <= 2 && "Can't have LLLL modifier");
      ++HowLong;
      break;
    }
  }

  QualType Type;

  // Read the base type.
  switch (*Str++) {
  default: assert(0 && "Unknown builtin type letter!");
  case 'v':
    assert(HowLong == 0 && !Signed && !Unsigned &&
           "Bad modifiers used with 'v'!");
    Type = Context.VoidTy;
    break;
  case 'f':
    assert(HowLong == 0 && !Signed && !Unsigned &&
           "Bad modifiers used with 'f'!");
    Type = Context.FloatTy;
    break;
  case 'd':
    assert(HowLong < 2 && !Signed && !Unsigned &&
           "Bad modifiers used with 'd'!");
    if (HowLong)
      Type = Context.LongDoubleTy;
    else
      Type = Context.DoubleTy;
    break;
  case 's':
    assert(HowLong == 0 && "Bad modifiers used with 's'!");
    if (Unsigned)
      Type = Context.UnsignedShortTy;
    else
      Type = Context.ShortTy;
    break;
  case 'i':
    if (HowLong == 3)
      Type = Unsigned ? Context.UnsignedInt128Ty : Context.Int128Ty;
    else if (HowLong == 2)
      Type = Unsigned ? Context.UnsignedLongLongTy : Context.LongLongTy;
    else if (HowLong == 1)
      Type = Unsigned ? Context.UnsignedLongTy : Context.LongTy;
    else
      Type = Unsigned ? Context.UnsignedIntTy : Context.IntTy;
    break;
  case 'c':
    assert(HowLong == 0 && "Bad modifiers used with 'c'!");
    if (Signed)
      Type = Context.SignedCharTy;
    else if (Unsigned)
      Type = Context.UnsignedCharTy;
    else
      Type = Context.CharTy;
    break;
  case 'b': // boolean
    assert(HowLong == 0 && !Signed && !Unsigned && "Bad modifiers for 'b'!");
    Type = Context.BoolTy;
    break;
  case 'z':  // size_t.
    assert(HowLong == 0 && !Signed && !Unsigned && "Bad modifiers for 'z'!");
    Type = Context.getSizeType();
    break;
  case 'F':
    Type = Context.getCFConstantStringType();
    break;
  case 'a':
    Type = Context.getBuiltinVaListType();
    assert(!Type.isNull() && "builtin va list type not initialized!");
    break;
  case 'A':
    // This is a "reference" to a va_list; however, what exactly
    // this means depends on how va_list is defined. There are two
    // different kinds of va_list: ones passed by value, and ones
    // passed by reference.  An example of a by-value va_list is
    // x86, where va_list is a char*. An example of by-ref va_list
    // is x86-64, where va_list is a __va_list_tag[1]. For x86,
    // we want this argument to be a char*&; for x86-64, we want
    // it to be a __va_list_tag*.
    Type = Context.getBuiltinVaListType();
    assert(!Type.isNull() && "builtin va list type not initialized!");
    if (Type->isArrayType()) {
      Type = Context.getArrayDecayedType(Type);
    } else {
      Type = Context.getLValueReferenceType(Type);
    }
    break;
  case 'V': {
    char *End;
    unsigned NumElements = strtoul(Str, &End, 10);
    assert(End != Str && "Missing vector size");

    Str = End;

    QualType ElementType = DecodeTypeFromStr(Str, Context, Error, false);
    // FIXME: Don't know what to do about AltiVec.
    Type = Context.getVectorType(ElementType, NumElements, false, false);
    break;
  }
  case 'X': {
    QualType ElementType = DecodeTypeFromStr(Str, Context, Error, false);
    Type = Context.getComplexType(ElementType);
    break;
  }      
  case 'P':
    Type = Context.getFILEType();
    if (Type.isNull()) {
      Error = ASTContext::GE_Missing_stdio;
      return QualType();
    }
    break;
  case 'J':
    if (Signed)
      Type = Context.getsigjmp_bufType();
    else
      Type = Context.getjmp_bufType();

    if (Type.isNull()) {
      Error = ASTContext::GE_Missing_setjmp;
      return QualType();
    }
    break;
  }

  if (!AllowTypeModifiers)
    return Type;

  Done = false;
  while (!Done) {
    switch (char c = *Str++) {
      default: Done = true; --Str; break;
      case '*':
      case '&':
        {
          // Both pointers and references can have their pointee types
          // qualified with an address space.
          char *End;
          unsigned AddrSpace = strtoul(Str, &End, 10);
          if (End != Str && AddrSpace != 0) {
            Type = Context.getAddrSpaceQualType(Type, AddrSpace);
            Str = End;
          }
        }
        if (c == '*')
          Type = Context.getPointerType(Type);
        else
          Type = Context.getLValueReferenceType(Type);
        break;
      // FIXME: There's no way to have a built-in with an rvalue ref arg.
      case 'C':
        Type = Type.withConst();
        break;
      case 'D':
        Type = Context.getVolatileType(Type);
        break;
    }
  }

  return Type;
}

/// GetBuiltinType - Return the type for the specified builtin.
QualType ASTContext::GetBuiltinType(unsigned id,
                                    GetBuiltinTypeError &Error) {
  const char *TypeStr = BuiltinInfo.GetTypeString(id);

  llvm::SmallVector<QualType, 8> ArgTypes;

  Error = GE_None;
  QualType ResType = DecodeTypeFromStr(TypeStr, *this, Error);
  if (Error != GE_None)
    return QualType();
  while (TypeStr[0] && TypeStr[0] != '.') {
    QualType Ty = DecodeTypeFromStr(TypeStr, *this, Error);
    if (Error != GE_None)
      return QualType();

    // Do array -> pointer decay.  The builtin should use the decayed type.
    if (Ty->isArrayType())
      Ty = getArrayDecayedType(Ty);

    ArgTypes.push_back(Ty);
  }

  assert((TypeStr[0] != '.' || TypeStr[1] == 0) &&
         "'.' should only occur at end of builtin type list!");

  // handle untyped/variadic arguments "T c99Style();" or "T cppStyle(...);".
  if (ArgTypes.size() == 0 && TypeStr[0] == '.')
    return getFunctionNoProtoType(ResType);

  // FIXME: Should we create noreturn types?
  return getFunctionType(ResType, ArgTypes.data(), ArgTypes.size(),
                         TypeStr[0] == '.', 0, false, false, 0, 0,
                         FunctionType::ExtInfo());
}

QualType
ASTContext::UsualArithmeticConversionsType(QualType lhs, QualType rhs) {
  // Perform the usual unary conversions. We do this early so that
  // integral promotions to "int" can allow us to exit early, in the
  // lhs == rhs check. Also, for conversion purposes, we ignore any
  // qualifiers.  For example, "const float" and "float" are
  // equivalent.
  if (lhs->isPromotableIntegerType())
    lhs = getPromotedIntegerType(lhs);
  else
    lhs = lhs.getUnqualifiedType();
  if (rhs->isPromotableIntegerType())
    rhs = getPromotedIntegerType(rhs);
  else
    rhs = rhs.getUnqualifiedType();

  // If both types are identical, no conversion is needed.
  if (lhs == rhs)
    return lhs;

  // If either side is a non-arithmetic type (e.g. a pointer), we are done.
  // The caller can deal with this (e.g. pointer + int).
  if (!lhs->isArithmeticType() || !rhs->isArithmeticType())
    return lhs;

  // At this point, we have two different arithmetic types.

  // Handle complex types first (C99 6.3.1.8p1).
  if (lhs->isComplexType() || rhs->isComplexType()) {
    // if we have an integer operand, the result is the complex type.
    if (rhs->isIntegerType() || rhs->isComplexIntegerType()) {
      // convert the rhs to the lhs complex type.
      return lhs;
    }
    if (lhs->isIntegerType() || lhs->isComplexIntegerType()) {
      // convert the lhs to the rhs complex type.
      return rhs;
    }
    // This handles complex/complex, complex/float, or float/complex.
    // When both operands are complex, the shorter operand is converted to the
    // type of the longer, and that is the type of the result. This corresponds
    // to what is done when combining two real floating-point operands.
    // The fun begins when size promotion occur across type domains.
    // From H&S 6.3.4: When one operand is complex and the other is a real
    // floating-point type, the less precise type is converted, within it's
    // real or complex domain, to the precision of the other type. For example,
    // when combining a "long double" with a "double _Complex", the
    // "double _Complex" is promoted to "long double _Complex".
    int result = getFloatingTypeOrder(lhs, rhs);

    if (result > 0) { // The left side is bigger, convert rhs.
      rhs = getFloatingTypeOfSizeWithinDomain(lhs, rhs);
    } else if (result < 0) { // The right side is bigger, convert lhs.
      lhs = getFloatingTypeOfSizeWithinDomain(rhs, lhs);
    }
    // At this point, lhs and rhs have the same rank/size. Now, make sure the
    // domains match. This is a requirement for our implementation, C99
    // does not require this promotion.
    if (lhs != rhs) { // Domains don't match, we have complex/float mix.
      if (lhs->isRealFloatingType()) { // handle "double, _Complex double".
        return rhs;
      } else { // handle "_Complex double, double".
        return lhs;
      }
    }
    return lhs; // The domain/size match exactly.
  }
  // Now handle "real" floating types (i.e. float, double, long double).
  if (lhs->isRealFloatingType() || rhs->isRealFloatingType()) {
    // if we have an integer operand, the result is the real floating type.
    if (rhs->isIntegerType()) {
      // convert rhs to the lhs floating point type.
      return lhs;
    }
    if (rhs->isComplexIntegerType()) {
      // convert rhs to the complex floating point type.
      return getComplexType(lhs);
    }
    if (lhs->isIntegerType()) {
      // convert lhs to the rhs floating point type.
      return rhs;
    }
    if (lhs->isComplexIntegerType()) {
      // convert lhs to the complex floating point type.
      return getComplexType(rhs);
    }
    // We have two real floating types, float/complex combos were handled above.
    // Convert the smaller operand to the bigger result.
    int result = getFloatingTypeOrder(lhs, rhs);
    if (result > 0) // convert the rhs
      return lhs;
    assert(result < 0 && "illegal float comparison");
    return rhs;   // convert the lhs
  }
  if (lhs->isComplexIntegerType() || rhs->isComplexIntegerType()) {
    // Handle GCC complex int extension.
    const ComplexType *lhsComplexInt = lhs->getAsComplexIntegerType();
    const ComplexType *rhsComplexInt = rhs->getAsComplexIntegerType();

    if (lhsComplexInt && rhsComplexInt) {
      if (getIntegerTypeOrder(lhsComplexInt->getElementType(),
                              rhsComplexInt->getElementType()) >= 0)
        return lhs; // convert the rhs
      return rhs;
    } else if (lhsComplexInt && rhs->isIntegerType()) {
      // convert the rhs to the lhs complex type.
      return lhs;
    } else if (rhsComplexInt && lhs->isIntegerType()) {
      // convert the lhs to the rhs complex type.
      return rhs;
    }
  }
  // Finally, we have two differing integer types.
  // The rules for this case are in C99 6.3.1.8
  int compare = getIntegerTypeOrder(lhs, rhs);
  bool lhsSigned = lhs->isSignedIntegerType(),
       rhsSigned = rhs->isSignedIntegerType();
  QualType destType;
  if (lhsSigned == rhsSigned) {
    // Same signedness; use the higher-ranked type
    destType = compare >= 0 ? lhs : rhs;
  } else if (compare != (lhsSigned ? 1 : -1)) {
    // The unsigned type has greater than or equal rank to the
    // signed type, so use the unsigned type
    destType = lhsSigned ? rhs : lhs;
  } else if (getIntWidth(lhs) != getIntWidth(rhs)) {
    // The two types are different widths; if we are here, that
    // means the signed type is larger than the unsigned type, so
    // use the signed type.
    destType = lhsSigned ? lhs : rhs;
  } else {
    // The signed type is higher-ranked than the unsigned type,
    // but isn't actually any bigger (like unsigned int and long
    // on most 32-bit systems).  Use the unsigned type corresponding
    // to the signed type.
    destType = getCorrespondingUnsignedType(lhsSigned ? lhs : rhs);
  }
  return destType;
}
