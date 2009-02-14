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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;

enum FloatingRank {
  FloatRank, DoubleRank, LongDoubleRank
};

ASTContext::ASTContext(const LangOptions& LOpts, SourceManager &SM,
                       TargetInfo &t,
                       IdentifierTable &idents, SelectorTable &sels,
                       bool FreeMem, unsigned size_reserve) : 
  CFConstantStringTypeDecl(0), ObjCFastEnumerationStateTypeDecl(0),
  SourceMgr(SM), LangOpts(LOpts), FreeMemory(FreeMem), Target(t), 
  Idents(idents), Selectors(sels)
{  
  if (size_reserve > 0) Types.reserve(size_reserve);    
  InitBuiltinTypes();
  BuiltinInfo.InitializeBuiltins(idents, Target);
  TUDecl = TranslationUnitDecl::Create(*this);
}

ASTContext::~ASTContext() {
  // Deallocate all the types.
  while (!Types.empty()) {
    Types.back()->Destroy(*this);
    Types.pop_back();
  }

  {
    llvm::DenseMap<const RecordDecl*, const ASTRecordLayout*>::iterator
      I = ASTRecordLayouts.begin(), E = ASTRecordLayouts.end();
    while (I != E) {
      ASTRecordLayout *R = const_cast<ASTRecordLayout*>((I++)->second);
      delete R;
    }
  }

  {
    llvm::DenseMap<const ObjCInterfaceDecl*, const ASTRecordLayout*>::iterator
      I = ASTObjCInterfaces.begin(), E = ASTObjCInterfaces.end();
    while (I != E) {
      ASTRecordLayout *R = const_cast<ASTRecordLayout*>((I++)->second);
      delete R;
    }
  }

  {
    llvm::DenseMap<const ObjCInterfaceDecl*, const RecordDecl*>::iterator
      I = ASTRecordForInterface.begin(), E = ASTRecordForInterface.end();
    while (I != E) {
      RecordDecl *R = const_cast<RecordDecl*>((I++)->second);
      R->Destroy(*this);
    }
  }

  TUDecl->Destroy(*this);
}

void ASTContext::PrintStats() const {
  fprintf(stderr, "*** AST Context Stats:\n");
  fprintf(stderr, "  %d types total.\n", (int)Types.size());
  unsigned NumBuiltin = 0, NumPointer = 0, NumArray = 0, NumFunctionP = 0;
  unsigned NumVector = 0, NumComplex = 0, NumBlockPointer = 0;
  unsigned NumFunctionNP = 0, NumTypeName = 0, NumTagged = 0, NumReference = 0;
  unsigned NumMemberPointer = 0;
  
  unsigned NumTagStruct = 0, NumTagUnion = 0, NumTagEnum = 0, NumTagClass = 0;
  unsigned NumObjCInterfaces = 0, NumObjCQualifiedInterfaces = 0;
  unsigned NumObjCQualifiedIds = 0;
  unsigned NumTypeOfTypes = 0, NumTypeOfExprs = 0;
  
  for (unsigned i = 0, e = Types.size(); i != e; ++i) {
    Type *T = Types[i];
    if (isa<BuiltinType>(T))
      ++NumBuiltin;
    else if (isa<PointerType>(T))
      ++NumPointer;
    else if (isa<BlockPointerType>(T))
      ++NumBlockPointer;
    else if (isa<ReferenceType>(T))
      ++NumReference;
    else if (isa<MemberPointerType>(T))
      ++NumMemberPointer;
    else if (isa<ComplexType>(T))
      ++NumComplex;
    else if (isa<ArrayType>(T))
      ++NumArray;
    else if (isa<VectorType>(T))
      ++NumVector;
    else if (isa<FunctionTypeNoProto>(T))
      ++NumFunctionNP;
    else if (isa<FunctionTypeProto>(T))
      ++NumFunctionP;
    else if (isa<TypedefType>(T))
      ++NumTypeName;
    else if (TagType *TT = dyn_cast<TagType>(T)) {
      ++NumTagged;
      switch (TT->getDecl()->getTagKind()) {
      default: assert(0 && "Unknown tagged type!");
      case TagDecl::TK_struct: ++NumTagStruct; break;
      case TagDecl::TK_union:  ++NumTagUnion; break;
      case TagDecl::TK_class:  ++NumTagClass; break; 
      case TagDecl::TK_enum:   ++NumTagEnum; break;
      }
    } else if (isa<ObjCInterfaceType>(T))
      ++NumObjCInterfaces;
    else if (isa<ObjCQualifiedInterfaceType>(T))
      ++NumObjCQualifiedInterfaces;
    else if (isa<ObjCQualifiedIdType>(T))
      ++NumObjCQualifiedIds;
    else if (isa<TypeOfType>(T))
      ++NumTypeOfTypes;
    else if (isa<TypeOfExpr>(T))
      ++NumTypeOfExprs;
    else {
      QualType(T, 0).dump();
      assert(0 && "Unknown type!");
    }
  }

  fprintf(stderr, "    %d builtin types\n", NumBuiltin);
  fprintf(stderr, "    %d pointer types\n", NumPointer);
  fprintf(stderr, "    %d block pointer types\n", NumBlockPointer);
  fprintf(stderr, "    %d reference types\n", NumReference);
  fprintf(stderr, "    %d member pointer types\n", NumMemberPointer);
  fprintf(stderr, "    %d complex types\n", NumComplex);
  fprintf(stderr, "    %d array types\n", NumArray);
  fprintf(stderr, "    %d vector types\n", NumVector);
  fprintf(stderr, "    %d function types with proto\n", NumFunctionP);
  fprintf(stderr, "    %d function types with no proto\n", NumFunctionNP);
  fprintf(stderr, "    %d typename (typedef) types\n", NumTypeName);
  fprintf(stderr, "    %d tagged types\n", NumTagged);
  fprintf(stderr, "      %d struct types\n", NumTagStruct);
  fprintf(stderr, "      %d union types\n", NumTagUnion);
  fprintf(stderr, "      %d class types\n", NumTagClass);
  fprintf(stderr, "      %d enum types\n", NumTagEnum);
  fprintf(stderr, "    %d interface types\n", NumObjCInterfaces);
  fprintf(stderr, "    %d protocol qualified interface types\n",
          NumObjCQualifiedInterfaces);
  fprintf(stderr, "    %d protocol qualified id types\n",
          NumObjCQualifiedIds);
  fprintf(stderr, "    %d typeof types\n", NumTypeOfTypes);
  fprintf(stderr, "    %d typeof exprs\n", NumTypeOfExprs);
  
  fprintf(stderr, "Total bytes = %d\n", int(NumBuiltin*sizeof(BuiltinType)+
    NumPointer*sizeof(PointerType)+NumArray*sizeof(ArrayType)+
    NumComplex*sizeof(ComplexType)+NumVector*sizeof(VectorType)+
    NumMemberPointer*sizeof(MemberPointerType)+
    NumFunctionP*sizeof(FunctionTypeProto)+
    NumFunctionNP*sizeof(FunctionTypeNoProto)+
    NumTypeName*sizeof(TypedefType)+NumTagged*sizeof(TagType)+
    NumTypeOfTypes*sizeof(TypeOfType)+NumTypeOfExprs*sizeof(TypeOfExpr)));
}


void ASTContext::InitBuiltinType(QualType &R, BuiltinType::Kind K) {
  Types.push_back((R = QualType(new (*this,8) BuiltinType(K),0)).getTypePtr());
}

void ASTContext::InitBuiltinTypes() {
  assert(VoidTy.isNull() && "Context reinitialized?");
  
  // C99 6.2.5p19.
  InitBuiltinType(VoidTy,              BuiltinType::Void);
  
  // C99 6.2.5p2.
  InitBuiltinType(BoolTy,              BuiltinType::Bool);
  // C99 6.2.5p3.
  if (Target.isCharSigned())
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

  // C++ 3.9.1p5
  InitBuiltinType(WCharTy,             BuiltinType::WChar);

  // Placeholder type for functions.
  InitBuiltinType(OverloadTy,          BuiltinType::Overload);

  // Placeholder type for type-dependent expressions whose type is
  // completely unknown. No code should ever check a type against
  // DependentTy and users should never see it; however, it is here to
  // help diagnose failures to properly check for type-dependent
  // expressions.
  InitBuiltinType(DependentTy,         BuiltinType::Dependent);

  // C99 6.2.5p11.
  FloatComplexTy      = getComplexType(FloatTy);
  DoubleComplexTy     = getComplexType(DoubleTy);
  LongDoubleComplexTy = getComplexType(LongDoubleTy);

  BuiltinVaListType = QualType();
  ObjCIdType = QualType();
  IdStructType = 0;
  ObjCClassType = QualType();
  ClassStructType = 0;
  
  ObjCConstantStringType = QualType();
  
  // void * type
  VoidPtrTy = getPointerType(VoidTy);
}

//===----------------------------------------------------------------------===//
//                         Type Sizing and Analysis
//===----------------------------------------------------------------------===//

/// getFloatTypeSemantics - Return the APFloat 'semantics' for the specified
/// scalar floating point type.
const llvm::fltSemantics &ASTContext::getFloatTypeSemantics(QualType T) const {
  const BuiltinType *BT = T->getAsBuiltinType();
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
unsigned ASTContext::getDeclAlign(const Decl *D) {
  // FIXME: If attribute(align) is specified on the decl, round up to it.
  
  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    QualType T = VD->getType();
    // Incomplete or function types default to 1.
    if (T->isIncompleteType() || T->isFunctionType())
      return 1;
    
    while (isa<VariableArrayType>(T) || isa<IncompleteArrayType>(T))
      T = cast<ArrayType>(T)->getElementType();
    
    return getTypeAlign(T);
  }
  
  return 1;
}

/// getTypeSize - Return the size of the specified type, in bits.  This method
/// does not work on incomplete types.
std::pair<uint64_t, unsigned>
ASTContext::getTypeInfo(const Type *T) {
  T = getCanonicalType(T);
  uint64_t Width;
  unsigned Align;
  switch (T->getTypeClass()) {
  case Type::TypeName: assert(0 && "Not a canonical type!");
  case Type::FunctionNoProto:
  case Type::FunctionProto:
  default:
    assert(0 && "Incomplete types have no size!");
  case Type::VariableArray:
    assert(0 && "VLAs not implemented yet!");
  case Type::DependentSizedArray:
    assert(0 && "Dependently-sized arrays don't have a known size");
  case Type::ConstantArray: {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(T);
    
    std::pair<uint64_t, unsigned> EltInfo = getTypeInfo(CAT->getElementType());
    Width = EltInfo.first*CAT->getSize().getZExtValue();
    Align = EltInfo.second;
    break;
  }
  case Type::ExtVector:
  case Type::Vector: {
    std::pair<uint64_t, unsigned> EltInfo = 
      getTypeInfo(cast<VectorType>(T)->getElementType());
    Width = EltInfo.first*cast<VectorType>(T)->getNumElements();
    Align = Width;
    // If the alignment is not a power of 2, round up to the next power of 2.
    // This happens for non-power-of-2 length vectors.
    // FIXME: this should probably be a target property.
    Align = 1 << llvm::Log2_32_Ceil(Align);
    break;
  }

  case Type::Builtin:
    switch (cast<BuiltinType>(T)->getKind()) {
    default: assert(0 && "Unknown builtin type!");
    case BuiltinType::Void:
      assert(0 && "Incomplete types have no size!");
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
    }
    break;
  case Type::FixedWidthInt:
    // FIXME: This isn't precisely correct; the width/alignment should depend
    // on the available types for the target
    Width = cast<FixedWidthIntType>(T)->getWidth();
    Width = std::max(llvm::NextPowerOf2(Width - 1), 8ULL);
    Align = Width;
    break;
  case Type::ASQual:
    // FIXME: Pointers into different addr spaces could have different sizes and
    // alignment requirements: getPointerInfo should take an AddrSpace.
    return getTypeInfo(QualType(cast<ASQualType>(T)->getBaseType(), 0));
  case Type::ObjCQualifiedId:
    Width = Target.getPointerWidth(0);
    Align = Target.getPointerAlign(0);
    break;
  case Type::BlockPointer: {
    unsigned AS = cast<BlockPointerType>(T)->getPointeeType().getAddressSpace();
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
  case Type::Reference:
    // "When applied to a reference or a reference type, the result is the size
    // of the referenced type." C++98 5.3.3p2: expr.sizeof.
    // FIXME: This is wrong for struct layout: a reference in a struct has
    // pointer size.
    return getTypeInfo(cast<ReferenceType>(T)->getPointeeType());
  case Type::MemberPointer: {
    // FIXME: This is not only platform- but also ABI-dependent. We follow
    // the GCC ABI, where pointers to data are one pointer large, pointers to
    // functions two pointers. But if we want to support ABI compatibility with
    // other compilers too, we need to delegate this completely to TargetInfo
    // or some ABI abstraction layer.
    QualType Pointee = cast<MemberPointerType>(T)->getPointeeType();
    unsigned AS = Pointee.getAddressSpace();
    Width = Target.getPointerWidth(AS);
    if (Pointee->isFunctionType())
      Width *= 2;
    Align = Target.getPointerAlign(AS);
    // GCC aligns at single pointer width.
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
  case Type::Tagged: {
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
  }
  
  assert(Align && (Align & (Align-1)) == 0 && "Alignment must be power of 2");
  return std::make_pair(Width, Align);
}

/// getPreferredTypeAlign - Return the "preferred" alignment of the specified
/// type for the current target in bits.  This can be different than the ABI
/// alignment in cases where it is beneficial for performance to overalign
/// a data type.
unsigned ASTContext::getPreferredTypeAlign(const Type *T) {
  unsigned ABIAlign = getTypeAlign(T);
  
  // Doubles should be naturally aligned if possible.
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(getCanonicalType(T)))
    if (BT->getKind() == BuiltinType::Double)
      return std::max(ABIAlign, 64U);
  
  return ABIAlign;
}


/// LayoutField - Field layout.
void ASTRecordLayout::LayoutField(const FieldDecl *FD, unsigned FieldNo,
                                  bool IsUnion, unsigned StructPacking,
                                  ASTContext &Context) {
  unsigned FieldPacking = StructPacking;
  uint64_t FieldOffset = IsUnion ? 0 : Size;
  uint64_t FieldSize;
  unsigned FieldAlign;

  // FIXME: Should this override struct packing? Probably we want to
  // take the minimum?
  if (const PackedAttr *PA = FD->getAttr<PackedAttr>())
    FieldPacking = PA->getAlignment();
  
  if (const Expr *BitWidthExpr = FD->getBitWidth()) {
    // TODO: Need to check this algorithm on other targets!
    //       (tested on Linux-X86)
    FieldSize = 
      BitWidthExpr->getIntegerConstantExprValue(Context).getZExtValue();
    
    std::pair<uint64_t, unsigned> FieldInfo = 
      Context.getTypeInfo(FD->getType());
    uint64_t TypeSize = FieldInfo.first;
    
    // Determine the alignment of this bitfield. The packing
    // attributes define a maximum and the alignment attribute defines
    // a minimum.
    // FIXME: What is the right behavior when the specified alignment
    // is smaller than the specified packing?
    FieldAlign = FieldInfo.second;
    if (FieldPacking)
      FieldAlign = std::min(FieldAlign, FieldPacking);
    if (const AlignedAttr *AA = FD->getAttr<AlignedAttr>())
      FieldAlign = std::max(FieldAlign, AA->getAlignment());
    
    // Check if we need to add padding to give the field the correct
    // alignment.
    if (FieldSize == 0 || (FieldOffset & (FieldAlign-1)) + FieldSize > TypeSize)
      FieldOffset = (FieldOffset + (FieldAlign-1)) & ~(FieldAlign-1);
    
    // Padding members don't affect overall alignment
    if (!FD->getIdentifier())
      FieldAlign = 1;
  } else {
    if (FD->getType()->isIncompleteArrayType()) {
      // This is a flexible array member; we can't directly
      // query getTypeInfo about these, so we figure it out here.
      // Flexible array members don't have any size, but they
      // have to be aligned appropriately for their element type.
      FieldSize = 0;
      const ArrayType* ATy = Context.getAsArrayType(FD->getType());
      FieldAlign = Context.getTypeAlign(ATy->getElementType());
    } else {
      std::pair<uint64_t, unsigned> FieldInfo = 
        Context.getTypeInfo(FD->getType());
      FieldSize = FieldInfo.first;
      FieldAlign = FieldInfo.second;
    }
    
    // Determine the alignment of this bitfield. The packing
    // attributes define a maximum and the alignment attribute defines
    // a minimum. Additionally, the packing alignment must be at least
    // a byte for non-bitfields.
    //
    // FIXME: What is the right behavior when the specified alignment
    // is smaller than the specified packing?
    if (FieldPacking)
      FieldAlign = std::min(FieldAlign, std::max(8U, FieldPacking));
    if (const AlignedAttr *AA = FD->getAttr<AlignedAttr>())
      FieldAlign = std::max(FieldAlign, AA->getAlignment());
    
    // Round up the current record size to the field's alignment boundary.
    FieldOffset = (FieldOffset + (FieldAlign-1)) & ~(FieldAlign-1);
  }
  
  // Place this field at the current location.
  FieldOffsets[FieldNo] = FieldOffset;
  
  // Reserve space for this field.
  if (IsUnion) {
    Size = std::max(Size, FieldSize);
  } else {
    Size = FieldOffset + FieldSize;
  }
  
  // Remember max struct/class alignment.
  Alignment = std::max(Alignment, FieldAlign);
}

static void CollectObjCIvars(const ObjCInterfaceDecl *OI,
                             std::vector<FieldDecl*> &Fields) {
  const ObjCInterfaceDecl *SuperClass = OI->getSuperClass();
  if (SuperClass)
    CollectObjCIvars(SuperClass, Fields);
  for (ObjCInterfaceDecl::ivar_iterator I = OI->ivar_begin(),
       E = OI->ivar_end(); I != E; ++I) {
    ObjCIvarDecl *IVDecl = (*I);
    if (!IVDecl->isInvalidDecl())
      Fields.push_back(cast<FieldDecl>(IVDecl));
  }
}

/// addRecordToClass - produces record info. for the class for its
/// ivars and all those inherited.
///
const RecordDecl *ASTContext::addRecordToClass(const ObjCInterfaceDecl *D)
{
  const RecordDecl *&RD = ASTRecordForInterface[D];
  if (RD)
    return RD;
  std::vector<FieldDecl*> RecFields;
  CollectObjCIvars(D, RecFields);
  RecordDecl *NewRD = RecordDecl::Create(*this, TagDecl::TK_struct, 0,
                                         D->getLocation(),
                                         D->getIdentifier());
  /// FIXME! Can do collection of ivars and adding to the record while
  /// doing it.
  for (unsigned int i = 0; i != RecFields.size(); i++) {
    FieldDecl *Field =  FieldDecl::Create(*this, NewRD, 
                                          RecFields[i]->getLocation(), 
                                          RecFields[i]->getIdentifier(),
                                          RecFields[i]->getType(), 
                                          RecFields[i]->getBitWidth(), false);
    NewRD->addDecl(Field);
  }
  NewRD->completeDefinition(*this);
  RD = NewRD;
  return RD;
}

/// setFieldDecl - maps a field for the given Ivar reference node.
//
void ASTContext::setFieldDecl(const ObjCInterfaceDecl *OI,
                              const ObjCIvarDecl *Ivar,
                              const ObjCIvarRefExpr *MRef) {
  FieldDecl *FD = (const_cast<ObjCInterfaceDecl *>(OI))->
                    lookupFieldDeclForIvar(*this, Ivar);
  ASTFieldForIvarRef[MRef] = FD;
}

/// getASTObjcInterfaceLayout - Get or compute information about the layout of
/// the specified Objective C, which indicates its size and ivar
/// position information.
const ASTRecordLayout &
ASTContext::getASTObjCInterfaceLayout(const ObjCInterfaceDecl *D) {
  // Look up this layout, if already laid out, return what we have.
  const ASTRecordLayout *&Entry = ASTObjCInterfaces[D];
  if (Entry) return *Entry;

  // Allocate and assign into ASTRecordLayouts here.  The "Entry" reference can
  // be invalidated (dangle) if the ASTRecordLayouts hashtable is inserted into.
  ASTRecordLayout *NewEntry = NULL;
  unsigned FieldCount = D->ivar_size();
  if (ObjCInterfaceDecl *SD = D->getSuperClass()) {
    FieldCount++;
    const ASTRecordLayout &SL = getASTObjCInterfaceLayout(SD);
    unsigned Alignment = SL.getAlignment();
    uint64_t Size = SL.getSize();
    NewEntry = new ASTRecordLayout(Size, Alignment);
    NewEntry->InitializeLayout(FieldCount);
    // Super class is at the beginning of the layout.
    NewEntry->SetFieldOffset(0, 0);
  } else {
    NewEntry = new ASTRecordLayout();
    NewEntry->InitializeLayout(FieldCount);
  }
  Entry = NewEntry;

  unsigned StructPacking = 0;
  if (const PackedAttr *PA = D->getAttr<PackedAttr>())
    StructPacking = PA->getAlignment();

  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    NewEntry->SetAlignment(std::max(NewEntry->getAlignment(), 
                                    AA->getAlignment()));

  // Layout each ivar sequentially.
  unsigned i = 0;
  for (ObjCInterfaceDecl::ivar_iterator IVI = D->ivar_begin(), 
       IVE = D->ivar_end(); IVI != IVE; ++IVI) {
    const ObjCIvarDecl* Ivar = (*IVI);
    NewEntry->LayoutField(Ivar, i++, false, StructPacking, *this);
  }

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  NewEntry->FinalizeLayout();
  return *NewEntry;
}

/// getASTRecordLayout - Get or compute information about the layout of the
/// specified record (struct/union/class), which indicates its size and field
/// position information.
const ASTRecordLayout &ASTContext::getASTRecordLayout(const RecordDecl *D) {
  D = D->getDefinition(*this);
  assert(D && "Cannot get layout of forward declarations!");

  // Look up this layout, if already laid out, return what we have.
  const ASTRecordLayout *&Entry = ASTRecordLayouts[D];
  if (Entry) return *Entry;

  // Allocate and assign into ASTRecordLayouts here.  The "Entry" reference can
  // be invalidated (dangle) if the ASTRecordLayouts hashtable is inserted into.
  ASTRecordLayout *NewEntry = new ASTRecordLayout();
  Entry = NewEntry;

  // FIXME: Avoid linear walk through the fields, if possible.
  NewEntry->InitializeLayout(std::distance(D->field_begin(), D->field_end()));
  bool IsUnion = D->isUnion();

  unsigned StructPacking = 0;
  if (const PackedAttr *PA = D->getAttr<PackedAttr>())
    StructPacking = PA->getAlignment();

  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>())
    NewEntry->SetAlignment(std::max(NewEntry->getAlignment(), 
                                    AA->getAlignment()));

  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  unsigned FieldIdx = 0;
  for (RecordDecl::field_iterator Field = D->field_begin(),
                               FieldEnd = D->field_end();
       Field != FieldEnd; (void)++Field, ++FieldIdx)
    NewEntry->LayoutField(*Field, FieldIdx, IsUnion, StructPacking, *this);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  NewEntry->FinalizeLayout();
  return *NewEntry;
}

//===----------------------------------------------------------------------===//
//                   Type creation/memoization methods
//===----------------------------------------------------------------------===//

QualType ASTContext::getASQualType(QualType T, unsigned AddressSpace) {
  QualType CanT = getCanonicalType(T);
  if (CanT.getAddressSpace() == AddressSpace)
    return T;
  
  // Type's cannot have multiple ASQuals, therefore we know we only have to deal
  // with CVR qualifiers from here on out.
  assert(CanT.getAddressSpace() == 0 &&
         "Type is already address space qualified");
  
  // Check if we've already instantiated an address space qual'd type of this
  // type.
  llvm::FoldingSetNodeID ID;
  ASQualType::Profile(ID, T.getTypePtr(), AddressSpace);      
  void *InsertPos = 0;
  if (ASQualType *ASQy = ASQualTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(ASQy, 0);
    
  // If the base type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!T->isCanonical()) {
    Canonical = getASQualType(CanT, AddressSpace);
    
    // Get the new insert position for the node we care about.
    ASQualType *NewIP = ASQualTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  ASQualType *New = new (*this, 8) ASQualType(T.getTypePtr(), Canonical, 
                                              AddressSpace);
  ASQualTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, T.getCVRQualifiers());
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
  if (!T->isCanonical()) {
    Canonical = getComplexType(getCanonicalType(T));
    
    // Get the new insert position for the node we care about.
    ComplexType *NewIP = ComplexTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  ComplexType *New = new (*this,8) ComplexType(T, Canonical);
  Types.push_back(New);
  ComplexTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

QualType ASTContext::getFixedWidthIntType(unsigned Width, bool Signed) {
  llvm::DenseMap<unsigned, FixedWidthIntType*> &Map = Signed ?
     SignedFixedWidthIntTypes : UnsignedFixedWidthIntTypes;
  FixedWidthIntType *&Entry = Map[Width];
  if (!Entry)
    Entry = new FixedWidthIntType(Width, Signed);
  return QualType(Entry, 0);
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
  if (!T->isCanonical()) {
    Canonical = getPointerType(getCanonicalType(T));
   
    // Get the new insert position for the node we care about.
    PointerType *NewIP = PointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  PointerType *New = new (*this,8) PointerType(T, Canonical);
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
  if (!T->isCanonical()) {
    Canonical = getBlockPointerType(getCanonicalType(T));
    
    // Get the new insert position for the node we care about.
    BlockPointerType *NewIP =
      BlockPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  BlockPointerType *New = new (*this,8) BlockPointerType(T, Canonical);
  Types.push_back(New);
  BlockPointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getReferenceType - Return the uniqued reference to the type for a reference
/// to the specified type.
QualType ASTContext::getReferenceType(QualType T) {
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  ReferenceType::Profile(ID, T);

  void *InsertPos = 0;
  if (ReferenceType *RT = ReferenceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(RT, 0);
  
  // If the referencee type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;
  if (!T->isCanonical()) {
    Canonical = getReferenceType(getCanonicalType(T));
   
    // Get the new insert position for the node we care about.
    ReferenceType *NewIP = ReferenceTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  ReferenceType *New = new (*this,8) ReferenceType(T, Canonical);
  Types.push_back(New);
  ReferenceTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getMemberPointerType - Return the uniqued reference to the type for a
/// member pointer to the specified type, in the specified class.
QualType ASTContext::getMemberPointerType(QualType T, const Type *Cls)
{
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
  if (!T->isCanonical()) {
    Canonical = getMemberPointerType(getCanonicalType(T),getCanonicalType(Cls));

    // Get the new insert position for the node we care about.
    MemberPointerType *NewIP =
      MemberPointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  MemberPointerType *New = new (*this,8) MemberPointerType(T, Cls, Canonical);
  Types.push_back(New);
  MemberPointerTypes.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getConstantArrayType - Return the unique reference to the type for an 
/// array of the specified element type.
QualType ASTContext::getConstantArrayType(QualType EltTy, 
                                          const llvm::APInt &ArySize,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned EltTypeQuals) {
  llvm::FoldingSetNodeID ID;
  ConstantArrayType::Profile(ID, EltTy, ArySize);
      
  void *InsertPos = 0;
  if (ConstantArrayType *ATP = 
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(ATP, 0);
  
  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!EltTy->isCanonical()) {
    Canonical = getConstantArrayType(getCanonicalType(EltTy), ArySize, 
                                     ASM, EltTypeQuals);
    // Get the new insert position for the node we care about.
    ConstantArrayType *NewIP = 
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  
  ConstantArrayType *New =
    new(*this,8)ConstantArrayType(EltTy, Canonical, ArySize, ASM, EltTypeQuals);
  ConstantArrayTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getVariableArrayType - Returns a non-unique reference to the type for a
/// variable array of the specified element type.
QualType ASTContext::getVariableArrayType(QualType EltTy, Expr *NumElts,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned EltTypeQuals) {
  // Since we don't unique expressions, it isn't possible to unique VLA's
  // that have an expression provided for their size.

  VariableArrayType *New =
    new(*this,8)VariableArrayType(EltTy,QualType(), NumElts, ASM, EltTypeQuals);

  VariableArrayTypes.push_back(New);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getDependentSizedArrayType - Returns a non-unique reference to
/// the type for a dependently-sized array of the specified element
/// type. FIXME: We will need these to be uniqued, or at least
/// comparable, at some point.
QualType ASTContext::getDependentSizedArrayType(QualType EltTy, Expr *NumElts,
                                                ArrayType::ArraySizeModifier ASM,
                                                unsigned EltTypeQuals) {
  assert((NumElts->isTypeDependent() || NumElts->isValueDependent()) && 
         "Size must be type- or value-dependent!");

  // Since we don't unique expressions, it isn't possible to unique
  // dependently-sized array types.

  DependentSizedArrayType *New =
      new (*this,8) DependentSizedArrayType(EltTy, QualType(), NumElts, 
                                            ASM, EltTypeQuals);

  DependentSizedArrayTypes.push_back(New);
  Types.push_back(New);
  return QualType(New, 0);
}

QualType ASTContext::getIncompleteArrayType(QualType EltTy,
                                            ArrayType::ArraySizeModifier ASM,
                                            unsigned EltTypeQuals) {
  llvm::FoldingSetNodeID ID;
  IncompleteArrayType::Profile(ID, EltTy);

  void *InsertPos = 0;
  if (IncompleteArrayType *ATP = 
       IncompleteArrayTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(ATP, 0);

  // If the element type isn't canonical, this won't be a canonical type
  // either, so fill in the canonical type field.
  QualType Canonical;

  if (!EltTy->isCanonical()) {
    Canonical = getIncompleteArrayType(getCanonicalType(EltTy),
                                       ASM, EltTypeQuals);

    // Get the new insert position for the node we care about.
    IncompleteArrayType *NewIP =
      IncompleteArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }

  IncompleteArrayType *New = new (*this,8) IncompleteArrayType(EltTy, Canonical,
                                                           ASM, EltTypeQuals);

  IncompleteArrayTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getVectorType - Return the unique reference to a vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType ASTContext::getVectorType(QualType vecType, unsigned NumElts) {
  BuiltinType *baseType;
  
  baseType = dyn_cast<BuiltinType>(getCanonicalType(vecType).getTypePtr());
  assert(baseType != 0 && "getVectorType(): Expecting a built-in type");
         
  // Check if we've already instantiated a vector of this type.
  llvm::FoldingSetNodeID ID;
  VectorType::Profile(ID, vecType, NumElts, Type::Vector);      
  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType->isCanonical()) {
    Canonical = getVectorType(getCanonicalType(vecType), NumElts);
    
    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  VectorType *New = new (*this,8) VectorType(vecType, NumElts, Canonical);
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
  VectorType::Profile(ID, vecType, NumElts, Type::ExtVector);      
  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType->isCanonical()) {
    Canonical = getExtVectorType(getCanonicalType(vecType), NumElts);
    
    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  ExtVectorType *New = new (*this,8) ExtVectorType(vecType, NumElts, Canonical);
  VectorTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getFunctionTypeNoProto - Return a K&R style C function type like 'int()'.
///
QualType ASTContext::getFunctionTypeNoProto(QualType ResultTy) {
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionTypeNoProto::Profile(ID, ResultTy);
  
  void *InsertPos = 0;
  if (FunctionTypeNoProto *FT = 
        FunctionTypeNoProtos.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(FT, 0);
  
  QualType Canonical;
  if (!ResultTy->isCanonical()) {
    Canonical = getFunctionTypeNoProto(getCanonicalType(ResultTy));
    
    // Get the new insert position for the node we care about.
    FunctionTypeNoProto *NewIP =
      FunctionTypeNoProtos.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  
  FunctionTypeNoProto *New =new(*this,8)FunctionTypeNoProto(ResultTy,Canonical);
  Types.push_back(New);
  FunctionTypeNoProtos.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getFunctionType - Return a normal function type with a typed argument
/// list.  isVariadic indicates whether the argument list includes '...'.
QualType ASTContext::getFunctionType(QualType ResultTy,const QualType *ArgArray,
                                     unsigned NumArgs, bool isVariadic,
                                     unsigned TypeQuals) {
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionTypeProto::Profile(ID, ResultTy, ArgArray, NumArgs, isVariadic,
                             TypeQuals);

  void *InsertPos = 0;
  if (FunctionTypeProto *FTP = 
        FunctionTypeProtos.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(FTP, 0);
    
  // Determine whether the type being created is already canonical or not.  
  bool isCanonical = ResultTy->isCanonical();
  for (unsigned i = 0; i != NumArgs && isCanonical; ++i)
    if (!ArgArray[i]->isCanonical())
      isCanonical = false;

  // If this type isn't canonical, get the canonical version of it.
  QualType Canonical;
  if (!isCanonical) {
    llvm::SmallVector<QualType, 16> CanonicalArgs;
    CanonicalArgs.reserve(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      CanonicalArgs.push_back(getCanonicalType(ArgArray[i]));
    
    Canonical = getFunctionType(getCanonicalType(ResultTy),
                                &CanonicalArgs[0], NumArgs,
                                isVariadic, TypeQuals);
    
    // Get the new insert position for the node we care about.
    FunctionTypeProto *NewIP =
      FunctionTypeProtos.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!"); NewIP = NewIP;
  }
  
  // FunctionTypeProto objects are allocated with extra bytes after them
  // for a variable size array (for parameter types) at the end of them.
  FunctionTypeProto *FTP = 
    (FunctionTypeProto*)Allocate(sizeof(FunctionTypeProto) + 
                                 NumArgs*sizeof(QualType), 8);
  new (FTP) FunctionTypeProto(ResultTy, ArgArray, NumArgs, isVariadic,
                              TypeQuals, Canonical);
  Types.push_back(FTP);
  FunctionTypeProtos.InsertNode(FTP, InsertPos);
  return QualType(FTP, 0);
}

/// getTypeDeclType - Return the unique reference to the type for the
/// specified type declaration.
QualType ASTContext::getTypeDeclType(TypeDecl *Decl, TypeDecl* PrevDecl) {
  assert(Decl && "Passed null for Decl param");
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);
  
  if (TypedefDecl *Typedef = dyn_cast<TypedefDecl>(Decl))
    return getTypedefType(Typedef);
  else if (isa<TemplateTypeParmDecl>(Decl)) {
    assert(false && "Template type parameter types are always available.");
  } else if (ObjCInterfaceDecl *ObjCInterface = dyn_cast<ObjCInterfaceDecl>(Decl))
    return getObjCInterfaceType(ObjCInterface);

  if (CXXRecordDecl *CXXRecord = dyn_cast<CXXRecordDecl>(Decl)) {
    if (PrevDecl)
      Decl->TypeForDecl = PrevDecl->TypeForDecl;
    else
      Decl->TypeForDecl = new (*this,8) CXXRecordType(CXXRecord);
  }
  else if (RecordDecl *Record = dyn_cast<RecordDecl>(Decl)) {
    if (PrevDecl)
      Decl->TypeForDecl = PrevDecl->TypeForDecl;
    else
      Decl->TypeForDecl = new (*this,8) RecordType(Record);
  }
  else if (EnumDecl *Enum = dyn_cast<EnumDecl>(Decl)) {
    if (PrevDecl)
      Decl->TypeForDecl = PrevDecl->TypeForDecl;
    else
      Decl->TypeForDecl = new (*this,8) EnumType(Enum);
  }
  else
    assert(false && "TypeDecl without a type?");

  if (!PrevDecl) Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// getTypedefType - Return the unique reference to the type for the
/// specified typename decl.
QualType ASTContext::getTypedefType(TypedefDecl *Decl) {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);
  
  QualType Canonical = getCanonicalType(Decl->getUnderlyingType());
  Decl->TypeForDecl = new(*this,8) TypedefType(Type::TypeName, Decl, Canonical);
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// getObjCInterfaceType - Return the unique reference to the type for the
/// specified ObjC interface decl.
QualType ASTContext::getObjCInterfaceType(ObjCInterfaceDecl *Decl) {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);
  
  Decl->TypeForDecl = new(*this,8) ObjCInterfaceType(Type::ObjCInterface, Decl);
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// buildObjCInterfaceType - Returns a new type for the interface
/// declaration, regardless. It also removes any previously built 
/// record declaration so caller can rebuild it.
QualType ASTContext::buildObjCInterfaceType(ObjCInterfaceDecl *Decl) {
  const RecordDecl *&RD = ASTRecordForInterface[Decl];
  if (RD)
    RD = 0;
  Decl->TypeForDecl = new(*this,8) ObjCInterfaceType(Type::ObjCInterface, Decl);
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// \brief Retrieve the template type parameter type for a template
/// parameter with the given depth, index, and (optionally) name.
QualType ASTContext::getTemplateTypeParmType(unsigned Depth, unsigned Index, 
                                             IdentifierInfo *Name) {
  llvm::FoldingSetNodeID ID;
  TemplateTypeParmType::Profile(ID, Depth, Index, Name);
  void *InsertPos = 0;
  TemplateTypeParmType *TypeParm 
    = TemplateTypeParmTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (TypeParm)
    return QualType(TypeParm, 0);
  
  if (Name)
    TypeParm = new (*this, 8) TemplateTypeParmType(Depth, Index, Name,
                                         getTemplateTypeParmType(Depth, Index));
  else
    TypeParm = new (*this, 8) TemplateTypeParmType(Depth, Index);

  Types.push_back(TypeParm);
  TemplateTypeParmTypes.InsertNode(TypeParm, InsertPos);

  return QualType(TypeParm, 0);
}

QualType 
ASTContext::getClassTemplateSpecializationType(TemplateDecl *Template,
                                               unsigned NumArgs,
                                               uintptr_t *Args, bool *ArgIsType,
                                               QualType Canon) {
  llvm::FoldingSetNodeID ID;
  ClassTemplateSpecializationType::Profile(ID, Template, NumArgs, Args, 
                                           ArgIsType);
  void *InsertPos = 0;
  ClassTemplateSpecializationType *Spec
    = ClassTemplateSpecializationTypes.FindNodeOrInsertPos(ID, InsertPos);

  if (Spec)
    return QualType(Spec, 0);
  
  void *Mem = Allocate(sizeof(ClassTemplateSpecializationType) + 
                       (sizeof(uintptr_t) * 
                        (ClassTemplateSpecializationType::
                           getNumPackedWords(NumArgs) + 
                         NumArgs)), 8);
  Spec = new (Mem) ClassTemplateSpecializationType(Template, NumArgs, Args,
                                                   ArgIsType, Canon);
  Types.push_back(Spec);
  ClassTemplateSpecializationTypes.InsertNode(Spec, InsertPos);

  return QualType(Spec, 0);  
}

/// CmpProtocolNames - Comparison predicate for sorting protocols
/// alphabetically.
static bool CmpProtocolNames(const ObjCProtocolDecl *LHS,
                            const ObjCProtocolDecl *RHS) {
  return LHS->getDeclName() < RHS->getDeclName();
}

static void SortAndUniqueProtocols(ObjCProtocolDecl **&Protocols,
                                   unsigned &NumProtocols) {
  ObjCProtocolDecl **ProtocolsEnd = Protocols+NumProtocols;
  
  // Sort protocols, keyed by name.
  std::sort(Protocols, Protocols+NumProtocols, CmpProtocolNames);

  // Remove duplicates.
  ProtocolsEnd = std::unique(Protocols, ProtocolsEnd);
  NumProtocols = ProtocolsEnd-Protocols;
}


/// getObjCQualifiedInterfaceType - Return a ObjCQualifiedInterfaceType type for
/// the given interface decl and the conforming protocol list.
QualType ASTContext::getObjCQualifiedInterfaceType(ObjCInterfaceDecl *Decl,
                       ObjCProtocolDecl **Protocols, unsigned NumProtocols) {
  // Sort the protocol list alphabetically to canonicalize it.
  SortAndUniqueProtocols(Protocols, NumProtocols);
  
  llvm::FoldingSetNodeID ID;
  ObjCQualifiedInterfaceType::Profile(ID, Decl, Protocols, NumProtocols);
  
  void *InsertPos = 0;
  if (ObjCQualifiedInterfaceType *QT =
      ObjCQualifiedInterfaceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(QT, 0);
  
  // No Match;
  ObjCQualifiedInterfaceType *QType =
    new (*this,8) ObjCQualifiedInterfaceType(Decl, Protocols, NumProtocols);

  Types.push_back(QType);
  ObjCQualifiedInterfaceTypes.InsertNode(QType, InsertPos);
  return QualType(QType, 0);
}

/// getObjCQualifiedIdType - Return an ObjCQualifiedIdType for the 'id' decl
/// and the conforming protocol list.
QualType ASTContext::getObjCQualifiedIdType(ObjCProtocolDecl **Protocols, 
                                            unsigned NumProtocols) {
  // Sort the protocol list alphabetically to canonicalize it.
  SortAndUniqueProtocols(Protocols, NumProtocols);

  llvm::FoldingSetNodeID ID;
  ObjCQualifiedIdType::Profile(ID, Protocols, NumProtocols);
  
  void *InsertPos = 0;
  if (ObjCQualifiedIdType *QT =
        ObjCQualifiedIdTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(QT, 0);
  
  // No Match;
  ObjCQualifiedIdType *QType =
    new (*this,8) ObjCQualifiedIdType(Protocols, NumProtocols);
  Types.push_back(QType);
  ObjCQualifiedIdTypes.InsertNode(QType, InsertPos);
  return QualType(QType, 0);
}

/// getTypeOfExpr - Unlike many "get<Type>" functions, we can't unique
/// TypeOfExpr AST's (since expression's are never shared). For example,
/// multiple declarations that refer to "typeof(x)" all contain different
/// DeclRefExpr's. This doesn't effect the type checker, since it operates 
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfExpr(Expr *tofExpr) {
  QualType Canonical = getCanonicalType(tofExpr->getType());
  TypeOfExpr *toe = new (*this,8) TypeOfExpr(tofExpr, Canonical);
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
  TypeOfType *tot = new (*this,8) TypeOfType(tofType, Canonical);
  Types.push_back(tot);
  return QualType(tot, 0);
}

/// getTagDeclType - Return the unique reference to the type for the
/// specified TagDecl (struct/union/class/enum) decl.
QualType ASTContext::getTagDeclType(TagDecl *Decl) {
  assert (Decl);
  return getTypeDeclType(Decl);
}

/// getSizeType - Return the unique type for "size_t" (C99 7.17), the result 
/// of the sizeof operator (C99 6.5.3.4p4). The value is target dependent and 
/// needs to agree with the definition in <stddef.h>. 
QualType ASTContext::getSizeType() const {
  return getFromTargetType(Target.getSizeType());
}

/// getWCharType - Return the unique type for "wchar_t" (C99 7.17), the
/// width of characters in wide strings, The value is target dependent and 
/// needs to agree with the definition in <stddef.h>.
QualType ASTContext::getWCharType() const {
  if (LangOpts.CPlusPlus)
    return WCharTy;

  // FIXME: In C, shouldn't WCharTy just be a typedef of the target's
  // wide-character type?
  return getFromTargetType(Target.getWCharType());
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

/// getCanonicalType - Return the canonical (structural) type corresponding to
/// the specified potentially non-canonical type.  The non-canonical version
/// of a type may have many "decorated" versions of types.  Decorators can
/// include typedefs, 'typeof' operators, etc. The returned type is guaranteed
/// to be free of any of these, allowing two canonical types to be compared
/// for exact equality with a simple pointer comparison.
QualType ASTContext::getCanonicalType(QualType T) {
  QualType CanType = T.getTypePtr()->getCanonicalTypeInternal();
  
  // If the result has type qualifiers, make sure to canonicalize them as well.
  unsigned TypeQuals = T.getCVRQualifiers() | CanType.getCVRQualifiers();
  if (TypeQuals == 0) return CanType;

  // If the type qualifiers are on an array type, get the canonical type of the
  // array with the qualifiers applied to the element type.
  ArrayType *AT = dyn_cast<ArrayType>(CanType);
  if (!AT)
    return CanType.getQualifiedType(TypeQuals);
  
  // Get the canonical version of the element with the extra qualifiers on it.
  // This can recursively sink qualifiers through multiple levels of arrays.
  QualType NewEltTy=AT->getElementType().getWithAdditionalQualifiers(TypeQuals);
  NewEltTy = getCanonicalType(NewEltTy);
  
  if (ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT))
    return getConstantArrayType(NewEltTy, CAT->getSize(),CAT->getSizeModifier(),
                                CAT->getIndexTypeQualifier());
  if (IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT))
    return getIncompleteArrayType(NewEltTy, IAT->getSizeModifier(),
                                  IAT->getIndexTypeQualifier());
  
  if (DependentSizedArrayType *DSAT = dyn_cast<DependentSizedArrayType>(AT))
    return getDependentSizedArrayType(NewEltTy, DSAT->getSizeExpr(),
                                      DSAT->getSizeModifier(),
                                      DSAT->getIndexTypeQualifier());    

  VariableArrayType *VAT = cast<VariableArrayType>(AT);
  return getVariableArrayType(NewEltTy, VAT->getSizeExpr(),
                              VAT->getSizeModifier(),
                              VAT->getIndexTypeQualifier());
}


const ArrayType *ASTContext::getAsArrayType(QualType T) {
  // Handle the non-qualified case efficiently.
  if (T.getCVRQualifiers() == 0) {
    // Handle the common positive case fast.
    if (const ArrayType *AT = dyn_cast<ArrayType>(T))
      return AT;
  }
  
  // Handle the common negative case fast, ignoring CVR qualifiers.
  QualType CType = T->getCanonicalTypeInternal();
    
  // Make sure to look through type qualifiers (like ASQuals) for the negative
  // test.
  if (!isa<ArrayType>(CType) &&
      !isa<ArrayType>(CType.getUnqualifiedType()))
    return 0;
  
  // Apply any CVR qualifiers from the array type to the element type.  This
  // implements C99 6.7.3p8: "If the specification of an array type includes
  // any type qualifiers, the element type is so qualified, not the array type."
  
  // If we get here, we either have type qualifiers on the type, or we have
  // sugar such as a typedef in the way.  If we have type qualifiers on the type
  // we must propagate them down into the elemeng type.
  unsigned CVRQuals = T.getCVRQualifiers();
  unsigned AddrSpace = 0;
  Type *Ty = T.getTypePtr();
  
  // Rip through ASQualType's and typedefs to get to a concrete type.
  while (1) {
    if (const ASQualType *ASQT = dyn_cast<ASQualType>(Ty)) {
      AddrSpace = ASQT->getAddressSpace();
      Ty = ASQT->getBaseType();
    } else {
      T = Ty->getDesugaredType();
      if (T.getTypePtr() == Ty && T.getCVRQualifiers() == 0)
        break;
      CVRQuals |= T.getCVRQualifiers();
      Ty = T.getTypePtr();
    }
  }
  
  // If we have a simple case, just return now.
  const ArrayType *ATy = dyn_cast<ArrayType>(Ty);
  if (ATy == 0 || (AddrSpace == 0 && CVRQuals == 0))
    return ATy;
  
  // Otherwise, we have an array and we have qualifiers on it.  Push the
  // qualifiers into the array element type and return a new array type.
  // Get the canonical version of the element with the extra qualifiers on it.
  // This can recursively sink qualifiers through multiple levels of arrays.
  QualType NewEltTy = ATy->getElementType();
  if (AddrSpace)
    NewEltTy = getASQualType(NewEltTy, AddrSpace);
  NewEltTy = NewEltTy.getWithAdditionalQualifiers(CVRQuals);
  
  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(ATy))
    return cast<ArrayType>(getConstantArrayType(NewEltTy, CAT->getSize(),
                                                CAT->getSizeModifier(),
                                                CAT->getIndexTypeQualifier()));
  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(ATy))
    return cast<ArrayType>(getIncompleteArrayType(NewEltTy,
                                                  IAT->getSizeModifier(),
                                                 IAT->getIndexTypeQualifier()));

  if (const DependentSizedArrayType *DSAT 
        = dyn_cast<DependentSizedArrayType>(ATy))
    return cast<ArrayType>(
                     getDependentSizedArrayType(NewEltTy, 
                                                DSAT->getSizeExpr(),
                                                DSAT->getSizeModifier(),
                                                DSAT->getIndexTypeQualifier()));
  
  const VariableArrayType *VAT = cast<VariableArrayType>(ATy);
  return cast<ArrayType>(getVariableArrayType(NewEltTy, VAT->getSizeExpr(),
                                              VAT->getSizeModifier(),
                                              VAT->getIndexTypeQualifier()));
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
  return PtrTy.getQualifiedType(PrettyArrayType->getIndexTypeQualifier());
}

QualType ASTContext::getBaseElementType(const VariableArrayType *VAT) {
  QualType ElemTy = VAT->getElementType();
  
  if (const VariableArrayType *VAT = getAsVariableArrayType(ElemTy))
    return getBaseElementType(VAT);
  
  return ElemTy;
}

/// getFloatingRank - Return a relative rank for floating point types.
/// This routine will assert if passed a built-in type that isn't a float.
static FloatingRank getFloatingRank(QualType T) {
  if (const ComplexType *CT = T->getAsComplexType())
    return getFloatingRank(CT->getElementType());

  assert(T->getAsBuiltinType() && "getFloatingRank(): not a floating type");
  switch (T->getAsBuiltinType()->getKind()) {
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
  assert(T->isCanonical() && "T should be canonicalized");
  if (EnumType* ET = dyn_cast<EnumType>(T))
    T = ET->getDecl()->getIntegerType().getTypePtr();

  // There are two things which impact the integer rank: the width, and
  // the ordering of builtins.  The builtin ordering is encoded in the
  // bottom three bits; the width is encoded in the bits above that.
  if (FixedWidthIntType* FWIT = dyn_cast<FixedWidthIntType>(T)) {
    return FWIT->getWidth() << 3;
  }

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
  }
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

// getCFConstantStringType - Return the type used for constant CFStrings. 
QualType ASTContext::getCFConstantStringType() {
  if (!CFConstantStringTypeDecl) {
    CFConstantStringTypeDecl = 
      RecordDecl::Create(*this, TagDecl::TK_struct, TUDecl, SourceLocation(), 
                         &Idents.get("NSConstantString"));
    QualType FieldTypes[4];
  
    // const int *isa;
    FieldTypes[0] = getPointerType(IntTy.getQualifiedType(QualType::Const));  
    // int flags;
    FieldTypes[1] = IntTy;
    // const char *str;
    FieldTypes[2] = getPointerType(CharTy.getQualifiedType(QualType::Const));  
    // long length;
    FieldTypes[3] = LongTy;  
  
    // Create fields
    for (unsigned i = 0; i < 4; ++i) {
      FieldDecl *Field = FieldDecl::Create(*this, CFConstantStringTypeDecl, 
                                           SourceLocation(), 0,
                                           FieldTypes[i], /*BitWidth=*/0, 
                                           /*Mutable=*/false);
      CFConstantStringTypeDecl->addDecl(Field);
    }

    CFConstantStringTypeDecl->completeDefinition(*this);
  }
  
  return getTagDeclType(CFConstantStringTypeDecl);
}

QualType ASTContext::getObjCFastEnumerationStateType()
{
  if (!ObjCFastEnumerationStateTypeDecl) {
    ObjCFastEnumerationStateTypeDecl =
      RecordDecl::Create(*this, TagDecl::TK_struct, TUDecl, SourceLocation(),
                         &Idents.get("__objcFastEnumerationState"));
    
    QualType FieldTypes[] = {
      UnsignedLongTy,
      getPointerType(ObjCIdType),
      getPointerType(UnsignedLongTy),
      getConstantArrayType(UnsignedLongTy,
                           llvm::APInt(32, 5), ArrayType::Normal, 0)
    };
    
    for (size_t i = 0; i < 4; ++i) {
      FieldDecl *Field = FieldDecl::Create(*this, 
                                           ObjCFastEnumerationStateTypeDecl, 
                                           SourceLocation(), 0, 
                                           FieldTypes[i], /*BitWidth=*/0, 
                                           /*Mutable=*/false);
      ObjCFastEnumerationStateTypeDecl->addDecl(Field);
    }
    
    ObjCFastEnumerationStateTypeDecl->completeDefinition(*this);
  }
  
  return getTagDeclType(ObjCFastEnumerationStateTypeDecl);
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
int ASTContext::getObjCEncodingTypeSize(QualType type) {
  uint64_t sz = getTypeSize(type);
  
  // Make all integer and enum types at least as large as an int
  if (sz > 0 && type->isIntegralType())
    sz = std::max(sz, getTypeSize(IntTy));
  // Treat arrays as pointers, since that's how they're passed in.
  else if (type->isArrayType())
    sz = getTypeSize(VoidPtrTy);
  return sz / getTypeSize(CharTy);
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
  int PtrSize = getTypeSize(VoidPtrTy) / getTypeSize(CharTy);
  // The first two arguments (self and _cmd) are pointers; account for
  // their size.
  int ParmOffset = 2 * PtrSize;
  int NumOfParams = Decl->getNumParams();
  for (int i = 0; i < NumOfParams; i++) {
    QualType PType = Decl->getParamDecl(i)->getType();
    int sz = getObjCEncodingTypeSize (PType);
    assert (sz > 0 && "getObjCEncodingForMethodDecl - Incomplete param type");
    ParmOffset += sz;
  }
  S += llvm::utostr(ParmOffset);
  S += "@0:";
  S += llvm::utostr(PtrSize);
  
  // Argument types.
  ParmOffset = 2 * PtrSize;
  for (int i = 0; i < NumOfParams; i++) {
    ParmVarDecl *PVDecl = Decl->getParamDecl(i);
    QualType PType = PVDecl->getOriginalType(); 
    if (const ArrayType *AT =
          dyn_cast<ArrayType>(PType->getCanonicalTypeInternal())) 
        // Use array's original type only if it has known number of
        // elements.
        if (!dyn_cast<ConstantArrayType>(AT))
          PType = PVDecl->getType();
    // Process argument qualifiers for user supplied arguments; such as,
    // 'in', 'inout', etc.
    getObjCEncodingForTypeQualifier(PVDecl->getObjCDeclQualifier(), S);
    getObjCEncodingForType(PType, S);
    S += llvm::utostr(ParmOffset);
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
             i = CID->propimpl_begin(), e = CID->propimpl_end(); i != e; ++i) {
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
             i = OID->propimpl_begin(), e = OID->propimpl_end(); i != e; ++i) {
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
  getObjCEncodingForTypeImpl(PD->getType(), S, true, true, NULL, 
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
  if (dyn_cast<TypedefType>(PointeeTy.getTypePtr())) {
    if (const BuiltinType *BT = PointeeTy->getAsBuiltinType()) {
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
                                        FieldDecl *Field) const {
  // We follow the behavior of gcc, expanding structures which are
  // directly pointed to, and expanding embedded structures. Note that
  // these rules are sufficient to prevent recursive encoding of the
  // same type.
  getObjCEncodingForTypeImpl(T, S, true, true, Field, 
                             true /* outermost type */);
}

static void EncodeBitField(const ASTContext *Context, std::string& S, 
                           FieldDecl *FD) {
  const Expr *E = FD->getBitWidth();
  assert(E && "bitfield width not there - getObjCEncodingForTypeImpl");
  ASTContext *Ctx = const_cast<ASTContext*>(Context);
  unsigned N = E->getIntegerConstantExprValue(*Ctx).getZExtValue();
  S += 'b';
  S += llvm::utostr(N);
}

void ASTContext::getObjCEncodingForTypeImpl(QualType T, std::string& S,
                                            bool ExpandPointedToStructures,
                                            bool ExpandStructures,
                                            FieldDecl *FD,
                                            bool OutermostType,
                                            bool EncodingProperty) const {
  if (const BuiltinType *BT = T->getAsBuiltinType()) {
    if (FD && FD->isBitField()) {
      EncodeBitField(this, S, FD);
    }
    else {
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
      case BuiltinType::Float:      encoding = 'f'; break;
      case BuiltinType::Double:     encoding = 'd'; break;
      case BuiltinType::LongDouble: encoding = 'd'; break;
      }
    
      S += encoding;
    }
  }
  else if (T->isObjCQualifiedIdType()) {
    getObjCEncodingForTypeImpl(getObjCIdType(), S, 
                               ExpandPointedToStructures,
                               ExpandStructures, FD);
    if (FD || EncodingProperty) {
      // Note that we do extended encoding of protocol qualifer list
      // Only when doing ivar or property encoding.
      const ObjCQualifiedIdType *QIDT = T->getAsObjCQualifiedIdType();
      S += '"';
      for (unsigned i =0; i < QIDT->getNumProtocols(); i++) {
        ObjCProtocolDecl *Proto = QIDT->getProtocols(i);
        S += '<';
        S += Proto->getNameAsString();
        S += '>';
      }
      S += '"';
    }
    return;
  }
  else if (const PointerType *PT = T->getAsPointerType()) {
    QualType PointeeTy = PT->getPointeeType();
    bool isReadOnly = false;
    // For historical/compatibility reasons, the read-only qualifier of the
    // pointee gets emitted _before_ the '^'.  The read-only qualifier of
    // the pointer itself gets ignored, _unless_ we are looking at a typedef!
    // Also, do not emit the 'r' for anything but the outermost type! 
    if (dyn_cast<TypedefType>(T.getTypePtr())) {
      if (OutermostType && T.isConstQualified()) {
        isReadOnly = true;
        S += 'r';
      }
    }
    else if (OutermostType) {
      QualType P = PointeeTy;
      while (P->getAsPointerType())
        P = P->getAsPointerType()->getPointeeType();
      if (P.isConstQualified()) {
        isReadOnly = true;
        S += 'r';
      }
    }
    if (isReadOnly) {
      // Another legacy compatibility encoding. Some ObjC qualifier and type
      // combinations need to be rearranged.
      // Rewrite "in const" from "nr" to "rn"
      const char * s = S.c_str();
      int len = S.length();
      if (len >= 2 && s[len-2] == 'n' && s[len-1] == 'r') {
        std::string replace = "rn";
        S.replace(S.end()-2, S.end(), replace);
      }
    }
    if (isObjCIdStructType(PointeeTy)) {
      S += '@';
      return;
    }
    else if (PointeeTy->isObjCInterfaceType()) {
      if (dyn_cast<TypedefType>(PointeeTy.getTypePtr())) {
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
      if (FD || EncodingProperty) {
        const ObjCInterfaceType *OIT = PointeeTy->getAsObjCInterfaceType();
        ObjCInterfaceDecl *OI = OIT->getDecl();
        S += '"';
        S += OI->getNameAsCString();
        for (unsigned i =0; i < OIT->getNumProtocols(); i++) {
          ObjCProtocolDecl *Proto = OIT->getProtocol(i);
          S += '<';
          S += Proto->getNameAsString();
          S += '>';
        } 
        S += '"';
      }
      return;
    } else if (isObjCClassStructType(PointeeTy)) {
      S += '#';
      return;
    } else if (isObjCSelType(PointeeTy)) {
      S += ':';
      return;
    }
    
    if (PointeeTy->isCharType()) {
      // char pointer types should be encoded as '*' unless it is a
      // type that has been typedef'd to 'BOOL'.
      if (!isTypeTypedefedAsBOOL(PointeeTy)) {
        S += '*';
        return;
      }
    }
    
    S += '^';
    getLegacyIntegralTypeEncoding(PointeeTy);

    getObjCEncodingForTypeImpl(PointeeTy, S, 
                               false, ExpandPointedToStructures, 
                               NULL);
  } else if (const ArrayType *AT =
               // Ignore type qualifiers etc.
               dyn_cast<ArrayType>(T->getCanonicalTypeInternal())) {
    S += '[';
    
    if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT))
      S += llvm::utostr(CAT->getSize().getZExtValue());
    else
      assert(0 && "Unhandled array type!");
    
    getObjCEncodingForTypeImpl(AT->getElementType(), S, 
                               false, ExpandStructures, FD);
    S += ']';
  } else if (T->getAsFunctionType()) {
    S += '?';
  } else if (const RecordType *RTy = T->getAsRecordType()) {
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
  } else if (T->isEnumeralType()) {
    if (FD && FD->isBitField())
      EncodeBitField(this, S, FD);
    else
      S += 'i';
  } else if (T->isBlockPointerType()) {
    S += "@?"; // Unlike a pointer-to-function, which is "^?".
  } else if (T->isObjCInterfaceType()) {
    // @encode(class_name)
    ObjCInterfaceDecl *OI = T->getAsObjCInterfaceType()->getDecl();
    S += '{';
    const IdentifierInfo *II = OI->getIdentifier();
    S += II->getName();
    S += '=';
    std::vector<FieldDecl*> RecFields;
    CollectObjCIvars(OI, RecFields);
    for (unsigned int i = 0; i != RecFields.size(); i++) {
      if (RecFields[i]->isBitField())
        getObjCEncodingForTypeImpl(RecFields[i]->getType(), S, false, true, 
                                   RecFields[i]);
      else
        getObjCEncodingForTypeImpl(RecFields[i]->getType(), S, false, true, 
                                   FD);
    }
    S += '}';
  }
  else
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

void ASTContext::setBuiltinVaListType(QualType T)
{
  assert(BuiltinVaListType.isNull() && "__builtin_va_list type already set!");
    
  BuiltinVaListType = T;
}

void ASTContext::setObjCIdType(TypedefDecl *TD)
{
  ObjCIdType = getTypedefType(TD);

  // typedef struct objc_object *id;
  const PointerType *ptr = TD->getUnderlyingType()->getAsPointerType();
  // User error - caller will issue diagnostics.
  if (!ptr)
    return;
  const RecordType *rec = ptr->getPointeeType()->getAsStructureType();
  // User error - caller will issue diagnostics.
  if (!rec)
    return;
  IdStructType = rec;
}

void ASTContext::setObjCSelType(TypedefDecl *TD)
{
  ObjCSelType = getTypedefType(TD);

  // typedef struct objc_selector *SEL;
  const PointerType *ptr = TD->getUnderlyingType()->getAsPointerType();
  if (!ptr)
    return;
  const RecordType *rec = ptr->getPointeeType()->getAsStructureType();
  if (!rec)
    return;
  SelStructType = rec;
}

void ASTContext::setObjCProtoType(QualType QT)
{
  ObjCProtoType = QT;
}

void ASTContext::setObjCClassType(TypedefDecl *TD)
{
  ObjCClassType = getTypedefType(TD);

  // typedef struct objc_class *Class;
  const PointerType *ptr = TD->getUnderlyingType()->getAsPointerType();
  assert(ptr && "'Class' incorrectly typed");
  const RecordType *rec = ptr->getPointeeType()->getAsStructureType();
  assert(rec && "'Class' incorrectly typed");
  ClassStructType = rec;
}

void ASTContext::setObjCConstantStringInterface(ObjCInterfaceDecl *Decl) {
  assert(ObjCConstantStringType.isNull() && 
         "'NSConstantString' type already set!");
  
  ObjCConstantStringType = getObjCInterfaceType(Decl);
}

/// getFromTargetType - Given one of the integer types provided by
/// TargetInfo, produce the corresponding type. The unsigned @p Type
/// is actually a value of type @c TargetInfo::IntType.
QualType ASTContext::getFromTargetType(unsigned Type) const {
  switch (Type) {
  case TargetInfo::NoInt: return QualType(); 
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
  return QualType();
}

//===----------------------------------------------------------------------===//
//                        Type Predicates.
//===----------------------------------------------------------------------===//

/// isObjCNSObjectType - Return true if this is an NSObject object using
/// NSObject attribute on a c-style pointer type.
/// FIXME - Make it work directly on types.
///
bool ASTContext::isObjCNSObjectType(QualType Ty) const {
  if (TypedefType *TDT = dyn_cast<TypedefType>(Ty)) {
    if (TypedefDecl *TD = TDT->getDecl())
      if (TD->getAttr<ObjCNSObjectAttr>())
        return true;
  }
  return false;  
}

/// isObjCObjectPointerType - Returns true if type is an Objective-C pointer
/// to an object type.  This includes "id" and "Class" (two 'special' pointers
/// to struct), Interface* (pointer to ObjCInterfaceType) and id<P> (qualified
/// ID type).
bool ASTContext::isObjCObjectPointerType(QualType Ty) const {
  if (Ty->isObjCQualifiedIdType())
    return true;
  
  // Blocks are objects.
  if (Ty->isBlockPointerType())
    return true;
    
  // All other object types are pointers.
  if (!Ty->isPointerType())
    return false;
  
  // Check to see if this is 'id' or 'Class', both of which are typedefs for
  // pointer types.  This looks for the typedef specifically, not for the
  // underlying type.
  if (Ty == getObjCIdType() || Ty == getObjCClassType())
    return true;
  
  // If this a pointer to an interface (e.g. NSString*), it is ok.
  if (Ty->getAsPointerType()->getPointeeType()->isObjCInterfaceType())
    return true;
  
  // If is has NSObject attribute, OK as well.
  return isObjCNSObjectType(Ty);
}

//===----------------------------------------------------------------------===//
//                        Type Compatibility Testing
//===----------------------------------------------------------------------===//

/// typesAreBlockCompatible - This routine is called when comparing two
/// block types. Types must be strictly compatible here. For example,
/// C unfortunately doesn't produce an error for the following:
/// 
///   int (*emptyArgFunc)();
///   int (*intArgList)(int) = emptyArgFunc;
/// 
/// For blocks, we will produce an error for the following (similar to C++):
///
///   int (^emptyArgBlock)();
///   int (^intArgBlock)(int) = emptyArgBlock;
///
/// FIXME: When the dust settles on this integration, fold this into mergeTypes.
///
bool ASTContext::typesAreBlockCompatible(QualType lhs, QualType rhs) {
  const FunctionType *lbase = lhs->getAsFunctionType();
  const FunctionType *rbase = rhs->getAsFunctionType();
  const FunctionTypeProto *lproto = dyn_cast<FunctionTypeProto>(lbase);
  const FunctionTypeProto *rproto = dyn_cast<FunctionTypeProto>(rbase);
  if (lproto && rproto)
    return !mergeTypes(lhs, rhs).isNull();
  return false;
}

/// areCompatVectorTypes - Return true if the two specified vector types are 
/// compatible.
static bool areCompatVectorTypes(const VectorType *LHS,
                                 const VectorType *RHS) {
  assert(LHS->isCanonical() && RHS->isCanonical());
  return LHS->getElementType() == RHS->getElementType() &&
         LHS->getNumElements() == RHS->getNumElements();
}

/// canAssignObjCInterfaces - Return true if the two interface types are
/// compatible for assignment from RHS to LHS.  This handles validation of any
/// protocol qualifiers on the LHS or RHS.
///
bool ASTContext::canAssignObjCInterfaces(const ObjCInterfaceType *LHS,
                                         const ObjCInterfaceType *RHS) {
  // Verify that the base decls are compatible: the RHS must be a subclass of
  // the LHS.
  if (!LHS->getDecl()->isSuperClassOf(RHS->getDecl()))
    return false;
  
  // RHS must have a superset of the protocols in the LHS.  If the LHS is not
  // protocol qualified at all, then we are good.
  if (!isa<ObjCQualifiedInterfaceType>(LHS))
    return true;
  
  // Okay, we know the LHS has protocol qualifiers.  If the RHS doesn't, then it
  // isn't a superset.
  if (!isa<ObjCQualifiedInterfaceType>(RHS))
    return true;  // FIXME: should return false!
  
  // Finally, we must have two protocol-qualified interfaces.
  const ObjCQualifiedInterfaceType *LHSP =cast<ObjCQualifiedInterfaceType>(LHS);
  const ObjCQualifiedInterfaceType *RHSP =cast<ObjCQualifiedInterfaceType>(RHS);
  ObjCQualifiedInterfaceType::qual_iterator LHSPI = LHSP->qual_begin();
  ObjCQualifiedInterfaceType::qual_iterator LHSPE = LHSP->qual_end();
  ObjCQualifiedInterfaceType::qual_iterator RHSPI = RHSP->qual_begin();
  ObjCQualifiedInterfaceType::qual_iterator RHSPE = RHSP->qual_end();
  
  // All protocols in LHS must have a presence in RHS.  Since the protocol lists
  // are both sorted alphabetically and have no duplicates, we can scan RHS and
  // LHS in a single parallel scan until we run out of elements in LHS.
  assert(LHSPI != LHSPE && "Empty LHS protocol list?");
  ObjCProtocolDecl *LHSProto = *LHSPI;
  
  while (RHSPI != RHSPE) {
    ObjCProtocolDecl *RHSProto = *RHSPI++;
    // If the RHS has a protocol that the LHS doesn't, ignore it.
    if (RHSProto != LHSProto)
      continue;
    
    // Otherwise, the RHS does have this element.
    ++LHSPI;
    if (LHSPI == LHSPE)
      return true;  // All protocols in LHS exist in RHS.
    
    LHSProto = *LHSPI;
  }
  
  // If we got here, we didn't find one of the LHS's protocols in the RHS list.
  return false;
}

bool ASTContext::areComparableObjCPointerTypes(QualType LHS, QualType RHS) {
  // get the "pointed to" types
  const PointerType *LHSPT = LHS->getAsPointerType();
  const PointerType *RHSPT = RHS->getAsPointerType();
  
  if (!LHSPT || !RHSPT)
    return false;
    
  QualType lhptee = LHSPT->getPointeeType();
  QualType rhptee = RHSPT->getPointeeType();
  const ObjCInterfaceType* LHSIface = lhptee->getAsObjCInterfaceType();
  const ObjCInterfaceType* RHSIface = rhptee->getAsObjCInterfaceType();
  // ID acts sort of like void* for ObjC interfaces
  if (LHSIface && isObjCIdStructType(rhptee))
    return true;
  if (RHSIface && isObjCIdStructType(lhptee))
    return true;
  if (!LHSIface || !RHSIface)
    return false;
  return canAssignObjCInterfaces(LHSIface, RHSIface) ||
         canAssignObjCInterfaces(RHSIface, LHSIface);
}

/// typesAreCompatible - C99 6.7.3p9: For two qualified types to be compatible, 
/// both shall have the identically qualified version of a compatible type.
/// C99 6.2.7p1: Two types have compatible types if their types are the 
/// same. See 6.7.[2,3,5] for additional rules.
bool ASTContext::typesAreCompatible(QualType LHS, QualType RHS) {
  return !mergeTypes(LHS, RHS).isNull();
}

QualType ASTContext::mergeFunctionTypes(QualType lhs, QualType rhs) {
  const FunctionType *lbase = lhs->getAsFunctionType();
  const FunctionType *rbase = rhs->getAsFunctionType();
  const FunctionTypeProto *lproto = dyn_cast<FunctionTypeProto>(lbase);
  const FunctionTypeProto *rproto = dyn_cast<FunctionTypeProto>(rbase);
  bool allLTypes = true;
  bool allRTypes = true;

  // Check return type
  QualType retType = mergeTypes(lbase->getResultType(), rbase->getResultType());
  if (retType.isNull()) return QualType();
  if (getCanonicalType(retType) != getCanonicalType(lbase->getResultType()))
    allLTypes = false;
  if (getCanonicalType(retType) != getCanonicalType(rbase->getResultType()))
    allRTypes = false;

  if (lproto && rproto) { // two C99 style function prototypes
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
      QualType argtype = mergeTypes(largtype, rargtype);
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
                           lproto->isVariadic(), lproto->getTypeQuals());
  }

  if (lproto) allRTypes = false;
  if (rproto) allLTypes = false;

  const FunctionTypeProto *proto = lproto ? lproto : rproto;
  if (proto) {
    if (proto->isVariadic()) return QualType();
    // Check that the types are compatible with the types that
    // would result from default argument promotions (C99 6.7.5.3p15).
    // The only types actually affected are promotable integer
    // types and floats, which would be passed as a different
    // type depending on whether the prototype is visible.
    unsigned proto_nargs = proto->getNumArgs();
    for (unsigned i = 0; i < proto_nargs; ++i) {
      QualType argTy = proto->getArgType(i);
      if (argTy->isPromotableIntegerType() ||
          getCanonicalType(argTy).getUnqualifiedType() == FloatTy)
        return QualType();
    }

    if (allLTypes) return lhs;
    if (allRTypes) return rhs;
    return getFunctionType(retType, proto->arg_type_begin(),
                           proto->getNumArgs(), lproto->isVariadic(),
                           lproto->getTypeQuals());
  }

  if (allLTypes) return lhs;
  if (allRTypes) return rhs;
  return getFunctionTypeNoProto(retType);
}

QualType ASTContext::mergeTypes(QualType LHS, QualType RHS) {
  // C++ [expr]: If an expression initially has the type "reference to T", the
  // type is adjusted to "T" prior to any further analysis, the expression
  // designates the object or function denoted by the reference, and the
  // expression is an lvalue.
  // FIXME: C++ shouldn't be going through here!  The rules are different
  // enough that they should be handled separately.
  if (const ReferenceType *RT = LHS->getAsReferenceType())
    LHS = RT->getPointeeType();
  if (const ReferenceType *RT = RHS->getAsReferenceType())
    RHS = RT->getPointeeType();

  QualType LHSCan = getCanonicalType(LHS),
           RHSCan = getCanonicalType(RHS);

  // If two types are identical, they are compatible.
  if (LHSCan == RHSCan)
    return LHS;

  // If the qualifiers are different, the types aren't compatible
  if (LHSCan.getCVRQualifiers() != RHSCan.getCVRQualifiers() ||
      LHSCan.getAddressSpace() != RHSCan.getAddressSpace())
    return QualType();

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
  
  // Consider qualified interfaces and interfaces the same.
  if (LHSClass == Type::ObjCQualifiedInterface) LHSClass = Type::ObjCInterface;
  if (RHSClass == Type::ObjCQualifiedInterface) RHSClass = Type::ObjCInterface;

  // If the canonical type classes don't match.
  if (LHSClass != RHSClass) {
    // ID is compatible with all qualified id types.
    if (LHS->isObjCQualifiedIdType()) {
      if (const PointerType *PT = RHS->getAsPointerType()) {
        QualType pType = PT->getPointeeType();
        if (isObjCIdStructType(pType))
          return LHS;
        // FIXME: need to use ObjCQualifiedIdTypesAreCompatible(LHS, RHS, true).
        // Unfortunately, this API is part of Sema (which we don't have access
        // to. Need to refactor. The following check is insufficient, since we 
        // need to make sure the class implements the protocol.
        if (pType->isObjCInterfaceType())
          return LHS;
      }
    }
    if (RHS->isObjCQualifiedIdType()) {
      if (const PointerType *PT = LHS->getAsPointerType()) {
        QualType pType = PT->getPointeeType();
        if (isObjCIdStructType(pType))
          return RHS;
        // FIXME: need to use ObjCQualifiedIdTypesAreCompatible(LHS, RHS, true).
        // Unfortunately, this API is part of Sema (which we don't have access
        // to. Need to refactor. The following check is insufficient, since we 
        // need to make sure the class implements the protocol.
        if (pType->isObjCInterfaceType())
          return RHS;
      }
    }
    // C99 6.7.2.2p4: Each enumerated type shall be compatible with char,
    // a signed integer type, or an unsigned integer type. 
    if (const EnumType* ETy = LHS->getAsEnumType()) {
      if (ETy->getDecl()->getIntegerType() == RHSCan.getUnqualifiedType())
        return RHS;
    }
    if (const EnumType* ETy = RHS->getAsEnumType()) {
      if (ETy->getDecl()->getIntegerType() == LHSCan.getUnqualifiedType())
        return LHS;
    }

    return QualType();
  }

  // The canonical type classes match.
  switch (LHSClass) {
  case Type::Pointer:
  {
    // Merge two pointer types, while trying to preserve typedef info
    QualType LHSPointee = LHS->getAsPointerType()->getPointeeType();
    QualType RHSPointee = RHS->getAsPointerType()->getPointeeType();
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
    QualType LHSPointee = LHS->getAsBlockPointerType()->getPointeeType();
    QualType RHSPointee = RHS->getAsBlockPointerType()->getPointeeType();
    QualType ResultType = mergeTypes(LHSPointee, RHSPointee);
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
    return getIncompleteArrayType(ResultType, ArrayType::ArraySizeModifier(),0);
  }
  case Type::FunctionNoProto:
    return mergeFunctionTypes(LHS, RHS);
  case Type::Tagged:
    // FIXME: Why are these compatible?
    if (isObjCIdStructType(LHS) && isObjCClassStructType(RHS)) return LHS;
    if (isObjCClassStructType(LHS) && isObjCIdStructType(RHS)) return LHS;
    return QualType();
  case Type::Builtin:
    // Only exactly equal builtin types are compatible, which is tested above.
    return QualType();
  case Type::Complex:
    // Distinct complex types are incompatible.
    return QualType();
  case Type::Vector:
    if (areCompatVectorTypes(LHS->getAsVectorType(), RHS->getAsVectorType()))
      return LHS;
    return QualType();
  case Type::ObjCInterface:
    // Distinct ObjC interfaces are not compatible; see canAssignObjCInterfaces
    // for checking assignment/comparison safety
    return QualType();
  case Type::ObjCQualifiedId:
    // Distinct qualified id's are not compatible.
    return QualType();
  default:
    assert(0 && "unexpected type");
    return QualType();
  }
}

//===----------------------------------------------------------------------===//
//                         Integer Predicates
//===----------------------------------------------------------------------===//

unsigned ASTContext::getIntWidth(QualType T) {
  if (T == BoolTy)
    return 1;
  if (FixedWidthIntType* FWIT = dyn_cast<FixedWidthIntType>(T)) {
    return FWIT->getWidth();
  }
  // For builtin types, just use the standard type sizing method
  return (unsigned)getTypeSize(T);
}

QualType ASTContext::getCorrespondingUnsignedType(QualType T) {
  assert(T->isSignedIntegerType() && "Unexpected type");
  if (const EnumType* ETy = T->getAsEnumType())
    T = ETy->getDecl()->getIntegerType();
  const BuiltinType* BTy = T->getAsBuiltinType();
  assert (BTy && "Unexpected signed integer type");
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
  default:
    assert(0 && "Unexpected signed integer type");
    return QualType();
  }
}


//===----------------------------------------------------------------------===//
//                         Serialization Support
//===----------------------------------------------------------------------===//

/// Emit - Serialize an ASTContext object to Bitcode.
void ASTContext::Emit(llvm::Serializer& S) const {
  S.Emit(LangOpts);
  S.EmitRef(SourceMgr);
  S.EmitRef(Target);
  S.EmitRef(Idents);
  S.EmitRef(Selectors);

  // Emit the size of the type vector so that we can reserve that size
  // when we reconstitute the ASTContext object.
  S.EmitInt(Types.size());
  
  for (std::vector<Type*>::const_iterator I=Types.begin(), E=Types.end(); 
                                          I!=E;++I)    
    (*I)->Emit(S);

  S.EmitOwnedPtr(TUDecl);

  // FIXME: S.EmitOwnedPtr(CFConstantStringTypeDecl);
}

ASTContext* ASTContext::Create(llvm::Deserializer& D) {
  
  // Read the language options.
  LangOptions LOpts;
  LOpts.Read(D);
  
  SourceManager &SM = D.ReadRef<SourceManager>();
  TargetInfo &t = D.ReadRef<TargetInfo>();
  IdentifierTable &idents = D.ReadRef<IdentifierTable>();
  SelectorTable &sels = D.ReadRef<SelectorTable>();

  unsigned size_reserve = D.ReadInt();
  
  ASTContext* A = new ASTContext(LOpts, SM, t, idents, sels,
                                 size_reserve);
  
  for (unsigned i = 0; i < size_reserve; ++i)
    Type::Create(*A,i,D);
  
  A->TUDecl = cast<TranslationUnitDecl>(D.ReadOwnedPtr<Decl>(*A));

  // FIXME: A->CFConstantStringTypeDecl = D.ReadOwnedPtr<RecordDecl>();
  
  return A;
}
