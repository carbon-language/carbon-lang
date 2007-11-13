//===--- ASTContext.cpp - Context to hold long-lived AST nodes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;

enum FloatingRank {
  FloatRank, DoubleRank, LongDoubleRank
};

ASTContext::~ASTContext() {
  // Deallocate all the types.
  while (!Types.empty()) {
    if (FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(Types.back())) {
      // Destroy the object, but don't call delete.  These are malloc'd.
      FT->~FunctionTypeProto();
      free(FT);
    } else {
      delete Types.back();
    }
    Types.pop_back();
  }
}

void ASTContext::PrintStats() const {
  fprintf(stderr, "*** AST Context Stats:\n");
  fprintf(stderr, "  %d types total.\n", (int)Types.size());
  unsigned NumBuiltin = 0, NumPointer = 0, NumArray = 0, NumFunctionP = 0;
  unsigned NumVector = 0, NumComplex = 0;
  unsigned NumFunctionNP = 0, NumTypeName = 0, NumTagged = 0, NumReference = 0;
  
  unsigned NumTagStruct = 0, NumTagUnion = 0, NumTagEnum = 0, NumTagClass = 0;
  unsigned NumObjcInterfaces = 0;
  
  for (unsigned i = 0, e = Types.size(); i != e; ++i) {
    Type *T = Types[i];
    if (isa<BuiltinType>(T))
      ++NumBuiltin;
    else if (isa<PointerType>(T))
      ++NumPointer;
    else if (isa<ReferenceType>(T))
      ++NumReference;
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
      switch (TT->getDecl()->getKind()) {
      default: assert(0 && "Unknown tagged type!");
      case Decl::Struct: ++NumTagStruct; break;
      case Decl::Union:  ++NumTagUnion; break;
      case Decl::Class:  ++NumTagClass; break; 
      case Decl::Enum:   ++NumTagEnum; break;
      }
    } else if (isa<ObjcInterfaceType>(T))
      ++NumObjcInterfaces;
    else {
      assert(0 && "Unknown type!");
    }
  }

  fprintf(stderr, "    %d builtin types\n", NumBuiltin);
  fprintf(stderr, "    %d pointer types\n", NumPointer);
  fprintf(stderr, "    %d reference types\n", NumReference);
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
  fprintf(stderr, "    %d interface types\n", NumObjcInterfaces);
  fprintf(stderr, "Total bytes = %d\n", int(NumBuiltin*sizeof(BuiltinType)+
    NumPointer*sizeof(PointerType)+NumArray*sizeof(ArrayType)+
    NumComplex*sizeof(ComplexType)+NumVector*sizeof(VectorType)+
    NumFunctionP*sizeof(FunctionTypeProto)+
    NumFunctionNP*sizeof(FunctionTypeNoProto)+
    NumTypeName*sizeof(TypedefType)+NumTagged*sizeof(TagType)));
}


void ASTContext::InitBuiltinType(QualType &R, BuiltinType::Kind K) {
  Types.push_back((R = QualType(new BuiltinType(K),0)).getTypePtr());
}

void ASTContext::InitBuiltinTypes() {
  assert(VoidTy.isNull() && "Context reinitialized?");
  
  // C99 6.2.5p19.
  InitBuiltinType(VoidTy,              BuiltinType::Void);
  
  // C99 6.2.5p2.
  InitBuiltinType(BoolTy,              BuiltinType::Bool);
  // C99 6.2.5p3.
  if (Target.isCharSigned(SourceLocation()))
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
  
  // C99 6.2.5p11.
  FloatComplexTy      = getComplexType(FloatTy);
  DoubleComplexTy     = getComplexType(DoubleTy);
  LongDoubleComplexTy = getComplexType(LongDoubleTy);
  
  BuiltinVaListType = QualType();
  ObjcIdType = QualType();
  IdStructType = 0;
  ObjcClassType = QualType();
  ClassStructType = 0;
  
  ObjcConstantStringType = QualType();
  
  // void * type
  VoidPtrTy = getPointerType(VoidTy);
}

//===----------------------------------------------------------------------===//
//                         Type Sizing and Analysis
//===----------------------------------------------------------------------===//

/// getTypeSize - Return the size of the specified type, in bits.  This method
/// does not work on incomplete types.
std::pair<uint64_t, unsigned>
ASTContext::getTypeInfo(QualType T, SourceLocation L) {
  T = T.getCanonicalType();
  uint64_t Size;
  unsigned Align;
  switch (T->getTypeClass()) {
  case Type::TypeName: assert(0 && "Not a canonical type!");
  case Type::FunctionNoProto:
  case Type::FunctionProto:
  default:
    assert(0 && "Incomplete types have no size!");
  case Type::VariableArray:
    assert(0 && "VLAs not implemented yet!");
  case Type::ConstantArray: {
    ConstantArrayType *CAT = cast<ConstantArrayType>(T);
    
    std::pair<uint64_t, unsigned> EltInfo = 
      getTypeInfo(CAT->getElementType(), L);
    Size = EltInfo.first*CAT->getSize().getZExtValue();
    Align = EltInfo.second;
    break;
  }    
  case Type::Vector: {
    std::pair<uint64_t, unsigned> EltInfo = 
      getTypeInfo(cast<VectorType>(T)->getElementType(), L);
    Size = EltInfo.first*cast<VectorType>(T)->getNumElements();
    // FIXME: Vector alignment is not the alignment of its elements.
    Align = EltInfo.second;
    break;
  }

  case Type::Builtin: {
    // FIXME: need to use TargetInfo to derive the target specific sizes. This
    // implementation will suffice for play with vector support.
    const llvm::fltSemantics *F;
    switch (cast<BuiltinType>(T)->getKind()) {
    default: assert(0 && "Unknown builtin type!");
    case BuiltinType::Void:
      assert(0 && "Incomplete types have no size!");
    case BuiltinType::Bool:       Target.getBoolInfo(Size, Align, L); break;
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::SChar:      Target.getCharInfo(Size, Align, L); break;
    case BuiltinType::UShort:
    case BuiltinType::Short:      Target.getShortInfo(Size, Align, L); break;
    case BuiltinType::UInt:
    case BuiltinType::Int:        Target.getIntInfo(Size, Align, L); break;
    case BuiltinType::ULong:
    case BuiltinType::Long:       Target.getLongInfo(Size, Align, L); break;
    case BuiltinType::ULongLong:
    case BuiltinType::LongLong:   Target.getLongLongInfo(Size, Align, L); break;
    case BuiltinType::Float:      Target.getFloatInfo(Size, Align, F, L); break;
    case BuiltinType::Double:     Target.getDoubleInfo(Size, Align, F, L);break;
    case BuiltinType::LongDouble:Target.getLongDoubleInfo(Size,Align,F,L);break;
    }
    break;
  }
  case Type::Pointer: Target.getPointerInfo(Size, Align, L); break;
  case Type::Reference:
    // "When applied to a reference or a reference type, the result is the size
    // of the referenced type." C++98 5.3.3p2: expr.sizeof.
    // FIXME: This is wrong for struct layout!
    return getTypeInfo(cast<ReferenceType>(T)->getReferenceeType(), L);
    
  case Type::Complex: {
    // Complex types have the same alignment as their elements, but twice the
    // size.
    std::pair<uint64_t, unsigned> EltInfo = 
      getTypeInfo(cast<ComplexType>(T)->getElementType(), L);
    Size = EltInfo.first*2;
    Align = EltInfo.second;
    break;
  }
  case Type::Tagged:
    TagType *TT = cast<TagType>(T);
    if (RecordType *RT = dyn_cast<RecordType>(TT)) {
      const ASTRecordLayout &Layout = getASTRecordLayout(RT->getDecl(), L);
      Size = Layout.getSize();
      Align = Layout.getAlignment();
    } else if (EnumDecl *ED = dyn_cast<EnumDecl>(TT->getDecl())) {
      return getTypeInfo(ED->getIntegerType(), L);
    } else {
      assert(0 && "Unimplemented type sizes!");
    }
    break;
  }
  
  assert(Align && (Align & (Align-1)) == 0 && "Alignment must be power of 2");
  return std::make_pair(Size, Align);
}

/// getASTRecordLayout - Get or compute information about the layout of the
/// specified record (struct/union/class), which indicates its size and field
/// position information.
const ASTRecordLayout &ASTContext::getASTRecordLayout(const RecordDecl *D,
                                                      SourceLocation L) {
  assert(D->isDefinition() && "Cannot get layout of forward declarations!");
  
  // Look up this layout, if already laid out, return what we have.
  const ASTRecordLayout *&Entry = ASTRecordLayouts[D];
  if (Entry) return *Entry;
  
  // Allocate and assign into ASTRecordLayouts here.  The "Entry" reference can
  // be invalidated (dangle) if the ASTRecordLayouts hashtable is inserted into.
  ASTRecordLayout *NewEntry = new ASTRecordLayout();
  Entry = NewEntry;
  
  uint64_t *FieldOffsets = new uint64_t[D->getNumMembers()];
  uint64_t RecordSize = 0;
  unsigned RecordAlign = 8;  // Default alignment = 1 byte = 8 bits.

  if (D->getKind() != Decl::Union) {
    // Layout each field, for now, just sequentially, respecting alignment.  In
    // the future, this will need to be tweakable by targets.
    for (unsigned i = 0, e = D->getNumMembers(); i != e; ++i) {
      const FieldDecl *FD = D->getMember(i);
      std::pair<uint64_t, unsigned> FieldInfo = getTypeInfo(FD->getType(), L);
      uint64_t FieldSize = FieldInfo.first;
      unsigned FieldAlign = FieldInfo.second;
      
      // Round up the current record size to the field's alignment boundary.
      RecordSize = (RecordSize+FieldAlign-1) & ~(FieldAlign-1);
      
      // Place this field at the current location.
      FieldOffsets[i] = RecordSize;
      
      // Reserve space for this field.
      RecordSize += FieldSize;
      
      // Remember max struct/class alignment.
      RecordAlign = std::max(RecordAlign, FieldAlign);
    }
    
    // Finally, round the size of the total struct up to the alignment of the
    // struct itself.
    RecordSize = (RecordSize+RecordAlign-1) & ~(RecordAlign-1);
  } else {
    // Union layout just puts each member at the start of the record.
    for (unsigned i = 0, e = D->getNumMembers(); i != e; ++i) {
      const FieldDecl *FD = D->getMember(i);
      std::pair<uint64_t, unsigned> FieldInfo = getTypeInfo(FD->getType(), L);
      uint64_t FieldSize = FieldInfo.first;
      unsigned FieldAlign = FieldInfo.second;
      
      // Round up the current record size to the field's alignment boundary.
      RecordSize = std::max(RecordSize, FieldSize);
      
      // Place this field at the start of the record.
      FieldOffsets[i] = 0;
      
      // Remember max struct/class alignment.
      RecordAlign = std::max(RecordAlign, FieldAlign);
    }
  }
  
  NewEntry->SetLayout(RecordSize, RecordAlign, FieldOffsets);
  return *NewEntry;
}

//===----------------------------------------------------------------------===//
//                   Type creation/memoization methods
//===----------------------------------------------------------------------===//


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
    Canonical = getComplexType(T.getCanonicalType());
    
    // Get the new insert position for the node we care about.
    ComplexType *NewIP = ComplexTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  ComplexType *New = new ComplexType(T, Canonical);
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
  if (!T->isCanonical()) {
    Canonical = getPointerType(T.getCanonicalType());
   
    // Get the new insert position for the node we care about.
    PointerType *NewIP = PointerTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  PointerType *New = new PointerType(T, Canonical);
  Types.push_back(New);
  PointerTypes.InsertNode(New, InsertPos);
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
    Canonical = getReferenceType(T.getCanonicalType());
   
    // Get the new insert position for the node we care about.
    ReferenceType *NewIP = ReferenceTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }

  ReferenceType *New = new ReferenceType(T, Canonical);
  Types.push_back(New);
  ReferenceTypes.InsertNode(New, InsertPos);
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
    Canonical = getConstantArrayType(EltTy.getCanonicalType(), ArySize, 
                                     ASM, EltTypeQuals);
    // Get the new insert position for the node we care about.
    ConstantArrayType *NewIP = 
      ConstantArrayTypes.FindNodeOrInsertPos(ID, InsertPos);

    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  
  ConstantArrayType *New = new ConstantArrayType(EltTy, Canonical, ArySize,
                                                 ASM, EltTypeQuals);
  ConstantArrayTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getVariableArrayType - Returns a non-unique reference to the type for a
/// variable array of the specified element type.
QualType ASTContext::getVariableArrayType(QualType EltTy, Expr *NumElts,
                                          ArrayType::ArraySizeModifier ASM,
                                          unsigned EltTypeQuals) {
  if (NumElts) {
    // Since we don't unique expressions, it isn't possible to unique VLA's
    // that have an expression provided for their size.
    
    VariableArrayType *New = new VariableArrayType(EltTy, QualType(), NumElts, 
                                                   ASM, EltTypeQuals);
    
    CompleteVariableArrayTypes.push_back(New);
    Types.push_back(New);
    return QualType(New, 0);
  }
  else {
    // No size is provided for the VLA.  These we can unique.
    llvm::FoldingSetNodeID ID;
    VariableArrayType::Profile(ID, EltTy);
    
    void *InsertPos = 0;
    if (VariableArrayType *ATP = 
         IncompleteVariableArrayTypes.FindNodeOrInsertPos(ID, InsertPos))
      return QualType(ATP, 0);
    
    // If the element type isn't canonical, this won't be a canonical type
    // either, so fill in the canonical type field.
    QualType Canonical;
    
    if (!EltTy->isCanonical()) {
      Canonical = getVariableArrayType(EltTy.getCanonicalType(), NumElts,
                                       ASM, EltTypeQuals);
      
      // Get the new insert position for the node we care about.
      VariableArrayType *NewIP =
        IncompleteVariableArrayTypes.FindNodeOrInsertPos(ID, InsertPos);
      
      assert(NewIP == 0 && "Shouldn't be in the map!");
    }
    
    VariableArrayType *New = new VariableArrayType(EltTy, QualType(), NumElts, 
                                                   ASM, EltTypeQuals);
    
    IncompleteVariableArrayTypes.InsertNode(New, InsertPos);
    Types.push_back(New);
    return QualType(New, 0);            
  }
}

/// getVectorType - Return the unique reference to a vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType ASTContext::getVectorType(QualType vecType, unsigned NumElts) {
  BuiltinType *baseType;
  
  baseType = dyn_cast<BuiltinType>(vecType.getCanonicalType().getTypePtr());
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
    Canonical = getVectorType(vecType.getCanonicalType(), NumElts);
    
    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  VectorType *New = new VectorType(vecType, NumElts, Canonical);
  VectorTypes.InsertNode(New, InsertPos);
  Types.push_back(New);
  return QualType(New, 0);
}

/// getOCUVectorType - Return the unique reference to an OCU vector type of
/// the specified element type and size. VectorType must be a built-in type.
QualType ASTContext::getOCUVectorType(QualType vecType, unsigned NumElts) {
  BuiltinType *baseType;
  
  baseType = dyn_cast<BuiltinType>(vecType.getCanonicalType().getTypePtr());
  assert(baseType != 0 && "getOCUVectorType(): Expecting a built-in type");
         
  // Check if we've already instantiated a vector of this type.
  llvm::FoldingSetNodeID ID;
  VectorType::Profile(ID, vecType, NumElts, Type::OCUVector);      
  void *InsertPos = 0;
  if (VectorType *VTP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(VTP, 0);

  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  QualType Canonical;
  if (!vecType->isCanonical()) {
    Canonical = getOCUVectorType(vecType.getCanonicalType(), NumElts);
    
    // Get the new insert position for the node we care about.
    VectorType *NewIP = VectorTypes.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  OCUVectorType *New = new OCUVectorType(vecType, NumElts, Canonical);
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
    Canonical = getFunctionTypeNoProto(ResultTy.getCanonicalType());
    
    // Get the new insert position for the node we care about.
    FunctionTypeNoProto *NewIP =
      FunctionTypeNoProtos.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  
  FunctionTypeNoProto *New = new FunctionTypeNoProto(ResultTy, Canonical);
  Types.push_back(New);
  FunctionTypeProtos.InsertNode(New, InsertPos);
  return QualType(New, 0);
}

/// getFunctionType - Return a normal function type with a typed argument
/// list.  isVariadic indicates whether the argument list includes '...'.
QualType ASTContext::getFunctionType(QualType ResultTy, QualType *ArgArray,
                                     unsigned NumArgs, bool isVariadic) {
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  llvm::FoldingSetNodeID ID;
  FunctionTypeProto::Profile(ID, ResultTy, ArgArray, NumArgs, isVariadic);

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
      CanonicalArgs.push_back(ArgArray[i].getCanonicalType());
    
    Canonical = getFunctionType(ResultTy.getCanonicalType(),
                                &CanonicalArgs[0], NumArgs,
                                isVariadic);
    
    // Get the new insert position for the node we care about.
    FunctionTypeProto *NewIP =
      FunctionTypeProtos.FindNodeOrInsertPos(ID, InsertPos);
    assert(NewIP == 0 && "Shouldn't be in the map!");
  }
  
  // FunctionTypeProto objects are not allocated with new because they have a
  // variable size array (for parameter types) at the end of them.
  FunctionTypeProto *FTP = 
    (FunctionTypeProto*)malloc(sizeof(FunctionTypeProto) + 
                               NumArgs*sizeof(QualType));
  new (FTP) FunctionTypeProto(ResultTy, ArgArray, NumArgs, isVariadic,
                              Canonical);
  Types.push_back(FTP);
  FunctionTypeProtos.InsertNode(FTP, InsertPos);
  return QualType(FTP, 0);
}

/// getTypedefType - Return the unique reference to the type for the
/// specified typename decl.
QualType ASTContext::getTypedefType(TypedefDecl *Decl) {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);
  
  QualType Canonical = Decl->getUnderlyingType().getCanonicalType();
  Decl->TypeForDecl = new TypedefType(Decl, Canonical);
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// getObjcInterfaceType - Return the unique reference to the type for the
/// specified ObjC interface decl.
QualType ASTContext::getObjcInterfaceType(ObjcInterfaceDecl *Decl) {
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);
  
  Decl->TypeForDecl = new ObjcInterfaceType(Decl);
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// getObjcQualifiedInterfaceType - Return a 
/// ObjcQualifiedInterfaceType type for the given interface decl and
/// the conforming protocol list.
QualType ASTContext::getObjcQualifiedInterfaceType(ObjcInterfaceDecl *Decl,
                       ObjcProtocolDecl **Protocols, unsigned NumProtocols) {
  ObjcInterfaceType *IType = 
    cast<ObjcInterfaceType>(getObjcInterfaceType(Decl));
  
  llvm::FoldingSetNodeID ID;
  ObjcQualifiedInterfaceType::Profile(ID, IType, Protocols, NumProtocols);
  
  void *InsertPos = 0;
  if (ObjcQualifiedInterfaceType *QT =
      ObjcQualifiedInterfaceTypes.FindNodeOrInsertPos(ID, InsertPos))
    return QualType(QT, 0);
  
  // No Match;
  ObjcQualifiedInterfaceType *QType =
    new ObjcQualifiedInterfaceType(IType, Protocols, NumProtocols);
  Types.push_back(QType);
  ObjcQualifiedInterfaceTypes.InsertNode(QType, InsertPos);
  return QualType(QType, 0);
}

/// getTypeOfExpr - Unlike many "get<Type>" functions, we can't unique
/// TypeOfExpr AST's (since expression's are never shared). For example,
/// multiple declarations that refer to "typeof(x)" all contain different
/// DeclRefExpr's. This doesn't effect the type checker, since it operates 
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfExpr(Expr *tofExpr) {
  QualType Canonical = tofExpr->getType().getCanonicalType();
  TypeOfExpr *toe = new TypeOfExpr(tofExpr, Canonical);
  Types.push_back(toe);
  return QualType(toe, 0);
}

/// getTypeOfType -  Unlike many "get<Type>" functions, we don't unique
/// TypeOfType AST's. The only motivation to unique these nodes would be
/// memory savings. Since typeof(t) is fairly uncommon, space shouldn't be
/// an issue. This doesn't effect the type checker, since it operates 
/// on canonical type's (which are always unique).
QualType ASTContext::getTypeOfType(QualType tofType) {
  QualType Canonical = tofType.getCanonicalType();
  TypeOfType *tot = new TypeOfType(tofType, Canonical);
  Types.push_back(tot);
  return QualType(tot, 0);
}

/// getTagDeclType - Return the unique reference to the type for the
/// specified TagDecl (struct/union/class/enum) decl.
QualType ASTContext::getTagDeclType(TagDecl *Decl) {
  // The decl stores the type cache.
  if (Decl->TypeForDecl) return QualType(Decl->TypeForDecl, 0);
  
  Decl->TypeForDecl = new TagType(Decl, QualType());
  Types.push_back(Decl->TypeForDecl);
  return QualType(Decl->TypeForDecl, 0);
}

/// getSizeType - Return the unique type for "size_t" (C99 7.17), the result 
/// of the sizeof operator (C99 6.5.3.4p4). The value is target dependent and 
/// needs to agree with the definition in <stddef.h>. 
QualType ASTContext::getSizeType() const {
  // On Darwin, size_t is defined as a "long unsigned int". 
  // FIXME: should derive from "Target".
  return UnsignedLongTy; 
}

/// getPointerDiffType - Return the unique type for "ptrdiff_t" (ref?)
/// defined in <stddef.h>. Pointer - pointer requires this (C99 6.5.6p9).
QualType ASTContext::getPointerDiffType() const {
  // On Darwin, ptrdiff_t is defined as a "int". This seems like a bug...
  // FIXME: should derive from "Target".
  return IntTy; 
}

/// getIntegerRank - Return an integer conversion rank (C99 6.3.1.1p1). This
/// routine will assert if passed a built-in type that isn't an integer or enum.
static int getIntegerRank(QualType t) {
  if (const TagType *TT = dyn_cast<TagType>(t.getCanonicalType())) {
    assert(TT->getDecl()->getKind() == Decl::Enum && "not an int or enum");
    return 4;
  }
  
  const BuiltinType *BT = cast<BuiltinType>(t.getCanonicalType());
  switch (BT->getKind()) {
  default:
    assert(0 && "getIntegerRank(): not a built-in integer");
  case BuiltinType::Bool:
    return 1;
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
    return 2;
  case BuiltinType::Short:
  case BuiltinType::UShort:
    return 3;
  case BuiltinType::Int:
  case BuiltinType::UInt:
    return 4;
  case BuiltinType::Long:
  case BuiltinType::ULong:
    return 5;
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    return 6;
  }
}

/// getFloatingRank - Return a relative rank for floating point types.
/// This routine will assert if passed a built-in type that isn't a float.
static int getFloatingRank(QualType T) {
  T = T.getCanonicalType();
  if (ComplexType *CT = dyn_cast<ComplexType>(T))
    return getFloatingRank(CT->getElementType());
  
  switch (cast<BuiltinType>(T)->getKind()) {
  default:  assert(0 && "getFloatingRank(): not a floating type");
  case BuiltinType::Float:      return FloatRank;
  case BuiltinType::Double:     return DoubleRank;
  case BuiltinType::LongDouble: return LongDoubleRank;
  }
}

/// getFloatingTypeOfSizeWithinDomain - Returns a real floating 
/// point or a complex type (based on typeDomain/typeSize). 
/// 'typeDomain' is a real floating point or complex type.
/// 'typeSize' is a real floating point or complex type.
QualType ASTContext::getFloatingTypeOfSizeWithinDomain(
  QualType typeSize, QualType typeDomain) const {
  if (typeDomain->isComplexType()) {
    switch (getFloatingRank(typeSize)) {
    default: assert(0 && "getFloatingRank(): illegal value for rank");
    case FloatRank:      return FloatComplexTy;
    case DoubleRank:     return DoubleComplexTy;
    case LongDoubleRank: return LongDoubleComplexTy;
    }
  }
  if (typeDomain->isRealFloatingType()) {
    switch (getFloatingRank(typeSize)) {
    default: assert(0 && "getFloatingRank(): illegal value for rank");
    case FloatRank:      return FloatTy;
    case DoubleRank:     return DoubleTy;
    case LongDoubleRank: return LongDoubleTy;
    }
  }
  assert(0 && "getFloatingTypeOfSizeWithinDomain(): illegal domain");
  //an invalid return value, but the assert
  //will ensure that this code is never reached.
  return VoidTy;
}

/// compareFloatingType - Handles 3 different combos: 
/// float/float, float/complex, complex/complex. 
/// If lt > rt, return 1. If lt == rt, return 0. If lt < rt, return -1. 
int ASTContext::compareFloatingType(QualType lt, QualType rt) {
  if (getFloatingRank(lt) == getFloatingRank(rt))
    return 0;
  if (getFloatingRank(lt) > getFloatingRank(rt))
    return 1;
  return -1;
}

// maxIntegerType - Returns the highest ranked integer type. Handles 3 case:
// unsigned/unsigned, signed/signed, signed/unsigned. C99 6.3.1.8p1.
QualType ASTContext::maxIntegerType(QualType lhs, QualType rhs) {
  if (lhs == rhs) return lhs;
  
  bool t1Unsigned = lhs->isUnsignedIntegerType();
  bool t2Unsigned = rhs->isUnsignedIntegerType();
  
  if ((t1Unsigned && t2Unsigned) || (!t1Unsigned && !t2Unsigned))
    return getIntegerRank(lhs) >= getIntegerRank(rhs) ? lhs : rhs; 
  
  // We have two integer types with differing signs
  QualType unsignedType = t1Unsigned ? lhs : rhs;
  QualType signedType = t1Unsigned ? rhs : lhs;
  
  if (getIntegerRank(unsignedType) >= getIntegerRank(signedType))
    return unsignedType;
  else {
    // FIXME: Need to check if the signed type can represent all values of the 
    // unsigned type. If it can, then the result is the signed type. 
    // If it can't, then the result is the unsigned version of the signed type.  
    // Should probably add a helper that returns a signed integer type from 
    // an unsigned (and vice versa). C99 6.3.1.8.
    return signedType; 
  }
}

// getCFConstantStringType - Return the type used for constant CFStrings. 
QualType ASTContext::getCFConstantStringType() {
  if (!CFConstantStringTypeDecl) {
    CFConstantStringTypeDecl = new RecordDecl(Decl::Struct, SourceLocation(), 
                                              &Idents.get("NSConstantString"),
                                              0);
    QualType FieldTypes[3];
  
    // const int *isa;
    FieldTypes[0] = getPointerType(IntTy.getQualifiedType(QualType::Const));  
    // const char *str;
    FieldTypes[1] = getPointerType(CharTy.getQualifiedType(QualType::Const));  
    // long length;
    FieldTypes[2] = LongTy;  
    // Create fields
    FieldDecl *FieldDecls[3];
  
    for (unsigned i = 0; i < 3; ++i)
      FieldDecls[i] = new FieldDecl(SourceLocation(), 0, FieldTypes[i]);
  
    CFConstantStringTypeDecl->defineBody(FieldDecls, 4);
  }
  
  return getTagDeclType(CFConstantStringTypeDecl);
}

// This returns true if a type has been typedefed to BOOL:
// typedef <type> BOOL;
static bool isTypeTypedefedAsBOOL(QualType T) {
  if (const TypedefType *TT = dyn_cast<TypedefType>(T))
    return !strcmp(TT->getDecl()->getName(), "BOOL");
        
  return false;
}

/// getObjcEncodingTypeSize returns size of type for objective-c encoding
/// purpose.
int ASTContext::getObjcEncodingTypeSize(QualType type) {
  SourceLocation Loc;
  uint64_t sz = getTypeSize(type, Loc);
  
  // Make all integer and enum types at least as large as an int
  if (sz > 0 && type->isIntegralType())
    sz = std::max(sz, getTypeSize(IntTy, Loc));
  // Treat arrays as pointers, since that's how they're passed in.
  else if (type->isArrayType())
    sz = getTypeSize(VoidPtrTy, Loc);
  return sz / getTypeSize(CharTy, Loc);
}

/// getObjcEncodingForMethodDecl - Return the encoded type for this method
/// declaration.
void ASTContext::getObjcEncodingForMethodDecl(ObjcMethodDecl *Decl, 
                                              std::string& S)
{
  // Encode type qualifer, 'in', 'inout', etc. for the return type.
  getObjcEncodingForTypeQualifier(Decl->getObjcDeclQualifier(), S);
  // Encode result type.
  getObjcEncodingForType(Decl->getResultType(), S);
  // Compute size of all parameters.
  // Start with computing size of a pointer in number of bytes.
  // FIXME: There might(should) be a better way of doing this computation!
  SourceLocation Loc;
  int PtrSize = getTypeSize(VoidPtrTy, Loc) / getTypeSize(CharTy, Loc);
  // The first two arguments (self and _cmd) are pointers; account for
  // their size.
  int ParmOffset = 2 * PtrSize;
  int NumOfParams = Decl->getNumParams();
  for (int i = 0; i < NumOfParams; i++) {
    QualType PType = Decl->getParamDecl(i)->getType();
    int sz = getObjcEncodingTypeSize (PType);
    assert (sz > 0 && "getObjcEncodingForMethodDecl - Incomplete param type");
    ParmOffset += sz;
  }
  S += llvm::utostr(ParmOffset);
  S += "@0:";
  S += llvm::utostr(PtrSize);
  
  // Argument types.
  ParmOffset = 2 * PtrSize;
  for (int i = 0; i < NumOfParams; i++) {
    QualType PType = Decl->getParamDecl(i)->getType();
    // Process argument qualifiers for user supplied arguments; such as,
    // 'in', 'inout', etc.
    getObjcEncodingForTypeQualifier(
      Decl->getParamDecl(i)->getObjcDeclQualifier(), S);
    getObjcEncodingForType(PType, S);
    S += llvm::utostr(ParmOffset);
    ParmOffset += getObjcEncodingTypeSize(PType);
  }
}

void ASTContext::getObjcEncodingForType(QualType T, std::string& S) const
{
  // FIXME: This currently doesn't encode:
  // @ An object (whether statically typed or typed id)
  // # A class object (Class)
  // : A method selector (SEL)
  // {name=type...} A structure
  // (name=type...) A union
  // bnum A bit field of num bits
  
  if (const BuiltinType *BT = T->getAsBuiltinType()) {
    char encoding;
    switch (BT->getKind()) {
    case BuiltinType::Void:
      encoding = 'v';
      break;
    case BuiltinType::Bool:
      encoding = 'B';
      break;
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
      encoding = 'C';
      break;
    case BuiltinType::UShort:
      encoding = 'S';
      break;
    case BuiltinType::UInt:
      encoding = 'I';
      break;
    case BuiltinType::ULong:
      encoding = 'L';
      break;
    case BuiltinType::ULongLong:
      encoding = 'Q';
      break;
    case BuiltinType::Char_S:
    case BuiltinType::SChar:
      encoding = 'c';
      break;
    case BuiltinType::Short:
      encoding = 's';
      break;
    case BuiltinType::Int:
      encoding = 'i';
      break;
    case BuiltinType::Long:
      encoding = 'l';
      break;
    case BuiltinType::LongLong:
      encoding = 'q';
      break;
    case BuiltinType::Float:
      encoding = 'f';
      break;
    case BuiltinType::Double:
      encoding = 'd';
      break;
    case BuiltinType::LongDouble:
      encoding = 'd';
      break;
    default:
      assert(0 && "Unhandled builtin type kind");          
    }
    
    S += encoding;
  } else if (const PointerType *PT = T->getAsPointerType()) {
    QualType PointeeTy = PT->getPointeeType();
    if (isObjcIdType(PointeeTy) || PointeeTy->isObjcInterfaceType()) {
      S += '@';
      return;
    } else if (isObjcClassType(PointeeTy)) {
      S += '#';
      return;
    } else if (isObjcSelType(PointeeTy)) {
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
    getObjcEncodingForType(PT->getPointeeType(), S);
  } else if (const ArrayType *AT = T->getAsArrayType()) {
    S += '[';
    
    if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT))
      S += llvm::utostr(CAT->getSize().getZExtValue());
    else
      assert(0 && "Unhandled array type!");
    
    getObjcEncodingForType(AT->getElementType(), S);
    S += ']';
  } else if (T->getAsFunctionType()) {
    S += '?';
  } else
    assert(0 && "@encode for type not implemented!");
}

void ASTContext::getObjcEncodingForTypeQualifier(Decl::ObjcDeclQualifier QT, 
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

void ASTContext::setObjcIdType(TypedefDecl *TD)
{
  assert(ObjcIdType.isNull() && "'id' type already set!");
    
  ObjcIdType = getTypedefType(TD);

  // typedef struct objc_object *id;
  const PointerType *ptr = TD->getUnderlyingType()->getAsPointerType();
  assert(ptr && "'id' incorrectly typed");
  const RecordType *rec = ptr->getPointeeType()->getAsStructureType();
  assert(rec && "'id' incorrectly typed");
  IdStructType = rec;
}

void ASTContext::setObjcSelType(TypedefDecl *TD)
{
  assert(ObjcSelType.isNull() && "'SEL' type already set!");
    
  ObjcSelType = getTypedefType(TD);

  // typedef struct objc_selector *SEL;
  const PointerType *ptr = TD->getUnderlyingType()->getAsPointerType();
  assert(ptr && "'SEL' incorrectly typed");
  const RecordType *rec = ptr->getPointeeType()->getAsStructureType();
  assert(rec && "'SEL' incorrectly typed");
  SelStructType = rec;
}

void ASTContext::setObjcProtoType(TypedefDecl *TD)
{
  assert(ObjcProtoType.isNull() && "'Protocol' type already set!");
  
  // typedef struct Protocol Protocol;
  ObjcProtoType = TD->getUnderlyingType();
  // Protocol * type
  ObjcProtoType = getPointerType(ObjcProtoType);  
  ProtoStructType = TD->getUnderlyingType()->getAsStructureType();
}

void ASTContext::setObjcClassType(TypedefDecl *TD)
{
  assert(ObjcClassType.isNull() && "'Class' type already set!");
    
  ObjcClassType = getTypedefType(TD);

  // typedef struct objc_class *Class;
  const PointerType *ptr = TD->getUnderlyingType()->getAsPointerType();
  assert(ptr && "'Class' incorrectly typed");
  const RecordType *rec = ptr->getPointeeType()->getAsStructureType();
  assert(rec && "'Class' incorrectly typed");
  ClassStructType = rec;
}

void ASTContext::setObjcConstantStringInterface(ObjcInterfaceDecl *Decl) {
  assert(ObjcConstantStringType.isNull() && 
         "'NSConstantString' type already set!");
  
  ObjcConstantStringType = getObjcInterfaceType(Decl);
}

bool ASTContext::builtinTypesAreCompatible(QualType lhs, QualType rhs) {
  const BuiltinType *lBuiltin = lhs->getAsBuiltinType();
  const BuiltinType *rBuiltin = rhs->getAsBuiltinType();
  
  return lBuiltin->getKind() == rBuiltin->getKind();
}


bool ASTContext::objcTypesAreCompatible(QualType lhs, QualType rhs) {
  if (lhs->isObjcInterfaceType() && isObjcIdType(rhs))
    return true;
  else if (isObjcIdType(lhs) && rhs->isObjcInterfaceType())
    return true;
  return false;
}

bool ASTContext::interfaceTypesAreCompatible(QualType lhs, QualType rhs) {
  return true; // FIXME: IMPLEMENT.
}

bool ASTContext::vectorTypesAreCompatible(QualType lhs, QualType rhs) {
  const VectorType *lVector = lhs->getAsVectorType();
  const VectorType *rVector = rhs->getAsVectorType();
  
  if ((lVector->getElementType().getCanonicalType() ==
      rVector->getElementType().getCanonicalType()) &&
      (lVector->getNumElements() == rVector->getNumElements()))
    return true;
  return false;
}

// C99 6.2.7p1: If both are complete types, then the following additional
// requirements apply...FIXME (handle compatibility across source files).
bool ASTContext::tagTypesAreCompatible(QualType lhs, QualType rhs) {
  TagDecl *ldecl = cast<TagType>(lhs.getCanonicalType())->getDecl();
  TagDecl *rdecl = cast<TagType>(rhs.getCanonicalType())->getDecl();
  
  if (ldecl->getKind() == Decl::Struct && rdecl->getKind() == Decl::Struct) {
    if (ldecl->getIdentifier() == rdecl->getIdentifier())
      return true;
  }
  if (ldecl->getKind() == Decl::Union && rdecl->getKind() == Decl::Union) {
    if (ldecl->getIdentifier() == rdecl->getIdentifier())
      return true;
  }
  // "Class" and "id" are compatible built-in structure types.
  if (isObjcIdType(lhs) && isObjcClassType(rhs) ||
      isObjcClassType(lhs) && isObjcIdType(rhs))
    return true;
  return false;
}

bool ASTContext::pointerTypesAreCompatible(QualType lhs, QualType rhs) {
  // C99 6.7.5.1p2: For two pointer types to be compatible, both shall be 
  // identically qualified and both shall be pointers to compatible types.
  if (lhs.getQualifiers() != rhs.getQualifiers())
    return false;
    
  QualType ltype = cast<PointerType>(lhs.getCanonicalType())->getPointeeType();
  QualType rtype = cast<PointerType>(rhs.getCanonicalType())->getPointeeType();
  
  return typesAreCompatible(ltype, rtype);
}

// C++ 5.17p6: When the left opperand of an assignment operator denotes a
// reference to T, the operation assigns to the object of type T denoted by the
// reference.
bool ASTContext::referenceTypesAreCompatible(QualType lhs, QualType rhs) {
  QualType ltype = lhs;

  if (lhs->isReferenceType())
    ltype = cast<ReferenceType>(lhs.getCanonicalType())->getReferenceeType();

  QualType rtype = rhs;

  if (rhs->isReferenceType())
    rtype = cast<ReferenceType>(rhs.getCanonicalType())->getReferenceeType();

  return typesAreCompatible(ltype, rtype);
}

bool ASTContext::functionTypesAreCompatible(QualType lhs, QualType rhs) {
  const FunctionType *lbase = cast<FunctionType>(lhs.getCanonicalType());
  const FunctionType *rbase = cast<FunctionType>(rhs.getCanonicalType());
  const FunctionTypeProto *lproto = dyn_cast<FunctionTypeProto>(lbase);
  const FunctionTypeProto *rproto = dyn_cast<FunctionTypeProto>(rbase);

  // first check the return types (common between C99 and K&R).
  if (!typesAreCompatible(lbase->getResultType(), rbase->getResultType()))
    return false;

  if (lproto && rproto) { // two C99 style function prototypes
    unsigned lproto_nargs = lproto->getNumArgs();
    unsigned rproto_nargs = rproto->getNumArgs();
    
    if (lproto_nargs != rproto_nargs)
      return false;
      
    // both prototypes have the same number of arguments.
    if ((lproto->isVariadic() && !rproto->isVariadic()) ||
        (rproto->isVariadic() && !lproto->isVariadic()))
      return false;
      
    // The use of ellipsis agree...now check the argument types.
    for (unsigned i = 0; i < lproto_nargs; i++)
      if (!typesAreCompatible(lproto->getArgType(i), rproto->getArgType(i)))
        return false;
    return true;
  }
  if (!lproto && !rproto) // two K&R style function decls, nothing to do.
    return true;

  // we have a mixture of K&R style with C99 prototypes
  const FunctionTypeProto *proto = lproto ? lproto : rproto;
  
  if (proto->isVariadic())
    return false;
    
  // FIXME: Each parameter type T in the prototype must be compatible with the
  // type resulting from applying the usual argument conversions to T.
  return true;
}

bool ASTContext::arrayTypesAreCompatible(QualType lhs, QualType rhs) {
  QualType ltype = cast<ArrayType>(lhs.getCanonicalType())->getElementType();
  QualType rtype = cast<ArrayType>(rhs.getCanonicalType())->getElementType();
  
  if (!typesAreCompatible(ltype, rtype))
    return false;
    
  // FIXME: If both types specify constant sizes, then the sizes must also be 
  // the same. Even if the sizes are the same, GCC produces an error.
  return true;
}

/// typesAreCompatible - C99 6.7.3p9: For two qualified types to be compatible, 
/// both shall have the identically qualified version of a compatible type.
/// C99 6.2.7p1: Two types have compatible types if their types are the 
/// same. See 6.7.[2,3,5] for additional rules.
bool ASTContext::typesAreCompatible(QualType lhs, QualType rhs) {
  QualType lcanon = lhs.getCanonicalType();
  QualType rcanon = rhs.getCanonicalType();

  // If two types are identical, they are are compatible
  if (lcanon == rcanon)
    return true;
  
  // If the canonical type classes don't match, they can't be compatible
  if (lcanon->getTypeClass() != rcanon->getTypeClass()) {
    // For Objective-C, it is possible for two types to be compatible
    // when their classes don't match (when dealing with "id"). If either type
    // is an interface, we defer to objcTypesAreCompatible(). 
    if (lcanon->isObjcInterfaceType() || rcanon->isObjcInterfaceType())
      return objcTypesAreCompatible(lcanon, rcanon);
    return false;
  }
  switch (lcanon->getTypeClass()) {
    case Type::Pointer:
      return pointerTypesAreCompatible(lcanon, rcanon);
    case Type::Reference:
      return referenceTypesAreCompatible(lcanon, rcanon);
    case Type::ConstantArray:
    case Type::VariableArray:
      return arrayTypesAreCompatible(lcanon, rcanon);
    case Type::FunctionNoProto:
    case Type::FunctionProto:
      return functionTypesAreCompatible(lcanon, rcanon);
    case Type::Tagged: // handle structures, unions
      return tagTypesAreCompatible(lcanon, rcanon);
    case Type::Builtin:
      return builtinTypesAreCompatible(lcanon, rcanon); 
    case Type::ObjcInterface:
      return interfaceTypesAreCompatible(lcanon, rcanon); 
    case Type::Vector:
    case Type::OCUVector:
      return vectorTypesAreCompatible(lcanon, rcanon);
    default:
      assert(0 && "unexpected type");
  }
  return true; // should never get here...
}

/// Emit - Serialize an ASTContext object to Bitcode.
void ASTContext::Emit(llvm::Serializer& S) const {
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

  // FIXME: S.EmitOwnedPtr(CFConstantStringTypeDecl);
}

ASTContext* ASTContext::Create(llvm::Deserializer& D) {
  SourceManager &SM = D.ReadRef<SourceManager>();
  TargetInfo &t = D.ReadRef<TargetInfo>();
  IdentifierTable &idents = D.ReadRef<IdentifierTable>();
  SelectorTable &sels = D.ReadRef<SelectorTable>();
  
  unsigned size_reserve = D.ReadInt();
  
  ASTContext* A = new ASTContext(SM,t,idents,sels,size_reserve);
  
  for (unsigned i = 0; i < size_reserve; ++i)
    Type::Create(*A,i,D);

  // FIXME: A->CFConstantStringTypeDecl = D.ReadOwnedPtr<RecordDecl>();
  
  return A;
}
