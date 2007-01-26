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
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;
using namespace clang;

ASTContext::ASTContext(Preprocessor &pp)
  : PP(pp), Target(pp.getTargetInfo()) {
  NumSlowLookups = 0;
  InitBuiltinTypes();
}

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
  unsigned NumFunctionNP = 0, NumTypeName = 0, NumTagged = 0;
  
  unsigned NumTagStruct = 0, NumTagUnion = 0, NumTagEnum = 0, NumTagClass = 0;
  
  for (unsigned i = 0, e = Types.size(); i != e; ++i) {
    Type *T = Types[i];
    if (isa<BuiltinType>(T))
      ++NumBuiltin;
    else if (isa<PointerType>(T))
      ++NumPointer;
    else if (isa<ArrayType>(T))
      ++NumArray;
    else if (isa<FunctionTypeNoProto>(T))
      ++NumFunctionNP;
    else if (isa<FunctionTypeProto>(T))
      ++NumFunctionP;
    else if (isa<TypeNameType>(T))
      ++NumTypeName;
    else if (TaggedType *TT = dyn_cast<TaggedType>(T)) {
      ++NumTagged;
      switch (TT->getDecl()->getKind()) {
      default: assert(0 && "Unknown tagged type!");
      case Decl::Struct: ++NumTagStruct; break;
      case Decl::Union:  ++NumTagUnion; break;
      case Decl::Class:  ++NumTagClass; break; 
      case Decl::Enum:   ++NumTagEnum; break;
      }
    } else {
      assert(0 && "Unknown type!");
    }
  }

  fprintf(stderr, "    %d builtin types\n", NumBuiltin);
  fprintf(stderr, "    %d pointer types\n", NumPointer);
  fprintf(stderr, "    %d array types\n", NumArray);
  fprintf(stderr, "    %d function types with proto\n", NumFunctionP);
  fprintf(stderr, "    %d function types with no proto\n", NumFunctionNP);
  fprintf(stderr, "    %d typename (typedef) types\n", NumTypeName);
  fprintf(stderr, "    %d tagged types\n", NumTagged);
  fprintf(stderr, "      %d struct types\n", NumTagStruct);
  fprintf(stderr, "      %d union types\n", NumTagUnion);
  fprintf(stderr, "      %d class types\n", NumTagClass);
  fprintf(stderr, "      %d enum types\n", NumTagEnum);
  fprintf(stderr, "  %d slow type lookups\n", NumSlowLookups);
}


void ASTContext::InitBuiltinType(TypeRef &R, BuiltinType::Kind K) {
  Types.push_back((R = new BuiltinType(K)).getTypePtr());
}


void ASTContext::InitBuiltinTypes() {
  assert(VoidTy.isNull() && "Context reinitialized?");
  
  // C99 6.2.5p19.
  InitBuiltinType(VoidTy,              BuiltinType::Void);
  
  // C99 6.2.5p2.
  InitBuiltinType(BoolTy,              BuiltinType::Bool);
  // C99 6.2.5p3.
  InitBuiltinType(CharTy,              BuiltinType::Char);
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
  InitBuiltinType(FloatComplexTy,      BuiltinType::FloatComplex);
  InitBuiltinType(DoubleComplexTy,     BuiltinType::DoubleComplex);
  InitBuiltinType(LongDoubleComplexTy, BuiltinType::LongDoubleComplex);
}

/// getPointerType - Return the uniqued reference to the type for a pointer to
/// the specified type.
TypeRef ASTContext::getPointerType(TypeRef T) {
  // FIXME: This is obviously braindead!
  // Unique pointers, to guarantee there is only one pointer of a particular
  // structure.
  ++NumSlowLookups;
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (PointerType *PTy = dyn_cast<PointerType>(Types[i]))
      if (PTy->getPointeeType() == T)
        return Types[i];
  
  // If the pointee type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  Type *Canonical = 0;
  if (!T->isCanonical())
    Canonical = getPointerType(T.getCanonicalType()).getTypePtr();
  
  Types.push_back(new PointerType(T, Canonical));
  return Types.back();
}

/// getArrayType - Return the unique reference to the type for an array of the
/// specified element type.
TypeRef ASTContext::getArrayType(TypeRef EltTy,ArrayType::ArraySizeModifier ASM,
                                 unsigned EltTypeQuals, void *NumElts) {
#warning "IGNORING SIZE"
  
  // FIXME: This is obviously braindead!
  // Unique array, to guarantee there is only one array of a particular
  // structure.
  ++NumSlowLookups;
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (ArrayType *ATy = dyn_cast<ArrayType>(Types[i]))
      if (ATy->getElementType() == EltTy &&
          ATy->getSizeModifier() == ASM &&
          ATy->getIndexTypeQualifier() == EltTypeQuals)
        return Types[i];
  
  // If the element type isn't canonical, this won't be a canonical type either,
  // so fill in the canonical type field.
  Type *Canonical = 0;
  if (!EltTy->isCanonical())
    Canonical = getArrayType(EltTy.getCanonicalType(), ASM, EltTypeQuals,
                             NumElts).getTypePtr();
  
  Types.push_back(new ArrayType(EltTy, ASM, EltTypeQuals, Canonical));
  return Types.back();
}

/// getFunctionTypeNoProto - Return a K&R style C function type like 'int()'.
///
TypeRef ASTContext::getFunctionTypeNoProto(TypeRef ResultTy) {
  // FIXME: This is obviously braindead!
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  ++NumSlowLookups;
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (FunctionTypeNoProto *FTy = dyn_cast<FunctionTypeNoProto>(Types[i]))
      if (FTy->getResultType() == ResultTy)
        return Types[i];

  Type *Canonical = 0;
  if (!ResultTy->isCanonical())
    Canonical =getFunctionTypeNoProto(ResultTy.getCanonicalType()).getTypePtr();
  
  Types.push_back(new FunctionTypeNoProto(ResultTy, Canonical));
  return Types.back();
}

/// getFunctionType - Return a normal function type with a typed argument
/// list.  isVariadic indicates whether the argument list includes '...'.
TypeRef ASTContext::getFunctionType(TypeRef ResultTy, TypeRef *ArgArray,
                                    unsigned NumArgs, bool isVariadic) {
  // FIXME: This is obviously braindead!
  // Unique functions, to guarantee there is only one function of a particular
  // structure.
  ++NumSlowLookups;
  for (unsigned i = 0, e = Types.size(); i != e; ++i) {
    if (FunctionTypeProto *FTy = dyn_cast<FunctionTypeProto>(Types[i]))
      if (FTy->getResultType() == ResultTy &&
          FTy->getNumArgs() == NumArgs &&
          FTy->isVariadic() == isVariadic) {
        bool Match = true;
        for (unsigned arg = 0; arg != NumArgs; ++arg) {
          if (FTy->getArgType(arg) != ArgArray[arg]) {
            Match = false;
            break;
          }
        } 
        if (Match)
          return Types[i];
      }
  }
  
  // Determine whether the type being created is already canonical or not.  
  bool isCanonical = ResultTy->isCanonical();
  for (unsigned i = 0; i != NumArgs && isCanonical; ++i)
    if (!ArgArray[i]->isCanonical())
      isCanonical = false;

  // If this type isn't canonical, get the canonical version of it.
  Type *Canonical = 0;
  if (!isCanonical) {
    SmallVector<TypeRef, 16> CanonicalArgs;
    CanonicalArgs.reserve(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      CanonicalArgs.push_back(ArgArray[i].getCanonicalType());
    
    Canonical = getFunctionType(ResultTy.getCanonicalType(),
                                &CanonicalArgs[0], NumArgs,
                                isVariadic).getTypePtr();
  }
  
  // FunctionTypeProto objects are not allocated with new because they have a
  // variable size array (for parameter types) at the end of them.
  FunctionTypeProto *FTP = 
    (FunctionTypeProto*)malloc(sizeof(FunctionTypeProto) + 
                               (NumArgs-1)*sizeof(TypeRef));
  new (FTP) FunctionTypeProto(ResultTy, ArgArray, NumArgs, isVariadic,
                              Canonical);
  
  Types.push_back(FTP);
  return FTP;
}

/// getTypeDeclType - Return the unique reference to the type for the
/// specified typename decl.
TypeRef ASTContext::getTypeDeclType(TypeDecl *Decl) {
  // FIXME: This is obviously braindead!
  // Unique TypeDecl, to guarantee there is only one TypeDeclType.
  ++NumSlowLookups;
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (TypeNameType *Ty = dyn_cast<TypeNameType>(Types[i]))
      if (Ty->getDecl() == Decl)
        return Types[i];

  // FIXME: does this lose qualifiers from the typedef??
  
  Type *Canonical = Decl->getUnderlyingType().getTypePtr();
  Types.push_back(new TypeNameType(Decl, Canonical));
  return Types.back();
}

/// getTagDeclType - Return the unique reference to the type for the
/// specified TagDecl (struct/union/class/enum) decl.
TypeRef ASTContext::getTagDeclType(TagDecl *Decl) {
  // FIXME: This is obviously braindead!
  // Unique TypeDecl, to guarantee there is only one TaggedType.
  ++NumSlowLookups;
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    if (TaggedType *Ty = dyn_cast<TaggedType>(Types[i]))
      if (Ty->getDecl() == Decl)
        return Types[i];
  
  // FIXME: does this lose qualifiers from the typedef??
  
  Types.push_back(new TaggedType(Decl, 0));
  return Types.back();
}


