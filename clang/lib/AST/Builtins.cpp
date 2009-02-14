//===--- Builtins.cpp - Builtin function implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements various things for builtin functions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Builtins.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;

static const Builtin::Info BuiltinInfo[] = {
  { "not a builtin function", 0, 0 },
#define BUILTIN(ID, TYPE, ATTRS) { #ID, TYPE, ATTRS },
#include "clang/AST/Builtins.def"
};

const Builtin::Info &Builtin::Context::GetRecord(unsigned ID) const {
  if (ID < Builtin::FirstTSBuiltin)
    return BuiltinInfo[ID];
  assert(ID - Builtin::FirstTSBuiltin < NumTSRecords && "Invalid builtin ID!");
  return TSRecords[ID - Builtin::FirstTSBuiltin];
}


/// InitializeBuiltins - Mark the identifiers for all the builtins with their
/// appropriate builtin ID # and mark any non-portable builtin identifiers as
/// such.
void Builtin::Context::InitializeBuiltins(IdentifierTable &Table,
                                          const TargetInfo &Target) {
  // Step #1: mark all target-independent builtins with their ID's.
  for (unsigned i = Builtin::NotBuiltin+1; i != Builtin::FirstTSBuiltin; ++i)
    Table.get(BuiltinInfo[i].Name).setBuiltinID(i);
  
  // Step #2: Get target builtins.
  Target.getTargetBuiltins(TSRecords, NumTSRecords);

  // Step #3: Register target-specific builtins.
  for (unsigned i = 0, e = NumTSRecords; i != e; ++i)
    Table.get(TSRecords[i].Name).setBuiltinID(i+Builtin::FirstTSBuiltin);
}

std::string Builtin::Context::getHeaderName(unsigned ID) const {
  const char *Name = strchr(GetRecord(ID).Attributes, 'f');
  if (!Name)
    return 0;
  ++Name;

  if (*Name != ':')
    return 0;

  ++Name;
  const char *NameEnd = strchr(Name, ':');
  assert(NameEnd && "Missing ':' after header name");
  return std::string(Name, NameEnd);
}

bool 
Builtin::Context::isPrintfLike(unsigned ID, unsigned &FormatIdx, 
                               bool &HasVAListArg) {
  const char *Printf = strpbrk(GetRecord(ID).Attributes, "pP");
  if (!Printf)
    return false;

  HasVAListArg = (*Printf == 'P');

  ++Printf;
  assert(*Printf == ':' && "p or P specifier must have be followed by a ':'");
  ++Printf;

  const char *PrintfEnd = strchr(Printf, ':');
  assert(PrintfEnd && "printf specifier must end with a ':'");

  FormatIdx = strtol(Printf, 0, 10);
  return true;
}

/// DecodeTypeFromStr - This decodes one type descriptor from Str, advancing the
/// pointer over the consumed characters.  This returns the resultant type.
static QualType DecodeTypeFromStr(const char *&Str, ASTContext &Context, 
                                  Builtin::Context::GetBuiltinTypeError &Error,
                                  bool AllowTypeModifiers = true) {
  // Modifiers.
  bool Long = false, LongLong = false, Signed = false, Unsigned = false;
  
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
      assert(!LongLong && "Can't have LLL modifier");
      if (Long) 
        LongLong = true;
      else
        Long = true;
      break;
    }
  }

  QualType Type;
  
  // Read the base type.
  switch (*Str++) {
  default: assert(0 && "Unknown builtin type letter!");
  case 'v':
    assert(!Long && !Signed && !Unsigned && "Bad modifiers used with 'v'!");
    Type = Context.VoidTy;
    break;
  case 'f':
    assert(!Long && !Signed && !Unsigned && "Bad modifiers used with 'f'!");
    Type = Context.FloatTy;
    break;
  case 'd':
    assert(!LongLong && !Signed && !Unsigned && "Bad modifiers used with 'd'!");
    if (Long)
      Type = Context.LongDoubleTy;
    else
      Type = Context.DoubleTy;
    break;
  case 's':
    assert(!LongLong && "Bad modifiers used with 's'!");
    if (Unsigned)
      Type = Context.UnsignedShortTy;
    else
      Type = Context.ShortTy;
      break;
  case 'i':
    if (LongLong)
      Type = Unsigned ? Context.UnsignedLongLongTy : Context.LongLongTy;
    else if (Long)
      Type = Unsigned ? Context.UnsignedLongTy : Context.LongTy;
    else if (Unsigned)
      Type = Context.UnsignedIntTy;
    else 
      Type = Context.IntTy; // default is signed.
    break;
  case 'c':
    assert(!Long && !LongLong && "Bad modifiers used with 'c'!");
    if (Signed)
      Type = Context.SignedCharTy;
    else if (Unsigned)
      Type = Context.UnsignedCharTy;
    else
      Type = Context.CharTy;
    break;
  case 'b': // boolean
    assert(!Long && !Signed && !Unsigned && "Bad modifiers for 'b'!");
    Type = Context.BoolTy;
    break;
  case 'z':  // size_t.
    assert(!Long && !Signed && !Unsigned && "Bad modifiers for 'z'!");
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
      Type = Context.getReferenceType(Type);
    }
    break;
  case 'V': {
    char *End;
    
    unsigned NumElements = strtoul(Str, &End, 10);
    assert(End != Str && "Missing vector size");
    
    Str = End;
    
    QualType ElementType = DecodeTypeFromStr(Str, Context, Error, false);
    Type = Context.getVectorType(ElementType, NumElements);
    break;
  }
  case 'P': {
    IdentifierInfo *II = &Context.Idents.get("FILE");
    DeclContext::lookup_result Lookup 
      = Context.getTranslationUnitDecl()->lookup(II);
    if (Lookup.first != Lookup.second && isa<TypeDecl>(*Lookup.first)) {
      Type = Context.getTypeDeclType(cast<TypeDecl>(*Lookup.first));
      break;
    }
    else {
      Error = Builtin::Context::GE_Missing_FILE;
      return QualType();
    }
  }
  }
  
  if (!AllowTypeModifiers)
    return Type;
  
  Done = false;
  while (!Done) {
    switch (*Str++) {
      default: Done = true; --Str; break;
      case '*':
        Type = Context.getPointerType(Type);
        break;
      case '&':
        Type = Context.getReferenceType(Type);
        break;
      case 'C':
        Type = Type.getQualifiedType(QualType::Const);
        break;
    }
  }
  
  return Type;
}

/// GetBuiltinType - Return the type for the specified builtin.
QualType Builtin::Context::GetBuiltinType(unsigned id, ASTContext &Context,
                                          GetBuiltinTypeError &Error) const {
  const char *TypeStr = GetRecord(id).Type;
  
  llvm::SmallVector<QualType, 8> ArgTypes;
  
  Error = GE_None;
  QualType ResType = DecodeTypeFromStr(TypeStr, Context, Error);
  if (Error != GE_None)
    return QualType();
  while (TypeStr[0] && TypeStr[0] != '.') {
    QualType Ty = DecodeTypeFromStr(TypeStr, Context, Error);
    if (Error != GE_None)
      return QualType();

    // Do array -> pointer decay.  The builtin should use the decayed type.
    if (Ty->isArrayType())
      Ty = Context.getArrayDecayedType(Ty);
   
    ArgTypes.push_back(Ty);
  }

  assert((TypeStr[0] != '.' || TypeStr[1] == 0) &&
         "'.' should only occur at end of builtin type list!");

  // handle untyped/variadic arguments "T c99Style();" or "T cppStyle(...);".
  if (ArgTypes.size() == 0 && TypeStr[0] == '.')
    return Context.getFunctionTypeNoProto(ResType);
  return Context.getFunctionType(ResType, &ArgTypes[0], ArgTypes.size(),
                                 TypeStr[0] == '.', 0);
}
