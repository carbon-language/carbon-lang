//===--- NSAPI.cpp - NSFoundation APIs ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/NSAPI.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

NSAPI::NSAPI(ASTContext &ctx)
  : Ctx(ctx), ClassIds() {
}

IdentifierInfo *NSAPI::getNSClassId(NSClassIdKindKind K) const {
  static const char *ClassName[NumClassIds] = {
    "NSObject",
    "NSString",
    "NSArray",
    "NSMutableArray",
    "NSDictionary",
    "NSMutableDictionary",
    "NSNumber"
  };

  if (!ClassIds[K])
    return (ClassIds[K] = &Ctx.Idents.get(ClassName[K]));

  return ClassIds[K];
}

Selector NSAPI::getNSStringSelector(NSStringMethodKind MK) const {
  if (NSStringSelectors[MK].isNull()) {
    Selector Sel;
    switch (MK) {
    case NSStr_stringWithString:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("stringWithString"));
      break;
    case NSStr_initWithString:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("initWithString"));
      break;
    }
    return (NSStringSelectors[MK] = Sel);
  }

  return NSStringSelectors[MK];
}

Selector NSAPI::getNSArraySelector(NSArrayMethodKind MK) const {
  if (NSArraySelectors[MK].isNull()) {
    Selector Sel;
    switch (MK) {
    case NSArr_array:
      Sel = Ctx.Selectors.getNullarySelector(&Ctx.Idents.get("array"));
      break;
    case NSArr_arrayWithArray:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("arrayWithArray"));
      break;
    case NSArr_arrayWithObject:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("arrayWithObject"));
      break;
    case NSArr_arrayWithObjects:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("arrayWithObjects"));
      break;
    case NSArr_arrayWithObjectsCount: {
      IdentifierInfo *KeyIdents[] = {
        &Ctx.Idents.get("arrayWithObjects"),
        &Ctx.Idents.get("count")
      };
      Sel = Ctx.Selectors.getSelector(2, KeyIdents);
      break;
    }
    case NSArr_initWithArray:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("initWithArray"));
      break;
    case NSArr_initWithObjects:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("initWithObjects"));
      break;
    case NSArr_objectAtIndex:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("objectAtIndex"));
      break;
    case NSMutableArr_replaceObjectAtIndex: {
      IdentifierInfo *KeyIdents[] = {
        &Ctx.Idents.get("replaceObjectAtIndex"),
        &Ctx.Idents.get("withObject")
      };
      Sel = Ctx.Selectors.getSelector(2, KeyIdents);
      break;
    }
    }
    return (NSArraySelectors[MK] = Sel);
  }

  return NSArraySelectors[MK];
}

llvm::Optional<NSAPI::NSArrayMethodKind>
NSAPI::getNSArrayMethodKind(Selector Sel) {
  for (unsigned i = 0; i != NumNSArrayMethods; ++i) {
    NSArrayMethodKind MK = NSArrayMethodKind(i);
    if (Sel == getNSArraySelector(MK))
      return MK;
  }

  return llvm::Optional<NSArrayMethodKind>();
}

Selector NSAPI::getNSDictionarySelector(
                                       NSDictionaryMethodKind MK) const {
  if (NSDictionarySelectors[MK].isNull()) {
    Selector Sel;
    switch (MK) {
    case NSDict_dictionary:
      Sel = Ctx.Selectors.getNullarySelector(&Ctx.Idents.get("dictionary"));
      break;
    case NSDict_dictionaryWithDictionary:
      Sel = Ctx.Selectors.getUnarySelector(
                                   &Ctx.Idents.get("dictionaryWithDictionary"));
      break;
    case NSDict_dictionaryWithObjectForKey: {
      IdentifierInfo *KeyIdents[] = {
        &Ctx.Idents.get("dictionaryWithObject"),
        &Ctx.Idents.get("forKey")
      };
      Sel = Ctx.Selectors.getSelector(2, KeyIdents);
      break;
    }
    case NSDict_dictionaryWithObjectsForKeys: {
      IdentifierInfo *KeyIdents[] = {
        &Ctx.Idents.get("dictionaryWithObjects"),
        &Ctx.Idents.get("forKeys")
      };
      Sel = Ctx.Selectors.getSelector(2, KeyIdents);
      break;
    }
    case NSDict_dictionaryWithObjectsForKeysCount: {
      IdentifierInfo *KeyIdents[] = {
        &Ctx.Idents.get("dictionaryWithObjects"),
        &Ctx.Idents.get("forKeys"),
        &Ctx.Idents.get("count")
      };
      Sel = Ctx.Selectors.getSelector(3, KeyIdents);
      break;
    }
    case NSDict_dictionaryWithObjectsAndKeys:
      Sel = Ctx.Selectors.getUnarySelector(
                               &Ctx.Idents.get("dictionaryWithObjectsAndKeys"));
      break;
    case NSDict_initWithDictionary:
      Sel = Ctx.Selectors.getUnarySelector(
                                         &Ctx.Idents.get("initWithDictionary"));
      break;
    case NSDict_initWithObjectsAndKeys:
      Sel = Ctx.Selectors.getUnarySelector(
                                     &Ctx.Idents.get("initWithObjectsAndKeys"));
      break;
    case NSDict_objectForKey:
      Sel = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get("objectForKey"));
      break;
    case NSMutableDict_setObjectForKey: {
      IdentifierInfo *KeyIdents[] = {
        &Ctx.Idents.get("setObject"),
        &Ctx.Idents.get("forKey")
      };
      Sel = Ctx.Selectors.getSelector(2, KeyIdents);
      break;
    }
    }
    return (NSDictionarySelectors[MK] = Sel);
  }

  return NSDictionarySelectors[MK];
}

llvm::Optional<NSAPI::NSDictionaryMethodKind>
NSAPI::getNSDictionaryMethodKind(Selector Sel) {
  for (unsigned i = 0; i != NumNSDictionaryMethods; ++i) {
    NSDictionaryMethodKind MK = NSDictionaryMethodKind(i);
    if (Sel == getNSDictionarySelector(MK))
      return MK;
  }

  return llvm::Optional<NSDictionaryMethodKind>();
}

Selector NSAPI::getNSNumberLiteralSelector(NSNumberLiteralMethodKind MK,
                                           bool Instance) const {
  static const char *ClassSelectorName[NumNSNumberLiteralMethods] = {
    "numberWithChar",
    "numberWithUnsignedChar",
    "numberWithShort",
    "numberWithUnsignedShort",
    "numberWithInt",
    "numberWithUnsignedInt",
    "numberWithLong",
    "numberWithUnsignedLong",
    "numberWithLongLong",
    "numberWithUnsignedLongLong",
    "numberWithFloat",
    "numberWithDouble",
    "numberWithBool",
    "numberWithInteger",
    "numberWithUnsignedInteger"
  };
  static const char *InstanceSelectorName[NumNSNumberLiteralMethods] = {
    "initWithChar",
    "initWithUnsignedChar",
    "initWithShort",
    "initWithUnsignedShort",
    "initWithInt",
    "initWithUnsignedInt",
    "initWithLong",
    "initWithUnsignedLong",
    "initWithLongLong",
    "initWithUnsignedLongLong",
    "initWithFloat",
    "initWithDouble",
    "initWithBool",
    "initWithInteger",
    "initWithUnsignedInteger"
  };

  Selector *Sels;
  const char **Names;
  if (Instance) {
    Sels = NSNumberInstanceSelectors;
    Names = InstanceSelectorName;
  } else {
    Sels = NSNumberClassSelectors;
    Names = ClassSelectorName;
  }

  if (Sels[MK].isNull())
    Sels[MK] = Ctx.Selectors.getUnarySelector(&Ctx.Idents.get(Names[MK]));
  return Sels[MK];
}

llvm::Optional<NSAPI::NSNumberLiteralMethodKind>
NSAPI::getNSNumberLiteralMethodKind(Selector Sel) const {
  for (unsigned i = 0; i != NumNSNumberLiteralMethods; ++i) {
    NSNumberLiteralMethodKind MK = NSNumberLiteralMethodKind(i);
    if (isNSNumberLiteralSelector(MK, Sel))
      return MK;
  }

  return llvm::Optional<NSNumberLiteralMethodKind>();
}

llvm::Optional<NSAPI::NSNumberLiteralMethodKind>
NSAPI::getNSNumberFactoryMethodKind(QualType T) {
  const BuiltinType *BT = T->getAs<BuiltinType>();
  if (!BT)
    return llvm::Optional<NSAPI::NSNumberLiteralMethodKind>();
  
  switch (BT->getKind()) {
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
    return NSAPI::NSNumberWithChar;
  case BuiltinType::Char_U:
  case BuiltinType::UChar:
    return NSAPI::NSNumberWithUnsignedChar;
  case BuiltinType::Short:
    return NSAPI::NSNumberWithShort;
  case BuiltinType::UShort:
    return NSAPI::NSNumberWithUnsignedShort;
  case BuiltinType::Int:
    return NSAPI::NSNumberWithInt;
  case BuiltinType::UInt:
    return NSAPI::NSNumberWithUnsignedInt;
  case BuiltinType::Long:
    return NSAPI::NSNumberWithLong;
  case BuiltinType::ULong:
    return NSAPI::NSNumberWithUnsignedLong;
  case BuiltinType::LongLong:
    return NSAPI::NSNumberWithLongLong;
  case BuiltinType::ULongLong:
    return NSAPI::NSNumberWithUnsignedLongLong;
  case BuiltinType::Float:
    return NSAPI::NSNumberWithFloat;
  case BuiltinType::Double:
    return NSAPI::NSNumberWithDouble;
  case BuiltinType::Bool:
    return NSAPI::NSNumberWithBool;
    
  case BuiltinType::Void:
  case BuiltinType::WChar_U:
  case BuiltinType::WChar_S:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Int128:
  case BuiltinType::LongDouble:
  case BuiltinType::UInt128:
  case BuiltinType::NullPtr:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCId:
  case BuiltinType::ObjCSel:
  case BuiltinType::BoundMember:
  case BuiltinType::Dependent:
  case BuiltinType::Overload:
  case BuiltinType::UnknownAny:
  case BuiltinType::ARCUnbridgedCast:
  case BuiltinType::Half:
  case BuiltinType::PseudoObject:
    break;
  }
  
  return llvm::Optional<NSAPI::NSNumberLiteralMethodKind>();
}
