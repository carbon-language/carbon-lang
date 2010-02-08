//===- CocoaConventions.h - Special handling of Cocoa conventions -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines 
//
//===----------------------------------------------------------------------===//

#include "clang/Checker/DomainSpecific/CocoaConventions.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;

using llvm::StringRef;

// The "fundamental rule" for naming conventions of methods:
//  (url broken into two lines)
//  http://developer.apple.com/documentation/Cocoa/Conceptual/
//     MemoryMgmt/Tasks/MemoryManagementRules.html
//
// "You take ownership of an object if you create it using a method whose name
//  begins with "alloc" or "new" or contains "copy" (for example, alloc,
//  newObject, or mutableCopy), or if you send it a retain message. You are
//  responsible for relinquishing ownership of objects you own using release
//  or autorelease. Any other time you receive an object, you must
//  not release it."
//

static bool isWordEnd(char ch, char prev, char next) {
  return ch == '\0'
      || (islower(prev) && isupper(ch)) // xxxC
      || (isupper(prev) && isupper(ch) && islower(next)) // XXCreate
      || !isalpha(ch);
}

static const char* parseWord(const char* s) {
  char ch = *s, prev = '\0';
  assert(ch != '\0');
  char next = *(s+1);
  while (!isWordEnd(ch, prev, next)) {
    prev = ch;
    ch = next;
    next = *((++s)+1);
  }
  return s;
}

cocoa::NamingConvention cocoa::deriveNamingConvention(Selector S) {
  IdentifierInfo *II = S.getIdentifierInfoForSlot(0);

  if (!II)
    return NoConvention;

  const char *s = II->getNameStart();

  // A method/function name may contain a prefix.  We don't know it is there,
  // however, until we encounter the first '_'.
  bool InPossiblePrefix = true;
  bool AtBeginning = true;
  NamingConvention C = NoConvention;

  while (*s != '\0') {
    // Skip '_'.
    if (*s == '_') {
      if (InPossiblePrefix) {
        // If we already have a convention, return it.  Otherwise, skip
        // the prefix as if it wasn't there.
        if (C != NoConvention)
          break;
        
        InPossiblePrefix = false;
        AtBeginning = true;
        assert(C == NoConvention);
      }
      ++s;
      continue;
    }

    // Skip numbers, ':', etc.
    if (!isalpha(*s)) {
      ++s;
      continue;
    }

    const char *wordEnd = parseWord(s);
    assert(wordEnd > s);
    unsigned len = wordEnd - s;

    switch (len) {
    default:
      break;
    case 3:
      // Methods starting with 'new' follow the create rule.
      if (AtBeginning && StringRef(s, len).equals_lower("new"))
        C = CreateRule;
      break;
    case 4:
      // Methods starting with 'alloc' or contain 'copy' follow the
      // create rule
      if (C == NoConvention && StringRef(s, len).equals_lower("copy"))
        C = CreateRule;
      else // Methods starting with 'init' follow the init rule.
        if (AtBeginning && StringRef(s, len).equals_lower("init"))
          C = InitRule;
      break;
    case 5:
      if (AtBeginning && StringRef(s, len).equals_lower("alloc"))
        C = CreateRule;
      break;
    }

    // If we aren't in the prefix and have a derived convention then just
    // return it now.
    if (!InPossiblePrefix && C != NoConvention)
      return C;

    AtBeginning = false;
    s = wordEnd;
  }

  // We will get here if there wasn't more than one word
  // after the prefix.
  return C;
}

bool cocoa::isRefType(QualType RetTy, llvm::StringRef Prefix,
                      llvm::StringRef Name) {
  // Recursively walk the typedef stack, allowing typedefs of reference types.
  while (TypedefType* TD = dyn_cast<TypedefType>(RetTy.getTypePtr())) {
    llvm::StringRef TDName = TD->getDecl()->getIdentifier()->getName();
    if (TDName.startswith(Prefix) && TDName.endswith("Ref"))
      return true;
    
    RetTy = TD->getDecl()->getUnderlyingType();
  }
  
  if (Name.empty())
    return false;
  
  // Is the type void*?
  const PointerType* PT = RetTy->getAs<PointerType>();
  if (!(PT->getPointeeType().getUnqualifiedType()->isVoidType()))
    return false;
  
  // Does the name start with the prefix?
  return Name.startswith(Prefix);
}

bool cocoa::isCFObjectRef(QualType T) {
  return isRefType(T, "CF") || // Core Foundation.
         isRefType(T, "CG") || // Core Graphics.
         isRefType(T, "DADisk") || // Disk Arbitration API.
         isRefType(T, "DADissenter") ||
         isRefType(T, "DASessionRef");
}


bool cocoa::isCocoaObjectRef(QualType Ty) {
  if (!Ty->isObjCObjectPointerType())
    return false;
  
  const ObjCObjectPointerType *PT = Ty->getAs<ObjCObjectPointerType>();
  
  // Can be true for objects with the 'NSObject' attribute.
  if (!PT)
    return true;
  
  // We assume that id<..>, id, and "Class" all represent tracked objects.
  if (PT->isObjCIdType() || PT->isObjCQualifiedIdType() ||
      PT->isObjCClassType())
    return true;
  
  // Does the interface subclass NSObject?
  // FIXME: We can memoize here if this gets too expensive.
  const ObjCInterfaceDecl *ID = PT->getInterfaceDecl();
  
  // Assume that anything declared with a forward declaration and no
  // @interface subclasses NSObject.
  if (ID->isForwardDecl())
    return true;
  
  for ( ; ID ; ID = ID->getSuperClass())
    if (ID->getIdentifier()->getName() == "NSObject")
      return true;
  
  return false;
}
