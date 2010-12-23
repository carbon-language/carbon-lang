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

#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace ento;

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
  while (*s != '\0') {
    // Skip '_', numbers, ':', etc.
    if (*s == '_' || !isalpha(*s)) {      
      ++s;
      continue;
    }
    break;
  }

  // Parse the first word, and look for specific keywords.
  const char *wordEnd = parseWord(s);
  assert(wordEnd > s);
  unsigned len = wordEnd - s;

  switch (len) {
    default:
      return NoConvention;
    case 3:
      // Methods starting with 'new' follow the create rule.
      return (memcmp(s, "new", 3) == 0) ? CreateRule : NoConvention;
    case 4:
      // Methods starting with 'copy' follow the create rule.
      if (memcmp(s, "copy", 4) == 0)
        return CreateRule;
      // Methods starting with 'init' follow the init rule.
      if (memcmp(s, "init", 4) == 0)
        return InitRule;
      return NoConvention;
    case 5:
      return (memcmp(s, "alloc", 5) == 0) ? CreateRule : NoConvention;
    case 7:
      // Methods starting with 'mutableCopy' follow the create rule.
      if (memcmp(s, "mutable", 7) == 0) {
        // Look at the next word to see if it is "Copy".
        s = wordEnd;
        if (*s != '\0') {
          wordEnd = parseWord(s);
          len = wordEnd - s;
          if (len == 4 && memcmp(s, "Copy", 4) == 0)
            return CreateRule;
        }
      }
      return NoConvention;
  }
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
  
  // We assume that id<..>, id, Class, and Class<..> all represent tracked
  // objects.
  if (PT->isObjCIdType() || PT->isObjCQualifiedIdType() ||
      PT->isObjCClassType() || PT->isObjCQualifiedClassType())
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
