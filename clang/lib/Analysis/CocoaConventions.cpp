//===- CocoaConventions.h - Special handling of Cocoa conventions -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements cocoa naming convention analysis. 
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;
using namespace ento;

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

cocoa::NamingConvention cocoa::deriveNamingConvention(Selector S, 
                                                    const ObjCMethodDecl *MD) {
  switch (MD && MD->hasAttr<ObjCMethodFamilyAttr>()? MD->getMethodFamily() 
                                                   : S.getMethodFamily()) {
  case OMF_None:
  case OMF_autorelease:
  case OMF_dealloc:
  case OMF_release:
  case OMF_retain:
  case OMF_retainCount:
  case OMF_self:
  case OMF_performSelector:
    return NoConvention;

  case OMF_init:
    return InitRule;

  case OMF_alloc:
  case OMF_copy:
  case OMF_mutableCopy:
  case OMF_new:
    return CreateRule;
  }
  llvm_unreachable("unexpected naming convention");
  return NoConvention;
}

bool cocoa::isRefType(QualType RetTy, StringRef Prefix,
                      StringRef Name) {
  // Recursively walk the typedef stack, allowing typedefs of reference types.
  while (const TypedefType *TD = dyn_cast<TypedefType>(RetTy.getTypePtr())) {
    StringRef TDName = TD->getDecl()->getIdentifier()->getName();
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

bool coreFoundation::isCFObjectRef(QualType T) {
  return cocoa::isRefType(T, "CF") || // Core Foundation.
         cocoa::isRefType(T, "CG") || // Core Graphics.
         cocoa::isRefType(T, "DADisk") || // Disk Arbitration API.
         cocoa::isRefType(T, "DADissenter") ||
         cocoa::isRefType(T, "DASessionRef");
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

bool coreFoundation::followsCreateRule(StringRef functionName) {
  StringRef::iterator it = functionName.begin();
  StringRef::iterator start = it;
  StringRef::iterator endI = functionName.end();
    
  while (true) {
    // Scan for the start of 'create' or 'copy'.
    for ( ; it != endI ; ++it) {
      // Search for the first character.  It can either be 'C' or 'c'.
      char ch = *it;
      if (ch == 'C' || ch == 'c') {
        ++it;
        break;
      }
    }

    // Did we hit the end of the string?  If so, we didn't find a match.
    if (it == endI)
      return false;
    
    // Scan for *lowercase* 'reate' or 'opy', followed by no lowercase
    // character.
    StringRef suffix = functionName.substr(it - start);
    if (suffix.startswith("reate")) {
      it += 5;
    }
    else if (suffix.startswith("opy")) {
      it += 3;
    }
    else {
      // Keep scanning.
      continue;
    }
    
    if (it == endI || !islower(*it))
      return true;
  
    // If we matched a lowercase character, it isn't the end of the
    // word.  Keep scanning.
  }
  
  return false;
}
