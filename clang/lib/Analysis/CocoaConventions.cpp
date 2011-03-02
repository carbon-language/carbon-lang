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
#include "llvm/Support/ErrorHandling.h"

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

cocoa::NamingConvention cocoa::deriveNamingConvention(Selector S) {
  switch (S.getMethodFamily()) {
  case OMF_None:
  case OMF_autorelease:
  case OMF_dealloc:
  case OMF_release:
  case OMF_retain:
  case OMF_retainCount:
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

bool cocoa::isRefType(QualType RetTy, llvm::StringRef Prefix,
                      llvm::StringRef Name) {
  // Recursively walk the typedef stack, allowing typedefs of reference types.
  while (const TypedefType *TD = dyn_cast<TypedefType>(RetTy.getTypePtr())) {
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
