//==- Regions.cpp - Abstract memory locations ----------------------*- C++ -*-//
//             
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines Region and its subclasses.  Regions represent abstract
//  memory locations.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Regions.h"
#include "clang/Analysis/PathSensitive/BasicValueFactory.h"
#include "clang/AST/ASTContext.h"

using namespace clang;

RegionExtent VarRegion::getExtent(BasicValueFactory& BV) const {
  QualType T = getDecl()->getType();
  
  // FIXME: Add support for VLAs.  This may require passing in additional
  //  information, or tracking a different region type.
  if (!T.getTypePtr()->isConstantSizeType())
    return UnknownExtent();

  ASTContext& C = BV.getContext();
  assert (!T->isObjCInterfaceType()); // @interface not a possible VarDecl type.
  assert (T != C.VoidTy); // void not a possible VarDecl type.  
  return IntExtent(BV.getValue(C.getTypeSize(T), C.VoidPtrTy));
}

