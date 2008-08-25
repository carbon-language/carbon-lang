//== BasicStore.h - Basic map from Locations to Values ----------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the BasicStore and BasicStoreManager classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BASICSTORE_H
#define LLVM_CLANG_ANALYSIS_BASICSTORE_H

#include "clang/Analysis/PathSensitive/Store.h"

namespace llvm {
  class llvm::BumpPtrAllocator; 
  class ASTContext;
}
  
namespace clang {
  StoreManager* CreateBasicStoreManager(llvm::BumpPtrAllocator& Alloc,
                                        ASTContext& Ctx);
}

#endif
