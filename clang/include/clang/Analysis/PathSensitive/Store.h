//== Store.h - Interface for maps from Locations to Values ------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the types Store and StoreManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_STORE_H
#define LLVM_CLANG_ANALYSIS_STORE_H

#include "clang/Analysis/PathSensitive/RValues.h"

namespace clang {
  
typedef const void* Store;
  
class StoreManager {
public:
  virtual ~StoreManager() {}
  virtual RVal GetRVal(Store St, LVal LV, QualType T) = 0;
  virtual Store SetRVal(Store St, LVal LV, RVal V) = 0;
  virtual Store Remove(Store St, LVal LV) = 0;
  virtual Store getInitialStore() = 0;
};
  
} // end clang namespace

#endif
