//=-- GRExprEngineInternalChecks.h- Builtin GRExprEngine Checks -----*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions to instantiate and register the "built-in"
//  checks in GRExprEngine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GREXPRENGINE_INTERNAL_CHECKS
#define LLVM_CLANG_GREXPRENGINE_INTERNAL_CHECKS

namespace clang {

class GRExprEngine;

void RegisterDivZeroChecker(GRExprEngine &Eng);
void RegisterReturnPointerRangeChecker(GRExprEngine &Eng);
void RegisterReturnStackAddressChecker(GRExprEngine &Eng);  
void RegisterReturnUndefChecker(GRExprEngine &Eng);
  
} // end clang namespace
#endif
