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

void RegisterAttrNonNullChecker(GRExprEngine &Eng);
void RegisterDereferenceChecker(GRExprEngine &Eng);
void RegisterDivZeroChecker(GRExprEngine &Eng);
void RegisterReturnPointerRangeChecker(GRExprEngine &Eng);
void RegisterReturnStackAddressChecker(GRExprEngine &Eng);  
void RegisterReturnUndefChecker(GRExprEngine &Eng);
void RegisterVLASizeChecker(GRExprEngine &Eng);
void RegisterPointerSubChecker(GRExprEngine &Eng);
void RegisterPointerArithChecker(GRExprEngine &Eng);
void RegisterFixedAddressChecker(GRExprEngine &Eng);
void RegisterCastToStructChecker(GRExprEngine &Eng);
void RegisterCallAndMessageChecker(GRExprEngine &Eng);
void RegisterArrayBoundChecker(GRExprEngine &Eng);
void RegisterUndefinedArraySubscriptChecker(GRExprEngine &Eng);
void RegisterUndefinedAssignmentChecker(GRExprEngine &Eng);
void RegisterUndefBranchChecker(GRExprEngine &Eng);
void RegisterUndefResultChecker(GRExprEngine &Eng);

void RegisterNoReturnFunctionChecker(GRExprEngine &Eng);
void RegisterBuiltinFunctionChecker(GRExprEngine &Eng);
void RegisterOSAtomicChecker(GRExprEngine &Eng);
} // end clang namespace
#endif
