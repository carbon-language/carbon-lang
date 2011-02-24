//=-- InternalChecks.h- Builtin ExprEngine Checks -------------------*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions to instantiate and register the "built-in"
//  checks in ExprEngine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_ExprEngine_INTERNAL_CHECKS
#define LLVM_CLANG_GR_ExprEngine_INTERNAL_CHECKS

namespace clang {

namespace ento {

class ExprEngine;

// Foundational checks that handle basic semantics.
void RegisterAdjustedReturnValueChecker(ExprEngine &Eng);
void RegisterArrayBoundCheckerV2(ExprEngine &Eng);
void RegisterAttrNonNullChecker(ExprEngine &Eng);
void RegisterBuiltinFunctionChecker(ExprEngine &Eng);
void RegisterCallAndMessageChecker(ExprEngine &Eng);
void RegisterDereferenceChecker(ExprEngine &Eng);
void RegisterDivZeroChecker(ExprEngine &Eng);
void RegisterNoReturnFunctionChecker(ExprEngine &Eng);
void RegisterReturnUndefChecker(ExprEngine &Eng);
void RegisterUndefBranchChecker(ExprEngine &Eng);
void RegisterUndefCapturedBlockVarChecker(ExprEngine &Eng);
void RegisterUndefResultChecker(ExprEngine &Eng);
void RegisterUndefinedArraySubscriptChecker(ExprEngine &Eng);
void RegisterUndefinedAssignmentChecker(ExprEngine &Eng);
void RegisterVLASizeChecker(ExprEngine &Eng);

// API checks.
void RegisterOSAtomicChecker(ExprEngine &Eng);

} // end GR namespace

} // end clang namespace

#endif
