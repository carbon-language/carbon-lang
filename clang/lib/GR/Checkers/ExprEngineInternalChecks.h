//=-- ExprEngineInternalChecks.h- Builtin ExprEngine Checks -----*- C++ -*-=
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

namespace GR {

class ExprEngine;

// Foundational checks that handle basic semantics.
void RegisterAdjustedReturnValueChecker(ExprEngine &Eng);
void RegisterArrayBoundChecker(ExprEngine &Eng);
void RegisterArrayBoundCheckerV2(ExprEngine &Eng);
void RegisterAttrNonNullChecker(ExprEngine &Eng);
void RegisterBuiltinFunctionChecker(ExprEngine &Eng);
void RegisterCallAndMessageChecker(ExprEngine &Eng);
void RegisterCastToStructChecker(ExprEngine &Eng);
void RegisterCastSizeChecker(ExprEngine &Eng);
void RegisterDereferenceChecker(ExprEngine &Eng);
void RegisterDivZeroChecker(ExprEngine &Eng);
void RegisterFixedAddressChecker(ExprEngine &Eng);
void RegisterNoReturnFunctionChecker(ExprEngine &Eng);
void RegisterObjCAtSyncChecker(ExprEngine &Eng);
void RegisterPointerArithChecker(ExprEngine &Eng);
void RegisterPointerSubChecker(ExprEngine &Eng);
void RegisterReturnPointerRangeChecker(ExprEngine &Eng);
void RegisterReturnUndefChecker(ExprEngine &Eng);
void RegisterStackAddrLeakChecker(ExprEngine &Eng);
void RegisterUndefBranchChecker(ExprEngine &Eng);
void RegisterUndefCapturedBlockVarChecker(ExprEngine &Eng);
void RegisterUndefResultChecker(ExprEngine &Eng);
void RegisterUndefinedArraySubscriptChecker(ExprEngine &Eng);
void RegisterUndefinedAssignmentChecker(ExprEngine &Eng);
void RegisterVLASizeChecker(ExprEngine &Eng);

// API checks.
void RegisterMacOSXAPIChecker(ExprEngine &Eng);
void RegisterOSAtomicChecker(ExprEngine &Eng);
void RegisterUnixAPIChecker(ExprEngine &Eng);

} // end GR namespace

} // end clang namespace

#endif
