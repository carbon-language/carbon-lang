//===-- llvm/Assembly/AutoUpgrade.h - AutoUpgrade Helpers --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer is distributed under the University 
// of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These functions are implemented by the lib/VMCore/AutoUpgrade.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_AUTOUPGRADE_H
#define LLVM_ASSEMBLY_AUTOUPGRADE_H

#include <string>
#include <vector>

namespace llvm {
  class Function;
  class CallInst;
  class Instruction;
  class Value;
  class BasicBlock;

  /// This function inspects the Function \p F to see if it is an old overloaded
  /// intrinsic. If it is, the Function's name is changed to add a suffix that
  /// indicates the kind of arguments or result that it accepts. In LLVM release
  /// 1.7, the overloading of intrinsic functions was replaced with separate
  /// functions for each of the various argument sizes. This function implements
  /// the auto-upgrade feature from the old overloaded names to the new
  /// non-overloaded names. 
  /// @param F The Function to potentially auto-upgrade.
  /// @returns A corrected version of F, or 0 if no change necessary
  /// @brief Remove overloaded intrinsic function names.
  Function* UpgradeIntrinsicFunction(Function* F);

  /// In LLVM 1.7, the overloading of intrinsic functions was replaced with
  /// separate functions for each of the various argument sizes. This function
  /// implements the auto-upgrade feature from old overloaded names to the new
  /// non-overloaded names. This function inspects the CallInst \p CI to see 
  /// if it is a call to an old overloaded intrinsic. If it is, a new CallInst 
  /// is created that uses the correct Function and possibly casts the 
  /// argument and result to an unsigned type.
  /// @brief Get replacement instruction for overloaded intrinsic function call.
  void UpgradeIntrinsicCall(
    CallInst* CI, ///< The CallInst to potentially auto-upgrade.
    Function* newF = 0 ///< The new function for the call replacement.
  );

  /// Upgrade both the function and all the calls made to it, if that function
  /// needs to be upgraded. This is like a combination of the above two
  /// functions, UpgradeIntrinsicFunction and UpgradeIntrinsicCall. Note that
  /// the calls are replaced so this should only be used in a post-processing
  /// manner (i.e. after all assembly/bytecode has been read).
  bool UpgradeCallsToIntrinsic(Function* F);

} // End llvm namespace

#endif
