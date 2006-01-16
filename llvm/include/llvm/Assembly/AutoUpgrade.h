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

namespace llvm {
  class Function;

  /// This function inspects the Function \p F to see if it is an old overloaded
  /// intrinsic. If it is, the Function's name is changed to add a suffix that
  /// indicates the kind of arguments or result that it accepts. In LLVM release
  /// 1.7, the overloading of intrinsic functions was replaced with separate
  /// functions for each of the various argument sizes. This function implements
  /// the auto-upgrade feature from the old overloaded names to the new
  /// non-overloaded names. 
  /// @param F The Function to potentially auto-upgrade.
  /// @brief Remove overloaded intrinsic function names.
  bool UpgradeIntrinsicFunction(Function* F);

} // End llvm namespace

#endif
