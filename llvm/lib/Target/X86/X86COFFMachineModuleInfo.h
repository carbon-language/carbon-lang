//===-- llvm/CodeGen/X86COFFMachineModuleInfo.h -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an MMI implementation for X86 COFF (windows) targets.
//
//===----------------------------------------------------------------------===//

#ifndef X86COFF_MACHINEMODULEINFO_H
#define X86COFF_MACHINEMODULEINFO_H

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ADT/StringSet.h"
#include "X86MachineFunctionInfo.h"

namespace llvm {
  class X86MachineFunctionInfo;
  class TargetData;

/// X86COFFMachineModuleInfo - This is a MachineModuleInfoImpl implementation
/// for X86 COFF targets.
class X86COFFMachineModuleInfo : public MachineModuleInfoImpl {
  StringSet<> CygMingStubs;
  DenseMap<const Function*, unsigned> FnArgWords;
public:
  X86COFFMachineModuleInfo(const MachineModuleInfo &);
  ~X86COFFMachineModuleInfo();

  void DecorateCygMingName(MCSymbol* &Name, MCContext &Ctx,
                           const GlobalValue *GV, const TargetData &TD);
  void DecorateCygMingName(SmallVectorImpl<char> &Name, const GlobalValue *GV,
                           const TargetData &TD);

  void addExternalFunction(const StringRef& Name);
  typedef StringSet<>::const_iterator stub_iterator;
  stub_iterator stub_begin() const { return CygMingStubs.begin(); }
  stub_iterator stub_end() const { return CygMingStubs.end(); }
};



} // end namespace llvm

#endif
