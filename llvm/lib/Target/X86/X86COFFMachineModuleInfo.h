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
  
  // We have to propagate some information about MachineFunction to
  // AsmPrinter. It's ok, when we're printing the function, since we have
  // access to MachineFunction and can get the appropriate MachineFunctionInfo.
  // Unfortunately, this is not possible when we're printing reference to
  // Function (e.g. calling it and so on). Even more, there is no way to get the
  // corresponding MachineFunctions: it can even be not created at all. That's
  // why we should use additional structure, when we're collecting all necessary
  // information.
  //
  // This structure is using e.g. for name decoration for stdcall & fastcall'ed
  // function, since we have to use arguments' size for decoration.
  typedef std::map<const Function*, X86MachineFunctionInfo> FMFInfoMap;
  FMFInfoMap FunctionInfoMap;
  
public:
  X86COFFMachineModuleInfo(const MachineModuleInfo &);
  ~X86COFFMachineModuleInfo();
  
  
  void DecorateCygMingName(MCSymbol* &Name, MCContext &Ctx,
                           const GlobalValue *GV, const TargetData &TD);
  void DecorateCygMingName(SmallVectorImpl<char> &Name, const GlobalValue *GV,
                           const TargetData &TD);
  
  void AddFunctionInfo(const Function *F, const X86MachineFunctionInfo &Val);
  

  typedef StringSet<>::const_iterator stub_iterator;
  stub_iterator stub_begin() const { return CygMingStubs.begin(); }
  stub_iterator stub_end() const { return CygMingStubs.end(); }

  
};



} // end namespace llvm

#endif
