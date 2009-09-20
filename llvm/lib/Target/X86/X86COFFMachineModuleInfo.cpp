//===-- llvm/CodeGen/X86COFFMachineModuleInfo.cpp -------------------------===//
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

#include "X86COFFMachineModuleInfo.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

X86COFFMachineModuleInfo::X86COFFMachineModuleInfo(const MachineModuleInfo &) {
}
X86COFFMachineModuleInfo::~X86COFFMachineModuleInfo() {
  
}

void X86COFFMachineModuleInfo::AddFunctionInfo(const Function *F,
                                            const X86MachineFunctionInfo &Val) {
  FunctionInfoMap[F] = Val;
}



static X86MachineFunctionInfo calculateFunctionInfo(const Function *F,
                                                    const TargetData &TD) {
  X86MachineFunctionInfo Info;
  uint64_t Size = 0;
  
  switch (F->getCallingConv()) {
  case CallingConv::X86_StdCall:
    Info.setDecorationStyle(StdCall);
    break;
  case CallingConv::X86_FastCall:
    Info.setDecorationStyle(FastCall);
    break;
  default:
    return Info;
  }
  
  unsigned argNum = 1;
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI, ++argNum) {
    const Type* Ty = AI->getType();
    
    // 'Dereference' type in case of byval parameter attribute
    if (F->paramHasAttr(argNum, Attribute::ByVal))
      Ty = cast<PointerType>(Ty)->getElementType();
    
    // Size should be aligned to DWORD boundary
    Size += ((TD.getTypeAllocSize(Ty) + 3)/4)*4;
  }
  
  // We're not supporting tooooo huge arguments :)
  Info.setBytesToPopOnReturn((unsigned int)Size);
  return Info;
}


/// DecorateCygMingName - Query FunctionInfoMap and use this information for
/// various name decorations for Cygwin and MingW.
void X86COFFMachineModuleInfo::DecorateCygMingName(SmallVectorImpl<char> &Name,
                                                   const GlobalValue *GV,
                                                   const TargetData &TD) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;
  
  // Save function name for later type emission.
  if (F->isDeclaration())
    CygMingStubs.insert(StringRef(Name.data(), Name.size()));
  
  // We don't want to decorate non-stdcall or non-fastcall functions right now
  CallingConv::ID CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;
  
  const X86MachineFunctionInfo *Info;
  
  FMFInfoMap::const_iterator info_item = FunctionInfoMap.find(F);
  if (info_item == FunctionInfoMap.end()) {
    // Calculate apropriate function info and populate map
    FunctionInfoMap[F] = calculateFunctionInfo(F, TD);
    Info = &FunctionInfoMap[F];
  } else {
    Info = &info_item->second;
  }
  
  if (Info->getDecorationStyle() == None) return;
  const FunctionType *FT = F->getFunctionType();
  
  // "Pure" variadic functions do not receive @0 suffix.
  if (!FT->isVarArg() || FT->getNumParams() == 0 ||
      (FT->getNumParams() == 1 && F->hasStructRetAttr()))
    raw_svector_ostream(Name) << '@' << Info->getBytesToPopOnReturn();
  
  if (Info->getDecorationStyle() == FastCall) {
    if (Name[0] == '_')
      Name[0] = '@';
    else
      Name.insert(Name.begin(), '@');
  }    
}

/// DecorateCygMingName - Query FunctionInfoMap and use this information for
/// various name decorations for Cygwin and MingW.
void X86COFFMachineModuleInfo::DecorateCygMingName(std::string &Name,
                                                   const GlobalValue *GV,
                                                   const TargetData &TD) {
  SmallString<128> NameStr(Name.begin(), Name.end());
  DecorateCygMingName(NameStr, GV, TD);
  Name.assign(NameStr.begin(), NameStr.end());
}
