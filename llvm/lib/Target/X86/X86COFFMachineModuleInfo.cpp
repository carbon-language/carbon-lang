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
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

X86COFFMachineModuleInfo::X86COFFMachineModuleInfo(const MachineModuleInfo &) {
}
X86COFFMachineModuleInfo::~X86COFFMachineModuleInfo() {
}

void X86COFFMachineModuleInfo::addExternalFunction(const StringRef& Name) {
  CygMingStubs.insert(Name);
}

/// DecorateCygMingName - Apply various name decorations if the function uses
/// stdcall or fastcall calling convention.
void X86COFFMachineModuleInfo::DecorateCygMingName(SmallVectorImpl<char> &Name,
                                                   const GlobalValue *GV,
                                                   const TargetData &TD) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

  // We don't want to decorate non-stdcall or non-fastcall functions right now
  CallingConv::ID CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;

  unsigned ArgWords = 0;
  DenseMap<const Function*, unsigned>::const_iterator item = FnArgWords.find(F);
  if (item == FnArgWords.end()) {
    // Calculate arguments sizes
    for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
         AI != AE; ++AI) {
      const Type* Ty = AI->getType();

      // 'Dereference' type in case of byval parameter attribute
      if (AI->hasByValAttr())
        Ty = cast<PointerType>(Ty)->getElementType();

      // Size should be aligned to DWORD boundary
      ArgWords += ((TD.getTypeAllocSize(Ty) + 3)/4)*4;
    }

    FnArgWords[F] = ArgWords;
  } else
    ArgWords = item->second;

  const FunctionType *FT = F->getFunctionType();
  // "Pure" variadic functions do not receive @0 suffix.
  if (!FT->isVarArg() || FT->getNumParams() == 0 ||
      (FT->getNumParams() == 1 && F->hasStructRetAttr()))
    raw_svector_ostream(Name) << '@' << ArgWords;

  if (CC == CallingConv::X86_FastCall) {
    if (Name[0] == '_')
      Name[0] = '@';
    else
      Name.insert(Name.begin(), '@');
  }
}

/// DecorateCygMingName - Query FunctionInfoMap and use this information for
/// various name decorations for Cygwin and MingW.
void X86COFFMachineModuleInfo::DecorateCygMingName(MCSymbol *&Name,
                                                   MCContext &Ctx,
                                                   const GlobalValue *GV,
                                                   const TargetData &TD) {
  SmallString<128> NameStr(Name->getName().begin(), Name->getName().end());
  DecorateCygMingName(NameStr, GV, TD);

  Name = Ctx.GetOrCreateSymbol(NameStr.str());
}
