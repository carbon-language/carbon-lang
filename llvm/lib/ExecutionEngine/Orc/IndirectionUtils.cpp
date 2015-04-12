//===---- IndirectionUtils.cpp - Utilities for call indirection in Orc ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/CloneSubModule.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/IRBuilder.h"
#include <set>
#include <sstream>

namespace llvm {
namespace orc {

Constant* createIRTypedAddress(FunctionType &FT, TargetAddress Addr) {
  Constant *AddrIntVal =
    ConstantInt::get(Type::getInt64Ty(FT.getContext()), Addr);
  Constant *AddrPtrVal =
    ConstantExpr::getCast(Instruction::IntToPtr, AddrIntVal,
                          PointerType::get(&FT, 0));
  return AddrPtrVal;
}

GlobalVariable* createImplPointer(PointerType &PT, Module &M,
                                  const Twine &Name, Constant *Initializer) {
  if (!Initializer)
    Initializer = Constant::getNullValue(&PT);
  return new GlobalVariable(M, &PT, false, GlobalValue::ExternalLinkage,
                            Initializer, Name, nullptr,
                            GlobalValue::NotThreadLocal, 0, true);
}

void makeStub(Function &F, GlobalVariable &ImplPointer) {
  assert(F.isDeclaration() && "Can't turn a definition into a stub.");
  assert(F.getParent() && "Function isn't in a module.");
  Module &M = *F.getParent();
  BasicBlock *EntryBlock = BasicBlock::Create(M.getContext(), "entry", &F);
  IRBuilder<> Builder(EntryBlock);
  LoadInst *ImplAddr = Builder.CreateLoad(&ImplPointer);
  std::vector<Value*> CallArgs;
  for (auto &A : F.args())
    CallArgs.push_back(&A);
  CallInst *Call = Builder.CreateCall(ImplAddr, CallArgs);
  Call->setTailCall();
  Builder.CreateRet(Call);
}

// Utility class for renaming global values and functions during partitioning.
class GlobalRenamer {
public:

  static bool needsRenaming(const Value &New) {
    if (!New.hasName() || New.getName().startswith("\01L"))
      return true;
    return false;
  }

  const std::string& getRename(const Value &Orig) {
    // See if we have a name for this global.
    {
      auto I = Names.find(&Orig);
      if (I != Names.end())
        return I->second;
    }

    // Nope. Create a new one.
    // FIXME: Use a more robust uniquing scheme. (This may blow up if the user
    //        writes a "__orc_anon[[:digit:]]* method).
    unsigned ID = Names.size();
    std::ostringstream NameStream;
    NameStream << "__orc_anon" << ID++;
    auto I = Names.insert(std::make_pair(&Orig, NameStream.str()));
    return I.first->second;
  }
private:
  DenseMap<const Value*, std::string> Names;
};

void partition(Module &M, const ModulePartitionMap &PMap) {

  GlobalRenamer Renamer;

  for (auto &KVPair : PMap) {

    auto ExtractGlobalVars =
      [&](GlobalVariable &New, const GlobalVariable &Orig,
          ValueToValueMapTy &VMap) {
        if (KVPair.second.count(&Orig)) {
          copyGVInitializer(New, Orig, VMap);
        }
        if (New.hasLocalLinkage()) {
          if (Renamer.needsRenaming(New))
            New.setName(Renamer.getRename(Orig));
          New.setLinkage(GlobalValue::ExternalLinkage);
          New.setVisibility(GlobalValue::HiddenVisibility);
        }
        assert(!Renamer.needsRenaming(New) && "Invalid global name.");
      };

    auto ExtractFunctions =
      [&](Function &New, const Function &Orig, ValueToValueMapTy &VMap) {
        if (KVPair.second.count(&Orig))
          copyFunctionBody(New, Orig, VMap);
        if (New.hasLocalLinkage()) {
          if (Renamer.needsRenaming(New))
            New.setName(Renamer.getRename(Orig));
          New.setLinkage(GlobalValue::ExternalLinkage);
          New.setVisibility(GlobalValue::HiddenVisibility);
        }
        assert(!Renamer.needsRenaming(New) && "Invalid function name.");
      };

    CloneSubModule(*KVPair.first, M, ExtractGlobalVars, ExtractFunctions,
                   false);
  }
}

FullyPartitionedModule fullyPartition(Module &M) {
  FullyPartitionedModule MP;

  ModulePartitionMap PMap;

  for (auto &F : M) {

    if (F.isDeclaration())
      continue;

    std::string NewModuleName = (M.getName() + "." + F.getName()).str();
    MP.Functions.push_back(
      llvm::make_unique<Module>(NewModuleName, M.getContext()));
    MP.Functions.back()->setDataLayout(M.getDataLayout());
    PMap[MP.Functions.back().get()].insert(&F);
  }

  MP.GlobalVars =
    llvm::make_unique<Module>((M.getName() + ".globals_and_stubs").str(),
                              M.getContext());
  MP.GlobalVars->setDataLayout(M.getDataLayout());

  MP.Commons =
    llvm::make_unique<Module>((M.getName() + ".commons").str(), M.getContext());
  MP.Commons->setDataLayout(M.getDataLayout());

  // Make sure there's at least an empty set for the stubs map or we'll fail
  // to clone anything for it (including the decls).
  PMap[MP.GlobalVars.get()] = ModulePartitionMap::mapped_type();
  for (auto &GV : M.globals())
    if (GV.getLinkage() == GlobalValue::CommonLinkage)
      PMap[MP.Commons.get()].insert(&GV);
    else
      PMap[MP.GlobalVars.get()].insert(&GV);

  partition(M, PMap);

  return MP;
}

} // End namespace orc.
} // End namespace llvm.
