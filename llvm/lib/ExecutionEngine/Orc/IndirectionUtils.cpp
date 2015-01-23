#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/CloneSubModule.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/IRBuilder.h"
#include <set>

using namespace llvm;

namespace llvm {

JITIndirections makeCallsSingleIndirect(
    Module &M, const std::function<bool(const Function &)> &ShouldIndirect,
    const char *JITImplSuffix, const char *JITAddrSuffix) {
  std::vector<Function *> Worklist;
  std::vector<std::string> FuncNames;

  for (auto &F : M)
    if (ShouldIndirect(F) && (F.user_begin() != F.user_end())) {
      Worklist.push_back(&F);
      FuncNames.push_back(F.getName());
    }

  for (auto *F : Worklist) {
    GlobalVariable *FImplAddr = new GlobalVariable(
        M, F->getType(), false, GlobalValue::ExternalLinkage,
        Constant::getNullValue(F->getType()), F->getName() + JITAddrSuffix,
        nullptr, GlobalValue::NotThreadLocal, 0, true);
    FImplAddr->setVisibility(GlobalValue::HiddenVisibility);

    for (auto *U : F->users()) {
      assert(isa<Instruction>(U) && "Cannot indirect non-instruction use");
      IRBuilder<> Builder(cast<Instruction>(U));
      U->replaceUsesOfWith(F, Builder.CreateLoad(FImplAddr));
    }
  }

  return JITIndirections(
      FuncNames, [=](StringRef S) -> std::string { return std::string(S); },
      [=](StringRef S)
          -> std::string { return std::string(S) + JITAddrSuffix; });
}

JITIndirections makeCallsDoubleIndirect(
    Module &M, const std::function<bool(const Function &)> &ShouldIndirect,
    const char *JITImplSuffix, const char *JITAddrSuffix) {

  std::vector<Function *> Worklist;
  std::vector<std::string> FuncNames;

  for (auto &F : M)
    if (!F.isDeclaration() && !F.hasAvailableExternallyLinkage() &&
        ShouldIndirect(F))
      Worklist.push_back(&F);

  for (auto *F : Worklist) {
    std::string OrigName = F->getName();
    F->setName(OrigName + JITImplSuffix);
    FuncNames.push_back(OrigName);

    GlobalVariable *FImplAddr = new GlobalVariable(
        M, F->getType(), false, GlobalValue::ExternalLinkage,
        Constant::getNullValue(F->getType()), OrigName + JITAddrSuffix, nullptr,
        GlobalValue::NotThreadLocal, 0, true);
    FImplAddr->setVisibility(GlobalValue::HiddenVisibility);

    Function *FRedirect =
        Function::Create(F->getFunctionType(), F->getLinkage(), OrigName, &M);

    F->replaceAllUsesWith(FRedirect);

    BasicBlock *EntryBlock =
        BasicBlock::Create(M.getContext(), "entry", FRedirect);

    IRBuilder<> Builder(EntryBlock);
    LoadInst *FImplLoadedAddr = Builder.CreateLoad(FImplAddr);

    std::vector<Value *> CallArgs;
    for (Value &Arg : FRedirect->args())
      CallArgs.push_back(&Arg);
    CallInst *Call = Builder.CreateCall(FImplLoadedAddr, CallArgs);
    Call->setTailCall();
    Builder.CreateRet(Call);
  }

  return JITIndirections(
      FuncNames, [=](StringRef S)
                     -> std::string { return std::string(S) + JITImplSuffix; },
      [=](StringRef S)
          -> std::string { return std::string(S) + JITAddrSuffix; });
}

std::vector<std::unique_ptr<Module>>
explode(const Module &OrigMod,
        const std::function<bool(const Function &)> &ShouldExtract) {

  std::vector<std::unique_ptr<Module>> NewModules;

  // Split all the globals, non-indirected functions, etc. into a single module.
  auto ExtractGlobalVars = [&](GlobalVariable &New, const GlobalVariable &Orig,
                               ValueToValueMapTy &VMap) {
    copyGVInitializer(New, Orig, VMap);
    if (New.getLinkage() == GlobalValue::PrivateLinkage) {
      New.setLinkage(GlobalValue::ExternalLinkage);
      New.setVisibility(GlobalValue::HiddenVisibility);
    }
  };

  auto ExtractNonImplFunctions =
      [&](Function &New, const Function &Orig, ValueToValueMapTy &VMap) {
        if (!ShouldExtract(New))
          copyFunctionBody(New, Orig, VMap);
      };

  NewModules.push_back(CloneSubModule(OrigMod, ExtractGlobalVars,
                                      ExtractNonImplFunctions, true));

  // Preserve initializers for Common linkage vars, and make private linkage
  // globals external: they are now provided by the globals module extracted
  // above.
  auto DropGlobalVars = [&](GlobalVariable &New, const GlobalVariable &Orig,
                            ValueToValueMapTy &VMap) {
    if (New.getLinkage() == GlobalValue::CommonLinkage)
      copyGVInitializer(New, Orig, VMap);
    else if (New.getLinkage() == GlobalValue::PrivateLinkage)
      New.setLinkage(GlobalValue::ExternalLinkage);
  };

  // Split each 'impl' function out in to its own module.
  for (const auto &Func : OrigMod) {
    if (Func.isDeclaration() || !ShouldExtract(Func))
      continue;

    auto ExtractNamedFunction =
        [&](Function &New, const Function &Orig, ValueToValueMapTy &VMap) {
          if (New.getName() == Func.getName())
            copyFunctionBody(New, Orig, VMap);
        };

    NewModules.push_back(
        CloneSubModule(OrigMod, DropGlobalVars, ExtractNamedFunction, false));
  }

  return NewModules;
}

std::vector<std::unique_ptr<Module>>
explode(const Module &OrigMod, const JITIndirections &Indirections) {
  std::set<std::string> ImplNames;

  for (const auto &FuncName : Indirections.IndirectedNames)
    ImplNames.insert(Indirections.GetImplName(FuncName));

  return explode(
      OrigMod, [&](const Function &F) { return ImplNames.count(F.getName()); });
}
}
