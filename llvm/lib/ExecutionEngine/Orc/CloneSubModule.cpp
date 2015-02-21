#include "llvm/ExecutionEngine/Orc/CloneSubModule.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm {
namespace orc {

void copyGVInitializer(GlobalVariable &New, const GlobalVariable &Orig,
                             ValueToValueMapTy &VMap) {
  if (Orig.hasInitializer())
    New.setInitializer(MapValue(Orig.getInitializer(), VMap));
}

void copyFunctionBody(Function &New, const Function &Orig,
                            ValueToValueMapTy &VMap) {
  if (!Orig.isDeclaration()) {
    Function::arg_iterator DestI = New.arg_begin();
    for (Function::const_arg_iterator J = Orig.arg_begin(); J != Orig.arg_end();
         ++J) {
      DestI->setName(J->getName());
      VMap[J] = DestI++;
    }

    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(&New, &Orig, VMap, /*ModuleLevelChanges=*/true, Returns);
  }
}

void CloneSubModule(llvm::Module &Dst, const Module &Src,
                    HandleGlobalVariableFtor HandleGlobalVariable,
                    HandleFunctionFtor HandleFunction, bool CloneInlineAsm) {

  ValueToValueMapTy VMap;

  if (CloneInlineAsm)
    Dst.appendModuleInlineAsm(Src.getModuleInlineAsm());

  // Copy global variables (but not initializers, yet).
  for (Module::const_global_iterator I = Src.global_begin(), E = Src.global_end();
       I != E; ++I) {
    GlobalVariable *GV = new GlobalVariable(
        Dst, I->getType()->getElementType(), I->isConstant(), I->getLinkage(),
        (Constant *)nullptr, I->getName(), (GlobalVariable *)nullptr,
        I->getThreadLocalMode(), I->getType()->getAddressSpace());
    GV->copyAttributesFrom(I);
    VMap[I] = GV;
  }

  // Loop over the functions in the module, making external functions as before
  for (Module::const_iterator I = Src.begin(), E = Src.end(); I != E; ++I) {
    Function *NF =
        Function::Create(cast<FunctionType>(I->getType()->getElementType()),
                         I->getLinkage(), I->getName(), &Dst);
    NF->copyAttributesFrom(I);
    VMap[I] = NF;
  }

  // Loop over the aliases in the module
  for (Module::const_alias_iterator I = Src.alias_begin(), E = Src.alias_end();
       I != E; ++I) {
    auto *PTy = cast<PointerType>(I->getType());
    auto *GA =
        GlobalAlias::create(PTy->getElementType(), PTy->getAddressSpace(),
                            I->getLinkage(), I->getName(), &Dst);
    GA->copyAttributesFrom(I);
    VMap[I] = GA;
  }

  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  for (Module::const_global_iterator I = Src.global_begin(), E = Src.global_end();
       I != E; ++I) {
    GlobalVariable &GV = *cast<GlobalVariable>(VMap[I]);
    HandleGlobalVariable(GV, *I, VMap);
  }

  // Similarly, copy over function bodies now...
  //
  for (Module::const_iterator I = Src.begin(), E = Src.end(); I != E; ++I) {
    Function &F = *cast<Function>(VMap[I]);
    HandleFunction(F, *I, VMap);
  }

  // And aliases
  for (Module::const_alias_iterator I = Src.alias_begin(), E = Src.alias_end();
       I != E; ++I) {
    GlobalAlias *GA = cast<GlobalAlias>(VMap[I]);
    if (const Constant *C = I->getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }

  // And named metadata....
  for (Module::const_named_metadata_iterator I = Src.named_metadata_begin(),
                                             E = Src.named_metadata_end();
       I != E; ++I) {
    const NamedMDNode &NMD = *I;
    NamedMDNode *NewNMD = Dst.getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapMetadata(NMD.getOperand(i), VMap));
  }

}

} // End namespace orc.
} // End namespace llvm.
