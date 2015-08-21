//===- SplitModule.cpp - Split a module into partitions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the function llvm::SplitModule, which splits a module
// into multiple linkable partitions. It can be used to implement parallel code
// generation for link-time optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

static void externalize(GlobalValue *GV) {
  if (GV->hasLocalLinkage()) {
    GV->setLinkage(GlobalValue::ExternalLinkage);
    GV->setVisibility(GlobalValue::HiddenVisibility);
  }

  // Unnamed entities must be named consistently between modules. setName will
  // give a distinct name to each such entity.
  if (!GV->hasName())
    GV->setName("__llvmsplit_unnamed");
}

// Returns whether GV should be in partition (0-based) I of N.
static bool isInPartition(const GlobalValue *GV, unsigned I, unsigned N) {
  if (auto GA = dyn_cast<GlobalAlias>(GV))
    if (const GlobalObject *Base = GA->getBaseObject())
      GV = Base;

  StringRef Name;
  if (const Comdat *C = GV->getComdat())
    Name = C->getName();
  else
    Name = GV->getName();

  // Partition by MD5 hash. We only need a few bits for evenness as the number
  // of partitions will generally be in the 1-2 figure range; the low 16 bits
  // are enough.
  MD5 H;
  MD5::MD5Result R;
  H.update(Name);
  H.final(R);
  return (R[0] | (R[1] << 8)) % N == I;
}

void llvm::SplitModule(
    std::unique_ptr<Module> M, unsigned N,
    std::function<void(std::unique_ptr<Module> MPart)> ModuleCallback) {
  for (Function &F : *M)
    externalize(&F);
  for (GlobalVariable &GV : M->globals())
    externalize(&GV);
  for (GlobalAlias &GA : M->aliases())
    externalize(&GA);

  // FIXME: We should be able to reuse M as the last partition instead of
  // cloning it.
  for (unsigned I = 0; I != N; ++I) {
    ValueToValueMapTy VMap;
    std::unique_ptr<Module> MPart(
        CloneModule(M.get(), VMap, [=](const GlobalValue *GV) {
          return isInPartition(GV, I, N);
        }));
    if (I != 0)
      MPart->setModuleInlineAsm("");
    ModuleCallback(std::move(MPart));
  }
}
