//===--- ThreadSafeModuleTest.cpp - Test basic use of ThreadSafeModule ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "gtest/gtest.h"

#include <atomic>
#include <future>
#include <thread>

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(ThreadSafeModuleTest, ContextWhollyOwnedByOneModule) {
  // Test that ownership of a context can be transferred to a single
  // ThreadSafeModule.
  ThreadSafeContext TSCtx(llvm::make_unique<LLVMContext>());
  ThreadSafeModule TSM(llvm::make_unique<Module>("M", *TSCtx.getContext()),
                       std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ContextOwnershipSharedByTwoModules) {
  // Test that ownership of a context can be shared between more than one
  // ThreadSafeModule.
  ThreadSafeContext TSCtx(llvm::make_unique<LLVMContext>());

  ThreadSafeModule TSM1(llvm::make_unique<Module>("M1", *TSCtx.getContext()),
                        TSCtx);
  ThreadSafeModule TSM2(llvm::make_unique<Module>("M2", *TSCtx.getContext()),
                        std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ContextOwnershipSharedWithClient) {
  // Test that ownership of a context can be shared with a client-held
  // ThreadSafeContext so that it can be re-used for new modules.
  ThreadSafeContext TSCtx(llvm::make_unique<LLVMContext>());

  {
    // Create and destroy a module.
    ThreadSafeModule TSM1(llvm::make_unique<Module>("M1", *TSCtx.getContext()),
                          TSCtx);
  }

  // Verify that the context is still available for re-use.
  ThreadSafeModule TSM2(llvm::make_unique<Module>("M2", *TSCtx.getContext()),
                        std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, BasicContextLockAPI) {
  // Test that basic lock API calls work.
  ThreadSafeContext TSCtx(llvm::make_unique<LLVMContext>());
  ThreadSafeModule TSM(llvm::make_unique<Module>("M", *TSCtx.getContext()),
                       TSCtx);

  { auto L = TSCtx.getLock(); }

  { auto L = TSM.getContextLock(); }
}

TEST(ThreadSafeModuleTest, ContextLockPreservesContext) {
  // Test that the existence of a context lock preserves the attached
  // context.
  // The trick to verify this is a bit of a hack: We attach a Module
  // (without the ThreadSafeModule wrapper) to the context, then verify
  // that this Module destructs safely (which it will not if its context
  // has been destroyed) even though all references to the context have
  // been thrown away (apart from the lock).

  ThreadSafeContext TSCtx(llvm::make_unique<LLVMContext>());
  auto L = TSCtx.getLock();
  auto &Ctx = *TSCtx.getContext();
  auto M = llvm::make_unique<Module>("M", Ctx);
  TSCtx = ThreadSafeContext();
}

} // end anonymous namespace
