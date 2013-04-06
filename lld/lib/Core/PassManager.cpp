//===- lib/Core/PassManager.cpp - Manage linker passes --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/PassManager.h"

#include "lld/Core/Instrumentation.h"
#include "lld/Core/Pass.h"

#include "llvm/Support/ErrorOr.h"

namespace lld {
ErrorOr<void> PassManager::runOnFile(MutableFile &mf) {
  for (auto &pass : _passes) {
    ScopedTask task(getDefaultDomain(), "Pass");
    pass->perform(mf);
  }
  return llvm::error_code::success();
}
} // end namespace lld
