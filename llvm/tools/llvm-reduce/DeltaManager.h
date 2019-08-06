//===- llvm-reduce.cpp - The LLVM Delta Reduction utility -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class calls each specialized Delta pass by passing it as a template to
// the generic Delta Pass.
//
//===----------------------------------------------------------------------===//

#include "TestRunner.h"
#include "deltas/Delta.h"
#include "deltas/RemoveFunctions.h"

namespace llvm {

inline void runDeltaPasses(TestRunner &Tester) {
  outs() << "Reducing functions...\n";
  Delta<RemoveFunctions>::run(Tester);
  // TODO: Implement the rest of the Delta Passes
}

} // namespace llvm
