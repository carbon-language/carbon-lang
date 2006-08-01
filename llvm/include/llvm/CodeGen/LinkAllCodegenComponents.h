//===- llvm/Codegen/LinkAllCodegenComponents.h ------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all codegen related passes for tools like lli and
// llc that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LINKALLCODEGENCOMPONENTS_H
#define LLVM_CODEGEN_LINKALLCODEGENCOMPONENTS_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/ScheduleDAG.h"

namespace {
  struct ForceCodegenLinking {
    ForceCodegenLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      (void) llvm::createSimpleRegisterAllocator();
      (void) llvm::createLocalRegisterAllocator();
      (void) llvm::createLinearScanRegisterAllocator();
      
      (void) llvm::createBFS_DAGScheduler(NULL, NULL, NULL);
      (void) llvm::createSimpleDAGScheduler(NULL, NULL, NULL);
      (void) llvm::createNoItinsDAGScheduler(NULL, NULL, NULL);
      (void) llvm::createBURRListDAGScheduler(NULL, NULL, NULL);
      (void) llvm::createTDRRListDAGScheduler(NULL, NULL, NULL);
      (void) llvm::createTDListDAGScheduler(NULL, NULL, NULL);

    }
  } ForceCodegenLinking; // Force link by creating a global definition.
}

#endif
