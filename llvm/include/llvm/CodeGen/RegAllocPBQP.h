//===-- RegAllocPBQP.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PBQPBuilder interface, for classes which build PBQP
// instances to represent register allocation problems, and the RegAllocPBQP
// interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCPBQP_H
#define LLVM_CODEGEN_REGALLOCPBQP_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/PBQPRAConstraint.h"
#include "llvm/CodeGen/PBQP/RegAllocSolver.h"

namespace llvm {

  /// @brief Create a PBQP register allocator instance.
  FunctionPass *
  createPBQPRegisterAllocator(char *customPassID = nullptr);
}

#endif /* LLVM_CODEGEN_REGALLOCPBQP_H */
