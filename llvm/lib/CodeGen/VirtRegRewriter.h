//===-- llvm/CodeGen/VirtRegRewriter.h - VirtRegRewriter -*- C++ -*--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VIRTREGREWRITER_H
#define LLVM_CODEGEN_VIRTREGREWRITER_H

#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "VirtRegMap.h"

namespace llvm {
  
  /// VirtRegRewriter interface: Implementations of this interface assign
  /// spilled virtual registers to stack slots, rewriting the code.
  struct VirtRegRewriter {
    virtual ~VirtRegRewriter();
    virtual bool runOnMachineFunction(MachineFunction &MF, VirtRegMap &VRM,
                                      LiveIntervals* LIs) = 0;
  };

  /// createVirtRegRewriter - Create an return a rewriter object, as specified
  /// on the command line.
  VirtRegRewriter* createVirtRegRewriter();

}

#endif
