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

#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Streams.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "VirtRegMap.h"
#include <map>

// TODO:
//       - Finish renaming Spiller -> Rewriter
//         - SimpleSpiller
//         - LocalSpiller

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
