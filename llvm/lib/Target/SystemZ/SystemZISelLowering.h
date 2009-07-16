//==-- SystemZISelLowering.h - SystemZ DAG Lowering Interface ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that SystemZ uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_SystemZ_ISELLOWERING_H
#define LLVM_TARGET_SystemZ_ISELLOWERING_H

#include "SystemZ.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
  namespace SystemZISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END
    };
  }

  class SystemZSubtarget;
  class SystemZTargetMachine;

  class SystemZTargetLowering : public TargetLowering {
  public:
    explicit SystemZTargetLowering(SystemZTargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);

    /// getTargetNodeName - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

  private:
    const SystemZSubtarget &Subtarget;
    const SystemZTargetMachine &TM;
  };
} // namespace llvm

#endif // LLVM_TARGET_SystemZ_ISELLOWERING_H
