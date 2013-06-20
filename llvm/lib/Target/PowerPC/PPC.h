//===-- PPC.h - Top-level interface for PowerPC Target ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PowerPC back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_POWERPC_H
#define LLVM_TARGET_POWERPC_H

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include <string>

// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

namespace llvm {
  class PPCTargetMachine;
  class FunctionPass;
  class ImmutablePass;
  class JITCodeEmitter;
  class MachineInstr;
  class AsmPrinter;
  class MCInst;

  FunctionPass *createPPCCTRLoops(PPCTargetMachine &TM);
#ifndef NDEBUG
  FunctionPass *createPPCCTRLoopsVerify();
#endif
  FunctionPass *createPPCEarlyReturnPass();
  FunctionPass *createPPCBranchSelectionPass();
  FunctionPass *createPPCISelDag(PPCTargetMachine &TM);
  FunctionPass *createPPCJITCodeEmitterPass(PPCTargetMachine &TM,
                                            JITCodeEmitter &MCE);
  void LowerPPCMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                    AsmPrinter &AP);

  /// \brief Creates an PPC-specific Target Transformation Info pass.
  ImmutablePass *createPPCTargetTransformInfoPass(const PPCTargetMachine *TM);

  namespace PPCII {
    
  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // PPC Specific MachineOperand flags.
    MO_NO_FLAG,
    
    /// MO_DARWIN_STUB - On a symbol operand "FOO", this indicates that the
    /// reference is actually to the "FOO$stub" symbol.  This is used for calls
    /// and jumps to external functions on Tiger and earlier.
    MO_DARWIN_STUB = 1,
    
    /// MO_PIC_FLAG - If this bit is set, the symbol reference is relative to
    /// the function's picbase, e.g. lo16(symbol-picbase).
    MO_PIC_FLAG = 2,

    /// MO_NLP_FLAG - If this bit is set, the symbol reference is actually to
    /// the non_lazy_ptr for the global, e.g. lo16(symbol$non_lazy_ptr-picbase).
    MO_NLP_FLAG = 4,
    
    /// MO_NLP_HIDDEN_FLAG - If this bit is set, the symbol reference is to a
    /// symbol with hidden visibility.  This causes a different kind of
    /// non-lazy-pointer to be generated.
    MO_NLP_HIDDEN_FLAG = 8,

    /// The next are not flags but distinct values.
    MO_ACCESS_MASK = 0xf0,

    /// MO_LO16, MO_HA16 - lo16(symbol) and ha16(symbol)
    MO_LO16 = 1 << 4,
    MO_HA16 = 2 << 4,

    MO_TPREL16_HA = 3 << 4,
    MO_TPREL16_LO = 4 << 4,

    /// These values identify relocations on immediates folded
    /// into memory operations.
    MO_DTPREL16_LO = 5 << 4,
    MO_TLSLD16_LO  = 6 << 4,
    MO_TOC16_LO    = 7 << 4
  };
  } // end namespace PPCII
  
} // end namespace llvm;

#endif
