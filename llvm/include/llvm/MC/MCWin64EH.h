//===- MCWin64EH.h - Machine Code Win64 EH support --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCDwarfFile to support the dwarf
// .file directive and the .loc directive.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWIN64EH_H
#define LLVM_MC_MCWIN64EH_H

#include "llvm/Support/Win64EH.h"
#include <vector>

namespace llvm {
  class MCStreamer;
  class MCSymbol;

  class MCWin64EHInstruction {
  public:
    typedef Win64EH::UnwindOpcodes OpType;
  private:
    OpType Operation;
    unsigned Offset;
    unsigned Register;
  public:
    MCWin64EHInstruction(OpType Op, unsigned Reg)
      : Operation(Op), Offset(0), Register(Reg) {
      assert(Op == Win64EH::UOP_PushNonVol);
    }
    MCWin64EHInstruction(unsigned Size)
      : Operation(Size>128 ? Win64EH::UOP_AllocLarge : Win64EH::UOP_AllocSmall),
        Offset(Size) { }
    MCWin64EHInstruction(OpType Op, unsigned Reg,
                         unsigned Off)
      : Operation(Op), Offset(Off), Register(Reg) {
      assert(Op == Win64EH::UOP_SetFPReg ||
             Op == Win64EH::UOP_SaveNonVol ||
             Op == Win64EH::UOP_SaveNonVolBig ||
             Op == Win64EH::UOP_SaveXMM128 ||
             Op == Win64EH::UOP_SaveXMM128Big);
    }
    MCWin64EHInstruction(OpType Op, bool Code)
      : Operation(Op), Offset(Code ? 1 : 0) {
      assert(Op == Win64EH::UOP_PushMachFrame);
    }
    OpType getOperation() const { return Operation; }
    unsigned getOffset() const { return Offset; }
    unsigned getSize() const { return Offset; }
    unsigned getRegister() const { return Register; }
    bool isPushCodeFrame() const { return Offset == 1; }
  };

  struct MCWin64EHUnwindInfo {
    MCWin64EHUnwindInfo() : Begin(0), End(0), ExceptionHandler(0),
                            Function(0), PrologEnd(0), UnwindOnly(false),
                            LastFrameInst(-1), ChainedParent(0),
                            Instructions() {}
    MCSymbol *Begin;
    MCSymbol *End;
    const MCSymbol *ExceptionHandler;
    const MCSymbol *Function;
    MCSymbol *PrologEnd;
    bool UnwindOnly;
    int LastFrameInst;
    MCWin64EHUnwindInfo *ChainedParent;
    std::vector<MCWin64EHInstruction> Instructions;
  };

  class MCWin64EHUnwindEmitter {
  public:
    //
    // This emits the unwind info section (.xdata in PE/COFF).
    //
    static void Emit(MCStreamer &streamer);
  };
} // end namespace llvm

#endif
