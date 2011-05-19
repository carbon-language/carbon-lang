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

#include "llvm/CodeGen/MachineLocation.h" // FIXME
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
    MachineLocation Destination;
    MachineLocation Source;
  public:
    MCWin64EHInstruction(OpType Op, unsigned Register)
      : Operation(Op), Offset(0), Destination(0), Source(Register) {
      assert(Op == Win64EH::UOP_PushNonVol);
    }
    MCWin64EHInstruction(unsigned Size)
      : Operation(Size>128 ? Win64EH::UOP_AllocLarge : Win64EH::UOP_AllocSmall),
        Offset(Size) { }
    MCWin64EHInstruction(unsigned Register, unsigned Off)
      : Operation(Win64EH::UOP_SetFPReg), Offset(Off), Destination(Register) { }
    MCWin64EHInstruction(OpType Op, const MachineLocation &D,
                         unsigned S)
      : Operation(Op), Destination(D), Source(S) {
      assert(Op == Win64EH::UOP_SaveNonVol ||
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
    bool isPushCodeFrame() const { return Offset == 1; }
    const MachineLocation &getDestination() const { return Destination; }
    const MachineLocation &getSource() const { return Source; }
  };

  struct MCWin64EHUnwindInfo {
    MCWin64EHUnwindInfo() : Begin(0), End(0), ExceptionHandler(0), Lsda(0),
                            Function(0), UnwindOnly(false), LsdaSize(0),
                            PrologSize(0), LastFrameInst(-1), Chained(false),
                            Instructions() {}
    MCSymbol *Begin;
    MCSymbol *End;
    const MCSymbol *ExceptionHandler;
    const MCSymbol *Lsda;
    const MCSymbol *Function;
    bool UnwindOnly;
    unsigned LsdaSize;
    unsigned PrologSize;
    int LastFrameInst;
    bool Chained;
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
