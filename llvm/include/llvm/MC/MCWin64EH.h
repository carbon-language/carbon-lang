//===- MCWin64EH.h - Machine Code Win64 EH support --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations to support the Win64 Exception Handling
// scheme in MC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWIN64EH_H
#define LLVM_MC_MCWIN64EH_H

#include "llvm/MC/MCWinEH.h"
#include "llvm/Support/Win64EH.h"
#include <vector>

namespace llvm {
  class StringRef;
  class MCStreamer;
  class MCSymbol;

namespace Win64EH {
struct Instruction {
  static WinEH::Instruction PushNonVol(MCSymbol *L, unsigned Reg) {
    return WinEH::Instruction(Win64EH::UOP_PushNonVol, L, Reg, -1);
  }
  static WinEH::Instruction Alloc(MCSymbol *L, unsigned Size) {
    return WinEH::Instruction(Size > 128 ? UOP_AllocLarge : UOP_AllocSmall, L,
                              -1, Size);
  }
  static WinEH::Instruction PushMachFrame(MCSymbol *L, bool Code) {
    return WinEH::Instruction(UOP_PushMachFrame, L, -1, Code ? 1 : 0);
  }
  static WinEH::Instruction SaveNonVol(MCSymbol *L, unsigned Reg,
                                       unsigned Offset) {
    return WinEH::Instruction(Offset > 512 * 1024 - 8 ? UOP_SaveNonVolBig
                                                      : UOP_SaveNonVol,
                              L, Reg, Offset);
  }
  static WinEH::Instruction SaveXMM(MCSymbol *L, unsigned Reg,
                                    unsigned Offset) {
    return WinEH::Instruction(Offset > 512 * 1024 - 8 ? UOP_SaveXMM128Big
                                                      : UOP_SaveXMM128,
                              L, Reg, Offset);
  }
  static WinEH::Instruction SetFPReg(MCSymbol *L, unsigned Reg, unsigned Off) {
    return WinEH::Instruction(UOP_SetFPReg, L, Reg, Off);
  }
};
}

  struct MCWinFrameInfo {
    MCWinFrameInfo()
      : Begin(nullptr), End(nullptr),ExceptionHandler(nullptr),
        Function(nullptr), PrologEnd(nullptr), Symbol(nullptr),
        HandlesUnwind(false), HandlesExceptions(false), LastFrameInst(-1),
        ChainedParent(nullptr), Instructions() {}
    MCSymbol *Begin;
    MCSymbol *End;
    const MCSymbol *ExceptionHandler;
    const MCSymbol *Function;
    MCSymbol *PrologEnd;
    MCSymbol *Symbol;
    bool HandlesUnwind;
    bool HandlesExceptions;
    int LastFrameInst;
    MCWinFrameInfo *ChainedParent;
    std::vector<WinEH::Instruction> Instructions;
  };

  class MCWin64EHUnwindEmitter {
  public:
    static StringRef GetSectionSuffix(const MCSymbol *func);
    //
    // This emits the unwind info sections (.pdata and .xdata in PE/COFF).
    //
    static void Emit(MCStreamer &streamer);
    static void EmitUnwindInfo(MCStreamer &streamer, MCWinFrameInfo *info);
  };
} // end namespace llvm

#endif
