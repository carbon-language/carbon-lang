//===--- BinaryContext.h  - Interface for machine-level context -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_FLO_BINARY_CONTEXT_H
#define LLVM_TOOLS_LLVM_FLO_BINARY_CONTEXT_H

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"

#include <functional>
#include <map>
#include <string>
#include <system_error>

namespace llvm {

namespace flo {

/// Everything that's needed to process binaries lives here.
class BinaryContext {

  BinaryContext() = delete;

public:

  // [name] -> [address]
  typedef std::map<std::string, uint64_t> SymbolMapType;
  SymbolMapType GlobalSymbols;

  // [address] -> [name1], [name2], ...
  std::multimap<uint64_t, std::string> GlobalAddresses;

  std::unique_ptr<MCContext> Ctx;

  std::unique_ptr<Triple> TheTriple;

  const Target *TheTarget;

  MCCodeEmitter *MCE;

  std::unique_ptr<MCObjectFileInfo> MOFI;

  std::unique_ptr<const MCAsmInfo> AsmInfo;

  std::unique_ptr<const MCInstrInfo> MII;

  std::unique_ptr<const MCSubtargetInfo> STI;

  std::unique_ptr<MCInstPrinter> InstPrinter;

  std::unique_ptr<const MCInstrAnalysis> MIA;

  std::unique_ptr<const MCRegisterInfo> MRI;

  std::unique_ptr<MCDisassembler> DisAsm;

  std::function<void(std::error_code)> ErrorCheck;

  MCAsmBackend *MAB;

  BinaryContext(std::unique_ptr<MCContext> Ctx,
                std::unique_ptr<Triple> TheTriple,
                const Target *TheTarget,
                MCCodeEmitter *MCE,
                std::unique_ptr<MCObjectFileInfo> MOFI,
                std::unique_ptr<const MCAsmInfo> AsmInfo,
                std::unique_ptr<const MCInstrInfo> MII,
                std::unique_ptr<const MCSubtargetInfo> STI,
                std::unique_ptr<MCInstPrinter> InstPrinter,
                std::unique_ptr<const MCInstrAnalysis> MIA,
                std::unique_ptr<const MCRegisterInfo> MRI,
                std::unique_ptr<MCDisassembler> DisAsm,
                MCAsmBackend *MAB) :
      Ctx(std::move(Ctx)),
      TheTriple(std::move(TheTriple)),
      TheTarget(TheTarget),
      MCE(MCE),
      MOFI(std::move(MOFI)),
      AsmInfo(std::move(AsmInfo)),
      MII(std::move(MII)),
      STI(std::move(STI)),
      InstPrinter(std::move(InstPrinter)),
      MIA(std::move(MIA)),
      MRI(std::move(MRI)),
      DisAsm(std::move(DisAsm)),
      MAB(MAB) {}

  ~BinaryContext() {}
};

} // namespace flo

} // namespace llvm

#endif
