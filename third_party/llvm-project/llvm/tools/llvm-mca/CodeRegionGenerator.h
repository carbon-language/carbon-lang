//===----------------------- CodeRegionGenerator.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file declares classes responsible for generating llvm-mca
/// CodeRegions from various types of input. llvm-mca only analyzes CodeRegions,
/// so the classes here provide the input-to-CodeRegions translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_CODEREGION_GENERATOR_H
#define LLVM_TOOLS_LLVM_MCA_CODEREGION_GENERATOR_H

#include "CodeRegion.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include <memory>

namespace llvm {
namespace mca {

/// This class is responsible for parsing the input given to the llvm-mca
/// driver, and converting that into a CodeRegions instance.
class CodeRegionGenerator {
protected:
  CodeRegions Regions;
  CodeRegionGenerator(const CodeRegionGenerator &) = delete;
  CodeRegionGenerator &operator=(const CodeRegionGenerator &) = delete;

public:
  CodeRegionGenerator(llvm::SourceMgr &SM) : Regions(SM) {}
  virtual ~CodeRegionGenerator();
  virtual Expected<const CodeRegions &>
  parseCodeRegions(const std::unique_ptr<MCInstPrinter> &IP) = 0;
};

/// This class is responsible for parsing input ASM and generating
/// a CodeRegions instance.
class AsmCodeRegionGenerator final : public CodeRegionGenerator {
  const Target &TheTarget;
  MCContext &Ctx;
  const MCAsmInfo &MAI;
  const MCSubtargetInfo &STI;
  const MCInstrInfo &MCII;
  unsigned AssemblerDialect; // This is set during parsing.

public:
  AsmCodeRegionGenerator(const Target &T, llvm::SourceMgr &SM, MCContext &C,
                         const MCAsmInfo &A, const MCSubtargetInfo &S,
                         const MCInstrInfo &I)
      : CodeRegionGenerator(SM), TheTarget(T), Ctx(C), MAI(A), STI(S), MCII(I),
        AssemblerDialect(0) {}

  unsigned getAssemblerDialect() const { return AssemblerDialect; }
  Expected<const CodeRegions &>
  parseCodeRegions(const std::unique_ptr<MCInstPrinter> &IP) override;
};

} // namespace mca
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_MCA_CODEREGION_GENERATOR_H
