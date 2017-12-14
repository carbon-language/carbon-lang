//===-- ProfileWriter.cpp - Serialize profiling data ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "ProfileWriter.h"
#include "ProfileYAMLMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-prof"

namespace llvm {
namespace bolt {

std::error_code
ProfileWriter::writeProfile(std::map<uint64_t, BinaryFunction> &Functions) {
  std::error_code EC;
  OS = make_unique<raw_fd_ostream>(FileName, EC, sys::fs::F_None);
  if (EC) {
    errs() << "BOLT-WARNING: " << EC.message() << " : unable to open "
           << FileName << " for output.\n";
    return EC;
  }

  printBinaryFunctionsProfile(Functions);

  return std::error_code();
}

namespace {
void
convert(const BinaryFunction &BF, yaml::bolt::BinaryFunctionProfile &YamlBF) {
  auto &BC = BF.getBinaryContext();

  YamlBF.Name = BF.getPrintName();
  YamlBF.Id = BF.getFunctionNumber();
  YamlBF.Hash = BF.hash(true, true);
  YamlBF.ExecCount = BF.getKnownExecutionCount();
  YamlBF.NumBasicBlocks = BF.size();

  for (const auto *BB : BF.dfs()) {
    yaml::bolt::BinaryBasicBlockProfile YamlBB;
    YamlBB.Index = BB->getLayoutIndex();
    YamlBB.NumInstructions = BB->getNumNonPseudos();
    YamlBB.ExecCount = BB->getKnownExecutionCount();

    for (const auto &Instr : *BB) {
      if (!BC.MIA->isCall(Instr) && !BC.MIA->isIndirectBranch(Instr))
        continue;

      yaml::bolt::CallSiteInfo CSI;
      auto Offset = BC.MIA->tryGetAnnotationAs<uint64_t>(Instr, "Offset");
      if (!Offset || Offset.get() < BB->getInputOffset())
        continue;
      CSI.Offset = Offset.get() - BB->getInputOffset();

      if (BC.MIA->isIndirectCall(Instr) || BC.MIA->isIndirectBranch(Instr)) {
        auto ICSP =
          BC.MIA->tryGetAnnotationAs<IndirectCallSiteProfile>(Instr,
                                                              "CallProfile");
        if (!ICSP)
          continue;
        for (auto &CSP : ICSP.get()) {
          CSI.DestId = 0; // designated for unknown functions
          CSI.EntryDiscriminator = 0;
          if (CSP.IsFunction) {
            const auto *CalleeSymbol = BC.getGlobalSymbolByName(CSP.Name);
            if (CalleeSymbol) {
              const auto *Callee = BC.getFunctionForSymbol(CalleeSymbol);
              if (Callee) {
                CSI.DestId = Callee->getFunctionNumber();
              }
            }
          }
          CSI.Count = CSP.Count;
          CSI.Mispreds = CSP.Mispreds;
          YamlBB.CallSites.push_back(CSI);
        }
      } else { // direct call or a tail call
        const auto *CalleeSymbol = BC.MIA->getTargetSymbol(Instr);
        const auto Callee = BC.getFunctionForSymbol(CalleeSymbol);
        if (Callee) {
          CSI.DestId = Callee->getFunctionNumber();;
          CSI.EntryDiscriminator = Callee->getEntryForSymbol(CalleeSymbol);
        }

        if (BC.MIA->getConditionalTailCall(Instr)) {
          auto CTCCount =
            BC.MIA->tryGetAnnotationAs<uint64_t>(Instr, "CTCTakenCount");
          if (CTCCount) {
            CSI.Count = *CTCCount;
            auto CTCMispreds =
              BC.MIA->tryGetAnnotationAs<uint64_t>(Instr, "CTCMispredCount");
            if (CTCMispreds)
              CSI.Mispreds = *CTCMispreds;
          }
        } else {
          auto Count = BC.MIA->tryGetAnnotationAs<uint64_t>(Instr, "Count");
          if (Count)
            CSI.Count = *Count;
        }

        if (CSI.Count)
          YamlBB.CallSites.emplace_back(CSI);
      }
    }

    // Skip printing if there's no profile data for non-entry basic block.
    if (YamlBB.CallSites.empty() && !BB->isEntryPoint()) {
      uint64_t SuccessorExecCount = 0;
      for (auto &BranchInfo : BB->branch_info()) {
        SuccessorExecCount += BranchInfo.Count;
      }
      if (!SuccessorExecCount)
        continue;
    }

    auto BranchInfo = BB->branch_info_begin();
    for (const auto *Successor : BB->successors()) {
      yaml::bolt::SuccessorInfo YamlSI;
      YamlSI.Index = Successor->getLayoutIndex();
      YamlSI.Count = BranchInfo->Count;
      YamlSI.Mispreds = BranchInfo->MispredictedCount;

      YamlBB.Successors.emplace_back(YamlSI);

      ++BranchInfo;
    }

    YamlBF.Blocks.emplace_back(YamlBB);
  }
}
} // end anonymous namespace

void ProfileWriter::printBinaryFunctionProfile(const BinaryFunction &BF) {
  yaml::bolt::BinaryFunctionProfile YamlBF;
  convert(BF, YamlBF);

  yaml::Output Out(*OS);
  Out << YamlBF;
}

void ProfileWriter::printBinaryFunctionsProfile(
    std::map<uint64_t, BinaryFunction> &BFs) {
  std::vector<yaml::bolt::BinaryFunctionProfile> YamlBFs;
  for (auto &BFI : BFs) {
    const auto &BF = BFI.second;
    if (BF.hasProfile()) {
      yaml::bolt::BinaryFunctionProfile YamlBF;
      convert(BF, YamlBF);
      YamlBFs.emplace_back(YamlBF);
    }
  }

  yaml::Output Out(*OS);
  Out << YamlBFs;
}

} // namespace bolt
} // namespace llvm
