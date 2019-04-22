//===-- ProfileReader.cpp - BOLT profile de-serializer ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "Passes/MCF.h"
#include "ProfileReader.h"
#include "ProfileYAMLMapping.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace opts {

extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltOptCategory;

static llvm::cl::opt<bool>
IgnoreHash("profile-ignore-hash",
  cl::desc("ignore hash while reading function profile"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

void
ProfileReader::buildNameMaps(std::map<uint64_t, BinaryFunction> &Functions) {
  for (auto &YamlBF : YamlBP.Functions) {
    StringRef Name = YamlBF.Name;
    const auto Pos = Name.find("(*");
    if (Pos != StringRef::npos)
      Name = Name.substr(0, Pos);
    ProfileNameToProfile[Name] = &YamlBF;
    if (const auto CommonName = getLTOCommonName(Name)) {
      LTOCommonNameMap[*CommonName].push_back(&YamlBF);
    }
  }
  for (auto &BFI : Functions) {
    const auto &Function = BFI.second;
    for (auto &Name : Function.getNames()) {
      if (const auto CommonName = getLTOCommonName(Name)) {
        LTOCommonNameFunctionMap[*CommonName].insert(&Function);
      }
    }
  }
}

bool
ProfileReader::parseFunctionProfile(BinaryFunction &BF,
    const yaml::bolt::BinaryFunctionProfile &YamlBF) {
  auto &BC = BF.getBinaryContext();

  bool ProfileMatched = true;
  uint64_t MismatchedBlocks = 0;
  uint64_t MismatchedCalls = 0;
  uint64_t MismatchedEdges = 0;

  uint64_t FunctionExecutionCount = 0;

  BF.setExecutionCount(YamlBF.ExecCount);

  if (!opts::IgnoreHash && YamlBF.Hash != BF.hash(true, true)) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: function hash mismatch\n";
    ProfileMatched = false;
  }

  if (YamlBF.NumBasicBlocks != BF.size()) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: number of basic blocks mismatch\n";
    ProfileMatched = false;
  }

  auto DFSOrder = BF.dfs();

  for (const auto &YamlBB : YamlBF.Blocks) {
    if (YamlBB.Index >= DFSOrder.size()) {
      if (opts::Verbosity >= 2)
        errs() << "BOLT-WARNING: index " << YamlBB.Index
               << " is out of bounds\n";
      ++MismatchedBlocks;
      continue;
    }

    auto &BB = *DFSOrder[YamlBB.Index];

    // Basic samples profile (without LBR) does not have branches information
    // and needs a special processing.
    if (YamlBP.Header.Flags & BinaryFunction::PF_SAMPLE) {
      if (!YamlBB.EventCount) {
        BB.setExecutionCount(0);
        continue;
      }
      auto NumSamples = YamlBB.EventCount * 1000;
      if (NormalizeByInsnCount && BB.getNumNonPseudos()) {
        NumSamples /= BB.getNumNonPseudos();
      } else if (NormalizeByCalls) {
        NumSamples /= BB.getNumCalls() + 1;
      }
      BB.setExecutionCount(NumSamples);
      if (BB.isEntryPoint())
        FunctionExecutionCount += NumSamples;
      continue;
    }

    BB.setExecutionCount(YamlBB.ExecCount);

    for (const auto &YamlCSI: YamlBB.CallSites) {
      auto *Callee = YamlCSI.DestId < YamlProfileToFunction.size() ?
          YamlProfileToFunction[YamlCSI.DestId] : nullptr;
      bool IsFunction = Callee ? true : false;
      const MCSymbol *CalleeSymbol = nullptr;
      if (IsFunction) {
        CalleeSymbol = Callee->getSymbolForEntry(YamlCSI.EntryDiscriminator);
      }
      StringRef Name = CalleeSymbol ? CalleeSymbol->getName() : "<unknown>";
      BF.getAllCallSites().emplace_back(
          IsFunction, Name, YamlCSI.Count, YamlCSI.Mispreds, YamlCSI.Offset);

      if (YamlCSI.Offset >= BB.getOriginalSize()) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: offset " << YamlCSI.Offset
                 << " out of bounds in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }

      auto *Instr =
        BF.getInstructionAtOffset(BB.getInputOffset() + YamlCSI.Offset);
      if (!Instr) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: no instruction at offset " << YamlCSI.Offset
                 << " in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }
      if (!BC.MIB->isCall(*Instr) && !BC.MIB->isIndirectBranch(*Instr)) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: expected call at offset " << YamlCSI.Offset
                 << " in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }

      auto setAnnotation = [&](StringRef Name, uint64_t Count) {
        if (BC.MIB->hasAnnotation(*Instr, Name)) {
          if (opts::Verbosity >= 1)
            errs() << "BOLT-WARNING: ignoring duplicate " << Name
                   << " info for offset 0x" << Twine::utohexstr(YamlCSI.Offset)
                   << " in function " << BF << '\n';
          return;
        }
        BC.MIB->addAnnotation(*Instr, Name, Count);
      };

      if (BC.MIB->isIndirectCall(*Instr) || BC.MIB->isIndirectBranch(*Instr)) {
        IndirectCallSiteProfile &CSP =
          BC.MIB->getOrCreateAnnotationAs<IndirectCallSiteProfile>(
              *Instr, "CallProfile");
        CSP.emplace_back(IsFunction, Name, YamlCSI.Count, YamlCSI.Mispreds);
      } else if (BC.MIB->getConditionalTailCall(*Instr)) {
        setAnnotation("CTCTakenCount", YamlCSI.Count);
        setAnnotation("CTCMispredCount", YamlCSI.Mispreds);
      } else {
        setAnnotation("Count", YamlCSI.Count);
      }
    }

    for (const auto &YamlSI : YamlBB.Successors) {
      if (YamlSI.Index >= DFSOrder.size()) {
        if (opts::Verbosity >= 1)
          errs() << "BOLT-WARNING: index out of bounds for profiled block\n";
        ++MismatchedEdges;
        continue;
      }

      auto &SuccessorBB = *DFSOrder[YamlSI.Index];
      if (!BB.getSuccessor(SuccessorBB.getLabel())) {
        if (opts::Verbosity >= 1)
          errs() << "BOLT-WARNING: no successor for block " << BB.getName()
                 << " that matches index " << YamlSI.Index << " or block "
                 << SuccessorBB.getName() << '\n';
        ++MismatchedEdges;
        continue;
      }

      auto &BI = BB.getBranchInfo(SuccessorBB);
      BI.Count += YamlSI.Count;
      BI.MispredictedCount += YamlSI.Mispreds;
    }
  }

  // If basic block profile wasn't read it should be 0.
  for (auto &BB : BF) {
    if (BB.getExecutionCount() == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB.setExecutionCount(0);
  }

  if (YamlBP.Header.Flags & BinaryFunction::PF_SAMPLE) {
    BF.setExecutionCount(FunctionExecutionCount);
    estimateEdgeCounts(BF);
  }

  ProfileMatched &= !MismatchedBlocks && !MismatchedCalls && !MismatchedEdges;

  if (ProfileMatched)
    BF.markProfiled(YamlBP.Header.Flags);

  if (!ProfileMatched && opts::Verbosity >= 1) {
    errs() << "BOLT-WARNING: " << MismatchedBlocks << " blocks, "
           << MismatchedCalls << " calls, and " << MismatchedEdges
           << " edges in profile did not match function " << BF << '\n';
  }

  return ProfileMatched;
}

std::error_code
ProfileReader::readProfile(const std::string &FileName,
                           std::map<uint64_t, BinaryFunction> &Functions) {
  auto MB = MemoryBuffer::getFileOrSTDIN(FileName);
  if (std::error_code EC = MB.getError()) {
    errs() << "ERROR: cannot open " << FileName << ": " << EC.message() << "\n";
    return EC;
  }
  yaml::Input YamlInput(MB.get()->getBuffer());

  // Consume YAML file.
  YamlInput >> YamlBP;
  if (YamlInput.error()) {
    errs() << "BOLT-ERROR: syntax error parsing profile in " << FileName
           << " : " << YamlInput.error().message() << '\n';
    return YamlInput.error();
  }

  // Sanity check.
  if (YamlBP.Header.Version != 1) {
    errs() << "BOLT-ERROR: cannot read profile : unsupported version\n";
    return std::make_error_code(std::errc::executable_format_error);
  }
  if (YamlBP.Header.EventNames.find(',') != StringRef::npos) {
    errs() << "BOLT-ERROR: multiple events in profile are not supported\n";
    return std::make_error_code(std::errc::executable_format_error);
  }

  NormalizeByInsnCount = usesEvent("cycles") || usesEvent("instructions");
  NormalizeByCalls = usesEvent("branches");

  // Match profile to function based on a function name.
  buildNameMaps(Functions);

  YamlProfileToFunction.resize(YamlBP.Functions.size() + 1);

  auto profileMatches = [](const yaml::bolt::BinaryFunctionProfile &Profile,
                           BinaryFunction &BF) {
    if (opts::IgnoreHash && Profile.NumBasicBlocks == BF.size())
      return true;
    if (!opts::IgnoreHash &&
        Profile.Hash == static_cast<uint64_t>(BF.hash(/*Recompute = */false)))
      return true;
    return false;
  };

  // We have to do 2 passes since LTO introduces an ambiguity in function
  // names. The first pass assigns profiles that match 100% by name and
  // by hash. The second pass allows name ambiguity for LTO private functions.
  for (auto &BFI : Functions) {
    auto &Function = BFI.second;

    // Recompute hash once per function.
    if (!opts::IgnoreHash)
      Function.hash(/*Recompute = */true, true);

    for (auto &FunctionName : Function.getNames()) {
      auto PI = ProfileNameToProfile.find(FunctionName);
      if (PI == ProfileNameToProfile.end()) {
        continue;
      }
      auto &YamlBF = *PI->getValue();
      if (profileMatches(YamlBF, Function))
        matchProfileToFunction(YamlBF, Function);
    }
  }

  for (auto &BFI : Functions) {
    auto &Function = BFI.second;

    if (ProfiledFunctions.count(&Function))
      continue;

    for (auto &FunctionName : Function.getNames()) {
      const auto CommonName = getLTOCommonName(FunctionName);
      if (CommonName) {
        auto I = LTOCommonNameMap.find(*CommonName);
        if (I == LTOCommonNameMap.end())
          continue;

        bool ProfileMatched{false};
        auto &LTOProfiles = I->getValue();
        for (auto *YamlBF : LTOProfiles) {
          if (YamlBF->Used)
            continue;
          if ((ProfileMatched = profileMatches(*YamlBF, Function))) {
            matchProfileToFunction(*YamlBF, Function);
            break;
          }
        }
        if (ProfileMatched)
          break;

        // If there's only one function with a given name, try to
        // match it partially.
        if (LTOProfiles.size() == 1 &&
            LTOCommonNameFunctionMap[*CommonName].size() == 1 &&
            !LTOProfiles.front()->Used) {
          matchProfileToFunction(*LTOProfiles.front(), Function);
          break;
        }
      } else {
        auto PI = ProfileNameToProfile.find(FunctionName);
        if (PI == ProfileNameToProfile.end())
          continue;

        auto &YamlBF = *PI->getValue();
        if (!YamlBF.Used) {
          matchProfileToFunction(YamlBF, Function);
          break;
        }
      }
    }
  }
  for (auto &YamlBF : YamlBP.Functions) {
    if (!YamlBF.Used) {
      errs() << "BOLT-WARNING: profile ignored for function "
             << YamlBF.Name << '\n';
    }
  }

  for (auto &YamlBF : YamlBP.Functions) {
    if (YamlBF.Id >= YamlProfileToFunction.size()) {
      // Such profile was ignored.
      continue;
    }
    if (auto *BF = YamlProfileToFunction[YamlBF.Id]) {
      parseFunctionProfile(*BF, YamlBF);
    }
  }

  return YamlInput.error();
}

bool ProfileReader::usesEvent(StringRef Name) const {
  return YamlBP.Header.EventNames.find(Name) != StringRef::npos;
}

} // end namespace bolt
} // end namespace llvm
