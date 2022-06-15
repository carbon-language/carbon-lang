//===- bolt/Profile/YAMLProfileReader.cpp - YAML profile de-serializer ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/YAMLProfileReader.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/MCF.h"
#include "bolt/Profile/ProfileYAMLMapping.h"
#include "bolt/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace opts {

extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltOptCategory;

static llvm::cl::opt<bool>
    IgnoreHash("profile-ignore-hash",
               cl::desc("ignore hash while reading function profile"),
               cl::Hidden, cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

bool YAMLProfileReader::isYAML(const StringRef Filename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = MB.getError())
    report_error(Filename, EC);
  StringRef Buffer = MB.get()->getBuffer();
  if (Buffer.startswith("---\n"))
    return true;
  return false;
}

void YAMLProfileReader::buildNameMaps(
    std::map<uint64_t, BinaryFunction> &Functions) {
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    StringRef Name = YamlBF.Name;
    const size_t Pos = Name.find("(*");
    if (Pos != StringRef::npos)
      Name = Name.substr(0, Pos);
    ProfileNameToProfile[Name] = &YamlBF;
    if (const Optional<StringRef> CommonName = getLTOCommonName(Name))
      LTOCommonNameMap[*CommonName].push_back(&YamlBF);
  }
  for (auto &BFI : Functions) {
    const BinaryFunction &Function = BFI.second;
    for (StringRef Name : Function.getNames())
      if (const Optional<StringRef> CommonName = getLTOCommonName(Name))
        LTOCommonNameFunctionMap[*CommonName].insert(&Function);
  }
}

bool YAMLProfileReader::hasLocalsWithFileName() const {
  for (const StringMapEntry<yaml::bolt::BinaryFunctionProfile *> &KV :
       ProfileNameToProfile) {
    const StringRef &FuncName = KV.getKey();
    if (FuncName.count('/') == 2 && FuncName[0] != '/')
      return true;
  }
  return false;
}

bool YAMLProfileReader::parseFunctionProfile(
    BinaryFunction &BF, const yaml::bolt::BinaryFunctionProfile &YamlBF) {
  BinaryContext &BC = BF.getBinaryContext();

  bool ProfileMatched = true;
  uint64_t MismatchedBlocks = 0;
  uint64_t MismatchedCalls = 0;
  uint64_t MismatchedEdges = 0;

  uint64_t FunctionExecutionCount = 0;

  BF.setExecutionCount(YamlBF.ExecCount);

  if (!opts::IgnoreHash && YamlBF.Hash != BF.computeHash(/*UseDFS=*/true)) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: function hash mismatch\n";
    ProfileMatched = false;
  }

  if (YamlBF.NumBasicBlocks != BF.size()) {
    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: number of basic blocks mismatch\n";
    ProfileMatched = false;
  }

  BinaryFunction::BasicBlockOrderType DFSOrder = BF.dfs();

  for (const yaml::bolt::BinaryBasicBlockProfile &YamlBB : YamlBF.Blocks) {
    if (YamlBB.Index >= DFSOrder.size()) {
      if (opts::Verbosity >= 2)
        errs() << "BOLT-WARNING: index " << YamlBB.Index
               << " is out of bounds\n";
      ++MismatchedBlocks;
      continue;
    }

    BinaryBasicBlock &BB = *DFSOrder[YamlBB.Index];

    // Basic samples profile (without LBR) does not have branches information
    // and needs a special processing.
    if (YamlBP.Header.Flags & BinaryFunction::PF_SAMPLE) {
      if (!YamlBB.EventCount) {
        BB.setExecutionCount(0);
        continue;
      }
      uint64_t NumSamples = YamlBB.EventCount * 1000;
      if (NormalizeByInsnCount && BB.getNumNonPseudos())
        NumSamples /= BB.getNumNonPseudos();
      else if (NormalizeByCalls)
        NumSamples /= BB.getNumCalls() + 1;

      BB.setExecutionCount(NumSamples);
      if (BB.isEntryPoint())
        FunctionExecutionCount += NumSamples;
      continue;
    }

    BB.setExecutionCount(YamlBB.ExecCount);

    for (const yaml::bolt::CallSiteInfo &YamlCSI : YamlBB.CallSites) {
      BinaryFunction *Callee = YamlCSI.DestId < YamlProfileToFunction.size()
                                   ? YamlProfileToFunction[YamlCSI.DestId]
                                   : nullptr;
      bool IsFunction = Callee ? true : false;
      MCSymbol *CalleeSymbol = nullptr;
      if (IsFunction)
        CalleeSymbol = Callee->getSymbolForEntryID(YamlCSI.EntryDiscriminator);

      BF.getAllCallSites().emplace_back(CalleeSymbol, YamlCSI.Count,
                                        YamlCSI.Mispreds, YamlCSI.Offset);

      if (YamlCSI.Offset >= BB.getOriginalSize()) {
        if (opts::Verbosity >= 2)
          errs() << "BOLT-WARNING: offset " << YamlCSI.Offset
                 << " out of bounds in block " << BB.getName() << '\n';
        ++MismatchedCalls;
        continue;
      }

      MCInst *Instr =
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
        auto &CSP = BC.MIB->getOrCreateAnnotationAs<IndirectCallSiteProfile>(
            *Instr, "CallProfile");
        CSP.emplace_back(CalleeSymbol, YamlCSI.Count, YamlCSI.Mispreds);
      } else if (BC.MIB->getConditionalTailCall(*Instr)) {
        setAnnotation("CTCTakenCount", YamlCSI.Count);
        setAnnotation("CTCMispredCount", YamlCSI.Mispreds);
      } else {
        setAnnotation("Count", YamlCSI.Count);
      }
    }

    for (const yaml::bolt::SuccessorInfo &YamlSI : YamlBB.Successors) {
      if (YamlSI.Index >= DFSOrder.size()) {
        if (opts::Verbosity >= 1)
          errs() << "BOLT-WARNING: index out of bounds for profiled block\n";
        ++MismatchedEdges;
        continue;
      }

      BinaryBasicBlock &SuccessorBB = *DFSOrder[YamlSI.Index];
      if (!BB.getSuccessor(SuccessorBB.getLabel())) {
        if (opts::Verbosity >= 1)
          errs() << "BOLT-WARNING: no successor for block " << BB.getName()
                 << " that matches index " << YamlSI.Index << " or block "
                 << SuccessorBB.getName() << '\n';
        ++MismatchedEdges;
        continue;
      }

      BinaryBasicBlock::BinaryBranchInfo &BI = BB.getBranchInfo(SuccessorBB);
      BI.Count += YamlSI.Count;
      BI.MispredictedCount += YamlSI.Mispreds;
    }
  }

  // If basic block profile wasn't read it should be 0.
  for (BinaryBasicBlock &BB : BF)
    if (BB.getExecutionCount() == BinaryBasicBlock::COUNT_NO_PROFILE)
      BB.setExecutionCount(0);

  if (YamlBP.Header.Flags & BinaryFunction::PF_SAMPLE) {
    BF.setExecutionCount(FunctionExecutionCount);
    estimateEdgeCounts(BF);
  }

  ProfileMatched &= !MismatchedBlocks && !MismatchedCalls && !MismatchedEdges;

  if (ProfileMatched)
    BF.markProfiled(YamlBP.Header.Flags);

  if (!ProfileMatched && opts::Verbosity >= 1)
    errs() << "BOLT-WARNING: " << MismatchedBlocks << " blocks, "
           << MismatchedCalls << " calls, and " << MismatchedEdges
           << " edges in profile did not match function " << BF << '\n';

  return ProfileMatched;
}

Error YAMLProfileReader::preprocessProfile(BinaryContext &BC) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = MB.getError()) {
    errs() << "ERROR: cannot open " << Filename << ": " << EC.message() << "\n";
    return errorCodeToError(EC);
  }
  yaml::Input YamlInput(MB.get()->getBuffer());

  // Consume YAML file.
  YamlInput >> YamlBP;
  if (YamlInput.error()) {
    errs() << "BOLT-ERROR: syntax error parsing profile in " << Filename
           << " : " << YamlInput.error().message() << '\n';
    return errorCodeToError(YamlInput.error());
  }

  // Sanity check.
  if (YamlBP.Header.Version != 1)
    return make_error<StringError>(
        Twine("cannot read profile : unsupported version"),
        inconvertibleErrorCode());

  if (YamlBP.Header.EventNames.find(',') != StringRef::npos)
    return make_error<StringError>(
        Twine("multiple events in profile are not supported"),
        inconvertibleErrorCode());

  // Match profile to function based on a function name.
  buildNameMaps(BC.getBinaryFunctions());

  // Preliminary assign function execution count.
  for (auto &KV : BC.getBinaryFunctions()) {
    BinaryFunction &BF = KV.second;
    for (StringRef Name : BF.getNames()) {
      auto PI = ProfileNameToProfile.find(Name);
      if (PI != ProfileNameToProfile.end()) {
        yaml::bolt::BinaryFunctionProfile &YamlBF = *PI->getValue();
        BF.setExecutionCount(YamlBF.ExecCount);
        break;
      }
    }
  }

  return Error::success();
}

bool YAMLProfileReader::mayHaveProfileData(const BinaryFunction &BF) {
  for (StringRef Name : BF.getNames()) {
    if (ProfileNameToProfile.find(Name) != ProfileNameToProfile.end())
      return true;
    if (const Optional<StringRef> CommonName = getLTOCommonName(Name)) {
      if (LTOCommonNameMap.find(*CommonName) != LTOCommonNameMap.end())
        return true;
    }
  }

  return false;
}

Error YAMLProfileReader::readProfile(BinaryContext &BC) {
  YamlProfileToFunction.resize(YamlBP.Functions.size() + 1);

  auto profileMatches = [](const yaml::bolt::BinaryFunctionProfile &Profile,
                           BinaryFunction &BF) {
    if (opts::IgnoreHash && Profile.NumBasicBlocks == BF.size())
      return true;
    if (!opts::IgnoreHash &&
        Profile.Hash == static_cast<uint64_t>(BF.getHash()))
      return true;
    return false;
  };

  // We have to do 2 passes since LTO introduces an ambiguity in function
  // names. The first pass assigns profiles that match 100% by name and
  // by hash. The second pass allows name ambiguity for LTO private functions.
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Clear function call count that may have been set while pre-processing
    // the profile.
    Function.setExecutionCount(BinaryFunction::COUNT_NO_PROFILE);

    // Recompute hash once per function.
    if (!opts::IgnoreHash)
      Function.computeHash(/*UseDFS=*/true);

    for (StringRef FunctionName : Function.getNames()) {
      auto PI = ProfileNameToProfile.find(FunctionName);
      if (PI == ProfileNameToProfile.end())
        continue;

      yaml::bolt::BinaryFunctionProfile &YamlBF = *PI->getValue();
      if (profileMatches(YamlBF, Function))
        matchProfileToFunction(YamlBF, Function);
    }
  }

  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    if (ProfiledFunctions.count(&Function))
      continue;

    for (StringRef FunctionName : Function.getNames()) {
      const Optional<StringRef> CommonName = getLTOCommonName(FunctionName);
      if (CommonName) {
        auto I = LTOCommonNameMap.find(*CommonName);
        if (I == LTOCommonNameMap.end())
          continue;

        bool ProfileMatched = false;
        std::vector<yaml::bolt::BinaryFunctionProfile *> &LTOProfiles =
            I->getValue();
        for (yaml::bolt::BinaryFunctionProfile *YamlBF : LTOProfiles) {
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

        yaml::bolt::BinaryFunctionProfile &YamlBF = *PI->getValue();
        if (!YamlBF.Used) {
          matchProfileToFunction(YamlBF, Function);
          break;
        }
      }
    }
  }

  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions)
    if (!YamlBF.Used && opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: profile ignored for function " << YamlBF.Name
             << '\n';

  // Set for parseFunctionProfile().
  NormalizeByInsnCount = usesEvent("cycles") || usesEvent("instructions");
  NormalizeByCalls = usesEvent("branches");

  uint64_t NumUnused = 0;
  for (yaml::bolt::BinaryFunctionProfile &YamlBF : YamlBP.Functions) {
    if (YamlBF.Id >= YamlProfileToFunction.size()) {
      // Such profile was ignored.
      ++NumUnused;
      continue;
    }
    if (BinaryFunction *BF = YamlProfileToFunction[YamlBF.Id])
      parseFunctionProfile(*BF, YamlBF);
    else
      ++NumUnused;
  }

  BC.setNumUnusedProfiledObjects(NumUnused);

  return Error::success();
}

bool YAMLProfileReader::usesEvent(StringRef Name) const {
  return YamlBP.Header.EventNames.find(std::string(Name)) != StringRef::npos;
}

} // end namespace bolt
} // end namespace llvm
