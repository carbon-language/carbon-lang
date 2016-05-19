//===-- ProfileSummary.h - Profile summary data structure. ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the profile summary data structure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROFILE_SUMMARY_H
#define LLVM_SUPPORT_PROFILE_SUMMARY_H

#include <cstdint>
#include <vector>

#include "llvm/Support/Casting.h"

namespace llvm {

class LLVMContext;
class Metadata;
class MDTuple;
class MDNode;

// The profile summary is one or more (Cutoff, MinCount, NumCounts) triplets.
// The semantics of counts depend on the type of profile. For instrumentation
// profile, counts are block counts and for sample profile, counts are
// per-line samples. Given a target counts percentile, we compute the minimum
// number of counts needed to reach this target and the minimum among these
// counts.
struct ProfileSummaryEntry {
  uint32_t Cutoff;    ///< The required percentile of counts.
  uint64_t MinCount;  ///< The minimum count for this percentile.
  uint64_t NumCounts; ///< Number of counts >= the minimum count.
  ProfileSummaryEntry(uint32_t TheCutoff, uint64_t TheMinCount,
                      uint64_t TheNumCounts)
      : Cutoff(TheCutoff), MinCount(TheMinCount), NumCounts(TheNumCounts) {}
};

typedef std::vector<ProfileSummaryEntry> SummaryEntryVector;

class ProfileSummary {
public:
  enum Kind { PSK_Instr, PSK_Sample };

private:
  const Kind PSK;
  static const char *KindStr[2];

protected:
  SummaryEntryVector DetailedSummary;
  uint64_t TotalCount, MaxCount, MaxFunctionCount;
  uint32_t NumCounts, NumFunctions;
  ProfileSummary(Kind K, SummaryEntryVector DetailedSummary,
                 uint64_t TotalCount, uint64_t MaxCount,
                 uint64_t MaxFunctionCount, uint32_t NumCounts,
                 uint32_t NumFunctions)
      : PSK(K), DetailedSummary(DetailedSummary), TotalCount(TotalCount),
        MaxCount(MaxCount), MaxFunctionCount(MaxFunctionCount),
        NumCounts(NumCounts), NumFunctions(NumFunctions) {}
  ~ProfileSummary() = default;
  /// \brief Return metadata specific to the profile format.
  /// Derived classes implement this method to return a vector of Metadata.
  virtual std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) = 0;
  /// \brief Return detailed summary as metadata.
  Metadata *getDetailedSummaryMD(LLVMContext &Context);

public:
  static const int Scale = 1000000;
  Kind getKind() const { return PSK; }
  const char *getKindStr() const { return KindStr[PSK]; }
  /// \brief Return summary information as metadata.
  Metadata *getMD(LLVMContext &Context);
  /// \brief Construct profile summary from metdata.
  static ProfileSummary *getFromMD(Metadata *MD);
  SummaryEntryVector &getDetailedSummary() { return DetailedSummary; }
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxFunctionCount() { return MaxFunctionCount; }
};

class InstrProfSummary final : public ProfileSummary {
  uint64_t MaxInternalBlockCount;

protected:
  std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) override;

public:
  InstrProfSummary(uint64_t TotalCount, uint64_t MaxBlockCount,
                   uint64_t MaxInternalBlockCount, uint64_t MaxFunctionCount,
                   uint32_t NumBlocks, uint32_t NumFunctions,
                   SummaryEntryVector Summary)
      : ProfileSummary(PSK_Instr, Summary, TotalCount, MaxBlockCount,
                       MaxFunctionCount, NumBlocks, NumFunctions),
        MaxInternalBlockCount(MaxInternalBlockCount) {}
  static bool classof(const ProfileSummary *PS) {
    return PS->getKind() == PSK_Instr;
  }
  uint32_t getNumBlocks() { return NumCounts; }
  uint64_t getTotalCount() { return TotalCount; }
  uint64_t getMaxBlockCount() { return MaxCount; }
  uint64_t getMaxInternalBlockCount() { return MaxInternalBlockCount; }
};

class SampleProfileSummary final : public ProfileSummary {
protected:
  std::vector<Metadata *> getFormatSpecificMD(LLVMContext &Context) override;

public:
  uint32_t getNumLinesWithSamples() { return NumCounts; }
  uint64_t getTotalSamples() { return TotalCount; }
  uint64_t getMaxSamplesPerLine() { return MaxCount; }
  SampleProfileSummary(uint64_t TotalSamples, uint64_t MaxSamplesPerLine,
                       uint64_t MaxFunctionCount, int32_t NumLinesWithSamples,
                       uint32_t NumFunctions,
                       SummaryEntryVector DetailedSummary)
      : ProfileSummary(PSK_Sample, DetailedSummary, TotalSamples,
                       MaxSamplesPerLine, MaxFunctionCount, NumLinesWithSamples,
                       NumFunctions) {}
  static bool classof(const ProfileSummary *PS) {
    return PS->getKind() == PSK_Sample;
  }
};

} // end namespace llvm
#endif
