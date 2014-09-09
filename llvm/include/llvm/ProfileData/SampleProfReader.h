//===- SampleProfReader.h - Read LLVM sample profile data -----------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions needed for reading sample profiles.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_PROFILEDATA_SAMPLEPROFREADER_H
#define LLVM_PROFILEDATA_SAMPLEPROFREADER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace sampleprof {

/// \brief Represents the relative location of an instruction.
///
/// Instruction locations are specified by the line offset from the
/// beginning of the function (marked by the line where the function
/// header is) and the discriminator value within that line.
///
/// The discriminator value is useful to distinguish instructions
/// that are on the same line but belong to different basic blocks
/// (e.g., the two post-increment instructions in "if (p) x++; else y++;").
struct LineLocation {
  LineLocation(int L, unsigned D) : LineOffset(L), Discriminator(D) {}
  int LineOffset;
  unsigned Discriminator;
};
} // End namespace sampleprof

namespace llvm {
template <> struct DenseMapInfo<sampleprof::LineLocation> {
  typedef DenseMapInfo<int> OffsetInfo;
  typedef DenseMapInfo<unsigned> DiscriminatorInfo;
  static inline sampleprof::LineLocation getEmptyKey() {
    return sampleprof::LineLocation(OffsetInfo::getEmptyKey(),
                                    DiscriminatorInfo::getEmptyKey());
  }
  static inline sampleprof::LineLocation getTombstoneKey() {
    return sampleprof::LineLocation(OffsetInfo::getTombstoneKey(),
                                    DiscriminatorInfo::getTombstoneKey());
  }
  static inline unsigned getHashValue(sampleprof::LineLocation Val) {
    return DenseMapInfo<std::pair<int, unsigned>>::getHashValue(
        std::pair<int, unsigned>(Val.LineOffset, Val.Discriminator));
  }
  static inline bool isEqual(sampleprof::LineLocation LHS,
                             sampleprof::LineLocation RHS) {
    return LHS.LineOffset == RHS.LineOffset &&
           LHS.Discriminator == RHS.Discriminator;
  }
};
}

namespace sampleprof {

typedef DenseMap<LineLocation, unsigned> BodySampleMap;

/// \brief Representation of the samples collected for a function.
///
/// This data structure contains all the collected samples for the body
/// of a function. Each sample corresponds to a LineLocation instance
/// within the body of the function.
class FunctionSamples {
public:
  FunctionSamples()
      : TotalSamples(0), TotalHeadSamples(0) {}
  void print(raw_ostream & OS);
  void addTotalSamples(unsigned Num) { TotalSamples += Num; }
  void addHeadSamples(unsigned Num) { TotalHeadSamples += Num; }
  void addBodySamples(int LineOffset, unsigned Discriminator, unsigned Num) {
    assert(LineOffset >= 0);
    BodySamples[LineLocation(LineOffset, Discriminator)] += Num;
  }

  /// \brief Return the number of samples collected at the given location.
  /// Each location is specified by \p LineOffset and \p Discriminator.
  unsigned samplesAt(int LineOffset, unsigned Discriminator) {
    return BodySamples.lookup(LineLocation(LineOffset, Discriminator));
  }

  bool empty() { return BodySamples.empty(); }

private:
  /// \brief Total number of samples collected inside this function.
  ///
  /// Samples are cumulative, they include all the samples collected
  /// inside this function and all its inlined callees.
  unsigned TotalSamples;

  /// \brief Total number of samples collected at the head of the function.
  unsigned TotalHeadSamples;

  /// \brief Map instruction locations to collected samples.
  ///
  /// Each entry in this map contains the number of samples
  /// collected at the corresponding line offset. All line locations
  /// are an offset from the start of the function.
  BodySampleMap BodySamples;
};

/// \brief Sample-based profile reader.
///
/// Each profile contains sample counts for all the functions
/// executed. Inside each function, statements are annotated with the
/// collected samples on all the instructions associated with that
/// statement.
///
/// For this to produce meaningful data, the program needs to be
/// compiled with some debug information (at minimum, line numbers:
/// -gline-tables-only). Otherwise, it will be impossible to match IR
/// instructions to the line numbers collected by the profiler.
///
/// From the profile file, we are interested in collecting the
/// following information:
///
/// * A list of functions included in the profile (mangled names).
///
/// * For each function F:
///   1. The total number of samples collected in F.
///
///   2. The samples collected at each line in F. To provide some
///      protection against source code shuffling, line numbers should
///      be relative to the start of the function.
///
/// The reader supports two file formats: text and bitcode. The text format
/// is useful for debugging and testing, while the bitcode format is more
/// compact. They can both be used interchangeably.
class SampleProfileReader {
public:
  SampleProfileReader(const Module &M, StringRef F)
      : Profiles(0), Filename(F), M(M) {}

  /// \brief Print all the profiles to dbgs().
  void dump();

  /// \brief Load sample profiles from the associated file.
  bool load();

  /// \brief Print the profile for \p FName on stream \p OS.
  void printFunctionProfile(raw_ostream &OS, StringRef FName);

  /// \brief Print the profile for \p FName on dbgs().
  void dumpFunctionProfile(StringRef FName);

  /// \brief Return the samples collected for function \p F.
  FunctionSamples *getSamplesFor(const Function &F) {
    return &Profiles[F.getName()];
  }

  /// \brief Report a parse error message.
  void reportParseError(int64_t LineNumber, Twine Msg) const {
    DiagnosticInfoSampleProfile Diag(Filename.data(), LineNumber, Msg);
    M.getContext().diagnose(Diag);
  }

protected:
  bool loadText();
  bool loadBitcode() { llvm_unreachable("not implemented"); }

  /// \brief Map every function to its associated profile.
  ///
  /// The profile of every function executed at runtime is collected
  /// in the structure FunctionSamples. This maps function objects
  /// to their corresponding profiles.
  StringMap<FunctionSamples> Profiles;

  /// \brief Path name to the file holding the profile data.
  StringRef Filename;

  /// \brief Module being compiled. Used to access the current
  /// LLVM context for diagnostics.
  const Module &M;
};

} // End namespace sampleprof

#endif // LLVM_PROFILEDATA_SAMPLEPROFREADER_H
