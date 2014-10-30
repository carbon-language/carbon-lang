//=-- SampleProf.h - Sampling profiling format support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in the reading and writing of
// sample profile data.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_PROFILEDATA_SAMPLEPROF_H_
#define LLVM_PROFILEDATA_SAMPLEPROF_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

namespace llvm {

const std::error_category &sampleprof_category();

enum class sampleprof_error {
  success = 0,
  bad_magic,
  unsupported_version,
  too_large,
  truncated,
  malformed
};

inline std::error_code make_error_code(sampleprof_error E) {
  return std::error_code(static_cast<int>(E), sampleprof_category());
}

} // end namespace llvm

namespace std {
template <>
struct is_error_code_enum<llvm::sampleprof_error> : std::true_type {};
}

namespace llvm {

namespace sampleprof {

static inline uint64_t SPMagic() {
  return uint64_t('S') << (64 - 8) | uint64_t('P') << (64 - 16) |
         uint64_t('R') << (64 - 24) | uint64_t('O') << (64 - 32) |
         uint64_t('F') << (64 - 40) | uint64_t('4') << (64 - 48) |
         uint64_t('2') << (64 - 56) | uint64_t(0xff);
}

static inline uint64_t SPVersion() { return 100; }

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

namespace sampleprof {

/// \brief Representation of a single sample record.
///
/// A sample record is represented by a positive integer value, which
/// indicates how frequently was the associated line location executed.
///
/// Additionally, if the associated location contains a function call,
/// the record will hold a list of all the possible called targets. For
/// direct calls, this will be the exact function being invoked. For
/// indirect calls (function pointers, virtual table dispatch), this
/// will be a list of one or more functions.
class SampleRecord {
public:
  typedef SmallVector<std::pair<std::string, unsigned>, 8> CallTargetList;

  SampleRecord() : NumSamples(0), CallTargets() {}

  /// \brief Increment the number of samples for this record by \p S.
  void addSamples(unsigned S) { NumSamples += S; }

  /// \brief Add called function \p F with samples \p S.
  void addCalledTarget(std::string F, unsigned S) {
    CallTargets.push_back(std::make_pair(F, S));
  }

  /// \brief Return true if this sample record contains function calls.
  bool hasCalls() const { return CallTargets.size() > 0; }

  unsigned getSamples() const { return NumSamples; }
  const CallTargetList &getCallTargets() const { return CallTargets; }

private:
  unsigned NumSamples;
  CallTargetList CallTargets;
};

typedef DenseMap<LineLocation, SampleRecord> BodySampleMap;

/// \brief Representation of the samples collected for a function.
///
/// This data structure contains all the collected samples for the body
/// of a function. Each sample corresponds to a LineLocation instance
/// within the body of the function.
class FunctionSamples {
public:
  FunctionSamples() : TotalSamples(0), TotalHeadSamples(0) {}
  void print(raw_ostream &OS);
  void addTotalSamples(unsigned Num) { TotalSamples += Num; }
  void addHeadSamples(unsigned Num) { TotalHeadSamples += Num; }
  void addBodySamples(int LineOffset, unsigned Discriminator, unsigned Num) {
    assert(LineOffset >= 0);
    // When dealing with instruction weights, we use the value
    // zero to indicate the absence of a sample. If we read an
    // actual zero from the profile file, use the value 1 to
    // avoid the confusion later on.
    if (Num == 0)
      Num = 1;
    BodySamples[LineLocation(LineOffset, Discriminator)].addSamples(Num);
  }
  void addCalledTargetSamples(int LineOffset, unsigned Discriminator,
                              std::string FName, unsigned Num) {
    assert(LineOffset >= 0);
    BodySamples[LineLocation(LineOffset, Discriminator)].addCalledTarget(FName,
                                                                         Num);
  }

  /// \brief Return the number of samples collected at the given location.
  /// Each location is specified by \p LineOffset and \p Discriminator.
  unsigned samplesAt(int LineOffset, unsigned Discriminator) {
    return BodySamples[LineLocation(LineOffset, Discriminator)].getSamples();
  }

  bool empty() const { return BodySamples.empty(); }

  /// \brief Return the total number of samples collected inside the function.
  unsigned getTotalSamples() const { return TotalSamples; }

  /// \brief Return the total number of samples collected at the head of the
  /// function.
  unsigned getHeadSamples() const { return TotalHeadSamples; }

  /// \brief Return all the samples collected in the body of the function.
  const BodySampleMap &getBodySamples() const { return BodySamples; }

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

} // End namespace sampleprof

} // End namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROF_H_
