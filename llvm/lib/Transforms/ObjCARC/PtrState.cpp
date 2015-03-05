//===--- PtrState.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PtrState.h"

using namespace llvm;
using namespace llvm::objcarc;

raw_ostream &operator<<(raw_ostream &OS, const Sequence S) {
  switch (S) {
  case S_None:
    return OS << "S_None";
  case S_Retain:
    return OS << "S_Retain";
  case S_CanRelease:
    return OS << "S_CanRelease";
  case S_Use:
    return OS << "S_Use";
  case S_Release:
    return OS << "S_Release";
  case S_MovableRelease:
    return OS << "S_MovableRelease";
  case S_Stop:
    return OS << "S_Stop";
  }
  llvm_unreachable("Unknown sequence type.");
}

static Sequence MergeSeqs(Sequence A, Sequence B, bool TopDown) {
  // The easy cases.
  if (A == B)
    return A;
  if (A == S_None || B == S_None)
    return S_None;

  if (A > B)
    std::swap(A, B);
  if (TopDown) {
    // Choose the side which is further along in the sequence.
    if ((A == S_Retain || A == S_CanRelease) &&
        (B == S_CanRelease || B == S_Use))
      return B;
  } else {
    // Choose the side which is further along in the sequence.
    if ((A == S_Use || A == S_CanRelease) &&
        (B == S_Use || B == S_Release || B == S_Stop || B == S_MovableRelease))
      return A;
    // If both sides are releases, choose the more conservative one.
    if (A == S_Stop && (B == S_Release || B == S_MovableRelease))
      return A;
    if (A == S_Release && B == S_MovableRelease)
      return A;
  }

  return S_None;
}

void RRInfo::clear() {
  KnownSafe = false;
  IsTailCallRelease = false;
  ReleaseMetadata = nullptr;
  Calls.clear();
  ReverseInsertPts.clear();
  CFGHazardAfflicted = false;
}

bool RRInfo::Merge(const RRInfo &Other) {
  // Conservatively merge the ReleaseMetadata information.
  if (ReleaseMetadata != Other.ReleaseMetadata)
    ReleaseMetadata = nullptr;

  // Conservatively merge the boolean state.
  KnownSafe &= Other.KnownSafe;
  IsTailCallRelease &= Other.IsTailCallRelease;
  CFGHazardAfflicted |= Other.CFGHazardAfflicted;

  // Merge the call sets.
  Calls.insert(Other.Calls.begin(), Other.Calls.end());

  // Merge the insert point sets. If there are any differences,
  // that makes this a partial merge.
  bool Partial = ReverseInsertPts.size() != Other.ReverseInsertPts.size();
  for (Instruction *Inst : Other.ReverseInsertPts)
    Partial |= ReverseInsertPts.insert(Inst).second;
  return Partial;
}

void PtrState::Merge(const PtrState &Other, bool TopDown) {
  Seq = MergeSeqs(GetSeq(), Other.GetSeq(), TopDown);
  KnownPositiveRefCount &= Other.KnownPositiveRefCount;

  // If we're not in a sequence (anymore), drop all associated state.
  if (Seq == S_None) {
    Partial = false;
    RRI.clear();
  } else if (Partial || Other.Partial) {
    // If we're doing a merge on a path that's previously seen a partial
    // merge, conservatively drop the sequence, to avoid doing partial
    // RR elimination. If the branch predicates for the two merge differ,
    // mixing them is unsafe.
    ClearSequenceProgress();
  } else {
    // Otherwise merge the other PtrState's RRInfo into our RRInfo. At this
    // point, we know that currently we are not partial. Stash whether or not
    // the merge operation caused us to undergo a partial merging of reverse
    // insertion points.
    Partial = RRI.Merge(Other.RRI);
  }
}
