//===- SampleProfWriter.cpp - Write LLVM sample profile data --------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that writes LLVM sample profiles. It
// supports two file formats: text and binary. The textual representation
// is useful for debugging and testing purposes. The binary representation
// is more compact, resulting in smaller file sizes. However, they can
// both be used interchangeably.
//
// See lib/ProfileData/SampleProfReader.cpp for documentation on each of the
// supported formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Regex.h"

using namespace llvm::sampleprof;
using namespace llvm;

/// \brief Write samples to a text file.
bool SampleProfileWriterText::write(const Function &F,
                                    const FunctionSamples &S) {
  if (S.empty())
    return true;

  OS << F.getName() << ":" << S.getTotalSamples() << ":" << S.getHeadSamples()
     << "\n";

  for (BodySampleMap::const_iterator I = S.getBodySamples().begin(),
                                     E = S.getBodySamples().end();
       I != E; ++I) {
    LineLocation Loc = I->first;
    SampleRecord Sample = I->second;
    if (Loc.Discriminator == 0)
      OS << Loc.LineOffset << ": ";
    else
      OS << Loc.LineOffset << "." << Loc.Discriminator << ": ";

    OS << Sample.getSamples();

    for (SampleRecord::CallTargetList::const_iterator
             I = Sample.getCallTargets().begin(),
             E = Sample.getCallTargets().end();
         I != E; ++I)
      OS << " " << (*I).first << ":" << (*I).second;
    OS << "\n";
  }

  return true;
}

SampleProfileWriterBinary::SampleProfileWriterBinary(StringRef F,
                                                     std::error_code &EC)
    : SampleProfileWriter(F, EC, sys::fs::F_None) {
  if (EC)
    return;

  // Write the file header.
  encodeULEB128(SPMagic(), OS);
  encodeULEB128(SPVersion(), OS);
}

/// \brief Write samples to a binary file.
///
/// \returns true if the samples were written successfully, false otherwise.
bool SampleProfileWriterBinary::write(const Function &F,
                                      const FunctionSamples &S) {
  if (S.empty())
    return true;

  OS << F.getName();
  encodeULEB128(0, OS);
  encodeULEB128(S.getTotalSamples(), OS);
  encodeULEB128(S.getHeadSamples(), OS);
  encodeULEB128(S.getBodySamples().size(), OS);
  for (BodySampleMap::const_iterator I = S.getBodySamples().begin(),
                                     E = S.getBodySamples().end();
       I != E; ++I) {
    LineLocation Loc = I->first;
    SampleRecord Sample = I->second;
    encodeULEB128(Loc.LineOffset, OS);
    encodeULEB128(Loc.Discriminator, OS);
    encodeULEB128(Sample.getSamples(), OS);
    encodeULEB128(Sample.getCallTargets().size(), OS);
    for (SampleRecord::CallTargetList::const_iterator
             I = Sample.getCallTargets().begin(),
             E = Sample.getCallTargets().end();
         I != E; ++I) {
      std::string Callee = (*I).first;
      unsigned CalleeSamples = (*I).second;
      OS << Callee;
      encodeULEB128(0, OS);
      encodeULEB128(CalleeSamples, OS);
    }
  }

  return true;
}
