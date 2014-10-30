//===- SampleProfWriter.h - Write LLVM sample profile data ----------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions needed for writing sample profiles.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_PROFILEDATA_SAMPLEPROFWRITER_H
#define LLVM_PROFILEDATA_SAMPLEPROFWRITER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace sampleprof {

/// \brief Sample-based profile writer. Base class.
class SampleProfileWriter {
public:
  SampleProfileWriter(StringRef Filename, std::error_code &EC,
                      sys::fs::OpenFlags Flags)
      : OS(Filename, EC, Flags) {}
  virtual ~SampleProfileWriter() {}

  /// \brief Write sample profiles in \p S for function \p F.
  ///
  /// \returns true if the file was updated successfully. False, otherwise.
  virtual bool write(const Function &F, const FunctionSamples &S) = 0;

  /// \brief Write all the sample profiles for all the functions in \p M.
  ///
  /// \returns true if the file was updated successfully. False, otherwise.
  bool write(const Module &M, StringMap<FunctionSamples> &P) {
    for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (!write((*I), P[I->getName()]))
        return false;
    return true;
  }

protected:
  /// \brief Output stream where to emit the profile to.
  raw_fd_ostream OS;
};

/// \brief Sample-based profile writer (text format).
class SampleProfileWriterText : public SampleProfileWriter {
public:
  SampleProfileWriterText(StringRef F, std::error_code &EC)
      : SampleProfileWriter(F, EC, sys::fs::F_Text) {}

  bool write(const Function &F, const FunctionSamples &S) override;
  bool write(const Module &M, StringMap<FunctionSamples> &P) {
    return SampleProfileWriter::write(M, P);
  }
};

/// \brief Sample-based profile writer (binary format).
class SampleProfileWriterBinary : public SampleProfileWriter {
public:
  SampleProfileWriterBinary(StringRef F, std::error_code &EC);

  bool write(const Function &F, const FunctionSamples &S) override;
  bool write(const Module &M, StringMap<FunctionSamples> &P) {
    return SampleProfileWriter::write(M, P);
  }
};

} // End namespace sampleprof

} // End namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROFWRITER_H
