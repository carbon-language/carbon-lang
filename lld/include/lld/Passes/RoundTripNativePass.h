//===--Passes/RoundTripNativePass.h - Write Native file/Read it back------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PASSES_ROUND_TRIP_NATIVE_PASS_H
#define LLD_PASSES_ROUND_TRIP_NATIVE_PASS_H

#include "lld/Core/File.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Pass.h"

#include <vector>

namespace lld {
class RoundTripNativePass : public Pass {
public:
  RoundTripNativePass(LinkingContext &context) : Pass(), _context(context) {}

  /// Writes to a native file and reads the atoms from the native file back.
  /// Replaces mergedFile with the contents of the native File.
  virtual void perform(std::unique_ptr<MutableFile> &mergedFile);

  virtual ~RoundTripNativePass() {}

private:
  LinkingContext &_context;
  // Keep the parsed file alive for the rest of the link. All atoms
  // that are created by the RoundTripNativePass are owned by the
  // nativeFile.
  std::vector<std::unique_ptr<File> > _nativeFile;
};

} // namespace lld

#endif // LLD_PASSES_ROUND_TRIP_NATIVE_PASS_H
