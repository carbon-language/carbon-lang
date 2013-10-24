//===------ Passes/RoundTripNativePass.h - Handles Layout of atoms
//------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PASSES_NATIVE_PASS_H
#define LLD_PASSES_NATIVE_PASS_H

#include "lld/Core/File.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Pass.h"

#include <map>
#include <vector>

namespace lld {
class RoundTripNativePass : public Pass {
public:
  RoundTripNativePass(LinkingContext &context) : Pass(), _context(context) {}

  /// Sorts atoms in mergedFile by content type then by command line order.
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

#endif // LLD_PASSES_NATIVE_PASS_H
