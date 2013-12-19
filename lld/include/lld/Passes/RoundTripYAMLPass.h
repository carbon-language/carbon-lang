//===--Passes/RoundTripYAMLPass.h- Write YAML file/Read it back-----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PASSES_ROUND_TRIP_YAML_PASS_H
#define LLD_PASSES_ROUND_TRIP_YAML_PASS_H

#include "lld/Core/File.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Pass.h"

#include <vector>

namespace lld {
class RoundTripYAMLPass : public Pass {
public:
  RoundTripYAMLPass(LinkingContext &context) : Pass(), _context(context) {}

  /// Writes to a YAML file and reads the atoms from the YAML file back.
  /// Replaces the mergedFile with new contents.
  virtual void perform(std::unique_ptr<MutableFile> &mergedFile);

  virtual ~RoundTripYAMLPass() {}

private:
  LinkingContext &_context;
  // Keep the parsed file alive for the rest of the link. All atoms
  // that are created by the RoundTripYAMLPass are owned by the
  // yamlFile.
  std::vector<std::unique_ptr<File> > _yamlFile;
};

} // namespace lld

#endif // LLD_PASSES_ROUND_TRIP_YAML_PASS_H
