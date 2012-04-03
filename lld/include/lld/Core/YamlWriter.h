//===- Core/YamlWriter.h - Writes YAML formatted object files -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_YAML_WRITER_H_
#define LLD_CORE_YAML_WRITER_H_

#include "lld/Core/LLVM.h"

#include "llvm/Support/raw_ostream.h"

namespace lld {

class Platform;
class File;

namespace yaml {

  /// writeObjectText - writes the lld::File object as in YAML
  /// format to the specified stream.
  void writeObjectText(const lld::File &, Platform &, raw_ostream &);

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_WRITER_H_
