//===- Core/YamlReader.h - Reads YAML -------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_YAML_READER_H_
#define LLD_CORE_YAML_READER_H_

#include "lld/Core/File.h"

#include "llvm/Support/system_error.h"

#include <vector>

namespace llvm { class MemoryBuffer; }

namespace lld {
namespace yaml {

llvm::error_code parseObjectText(  llvm::MemoryBuffer *mb
                                 , std::vector<File *>&);

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_READER_H_
