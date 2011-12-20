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

  /// parseObjectTextFileOrSTDIN - Open the specified YAML file (use stdin if 
  /// the path is "-") and parse into lld::File object(s) and append each to 
  /// the specified vector<File*>.
  llvm::error_code parseObjectTextFileOrSTDIN(llvm::StringRef path
                                 , std::vector<File *>&);


  /// parseObjectText - Parse the specified YAML formatted MemoryBuffer
  /// into lld::File object(s) and append each to the specified vector<File*>.
  llvm::error_code parseObjectText(llvm::MemoryBuffer *mb
                                 , std::vector<File *>&);

} // namespace yaml
} // namespace lld

#endif // LLD_CORE_YAML_READER_H_
