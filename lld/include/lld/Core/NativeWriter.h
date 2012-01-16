//===- Core/NativeWriter.h - Writes native object file --------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_NATIVE_WRITER_H_
#define LLD_CORE_NATIVE_WRITER_H_

#include "lld/Core/File.h"

#include "llvm/Support/raw_ostream.h"

namespace llvm { 
  class StringRef;
}


namespace lld {

  /// writeNativeObjectFile - writes the lld::File object in native object
  /// file format to the specified file path.
  int writeNativeObjectFile(const lld::File &, llvm::StringRef path);

  /// writeNativeObjectFile - writes the lld::File object in native object
  /// file format to the specified stream.
  int writeNativeObjectFile(const lld::File &, llvm::raw_ostream &);

} // namespace lld

#endif // LLD_CORE_NATIVE_WRITER_H_
