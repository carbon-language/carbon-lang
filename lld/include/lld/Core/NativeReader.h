//===- Core/NativeReader.h - Reads llvm native object files ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_NATIVE_READER_H_
#define LLD_CORE_NATIVE_READER_H_

#include "lld/Core/File.h"

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/system_error.h"

#include <vector>

namespace llvm { 
  class MemoryBuffer;
  class StringRef;
}

namespace lld {

  /// parseNativeObjectFileOrSTDIN - Open the specified native object file (use 
  /// stdin if the path is "-") and instantiate into an lld::File object.
  llvm::error_code parseNativeObjectFileOrSTDIN(llvm::StringRef path
                                              , llvm::OwningPtr<File>& result);


  /// parseNativeObjectFile - Parse the specified native object file 
  /// (in a buffer) and instantiate into an lld::File object.
  llvm::error_code parseNativeObjectFile(llvm::OwningPtr<llvm::MemoryBuffer>& mb
                                        ,llvm::StringRef path
                                        ,llvm::OwningPtr<File>& result);

} // namespace lld

#endif // LLD_CORE_NATIVE_READER_H_
