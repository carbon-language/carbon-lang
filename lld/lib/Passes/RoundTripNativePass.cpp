//===--Passes/RoundTripNativePass.cpp - Write Native file/Read it back-----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "RoundTripNativePass"

#include "lld/Core/Instrumentation.h"
#include "lld/Passes/RoundTripNativePass.h"
#include "lld/ReaderWriter/Simple.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/Path.h"

using namespace lld;

/// Perform the actual pass
void RoundTripNativePass::perform(std::unique_ptr<MutableFile> &mergedFile) {
  ScopedTask task(getDefaultDomain(), "RoundTripNativePass");
  std::unique_ptr<Writer> nativeWriter = createWriterNative(_context);
  SmallString<128> tmpNativeFile;
  // Separate the directory from the filename
  StringRef outFile = llvm::sys::path::filename(_context.outputPath());
  if (llvm::sys::fs::createTemporaryFile(outFile, "native", tmpNativeFile))
    return;

  // The file that is written would be kept around if there is a problem
  // writing to the file or when reading atoms back from the file.
  nativeWriter->writeFile(*mergedFile, tmpNativeFile.str());
  OwningPtr<MemoryBuffer> buff;
  if (MemoryBuffer::getFileOrSTDIN(tmpNativeFile.str(), buff))
    return;

  std::unique_ptr<MemoryBuffer> mb(buff.take());
  _context.getNativeReader().parseFile(mb, _nativeFile);

  mergedFile.reset(new FileToMutable(_context, *_nativeFile[0].get()));

  llvm::sys::fs::remove(tmpNativeFile.str());
}
