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

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Debug.h"
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
  DEBUG_WITH_TYPE("RoundTripNativePass", {
    llvm::dbgs() << "RoundTripNativePass: " << tmpNativeFile << "\n";
  });

  // The file that is written would be kept around if there is a problem
  // writing to the file or when reading atoms back from the file.
  nativeWriter->writeFile(*mergedFile, tmpNativeFile.str());
  OwningPtr<MemoryBuffer> buff;
  if (MemoryBuffer::getFile(tmpNativeFile.str(), buff))
    return;

  std::unique_ptr<MemoryBuffer> mb(buff.take());
  error_code ec = _context.registry().parseFile(mb, _nativeFile);
  if (ec) {
    // Note: we need a way for Passes to report errors.
    llvm_unreachable("native reader not registered or read error");
  }
  File *objFile = _nativeFile[0].get();
  mergedFile.reset(new FileToMutable(_context, *objFile));

  llvm::sys::fs::remove(tmpNativeFile.str());
}
