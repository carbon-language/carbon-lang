//===- Passes/RoundTripYAMLPass.cpp - Layout atoms
//-------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "RoundTripYAMLPass"

#include "lld/Core/Instrumentation.h"
#include "lld/Passes/RoundTripYAMLPass.h"
#include "lld/ReaderWriter/Simple.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/Path.h"

using namespace lld;

/// Perform the actual pass
void RoundTripYAMLPass::perform(std::unique_ptr<MutableFile> &mergedFile) {
  ScopedTask task(getDefaultDomain(), "RoundTripYAMLPass");
  std::unique_ptr<Writer> yamlWriter = createWriterYAML(_context);
  SmallString<128> tmpYAMLFile;
  // Separate the directory from the filename
  StringRef outFile = llvm::sys::path::filename(_context.outputPath());
  if (llvm::sys::fs::createTemporaryFile(outFile, "yaml", tmpYAMLFile))
    return;

  yamlWriter->writeFile(*mergedFile, tmpYAMLFile.str());
  llvm::OwningPtr<llvm::MemoryBuffer> buff;
  if (llvm::MemoryBuffer::getFileOrSTDIN(tmpYAMLFile.str(), buff))
    return;

  std::unique_ptr<MemoryBuffer> mb(buff.take());
  _context.getYAMLReader().parseFile(mb, _yamlFile);

  mergedFile.reset(new FileToMutable(_context, *_yamlFile[0].get()));

  llvm::sys::fs::remove(tmpYAMLFile.str());
}
