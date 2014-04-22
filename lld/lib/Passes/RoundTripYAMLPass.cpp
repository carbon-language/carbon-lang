//===--Passes/RoundTripYAMLPass.cpp - Write YAML file/Read it back---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lld/Core/Instrumentation.h"
#include "lld/Passes/RoundTripYAMLPass.h"
#include "lld/ReaderWriter/Simple.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include <memory>

using namespace lld;

#define DEBUG_TYPE "RoundTripYAMLPass"

/// Perform the actual pass
void RoundTripYAMLPass::perform(std::unique_ptr<MutableFile> &mergedFile) {
  ScopedTask task(getDefaultDomain(), "RoundTripYAMLPass");
  std::unique_ptr<Writer> yamlWriter = createWriterYAML(_context);
  SmallString<128> tmpYAMLFile;
  // Separate the directory from the filename
  StringRef outFile = llvm::sys::path::filename(_context.outputPath());
  if (llvm::sys::fs::createTemporaryFile(outFile, "yaml", tmpYAMLFile))
    return;
  DEBUG_WITH_TYPE("RoundTripYAMLPass", {
    llvm::dbgs() << "RoundTripYAMLPass: " << tmpYAMLFile << "\n";
  });

  // The file that is written would be kept around if there is a problem
  // writing to the file or when reading atoms back from the file.
  yamlWriter->writeFile(*mergedFile, tmpYAMLFile.str());
  std::unique_ptr<MemoryBuffer> mb;
  if (MemoryBuffer::getFile(tmpYAMLFile.str(), mb))
    return;

  error_code ec = _context.registry().parseFile(mb, _yamlFile);
  if (ec) {
    // Note: we need a way for Passes to report errors.
    llvm_unreachable("yaml reader not registered or read error");
  }
  File *objFile = _yamlFile[0].get();
  mergedFile.reset(new FileToMutable(_context, *objFile));
  llvm::sys::fs::remove(tmpYAMLFile.str());
}
