//===- lib/Driver/LinkerInvocation.cpp - Linker Invocation ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/LinkerInvocation.h"

#include "lld/Core/InputFiles.h"
#include "lld/Core/Resolver.h"
#include "lld/Driver/Target.h"

#include "llvm/Support/raw_ostream.h"

using namespace lld;

void LinkerInvocation::operator()() {
    // Create target.
    std::unique_ptr<Target> target(Target::create(_options));

    if (!target) {
      llvm::errs() << "Failed to create target for " << _options._target
                   << "\n";
      return;
    }

    // Read inputs
    InputFiles inputs;
    for (const auto &input : _options._input) {
      auto reader = target->getReader(input);
      if (error_code ec = reader) {
        llvm::errs() << "Failed to get reader for: " << input.getPath() << ": "
                     << ec.message() << "\n";
        return;
      }

      auto buffer = input.getBuffer();
      if (error_code ec = buffer) {
        llvm::errs() << "Failed to read file: " << input.getPath() << ": "
                     << ec.message() << "\n";
        return;
      }

      std::vector<std::unique_ptr<File>> files;
      if (llvm::error_code ec = reader->readFile(
            buffer->getBufferIdentifier(), files)) {
        llvm::errs() << "Failed to read file: " << input.getPath() << ": "
                     << ec.message() << "\n";
        return;
      }
      inputs.appendFiles(files);
    }

    ResolverOptions ro;
    Resolver resolver(ro, inputs);
    resolver.resolve();
    File &merged = resolver.resultFile();

    auto writer = target->getWriter();
    if (error_code ec = writer) {
      llvm::errs() << "Failed to get writer: " << ec.message() << ".\n";
      return;
    }

    if (error_code ec = writer->writeFile(merged, _options._outputPath))
      llvm::errs() << "Failed to write file: " << ec.message() << "\n";
  }
