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
#include "lld/Core/TargetInfo.h"
#include "lld/Driver/Target.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;

void LinkerInvocation::operator()() {
  // Honor -mllvm
  if (!_options._llvmArgs.empty()) {
    unsigned NumArgs = _options._llvmArgs.size();
    const char **Args = new const char*[NumArgs + 2];
    Args[0] = "lld (LLVM option parsing)";
    for (unsigned i = 0; i != NumArgs; ++i)
      Args[i + 1] = _options._llvmArgs[i].c_str();
    Args[NumArgs + 1] = 0;
    llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args);
  }

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

  class TestingTargetInfo LLVM_FINAL : public TargetInfo {
  public:
    TestingTargetInfo(const LinkerOptions &lo) : TargetInfo(lo) {}

    virtual uint64_t getPageSize() const { return 0x1000; }
  } tti(_options);

  auto writer = target->getWriter();

  // Give writer a chance to add files
  writer->addFiles(inputs);

  Resolver resolver(tti, inputs);
  resolver.resolve();
  File &merged = resolver.resultFile();

  if (error_code ec = writer) {
    llvm::errs() << "Failed to get writer: " << ec.message() << ".\n";
    return;
  }

  if (error_code ec = writer->writeFile(merged, _options._outputPath))
    llvm::errs() << "Failed to write file: " << ec.message() << "\n";
}
