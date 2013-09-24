//===- lib/Driver/Driver.cpp - Linker Driver Emulator ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"

#include "lld/Core/LLVM.h"
#include "lld/Core/InputFiles.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Parallel.h"
#include "lld/Core/Resolver.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace lld {

/// This is where the link is actually performed.
bool Driver::link(const LinkingContext &context, raw_ostream &diagnostics) {
  // Honor -mllvm
  if (!context.llvmOptions().empty()) {
    unsigned numArgs = context.llvmOptions().size();
    const char **args = new const char *[numArgs + 2];
    args[0] = "lld (LLVM option parsing)";
    for (unsigned i = 0; i != numArgs; ++i)
      args[i + 1] = context.llvmOptions()[i];
    args[numArgs + 1] = 0;
    llvm::cl::ParseCommandLineOptions(numArgs + 1, args);
  }
  InputGraph &inputGraph = context.inputGraph();
  if (!inputGraph.numFiles())
    return false;

  // Read inputs
  ScopedTask readTask(getDefaultDomain(), "Read Args");
  std::vector<std::vector<std::unique_ptr<File> > > files(
      inputGraph.numFiles());
  size_t index = 0;
  std::atomic<bool> fail(false);
  TaskGroup tg;
  std::vector<std::unique_ptr<LinkerInput> > linkerInputs;
  for (auto &ie : inputGraph.inputElements()) {
    if (ie->kind() == InputElement::Kind::File) {
      FileNode *fileNode = (llvm::dyn_cast<FileNode>)(ie.get());
      auto linkerInput = fileNode->createLinkerInput(context);
      if (!linkerInput) {
        llvm::outs() << fileNode->errStr(error_code(linkerInput)) << "\n";
        return false;
      }
      linkerInputs.push_back(std::move(*linkerInput));
    }
    else {
      llvm_unreachable("Not handling other types of InputElements");
    }
  }
  for (const auto &input : linkerInputs) {
    if (context.logInputFiles())
      llvm::outs() << input->getUserPath() << "\n";

    tg.spawn([ &, index]{
      if (error_code ec = context.parseFile(*input, files[index])) {
        diagnostics << "Failed to read file: " << input->getUserPath() << ": "
                    << ec.message() << "\n";
        fail = true;
        return;
      }
    });
    ++index;
  }
  tg.sync();
  readTask.end();

  if (fail)
    return false;

  InputFiles inputs;

  for (auto &f : inputGraph.internalFiles())
    inputs.appendFile(*f.get());

  for (auto &f : files)
    inputs.appendFiles(f);

  // Give target a chance to add files.
  context.addImplicitFiles(inputs);

  // assign an ordinal to each file so sort() can preserve command line order
  inputs.assignFileOrdinals();

  // Do core linking.
  ScopedTask resolveTask(getDefaultDomain(), "Resolve");
  Resolver resolver(context, inputs);
  if (resolver.resolve()) {
    if (!context.allowRemainingUndefines())
      return false;
  }
  MutableFile &merged = resolver.resultFile();
  resolveTask.end();

  // Run passes on linked atoms.
  ScopedTask passTask(getDefaultDomain(), "Passes");
  PassManager pm;
  context.addPasses(pm);
  pm.runOnFile(merged);
  passTask.end();

  // Give linked atoms to Writer to generate output file.
  ScopedTask writeTask(getDefaultDomain(), "Write");
  if (error_code ec = context.writeFile(merged)) {
    diagnostics << "Failed to write file '" << context.outputPath()
                << "': " << ec.message() << "\n";
    return false;
  }

  return true;
}

} // namespace
