//===- lib/Driver/Driver.cpp - Linker Driver Emulator ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/File.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Parallel.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Resolver.h"
#include "lld/Passes/RoundTripNativePass.h"
#include "lld/Passes/RoundTripYAMLPass.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

namespace lld {

FileVector makeErrorFile(StringRef path, std::error_code ec) {
  std::vector<std::unique_ptr<File>> result;
  result.push_back(llvm::make_unique<ErrorFile>(path, ec));
  return result;
}

FileVector parseMemberFiles(FileVector &files) {
  std::vector<std::unique_ptr<File>> members;
  for (std::unique_ptr<File> &file : files) {
    if (auto *archive = dyn_cast<ArchiveLibraryFile>(file.get())) {
      if (std::error_code ec = archive->parseAllMembers(members))
        return makeErrorFile(file->path(), ec);
    } else {
      members.push_back(std::move(file));
    }
  }
  return members;
}

FileVector parseFile(LinkingContext &ctx, StringRef path, bool wholeArchive) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> mb
      = MemoryBuffer::getFileOrSTDIN(path);
  if (std::error_code ec = mb.getError())
    return makeErrorFile(path, ec);
  std::vector<std::unique_ptr<File>> files;
  if (std::error_code ec = ctx.registry().parseFile(std::move(mb.get()), files))
    return makeErrorFile(path, ec);
  if (wholeArchive)
    return parseMemberFiles(files);
  return files;
}

/// This is where the link is actually performed.
bool Driver::link(LinkingContext &context, raw_ostream &diagnostics) {
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
  InputGraph &inputGraph = context.getInputGraph();
  if (!inputGraph.size())
    return false;
  inputGraph.normalize();

  bool fail = false;

  // Read inputs
  ScopedTask readTask(getDefaultDomain(), "Read Args");
  TaskGroup tg;
  std::mutex diagnosticsMutex;
  for (std::unique_ptr<InputElement> &ie : inputGraph.inputElements()) {
    tg.spawn([&] {
      // Writes to the same output stream is not guaranteed to be thread-safe.
      // We buffer the diagnostics output to a separate string-backed output
      // stream, acquire the lock, and then print it out.
      std::string buf;
      llvm::raw_string_ostream stream(buf);

      if (std::error_code ec = ie->parse(context, stream)) {
        if (FileNode *fileNode = dyn_cast<FileNode>(ie.get()))
          stream << fileNode->errStr(ec) << "\n";
        else
          llvm_unreachable("Unknown type of input element");
        fail = true;
      }

      stream.flush();
      if (!buf.empty()) {
        std::lock_guard<std::mutex> lock(diagnosticsMutex);
        diagnostics << buf;
      }
    });
  }
  tg.sync();
  readTask.end();

  if (fail)
    return false;

  InputGraph::FileVectorT internalFiles;
  context.createInternalFiles(internalFiles);
  for (auto i = internalFiles.rbegin(), e = internalFiles.rend(); i != e; ++i) {
    context.getInputGraph().addInputElementFront(
        llvm::make_unique<SimpleFileNode>("internal", std::move(*i)));
  }

  // Give target a chance to add files.
  InputGraph::FileVectorT implicitFiles;
  context.createImplicitFiles(implicitFiles);
  for (auto i = implicitFiles.rbegin(), e = implicitFiles.rend(); i != e; ++i) {
    context.getInputGraph().addInputElementFront(
        llvm::make_unique<SimpleFileNode>("implicit", std::move(*i)));
  }

  // Give target a chance to sort the input files.
  // Mach-O uses this chance to move all object files before library files.
  context.maybeSortInputFiles();

  // Do core linking.
  ScopedTask resolveTask(getDefaultDomain(), "Resolve");
  Resolver resolver(context);
  if (!resolver.resolve())
    return false;
  std::unique_ptr<MutableFile> merged = resolver.resultFile();
  resolveTask.end();

  // Run passes on linked atoms.
  ScopedTask passTask(getDefaultDomain(), "Passes");
  PassManager pm;
  context.addPasses(pm);

#ifndef NDEBUG
  if (context.runRoundTripPass()) {
    pm.add(std::unique_ptr<Pass>(new RoundTripYAMLPass(context)));
    pm.add(std::unique_ptr<Pass>(new RoundTripNativePass(context)));
  }
#endif

  pm.runOnFile(merged);
  passTask.end();

  // Give linked atoms to Writer to generate output file.
  ScopedTask writeTask(getDefaultDomain(), "Write");
  if (std::error_code ec = context.writeFile(*merged)) {
    diagnostics << "Failed to write file '" << context.outputPath()
                << "': " << ec.message() << "\n";
    return false;
  }

  return true;
}

} // namespace
