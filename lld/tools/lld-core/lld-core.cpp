//===- tools/lld/lld-core.cpp - Linker Core Test Driver -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Atom.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Resolver.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/ReaderNative.h"
#include "lld/ReaderWriter/ReaderYAML.h"
#include "lld/ReaderWriter/Writer.h"
#include "lld/ReaderWriter/WriterELF.h"
#include "lld/ReaderWriter/WriterMachO.h"
#include "lld/ReaderWriter/WriterNative.h"
#include "lld/ReaderWriter/WriterPECOFF.h"
#include "lld/ReaderWriter/WriterYAML.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"

#include <vector>

#include "TestingHelpers.hpp"

using namespace lld;

static void error(Twine message) {
  llvm::errs() << "lld-core: " << message << ".\n";
}

static bool error(error_code ec) {
  if (ec) {
    error(ec.message());
    return true;
  }
  return false;
}


llvm::cl::list<std::string>
cmdLineInputFilePaths(llvm::cl::Positional,
              llvm::cl::desc("<input file>"));

llvm::cl::opt<std::string> 
cmdLineOutputFilePath("o", 
              llvm::cl::desc("Specify output filename"), 
              llvm::cl::value_desc("filename"));

llvm::cl::opt<bool> 
cmdLineDoStubsPass("stubs-pass", 
          llvm::cl::desc("Run pass to create stub atoms"));

llvm::cl::opt<bool> 
cmdLineDoGotPass("got-pass", 
          llvm::cl::desc("Run pass to create GOT atoms"));

llvm::cl::opt<bool> 
cmdLineUndefinesIsError("undefines-are-errors", 
          llvm::cl::desc("Any undefined symbols at end is an error"));

llvm::cl::opt<bool> 
cmdLineCommonsSearchArchives("commons-search-archives", 
          llvm::cl::desc("Tentative definitions trigger archive search"));

llvm::cl::opt<bool> 
cmdLineDeadStrip("dead-strip", 
          llvm::cl::desc("Remove unreachable code and data"));

llvm::cl::opt<bool> 
cmdLineGlobalsNotDeadStrip("keep-globals", 
          llvm::cl::desc("All global symbols are roots for dead-strip"));


enum WriteChoice {
  writeYAML, writeMachO, writePECOFF, writeELF
};

llvm::cl::opt<WriteChoice> 
writeSelected("writer",
  llvm::cl::desc("Select writer"),
  llvm::cl::values(
    clEnumValN(writeYAML,   "YAML",   "link assuming YAML format"),
    clEnumValN(writeMachO,  "mach-o", "link as darwin would"),
    clEnumValN(writePECOFF, "PECOFF", "link as windows would"),
    clEnumValN(writeELF,    "ELF",    "link as linux would"),
    clEnumValEnd));
    


class TestingResolverOptions : public ResolverOptions {
public:
  TestingResolverOptions() {
    _undefinesAreErrors = cmdLineUndefinesIsError;
    _searchArchivesToOverrideTentativeDefinitions = cmdLineCommonsSearchArchives;
    _deadCodeStrip = cmdLineDeadStrip;
    _globalsAreDeadStripRoots = cmdLineGlobalsNotDeadStrip;
  }

};



int main(int argc, char *argv[]) {
  // Print a stack trace if we signal out.
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // parse options
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // if no input file specified, read from stdin
  if (cmdLineInputFilePaths.empty())
    cmdLineInputFilePaths.emplace_back("-");

  // if no output path specified, write to stdout
  if (cmdLineOutputFilePath.empty())
    cmdLineOutputFilePath.assign("-");

  // create writer for final output
  TestingWriterOptionsYAML  writerOptionsYAML(cmdLineDoStubsPass, 
                                              cmdLineDoGotPass);
  WriterOptionsMachO        writerOptionsMachO;
  WriterOptionsPECOFF       writerOptionsPECOFF;
  WriterOptionsELF          writerOptionsELF;
  Writer* writer = nullptr;
  switch ( writeSelected ) {
    case writeYAML:
      writer = createWriterYAML(writerOptionsYAML);
      break;
    case writeMachO:
      writer = createWriterMachO(writerOptionsMachO);
      break;
    case writePECOFF:
      writer = createWriterPECOFF(writerOptionsPECOFF);
      break;
    case writeELF:
      writer = createWriterELF(writerOptionsELF);
      break;
  }
  
  // create object to mange input files
  InputFiles inputFiles;

  // read input files into in-memory File objects
  TestingReaderOptionsYAML  readerOptionsYAML;
  Reader *reader = createReaderYAML(readerOptionsYAML);
  for (auto path : cmdLineInputFilePaths) {
    std::vector<std::unique_ptr<File>> files;
    if ( error(reader->readFile(path, files)) )
      return 1;
    inputFiles.appendFiles(files);
  }
    
  // given writer a chance to add files
  writer->addFiles(inputFiles);

  // create options for resolving
  TestingResolverOptions options;

  // merge all atom graphs
  Resolver resolver(options, inputFiles);
  resolver.resolve();
  File &mergedMasterFile = resolver.resultFile();

  // run passes
  if ( GOTPass *pass = writer->gotPass() ) {
    pass->perform(mergedMasterFile);
  }
  if ( StubsPass *pass = writer->stubPass() ) {
    pass->perform(mergedMasterFile);
  }

  // showing yaml at this stage can help when debugging
  const bool dumpIntermediateYAML = false;
  if ( dumpIntermediateYAML )
    writer->writeFile(mergedMasterFile, "-");

  // make unique temp file to put generated native object file
  llvm::sys::Path tmpNativePath = llvm::sys::Path::GetTemporaryDirectory();
  if (tmpNativePath.createTemporaryFileOnDisk()) {
    error("createTemporaryFileOnDisk() failed");
    return 1;
  }
  
  // write as native file
  WriterOptionsNative  optionsNativeWriter;
  Writer *natWriter = createWriterNative(optionsNativeWriter);
  if (error(natWriter->writeFile(mergedMasterFile, tmpNativePath.c_str())))
    return 1;
  
  // read as native file
  ReaderOptionsNative  optionsNativeReader;
  Reader *natReader = createReaderNative(optionsNativeReader);
  std::vector<std::unique_ptr<File>> readNativeFiles;
  if (error(natReader->readFile(tmpNativePath.c_str(), readNativeFiles)))
    return 1;
  
  // write new atom graph
  const File *parsedNativeFile = readNativeFiles[0].get();
  if (error(writer->writeFile(*parsedNativeFile, cmdLineOutputFilePath)))
    return 1;
   
  return 0;
}
