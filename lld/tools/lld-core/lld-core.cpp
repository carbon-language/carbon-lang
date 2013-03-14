//===- tools/lld/lld-core.cpp - Linker Core Test Driver -----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Atom.h"
#include "lld/Core/LinkerOptions.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Resolver.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/ReaderWriter/ELFTargetInfo.h"
#include "lld/ReaderWriter/MachOTargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/ReaderArchive.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
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

llvm::cl::opt<bool> cmdLineDoStubsPass(
    "stubs-pass", llvm::cl::desc("Run pass to create stub atoms"));

llvm::cl::opt<bool>
cmdLineDoGotPass("got-pass", llvm::cl::desc("Run pass to create GOT atoms"));

llvm::cl::opt<bool>
cmdLineDoLayoutPass("layout-pass", llvm::cl::desc("Run pass to layout atoms"));

llvm::cl::opt<bool>
cmdLineDoMergeStrings(
  "merge-strings",
  llvm::cl::desc("make common strings merge possible"));

llvm::cl::opt<bool> cmdLineUndefinesIsError(
    "undefines-are-errors",
    llvm::cl::desc("Any undefined symbols at end is an error"));

llvm::cl::opt<bool> cmdLineForceLoad(
    "force-load", llvm::cl::desc("force load all members of the archive"));

llvm::cl::opt<bool>
cmdLineCommonsSearchArchives("commons-search-archives",
          llvm::cl::desc("Tentative definitions trigger archive search"));

llvm::cl::opt<bool>
cmdLineDeadStrip("dead-strip",
          llvm::cl::desc("Remove unreachable code and data"));

llvm::cl::opt<bool>
cmdLineGlobalsNotDeadStrip("keep-globals",
          llvm::cl::desc("All global symbols are roots for dead-strip"));

llvm::cl::opt<std::string>
cmdLineEntryPoint("entry",
              llvm::cl::desc("Specify entry point symbol"),
              llvm::cl::value_desc("symbol"));


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
    clEnumValEnd),
  llvm::cl::init(writeYAML));

enum ReaderChoice {
  readerYAML, readerMachO, readerPECOFF, readerELF
};
llvm::cl::opt<ReaderChoice>
readerSelected("reader",
  llvm::cl::desc("Select reader"),
  llvm::cl::values(
    clEnumValN(readerYAML,   "YAML",   "read assuming YAML format"),
    clEnumValN(readerMachO,  "mach-o", "read as darwin would"),
    clEnumValN(readerPECOFF, "PECOFF", "read as windows would"),
    clEnumValN(readerELF,    "ELF",    "read as linux would"),
    clEnumValEnd),
  llvm::cl::init(readerYAML));

enum ArchChoice {
  i386 = llvm::ELF::EM_386,
  x86_64 = llvm::ELF::EM_X86_64,
  hexagon = llvm::ELF::EM_HEXAGON,
  ppc = llvm::ELF::EM_PPC
};
llvm::cl::opt<ArchChoice>
archSelected("arch",
  llvm::cl::desc("Select architecture, only valid with ELF output"),
  llvm::cl::values(
    clEnumValN(i386, "i386",
               "output i386, EM_386 file"),
    clEnumValN(x86_64,
               "x86_64", "output x86_64, EM_X86_64 file"),
    clEnumValN(hexagon,
               "hexagon", "output Hexagon, EM_HEXAGON file"),
    clEnumValN(ppc,
               "ppc", "output PowerPC, EM_PPC file"),
    clEnumValEnd),
  llvm::cl::init(i386));


enum endianChoice {
  little, big
};
llvm::cl::opt<endianChoice> endianSelected(
    "endian", llvm::cl::desc("Select endianness of ELF output"),
    llvm::cl::values(clEnumValN(big, "big", "output big endian format"),
                     clEnumValN(little, "little",
                                "output little endian format"), clEnumValEnd));

class TestingTargetInfo : public TargetInfo {
public:
  TestingTargetInfo(const LinkerOptions &lo, bool stubs, bool got, bool layout)
      : TargetInfo(lo), _doStubs(stubs), _doGOT(got), _doLayout(layout) {
  }

  virtual uint64_t getPageSize() const { return 0x1000; }

  virtual void addPasses(PassManager &pm) const {
    if (_doStubs)
      pm.add(std::unique_ptr<Pass>(new TestingStubsPass(*this)));
    if (_doGOT)
      pm.add(std::unique_ptr<Pass>(new TestingGOTPass(*this)));
    if (_doLayout)
      pm.add(std::unique_ptr<Pass>(new LayoutPass()));
  }

  virtual ErrorOr<int32_t> relocKindFromString(StringRef str) const {
    // Try parsing as a number.
    if (auto kind = TargetInfo::relocKindFromString(str))
      return kind;
    for (const auto *kinds = sKinds; kinds->string; ++kinds)
      if (str == kinds->string)
        return kinds->value;
    return llvm::make_error_code(llvm::errc::invalid_argument);
  }

  virtual ErrorOr<std::string> stringFromRelocKind(int32_t kind) const {
    for (const TestingKindMapping *p = sKinds; p->string != nullptr; ++p) {
      if (kind == p->value)
        return std::string(p->string);
    }
    return llvm::make_error_code(llvm::errc::invalid_argument);
  }

  virtual ErrorOr<Reader &> getReader(const LinkerInput &input) const {
    llvm_unreachable("Unimplemented!");
  }

  virtual ErrorOr<Writer &> getWriter() const {
    llvm_unreachable("Unimplemented!");
  }

private:
  bool _doStubs;
  bool _doGOT;
  bool _doLayout;
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

  LinkerOptions lo;
  lo._noInhibitExec = !cmdLineUndefinesIsError;
  lo._searchArchivesToOverrideTentativeDefinitions =
      cmdLineCommonsSearchArchives;
  lo._deadStrip = cmdLineDeadStrip;
  lo._globalsAreDeadStripRoots = cmdLineGlobalsNotDeadStrip;
  lo._forceLoadArchives = cmdLineForceLoad;
  lo._outputKind = OutputKind::StaticExecutable;
  lo._entrySymbol = cmdLineEntryPoint;
  lo._mergeCommonStrings = cmdLineDoMergeStrings;

  switch (archSelected) {
  case i386:
    lo._target = "i386";
    break;
  case x86_64:
    lo._target = "x86_64";
    break;
  case hexagon:
    lo._target = "hexagon";
    break;
  case ppc:
    lo._target = "powerpc";
    break;
  }

  TestingTargetInfo tti(lo, cmdLineDoStubsPass, cmdLineDoGotPass,
                        cmdLineDoLayoutPass);

  std::unique_ptr<ELFTargetInfo> eti = ELFTargetInfo::create(lo);
  std::unique_ptr<MachOTargetInfo> mti = MachOTargetInfo::create(lo);
  std::unique_ptr<Writer> writer;
  const TargetInfo *ti = 0;
  switch ( writeSelected ) {
    case writeYAML:
      writer = createWriterYAML(tti);
      ti = &tti;
      break;
    case writeMachO:
      writer = createWriterMachO(*mti);
      ti = mti.get();
      break;
    case writePECOFF:
      writer = createWriterPECOFF(tti);
      ti = &tti;
      break;
    case writeELF:
      writer = createWriterELF(*eti);
      ti = eti.get();
      break;
  }

  // create object to mange input files
  InputFiles inputFiles;

  // read input files into in-memory File objects
  std::unique_ptr<Reader> reader;
  switch ( readerSelected ) {
    case readerYAML:
      reader = createReaderYAML(tti);
      break;
#if 0
    case readerMachO:
      reader = createReaderMachO(lld::readerOptionsMachO);
      break;
#endif
    case readerPECOFF:
      reader = createReaderPECOFF(tti,
      [&] (const LinkerInput &) -> ErrorOr<Reader&> {
        return *reader;
      });
      break;
    case readerELF:
      reader = createReaderELF(*eti,
      [&] (const LinkerInput &) -> ErrorOr<Reader&> {
        return *reader;
      });
      break;
    default:
      reader = createReaderYAML(tti);
      break;
  }

  for (auto path : cmdLineInputFilePaths) {
    std::vector<std::unique_ptr<File>> files;
    if ( error(reader->readFile(path, files)) )
      return 1;
    inputFiles.appendFiles(files);
  }

  // given writer a chance to add files
  writer->addFiles(inputFiles);

  // assign an ordinal to each file so sort() can preserve command line order
  inputFiles.assignFileOrdinals();

  // merge all atom graphs
  Resolver resolver(tti, inputFiles);
  resolver.resolve();
  MutableFile &mergedMasterFile = resolver.resultFile();

  PassManager pm;
  if (ti)
    ti->addPasses(pm);
  pm.runOnFile(mergedMasterFile);

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
  std::unique_ptr<Writer> natWriter = createWriterNative(tti);
  if (error(natWriter->writeFile(mergedMasterFile, tmpNativePath.c_str())))
    return 1;

  // read as native file
  std::unique_ptr<Reader> natReader = createReaderNative(tti);
  std::vector<std::unique_ptr<File>> readNativeFiles;
  if (error(natReader->readFile(tmpNativePath.c_str(), readNativeFiles)))
    return 1;

  // write new atom graph
  const File *parsedNativeFile = readNativeFiles[0].get();
  if (error(writer->writeFile(*parsedNativeFile, cmdLineOutputFilePath)))
    return 1;

  return 0;
}
