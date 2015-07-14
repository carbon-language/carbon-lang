//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "Writer.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/LibDriver/LibDriver.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using llvm::sys::fs::file_magic;
using llvm::sys::fs::identify_magic;

using namespace lld;
using namespace lld::elfv2;

namespace lld {
namespace elfv2 {
Configuration *Config;
LinkerDriver *Driver;

bool link(llvm::ArrayRef<const char *> Args) {
  auto C = make_unique<Configuration>();
  Config = C.get();
  auto D = make_unique<LinkerDriver>();
  Driver = D.get();
  return Driver->link(Args);
}
}
}

// Drop directory components and replace extension with ".exe".
static std::string getOutputPath(StringRef Path) {
  auto P = Path.find_last_of("\\/");
  StringRef S = (P == StringRef::npos) ? Path : Path.substr(P + 1);
  return (S.substr(0, S.rfind('.')) + ".exe").str();
}

// Opens a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
ErrorOr<MemoryBufferRef> LinkerDriver::openFile(StringRef Path) {
  auto MBOrErr = MemoryBuffer::getFile(Path);
  if (auto EC = MBOrErr.getError())
    return EC;
  std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take ownership
  return MBRef;
}

static std::unique_ptr<InputFile> createFile(MemoryBufferRef MB) {
  // File type is detected by contents, not by file extension.
  file_magic Magic = identify_magic(MB.getBuffer());
  if (Magic == file_magic::archive)
    return std::unique_ptr<InputFile>(new ArchiveFile(MB));
  if (Magic == file_magic::bitcode)
    return std::unique_ptr<InputFile>(new BitcodeFile(MB));
  if (Config->OutputFile == "")
    Config->OutputFile = getOutputPath(MB.getBufferIdentifier());
  return std::unique_ptr<InputFile>(new ObjectFile<llvm::object::ELF64LE>(MB));
}

// Find file from search paths. You can omit ".obj", this function takes
// care of that. Note that the returned path is not guaranteed to exist.
StringRef LinkerDriver::doFindFile(StringRef Filename) {
  bool hasPathSep = (Filename.find_first_of("/\\") != StringRef::npos);
  if (hasPathSep)
    return Filename;
  bool hasExt = (Filename.find('.') != StringRef::npos);
  for (StringRef Dir : SearchPaths) {
    SmallString<128> Path = Dir;
    llvm::sys::path::append(Path, Filename);
    if (llvm::sys::fs::exists(Path.str()))
      return Alloc.save(Path.str());
    if (!hasExt) {
      Path.append(".obj");
      if (llvm::sys::fs::exists(Path.str()))
        return Alloc.save(Path.str());
    }
  }
  return Filename;
}

// Resolves a file path. This never returns the same path
// (in that case, it returns None).
Optional<StringRef> LinkerDriver::findFile(StringRef Filename) {
  StringRef Path = doFindFile(Filename);
  bool Seen = !VisitedFiles.insert(Path.lower()).second;
  if (Seen)
    return None;
  return Path;
}

// Find library file from search path.
StringRef LinkerDriver::doFindLib(StringRef Filename) {
  // Add ".lib" to Filename if that has no file extension.
  bool hasExt = (Filename.find('.') != StringRef::npos);
  if (!hasExt)
    Filename = Alloc.save(Filename + ".lib");
  return doFindFile(Filename);
}

// Resolves a library path. /nodefaultlib options are taken into
// consideration. This never returns the same path (in that case,
// it returns None).
Optional<StringRef> LinkerDriver::findLib(StringRef Filename) {
  StringRef Path = doFindLib(Filename);
  bool Seen = !VisitedFiles.insert(Path.lower()).second;
  if (Seen)
    return None;
  return Path;
}

bool LinkerDriver::link(llvm::ArrayRef<const char *> ArgsArr) {
  // Needed for LTO.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();

  // Parse command line options.
  auto ArgsOrErr = Parser.parse(ArgsArr);
  if (auto EC = ArgsOrErr.getError()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  llvm::opt::InputArgList Args = std::move(ArgsOrErr.get());

  // Handle /help
  if (Args.hasArg(OPT_help)) {
    printHelp(ArgsArr[0]);
    return true;
  }

  if (Args.filtered_begin(OPT_INPUT) == Args.filtered_end()) {
    llvm::errs() << "no input files.\n";
    return false;
  }

  // Construct search path list.
  SearchPaths.push_back("");
  for (auto *Arg : Args.filtered(OPT_L))
    SearchPaths.push_back(Arg->getValue());

  // Handle /out
  if (auto *Arg = Args.getLastArg(OPT_output))
    Config->OutputFile = Arg->getValue();

  // Handle /entry
  if (auto *Arg = Args.getLastArg(OPT_e))
    Config->EntryName = Arg->getValue();

  // Create a list of input files. Files can be given as arguments
  // for /defaultlib option.
  std::vector<StringRef> InputPaths;
  std::vector<MemoryBufferRef> Inputs;
  for (auto *Arg : Args.filtered(OPT_INPUT))
    if (Optional<StringRef> Path = findFile(Arg->getValue()))
      InputPaths.push_back(*Path);

  for (StringRef Path : InputPaths) {
    ErrorOr<MemoryBufferRef> MBOrErr = openFile(Path);
    if (auto EC = MBOrErr.getError()) {
      llvm::errs() << "cannot open " << Path << ": " << EC.message() << "\n";
      return false;
    }
    Inputs.push_back(MBOrErr.get());
  }

  // Create a symbol table.
  SymbolTable<llvm::object::ELF64LE> Symtab;

  // Parse all input files and put all symbols to the symbol table.
  // The symbol table will take care of name resolution.
  for (MemoryBufferRef MB : Inputs) {
    std::unique_ptr<InputFile> File = createFile(MB);
    if (Config->Verbose)
      llvm::outs() << "Reading " << File->getName() << "\n";
    if (auto EC = Symtab.addFile(std::move(File))) {
      llvm::errs() << MB.getBufferIdentifier() << ": " << EC.message() << "\n";
      return false;
    }
  }

  // Make sure we have resolved all symbols.
  if (Symtab.reportRemainingUndefines())
    return false;

  // Initialize a list of GC root.
  Config->GCRoots.insert(Config->EntryName);

  // Do LTO by compiling bitcode input files to a native ELF file
  // then link that file.
  if (auto EC = Symtab.addCombinedLTOObject()) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }

  // Write the result.
  Writer<llvm::object::ELF64LE> Out(&Symtab);
  if (auto EC = Out.write(Config->OutputFile)) {
    llvm::errs() << EC.message() << "\n";
    return false;
  }
  return true;
}
