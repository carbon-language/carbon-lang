//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "Error.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

namespace lld {
namespace elf2 {
Configuration *Config;

void link(ArrayRef<const char *> Args) {
  Configuration C;
  Config = &C;
  LinkerDriver().link(Args.slice(1));
}

}
}

// Opens a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
MemoryBufferRef LinkerDriver::openFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getFile(Path);
  error(MBOrErr, Twine("cannot open ") + Path);
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take ownership
  return MBRef;
}

static std::unique_ptr<InputFile> createFile(MemoryBufferRef MB) {
  using namespace llvm::sys::fs;
  file_magic Magic = identify_magic(MB.getBuffer());

  if (Magic == file_magic::archive)
    return make_unique<ArchiveFile>(MB);

  if (Magic == file_magic::elf_shared_object)
    return createELFFile<SharedFile>(MB);

  return createELFFile<ObjectFile>(MB);
}

void LinkerDriver::link(ArrayRef<const char *> ArgsArr) {
  // Parse command line options.
  opt::InputArgList Args = Parser.parse(ArgsArr);

  // Handle -o
  if (auto *Arg = Args.getLastArg(OPT_output))
    Config->OutputFile = Arg->getValue();
  if (Config->OutputFile.empty())
    error("-o must be specified.");

  // Handle -dynamic-linker
  if (auto *Arg = Args.getLastArg(OPT_dynamic_linker))
    Config->DynamicLinker = Arg->getValue();

  std::vector<StringRef> RPaths;
  for (auto *Arg : Args.filtered(OPT_rpath))
    RPaths.push_back(Arg->getValue());
  if (!RPaths.empty())
    Config->RPath = llvm::join(RPaths.begin(), RPaths.end(), ":");

  if (Args.hasArg(OPT_shared))
    Config->Shared = true;

  if (Args.hasArg(OPT_discard_all))
    Config->DiscardAll = true;

  if (Args.hasArg(OPT_discard_locals))
    Config->DiscardLocals = true;

  // Create a list of input files.
  std::vector<MemoryBufferRef> Inputs;

  for (auto *Arg : Args.filtered(OPT_INPUT)) {
    StringRef Path = Arg->getValue();
    Inputs.push_back(openFile(Path));
  }

  if (Inputs.empty())
    error("no input files.");

  // Create a symbol table.
  SymbolTable Symtab;

  // Parse all input files and put all symbols to the symbol table.
  // The symbol table will take care of name resolution.
  for (MemoryBufferRef MB : Inputs) {
    std::unique_ptr<InputFile> File = createFile(MB);
    Symtab.addFile(std::move(File));
  }

  // Write the result.
  const ELFFileBase *FirstObj = Symtab.getFirstELF();
  switch (FirstObj->getELFKind()) {
  case ELF32LEKind:
    writeResult<object::ELF32LE>(&Symtab);
    return;
  case ELF32BEKind:
    writeResult<object::ELF32BE>(&Symtab);
    return;
  case ELF64LEKind:
    writeResult<object::ELF64LE>(&Symtab);
    return;
  case ELF64BEKind:
    writeResult<object::ELF64BE>(&Symtab);
    return;
  }
}
