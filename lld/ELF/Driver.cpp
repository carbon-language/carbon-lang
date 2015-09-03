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

template <class ELFT>
static std::unique_ptr<InputFile> createFile(MemoryBufferRef MB,
                                             bool IsShared) {
  if (IsShared)
    return make_unique<SharedFile<ELFT>>(MB);
  return make_unique<ObjectFile<ELFT>>(MB);
}

static std::unique_ptr<InputFile> createFile(MemoryBufferRef MB) {
  using namespace llvm::sys::fs;
  file_magic Magic = identify_magic(MB.getBuffer());

  std::pair<unsigned char, unsigned char> Type =
      object::getElfArchType(MB.getBuffer());
  if (Type.second != ELF::ELFDATA2LSB && Type.second != ELF::ELFDATA2MSB)
    error("Invalid data encoding");

  bool IsShared = Magic == file_magic::elf_shared_object;
  if (Type.first == ELF::ELFCLASS32) {
    if (Type.second == ELF::ELFDATA2LSB)
      return createFile<object::ELF32LE>(MB, IsShared);
    return createFile<object::ELF32BE>(MB, IsShared);
  }
  if (Type.first == ELF::ELFCLASS64) {
    if (Type.second == ELF::ELFDATA2LSB)
      return createFile<object::ELF64LE>(MB, IsShared);
    return createFile<object::ELF64BE>(MB, IsShared);
  }
  error("Invalid file class");
}

void LinkerDriver::link(ArrayRef<const char *> ArgsArr) {
  // Parse command line options.
  opt::InputArgList Args = Parser.parse(ArgsArr);

  // Handle -o
  if (auto *Arg = Args.getLastArg(OPT_output))
    Config->OutputFile = Arg->getValue();
  if (Config->OutputFile.empty())
    error("-o must be specified.");

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

  // Make sure we have resolved all symbols.
  Symtab.reportRemainingUndefines();

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
