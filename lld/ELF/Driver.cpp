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
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"

using namespace llvm;

using namespace lld;
using namespace lld::elf2;

namespace lld {
namespace elf2 {
Configuration *Config;
LinkerDriver *Driver;

void link(ArrayRef<const char *> Args) {
  auto C = make_unique<Configuration>();
  Config = C.get();
  auto D = make_unique<LinkerDriver>();
  Driver = D.get();
  Driver->link(Args.slice(1));
}

void error(Twine Msg) {
  errs() << Msg << "\n";
  exit(1);
}

void error(std::error_code EC, Twine Prefix) {
  if (!EC)
    return;
  error(Prefix + ": " + EC.message());
}

void error(std::error_code EC) {
  if (!EC)
    return;
  error(EC.message());
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
  std::pair<unsigned char, unsigned char> Type =
      object::getElfArchType(MB.getBuffer());
  if (Type.second != ELF::ELFDATA2LSB && Type.second != ELF::ELFDATA2MSB)
    error("Invalid data encoding");

  if (Type.first == ELF::ELFCLASS32) {
    if (Type.second == ELF::ELFDATA2LSB)
      return make_unique<ObjectFile<object::ELF32LE>>(MB);
    return make_unique<ObjectFile<object::ELF32BE>>(MB);
  }
  if (Type.first == ELF::ELFCLASS64) {
    if (Type.second == ELF::ELFDATA2LSB)
      return make_unique<ObjectFile<object::ELF64LE>>(MB);
    return make_unique<ObjectFile<object::ELF64BE>>(MB);
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
  ObjectFileBase &FirstObj = *Symtab.ObjectFiles[0];
  switch (FirstObj.kind()) {
  case InputFile::Object32LEKind: {
    Writer<object::ELF32LE> Out(&Symtab);
    Out.write(Config->OutputFile);
    return;
  }
  case InputFile::Object32BEKind: {
    Writer<object::ELF32BE> Out(&Symtab);
    Out.write(Config->OutputFile);
    return;
  }
  case InputFile::Object64LEKind: {
    Writer<object::ELF64LE> Out(&Symtab);
    Out.write(Config->OutputFile);
    return;
  }
  case InputFile::Object64BEKind: {
    Writer<object::ELF64BE> Out(&Symtab);
    Out.write(Config->OutputFile);
    return;
  }
  }
}
