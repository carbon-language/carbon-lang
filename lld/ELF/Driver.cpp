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
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

Configuration *lld::elf2::Config;
LinkerDriver *lld::elf2::Driver;

void lld::elf2::link(ArrayRef<const char *> Args) {
  Configuration C;
  LinkerDriver D;
  Config = &C;
  Driver = &D;
  Driver->main(Args.slice(1));
}

static void setELFType(StringRef Emul) {
  if (Emul == "elf_i386") {
    Config->ElfKind = ELF32LEKind;
    Config->EMachine = EM_386;
    return;
  }
  if (Emul == "elf_x86_64") {
    Config->ElfKind = ELF64LEKind;
    Config->EMachine = EM_X86_64;
    return;
  }
  if (Emul == "elf32ltsmip") {
    Config->ElfKind = ELF32LEKind;
    Config->EMachine = EM_MIPS;
    return;
  }
  if (Emul == "elf32btsmip") {
    Config->ElfKind = ELF32BEKind;
    Config->EMachine = EM_MIPS;
    return;
  }
  if (Emul == "elf32ppc") {
    Config->ElfKind = ELF32BEKind;
    Config->EMachine = EM_PPC;
    return;
  }
  if (Emul == "elf64ppc") {
    Config->ElfKind = ELF64BEKind;
    Config->EMachine = EM_PPC64;
    return;
  }
  error(Twine("Unknown emulation: ") + Emul);
}

// Makes a path by concatenating Dir and File.
// If Dir starts with '=' the result will be preceded by Sysroot,
// which can be set with --sysroot command line switch.
static std::string buildSysrootedPath(StringRef Dir, StringRef File) {
  SmallString<128> Path;
  if (Dir.startswith("="))
    sys::path::append(Path, Config->Sysroot, Dir.substr(1), File);
  else
    sys::path::append(Path, Dir, File);
  return Path.str().str();
}

// Searches a given library from input search paths, which are filled
// from -L command line switches. Returns a path to an existent library file.
static std::string searchLibrary(StringRef Path) {
  std::vector<std::string> Names;
  if (Path[0] == ':') {
    Names.push_back(Path.drop_front().str());
  } else {
    if (!Config->Static)
      Names.push_back((Twine("lib") + Path + ".so").str());
    Names.push_back((Twine("lib") + Path + ".a").str());
  }
  for (StringRef Dir : Config->InputSearchPaths) {
    for (const std::string &Name : Names) {
      std::string FullPath = buildSysrootedPath(Dir, Name);
      if (sys::fs::exists(FullPath))
        return FullPath;
    }
  }
  error(Twine("Unable to find library -l") + Path);
}

// Opens and parses a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
void LinkerDriver::addFile(StringRef Path) {
  using namespace llvm::sys::fs;
  auto MBOrErr = MemoryBuffer::getFile(Path);
  error(MBOrErr, Twine("cannot open ") + Path);
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take MB ownership

  switch (identify_magic(MBRef.getBuffer())) {
  case file_magic::unknown:
    readLinkerScript(MBRef);
    return;
  case file_magic::archive:
    if (WholeArchive) {
      auto File = make_unique<ArchiveFile>(MBRef);
      for (MemoryBufferRef &MB : File->getMembers())
        Files.push_back(createELFFile<ObjectFile>(MB));
      OwningArchives.emplace_back(std::move(File));
      return;
    }
    Files.push_back(make_unique<ArchiveFile>(MBRef));
    return;
  case file_magic::elf_shared_object:
    Files.push_back(createELFFile<SharedFile>(MBRef));
    return;
  default:
    Files.push_back(createELFFile<ObjectFile>(MBRef));
  }
}

static StringRef
getString(opt::InputArgList &Args, unsigned Key, StringRef Default = "") {
  if (auto *Arg = Args.getLastArg(Key))
    return Arg->getValue();
  return Default;
}

void LinkerDriver::main(ArrayRef<const char *> ArgsArr) {
  initSymbols();

  opt::InputArgList Args = Parser.parse(ArgsArr);
  createFiles(Args);

  switch (Config->ElfKind) {
  case ELF32LEKind:
    link<ELF32LE>(Args);
    return;
  case ELF32BEKind:
    link<ELF32BE>(Args);
    return;
  case ELF64LEKind:
    link<ELF64LE>(Args);
    return;
  case ELF64BEKind:
    link<ELF64BE>(Args);
    return;
  default:
    error("-m or at least a .o file required");
  }
}

void LinkerDriver::createFiles(opt::InputArgList &Args) {
  for (auto *Arg : Args.filtered(OPT_L))
    Config->InputSearchPaths.push_back(Arg->getValue());

  std::vector<StringRef> RPaths;
  for (auto *Arg : Args.filtered(OPT_rpath))
    RPaths.push_back(Arg->getValue());
  if (!RPaths.empty())
    Config->RPath = llvm::join(RPaths.begin(), RPaths.end(), ":");

  if (auto *Arg = Args.getLastArg(OPT_m))
    setELFType(Arg->getValue());

  Config->AllowMultipleDefinition = Args.hasArg(OPT_allow_multiple_definition);
  Config->DiscardAll = Args.hasArg(OPT_discard_all);
  Config->DiscardLocals = Args.hasArg(OPT_discard_locals);
  Config->DiscardNone = Args.hasArg(OPT_discard_none);
  Config->EnableNewDtags = !Args.hasArg(OPT_disable_new_dtags);
  Config->ExportDynamic = Args.hasArg(OPT_export_dynamic);
  Config->NoInhibitExec = Args.hasArg(OPT_noinhibit_exec);
  Config->NoUndefined = Args.hasArg(OPT_no_undefined);
  Config->Shared = Args.hasArg(OPT_shared);

  Config->DynamicLinker = getString(Args, OPT_dynamic_linker);
  Config->Entry = getString(Args, OPT_entry);
  Config->Fini = getString(Args, OPT_fini, "_fini");
  Config->Init = getString(Args, OPT_init, "_init");
  Config->OutputFile = getString(Args, OPT_o);
  Config->SoName = getString(Args, OPT_soname);
  Config->Sysroot = getString(Args, OPT_sysroot);

  for (auto *Arg : Args.filtered(OPT_z))
    if (Arg->getValue() == StringRef("now"))
      Config->ZNow = true;

  for (auto *Arg : Args) {
    switch (Arg->getOption().getID()) {
    case OPT_l:
      addFile(searchLibrary(Arg->getValue()));
      break;
    case OPT_INPUT:
      addFile(Arg->getValue());
      break;
    case OPT_Bstatic:
      Config->Static = true;
      break;
    case OPT_Bdynamic:
      Config->Static = false;
      break;
    case OPT_whole_archive:
      WholeArchive = true;
      break;
    case OPT_no_whole_archive:
      WholeArchive = false;
      break;
    }
  }

  if (Files.empty())
    error("no input files.");

  // Set machine type if -m is not given.
  if (Config->ElfKind == ELFNoneKind) {
    for (std::unique_ptr<InputFile> &File : Files) {
      auto *F = dyn_cast<ELFFileBase>(File.get());
      if (!F)
        continue;
      Config->ElfKind = F->getELFKind();
      Config->EMachine = F->getEMachine();
      break;
    }
  }

  // Check if all files are for the same machine type.
  for (std::unique_ptr<InputFile> &File : Files) {
    auto *F = dyn_cast<ELFFileBase>(File.get());
    if (!F)
      continue;
    if (F->getELFKind() == Config->ElfKind &&
        F->getEMachine() == Config->EMachine)
      continue;
    StringRef A = F->getName();
    StringRef B = Files[0]->getName();
    if (auto *Arg = Args.getLastArg(OPT_m))
      B = Arg->getValue();
    error(A + " is incompatible with " + B);
  }
}

template <class ELFT> void LinkerDriver::link(opt::InputArgList &Args) {
  SymbolTable<ELFT> Symtab;

  for (std::unique_ptr<InputFile> &F : Files)
    Symtab.addFile(std::move(F));

  for (auto *Arg : Args.filtered(OPT_undefined))
    Symtab.addUndefinedSym(Arg->getValue());

  if (Config->OutputFile.empty())
    Config->OutputFile = "a.out";

  // Write the result to the file.
  writeResult<ELFT>(&Symtab);
}
