//===-- llvm-nm.cpp - Symbol table dumping utility for llvm ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like traditional Unix "nm", that is, it
// prints out the names of symbols in a bitcode or object file, along with some
// information about each symbol.
//
// This "nm" supports many of the features of GNU "nm", including its different
// output formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <vector>
using namespace llvm;
using namespace object;

namespace {
enum OutputFormatTy { bsd, sysv, posix };
cl::opt<OutputFormatTy> OutputFormat(
    "format", cl::desc("Specify output format"),
    cl::values(clEnumVal(bsd, "BSD format"), clEnumVal(sysv, "System V format"),
               clEnumVal(posix, "POSIX.2 format"), clEnumValEnd),
    cl::init(bsd));
cl::alias OutputFormat2("f", cl::desc("Alias for --format"),
                        cl::aliasopt(OutputFormat));

cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input bitcode files>"),
                                     cl::ZeroOrMore);

cl::opt<bool> UndefinedOnly("undefined-only",
                            cl::desc("Show only undefined symbols"));
cl::alias UndefinedOnly2("u", cl::desc("Alias for --undefined-only"),
                         cl::aliasopt(UndefinedOnly));

cl::opt<bool> DynamicSyms("dynamic",
                          cl::desc("Display the dynamic symbols instead "
                                   "of normal symbols."));
cl::alias DynamicSyms2("D", cl::desc("Alias for --dynamic"),
                       cl::aliasopt(DynamicSyms));

cl::opt<bool> DefinedOnly("defined-only",
                          cl::desc("Show only defined symbols"));

cl::opt<bool> ExternalOnly("extern-only",
                           cl::desc("Show only external symbols"));
cl::alias ExternalOnly2("g", cl::desc("Alias for --extern-only"),
                        cl::aliasopt(ExternalOnly));

cl::opt<bool> BSDFormat("B", cl::desc("Alias for --format=bsd"));
cl::opt<bool> POSIXFormat("P", cl::desc("Alias for --format=posix"));

cl::opt<bool> PrintFileName(
    "print-file-name",
    cl::desc("Precede each symbol with the object file it came from"));

cl::alias PrintFileNameA("A", cl::desc("Alias for --print-file-name"),
                         cl::aliasopt(PrintFileName));
cl::alias PrintFileNameo("o", cl::desc("Alias for --print-file-name"),
                         cl::aliasopt(PrintFileName));

cl::opt<bool> DebugSyms("debug-syms",
                        cl::desc("Show all symbols, even debugger only"));
cl::alias DebugSymsa("a", cl::desc("Alias for --debug-syms"),
                     cl::aliasopt(DebugSyms));

cl::opt<bool> NumericSort("numeric-sort", cl::desc("Sort symbols by address"));
cl::alias NumericSortn("n", cl::desc("Alias for --numeric-sort"),
                       cl::aliasopt(NumericSort));
cl::alias NumericSortv("v", cl::desc("Alias for --numeric-sort"),
                       cl::aliasopt(NumericSort));

cl::opt<bool> NoSort("no-sort", cl::desc("Show symbols in order encountered"));
cl::alias NoSortp("p", cl::desc("Alias for --no-sort"), cl::aliasopt(NoSort));

cl::opt<bool> PrintSize("print-size",
                        cl::desc("Show symbol size instead of address"));
cl::alias PrintSizeS("S", cl::desc("Alias for --print-size"),
                     cl::aliasopt(PrintSize));

cl::opt<bool> SizeSort("size-sort", cl::desc("Sort symbols by size"));

cl::opt<bool> WithoutAliases("without-aliases", cl::Hidden,
                             cl::desc("Exclude aliases from output"));

cl::opt<bool> ArchiveMap("print-armap", cl::desc("Print the archive map"));
cl::alias ArchiveMaps("s", cl::desc("Alias for --print-armap"),
                      cl::aliasopt(ArchiveMap));
bool PrintAddress = true;

bool MultipleFiles = false;

bool HadError = false;

std::string ToolName;
}

static void error(Twine Message, Twine Path = Twine()) {
  HadError = true;
  errs() << ToolName << ": " << Path << ": " << Message << ".\n";
}

static bool error(error_code EC, Twine Path = Twine()) {
  if (EC) {
    error(EC.message(), Path);
    return true;
  }
  return false;
}

namespace {
struct NMSymbol {
  uint64_t Address;
  uint64_t Size;
  char TypeChar;
  StringRef Name;
};
}

static bool compareSymbolAddress(const NMSymbol &A, const NMSymbol &B) {
  if (A.Address < B.Address)
    return true;
  else if (A.Address == B.Address && A.Name < B.Name)
    return true;
  else if (A.Address == B.Address && A.Name == B.Name && A.Size < B.Size)
    return true;
  else
    return false;
}

static bool compareSymbolSize(const NMSymbol &A, const NMSymbol &B) {
  if (A.Size < B.Size)
    return true;
  else if (A.Size == B.Size && A.Name < B.Name)
    return true;
  else if (A.Size == B.Size && A.Name == B.Name && A.Address < B.Address)
    return true;
  else
    return false;
}

static bool compareSymbolName(const NMSymbol &A, const NMSymbol &B) {
  if (A.Name < B.Name)
    return true;
  else if (A.Name == B.Name && A.Size < B.Size)
    return true;
  else if (A.Name == B.Name && A.Size == B.Size && A.Address < B.Address)
    return true;
  else
    return false;
}

static StringRef CurrentFilename;
typedef std::vector<NMSymbol> SymbolListT;
static SymbolListT SymbolList;

static void sortAndPrintSymbolList() {
  if (!NoSort) {
    if (NumericSort)
      std::sort(SymbolList.begin(), SymbolList.end(), compareSymbolAddress);
    else if (SizeSort)
      std::sort(SymbolList.begin(), SymbolList.end(), compareSymbolSize);
    else
      std::sort(SymbolList.begin(), SymbolList.end(), compareSymbolName);
  }

  if (OutputFormat == posix && MultipleFiles) {
    outs() << '\n' << CurrentFilename << ":\n";
  } else if (OutputFormat == bsd && MultipleFiles) {
    outs() << "\n" << CurrentFilename << ":\n";
  } else if (OutputFormat == sysv) {
    outs() << "\n\nSymbols from " << CurrentFilename << ":\n\n"
           << "Name                  Value   Class        Type"
           << "         Size   Line  Section\n";
  }

  for (SymbolListT::iterator I = SymbolList.begin(), E = SymbolList.end();
       I != E; ++I) {
    if ((I->TypeChar != 'U') && UndefinedOnly)
      continue;
    if ((I->TypeChar == 'U') && DefinedOnly)
      continue;
    if (SizeSort && !PrintAddress && I->Size == UnknownAddressOrSize)
      continue;

    char SymbolAddrStr[10] = "";
    char SymbolSizeStr[10] = "";

    if (OutputFormat == sysv || I->Address == object::UnknownAddressOrSize)
      strcpy(SymbolAddrStr, "        ");
    if (OutputFormat == sysv)
      strcpy(SymbolSizeStr, "        ");

    if (I->Address != object::UnknownAddressOrSize)
      format("%08" PRIx64, I->Address)
          .print(SymbolAddrStr, sizeof(SymbolAddrStr));
    if (I->Size != object::UnknownAddressOrSize)
      format("%08" PRIx64, I->Size).print(SymbolSizeStr, sizeof(SymbolSizeStr));

    if (OutputFormat == posix) {
      outs() << I->Name << " " << I->TypeChar << " " << SymbolAddrStr
             << SymbolSizeStr << "\n";
    } else if (OutputFormat == bsd) {
      if (PrintAddress)
        outs() << SymbolAddrStr << ' ';
      if (PrintSize) {
        outs() << SymbolSizeStr;
        if (I->Size != object::UnknownAddressOrSize)
          outs() << ' ';
      }
      outs() << I->TypeChar << " " << I->Name << "\n";
    } else if (OutputFormat == sysv) {
      std::string PaddedName(I->Name);
      while (PaddedName.length() < 20)
        PaddedName += " ";
      outs() << PaddedName << "|" << SymbolAddrStr << "|   " << I->TypeChar
             << "  |                  |" << SymbolSizeStr << "|     |\n";
    }
  }

  SymbolList.clear();
}

static char typeCharForSymbol(GlobalValue &GV) {
  if (GV.isDeclaration())
    return 'U';
  if (GV.hasLinkOnceLinkage())
    return 'C';
  if (GV.hasCommonLinkage())
    return 'C';
  if (GV.hasWeakLinkage())
    return 'W';
  if (isa<Function>(GV) && GV.hasInternalLinkage())
    return 't';
  if (isa<Function>(GV))
    return 'T';
  if (isa<GlobalVariable>(GV) && GV.hasInternalLinkage())
    return 'd';
  if (isa<GlobalVariable>(GV))
    return 'D';
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(&GV)) {
    const GlobalValue *AliasedGV = GA->getAliasedGlobal();
    if (isa<Function>(AliasedGV))
      return 'T';
    if (isa<GlobalVariable>(AliasedGV))
      return 'D';
  }
  return '?';
}

static void dumpSymbolNameForGlobalValue(GlobalValue &GV) {
  // Private linkage and available_externally linkage don't exist in symtab.
  if (GV.hasPrivateLinkage() || GV.hasLinkerPrivateLinkage() ||
      GV.hasLinkerPrivateWeakLinkage() || GV.hasAvailableExternallyLinkage())
    return;
  char TypeChar = typeCharForSymbol(GV);
  if (GV.hasLocalLinkage() && ExternalOnly)
    return;

  NMSymbol S;
  S.Address = object::UnknownAddressOrSize;
  S.Size = object::UnknownAddressOrSize;
  S.TypeChar = TypeChar;
  S.Name = GV.getName();
  SymbolList.push_back(S);
}

static void dumpSymbolNamesFromModule(Module *M) {
  CurrentFilename = M->getModuleIdentifier();
  std::for_each(M->begin(), M->end(), dumpSymbolNameForGlobalValue);
  std::for_each(M->global_begin(), M->global_end(),
                dumpSymbolNameForGlobalValue);
  if (!WithoutAliases)
    std::for_each(M->alias_begin(), M->alias_end(),
                  dumpSymbolNameForGlobalValue);

  sortAndPrintSymbolList();
}

template <class ELFT>
static error_code getSymbolNMTypeChar(ELFObjectFile<ELFT> &Obj,
                                      symbol_iterator I, char &Result) {
  typedef typename ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;

  DataRefImpl Symb = I->getRawDataRefImpl();
  const Elf_Sym *ESym = Obj.getSymbol(Symb);
  const ELFFile<ELFT> &EF = *Obj.getELFFile();
  const Elf_Shdr *ESec = EF.getSection(ESym);

  char Ret = '?';

  if (ESec) {
    switch (ESec->sh_type) {
    case ELF::SHT_PROGBITS:
    case ELF::SHT_DYNAMIC:
      switch (ESec->sh_flags) {
      case(ELF::SHF_ALLOC | ELF::SHF_EXECINSTR) :
        Ret = 't';
        break;
      case(ELF::SHF_TLS | ELF::SHF_ALLOC | ELF::SHF_WRITE) :
      case(ELF::SHF_ALLOC | ELF::SHF_WRITE) :
        Ret = 'd';
        break;
      case ELF::SHF_ALLOC:
      case(ELF::SHF_ALLOC | ELF::SHF_MERGE) :
      case(ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS) :
        Ret = 'r';
        break;
      }
      break;
    case ELF::SHT_NOBITS:
      Ret = 'b';
    }
  }

  switch (EF.getSymbolTableIndex(ESym)) {
  case ELF::SHN_UNDEF:
    if (Ret == '?')
      Ret = 'U';
    break;
  case ELF::SHN_ABS:
    Ret = 'a';
    break;
  case ELF::SHN_COMMON:
    Ret = 'c';
    break;
  }

  switch (ESym->getBinding()) {
  case ELF::STB_GLOBAL:
    Ret = ::toupper(Ret);
    break;
  case ELF::STB_WEAK:
    if (EF.getSymbolTableIndex(ESym) == ELF::SHN_UNDEF)
      Ret = 'w';
    else if (ESym->getType() == ELF::STT_OBJECT)
      Ret = 'V';
    else
      Ret = 'W';
  }

  if (Ret == '?' && ESym->getType() == ELF::STT_SECTION) {
    StringRef Name;
    error_code EC = I->getName(Name);
    if (EC)
      return EC;
    Result = StringSwitch<char>(Name)
                 .StartsWith(".debug", 'N')
                 .StartsWith(".note", 'n')
                 .Default('?');
    return object_error::success;
  }

  Result = Ret;
  return object_error::success;
}

static error_code getSymbolNMTypeChar(COFFObjectFile &Obj, symbol_iterator I,
                                      char &Result) {
  const coff_symbol *Symb = Obj.getCOFFSymbol(I);
  StringRef Name;
  if (error_code EC = I->getName(Name))
    return EC;
  char Ret = StringSwitch<char>(Name)
                 .StartsWith(".debug", 'N')
                 .StartsWith(".sxdata", 'N')
                 .Default('?');

  if (Ret != '?') {
    Result = Ret;
    return object_error::success;
  }

  uint32_t Characteristics = 0;
  if (Symb->SectionNumber > 0) {
    section_iterator SecI = Obj.end_sections();
    if (error_code EC = I->getSection(SecI))
      return EC;
    const coff_section *Section = Obj.getCOFFSection(SecI);
    Characteristics = Section->Characteristics;
  }

  switch (Symb->SectionNumber) {
  case COFF::IMAGE_SYM_UNDEFINED:
    // Check storage classes.
    if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL) {
      Result = 'w';
      return object_error::success; // Don't do ::toupper.
    } else if (Symb->Value != 0)    // Check for common symbols.
      Ret = 'c';
    else
      Ret = 'u';
    break;
  case COFF::IMAGE_SYM_ABSOLUTE:
    Ret = 'a';
    break;
  case COFF::IMAGE_SYM_DEBUG:
    Ret = 'n';
    break;
  default:
    // Check section type.
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      Ret = 't';
    else if (Characteristics & COFF::IMAGE_SCN_MEM_READ &&
             ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
      Ret = 'r';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      Ret = 'd';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      Ret = 'b';
    else if (Characteristics & COFF::IMAGE_SCN_LNK_INFO)
      Ret = 'i';

    // Check for section symbol.
    else if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_STATIC &&
             Symb->Value == 0)
      Ret = 's';
  }

  if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    Ret = ::toupper(static_cast<unsigned char>(Ret));

  Result = Ret;
  return object_error::success;
}

static uint8_t getNType(MachOObjectFile &Obj, DataRefImpl Symb) {
  if (Obj.is64Bit()) {
    MachO::nlist_64 STE = Obj.getSymbol64TableEntry(Symb);
    return STE.n_type;
  }
  MachO::nlist STE = Obj.getSymbolTableEntry(Symb);
  return STE.n_type;
}

static error_code getSymbolNMTypeChar(MachOObjectFile &Obj, symbol_iterator I,
                                      char &Res) {
  DataRefImpl Symb = I->getRawDataRefImpl();
  uint8_t NType = getNType(Obj, Symb);

  char Char;
  switch (NType & MachO::N_TYPE) {
  case MachO::N_UNDF:
    Char = 'u';
    break;
  case MachO::N_ABS:
    Char = 's';
    break;
  case MachO::N_SECT: {
    section_iterator Sec = Obj.end_sections();
    Obj.getSymbolSection(Symb, Sec);
    DataRefImpl Ref = Sec->getRawDataRefImpl();
    StringRef SectionName;
    Obj.getSectionName(Ref, SectionName);
    StringRef SegmentName = Obj.getSectionFinalSegmentName(Ref);
    if (SegmentName == "__TEXT" && SectionName == "__text")
      Char = 't';
    else
      Char = 's';
  } break;
  default:
    Char = '?';
    break;
  }

  if (NType & (MachO::N_EXT | MachO::N_PEXT))
    Char = toupper(static_cast<unsigned char>(Char));
  Res = Char;
  return object_error::success;
}

static char getNMTypeChar(ObjectFile *Obj, symbol_iterator I) {
  char Res = '?';
  if (COFFObjectFile *COFF = dyn_cast<COFFObjectFile>(Obj)) {
    error(getSymbolNMTypeChar(*COFF, I, Res));
    return Res;
  }
  if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(Obj)) {
    error(getSymbolNMTypeChar(*MachO, I, Res));
    return Res;
  }

  if (ELF32LEObjectFile *ELF = dyn_cast<ELF32LEObjectFile>(Obj)) {
    error(getSymbolNMTypeChar(*ELF, I, Res));
    return Res;
  }
  if (ELF64LEObjectFile *ELF = dyn_cast<ELF64LEObjectFile>(Obj)) {
    error(getSymbolNMTypeChar(*ELF, I, Res));
    return Res;
  }
  if (ELF32BEObjectFile *ELF = dyn_cast<ELF32BEObjectFile>(Obj)) {
    error(getSymbolNMTypeChar(*ELF, I, Res));
    return Res;
  }
  ELF64BEObjectFile *ELF = cast<ELF64BEObjectFile>(Obj);
  error(getSymbolNMTypeChar(*ELF, I, Res));
  return Res;
}

static void dumpSymbolNamesFromObject(ObjectFile *Obj) {
  symbol_iterator IBegin = Obj->begin_symbols();
  symbol_iterator IEnd = Obj->end_symbols();
  if (DynamicSyms) {
    IBegin = Obj->begin_dynamic_symbols();
    IEnd = Obj->end_dynamic_symbols();
  }
  for (symbol_iterator I = IBegin; I != IEnd; ++I) {
    uint32_t SymFlags;
    if (error(I->getFlags(SymFlags)))
      break;
    if (!DebugSyms && (SymFlags & SymbolRef::SF_FormatSpecific))
      continue;
    NMSymbol S;
    S.Size = object::UnknownAddressOrSize;
    S.Address = object::UnknownAddressOrSize;
    if (PrintSize || SizeSort) {
      if (error(I->getSize(S.Size)))
        break;
    }
    if (PrintAddress)
      if (error(I->getAddress(S.Address)))
        break;
    S.TypeChar = getNMTypeChar(Obj, I);
    if (error(I->getName(S.Name)))
      break;
    SymbolList.push_back(S);
  }

  CurrentFilename = Obj->getFileName();
  sortAndPrintSymbolList();
}

static void dumpSymbolNamesFromFile(std::string &Filename) {
  OwningPtr<MemoryBuffer> Buffer;
  if (error(MemoryBuffer::getFileOrSTDIN(Filename, Buffer), Filename))
    return;

  sys::fs::file_magic Magic = sys::fs::identify_magic(Buffer->getBuffer());

  LLVMContext &Context = getGlobalContext();
  if (Magic == sys::fs::file_magic::bitcode) {
    ErrorOr<Module *> ModuleOrErr = parseBitcodeFile(Buffer.get(), Context);
    if (error(ModuleOrErr.getError(), Filename)) {
      return;
    } else {
      Module *Result = ModuleOrErr.get();
      dumpSymbolNamesFromModule(Result);
      delete Result;
    }
    return;
  }

  ErrorOr<Binary *> BinaryOrErr = object::createBinary(Buffer.take(), Magic);
  if (error(BinaryOrErr.getError(), Filename))
    return;
  OwningPtr<Binary> Bin(BinaryOrErr.get());

  if (object::Archive *A = dyn_cast<object::Archive>(Bin.get())) {
    if (ArchiveMap) {
      object::Archive::symbol_iterator I = A->symbol_begin();
      object::Archive::symbol_iterator E = A->symbol_end();
      if (I != E) {
        outs() << "Archive map"
               << "\n";
        for (; I != E; ++I) {
          object::Archive::child_iterator C;
          StringRef SymName;
          StringRef FileName;
          if (error(I->getMember(C)))
            return;
          if (error(I->getName(SymName)))
            return;
          if (error(C->getName(FileName)))
            return;
          outs() << SymName << " in " << FileName << "\n";
        }
        outs() << "\n";
      }
    }

    for (object::Archive::child_iterator I = A->child_begin(),
                                         E = A->child_end();
         I != E; ++I) {
      OwningPtr<Binary> Child;
      if (I->getAsBinary(Child)) {
        // Try opening it as a bitcode file.
        OwningPtr<MemoryBuffer> Buff;
        if (error(I->getMemoryBuffer(Buff)))
          return;

        ErrorOr<Module *> ModuleOrErr = parseBitcodeFile(Buff.get(), Context);
        if (ModuleOrErr) {
          Module *Result = ModuleOrErr.get();
          dumpSymbolNamesFromModule(Result);
          delete Result;
        }
        continue;
      }
      if (object::ObjectFile *O = dyn_cast<ObjectFile>(Child.get())) {
        outs() << O->getFileName() << ":\n";
        dumpSymbolNamesFromObject(O);
      }
    }
    return;
  }
  if (object::MachOUniversalBinary *UB =
          dyn_cast<object::MachOUniversalBinary>(Bin.get())) {
    for (object::MachOUniversalBinary::object_iterator I = UB->begin_objects(),
                                                       E = UB->end_objects();
         I != E; ++I) {
      OwningPtr<ObjectFile> Obj;
      if (!I->getAsObjectFile(Obj)) {
        outs() << Obj->getFileName() << ":\n";
        dumpSymbolNamesFromObject(Obj.get());
      }
    }
    return;
  }
  if (object::ObjectFile *O = dyn_cast<ObjectFile>(Bin.get())) {
    dumpSymbolNamesFromObject(O);
    return;
  }
  error("unrecognizable file type", Filename);
  return;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm symbol table dumper\n");

  // llvm-nm only reads binary files.
  if (error(sys::ChangeStdinToBinary()))
    return 1;

  ToolName = argv[0];
  if (BSDFormat)
    OutputFormat = bsd;
  if (POSIXFormat)
    OutputFormat = posix;

  // The relative order of these is important. If you pass --size-sort it should
  // only print out the size. However, if you pass -S --size-sort, it should
  // print out both the size and address.
  if (SizeSort && !PrintSize)
    PrintAddress = false;
  if (OutputFormat == sysv || SizeSort)
    PrintSize = true;

  switch (InputFilenames.size()) {
  case 0:
    InputFilenames.push_back("-");
  case 1:
    break;
  default:
    MultipleFiles = true;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                dumpSymbolNamesFromFile);

  if (HadError)
    return 1;

  return 0;
}
