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

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/IRObjectFile.h"
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

cl::list<std::string> InputFilenames(cl::Positional, cl::desc("<input files>"),
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

    if (OutputFormat == sysv || I->Address == UnknownAddressOrSize)
      strcpy(SymbolAddrStr, "        ");
    if (OutputFormat == sysv)
      strcpy(SymbolSizeStr, "        ");

    if (I->Address != UnknownAddressOrSize)
      format("%08" PRIx64, I->Address)
          .print(SymbolAddrStr, sizeof(SymbolAddrStr));
    if (I->Size != UnknownAddressOrSize)
      format("%08" PRIx64, I->Size).print(SymbolSizeStr, sizeof(SymbolSizeStr));

    if (OutputFormat == posix) {
      outs() << I->Name << " " << I->TypeChar << " " << SymbolAddrStr
             << SymbolSizeStr << "\n";
    } else if (OutputFormat == bsd) {
      if (PrintAddress)
        outs() << SymbolAddrStr << ' ';
      if (PrintSize) {
        outs() << SymbolSizeStr;
        if (I->Size != UnknownAddressOrSize)
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

template <class ELFT>
static char getSymbolNMTypeChar(ELFObjectFile<ELFT> &Obj,
                                basic_symbol_iterator I) {
  typedef typename ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;

  // OK, this is ELF
  symbol_iterator SymI(I);

  DataRefImpl Symb = I->getRawDataRefImpl();
  const Elf_Sym *ESym = Obj.getSymbol(Symb);
  const ELFFile<ELFT> &EF = *Obj.getELFFile();
  const Elf_Shdr *ESec = EF.getSection(ESym);

  if (ESec) {
    switch (ESec->sh_type) {
    case ELF::SHT_PROGBITS:
    case ELF::SHT_DYNAMIC:
      switch (ESec->sh_flags) {
      case(ELF::SHF_ALLOC | ELF::SHF_EXECINSTR) :
        return 't';
      case(ELF::SHF_TLS | ELF::SHF_ALLOC | ELF::SHF_WRITE) :
      case(ELF::SHF_ALLOC | ELF::SHF_WRITE) :
        return 'd';
      case ELF::SHF_ALLOC:
      case(ELF::SHF_ALLOC | ELF::SHF_MERGE) :
      case(ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS) :
        return 'r';
      }
      break;
    case ELF::SHT_NOBITS:
      return 'b';
    }
  }

  if (ESym->getType() == ELF::STT_SECTION) {
    StringRef Name;
    if (error(SymI->getName(Name)))
      return '?';
    return StringSwitch<char>(Name)
        .StartsWith(".debug", 'N')
        .StartsWith(".note", 'n')
        .Default('?');
  }

  return '?';
}

static char getSymbolNMTypeChar(COFFObjectFile &Obj, symbol_iterator I) {
  const coff_symbol *Symb = Obj.getCOFFSymbol(I);
  // OK, this is COFF.
  symbol_iterator SymI(I);

  StringRef Name;
  if (error(SymI->getName(Name)))
    return '?';

  char Ret = StringSwitch<char>(Name)
                 .StartsWith(".debug", 'N')
                 .StartsWith(".sxdata", 'N')
                 .Default('?');

  if (Ret != '?')
    return Ret;

  uint32_t Characteristics = 0;
  if (Symb->SectionNumber > 0) {
    section_iterator SecI = Obj.section_end();
    if (error(SymI->getSection(SecI)))
      return '?';
    const coff_section *Section = Obj.getCOFFSection(SecI);
    Characteristics = Section->Characteristics;
  }

  switch (Symb->SectionNumber) {
  case COFF::IMAGE_SYM_DEBUG:
    return 'n';
  default:
    // Check section type.
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      return 't';
    else if (Characteristics & COFF::IMAGE_SCN_MEM_READ &&
             ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
      return 'r';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      return 'd';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      return 'b';
    else if (Characteristics & COFF::IMAGE_SCN_LNK_INFO)
      return 'i';

    // Check for section symbol.
    else if (Symb->StorageClass == COFF::IMAGE_SYM_CLASS_STATIC &&
             Symb->Value == 0)
      return 's';
  }

  return '?';
}

static uint8_t getNType(MachOObjectFile &Obj, DataRefImpl Symb) {
  if (Obj.is64Bit()) {
    MachO::nlist_64 STE = Obj.getSymbol64TableEntry(Symb);
    return STE.n_type;
  }
  MachO::nlist STE = Obj.getSymbolTableEntry(Symb);
  return STE.n_type;
}

static char getSymbolNMTypeChar(MachOObjectFile &Obj, basic_symbol_iterator I) {
  DataRefImpl Symb = I->getRawDataRefImpl();
  uint8_t NType = getNType(Obj, Symb);

  switch (NType & MachO::N_TYPE) {
  case MachO::N_ABS:
    return 's';
  case MachO::N_SECT: {
    section_iterator Sec = Obj.section_end();
    Obj.getSymbolSection(Symb, Sec);
    DataRefImpl Ref = Sec->getRawDataRefImpl();
    StringRef SectionName;
    Obj.getSectionName(Ref, SectionName);
    StringRef SegmentName = Obj.getSectionFinalSegmentName(Ref);
    if (SegmentName == "__TEXT" && SectionName == "__text")
      return 't';
    else
      return 's';
  }
  }

  return '?';
}

static char getSymbolNMTypeChar(const GlobalValue &GV) {
  if (isa<Function>(GV))
    return 't';
  // FIXME: should we print 'b'? At the IR level we cannot be sure if this
  // will be in bss or not, but we could approximate.
  if (isa<GlobalVariable>(GV))
    return 'd';
  const GlobalAlias *GA = cast<GlobalAlias>(&GV);
  const GlobalValue *AliasedGV = GA->getAliasedGlobal();
  return getSymbolNMTypeChar(*AliasedGV);
}

static char getSymbolNMTypeChar(IRObjectFile &Obj, basic_symbol_iterator I) {
  const GlobalValue &GV = Obj.getSymbolGV(I->getRawDataRefImpl());
  return getSymbolNMTypeChar(GV);
}

template <class ELFT>
static bool isObject(ELFObjectFile<ELFT> &Obj, symbol_iterator I) {
  typedef typename ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;

  DataRefImpl Symb = I->getRawDataRefImpl();
  const Elf_Sym *ESym = Obj.getSymbol(Symb);

  return ESym->getType() == ELF::STT_OBJECT;
}

static bool isObject(SymbolicFile *Obj, basic_symbol_iterator I) {
  if (ELF32LEObjectFile *ELF = dyn_cast<ELF32LEObjectFile>(Obj))
    return isObject(*ELF, I);
  if (ELF64LEObjectFile *ELF = dyn_cast<ELF64LEObjectFile>(Obj))
    return isObject(*ELF, I);
  if (ELF32BEObjectFile *ELF = dyn_cast<ELF32BEObjectFile>(Obj))
    return isObject(*ELF, I);
  if (ELF64BEObjectFile *ELF = dyn_cast<ELF64BEObjectFile>(Obj))
    return isObject(*ELF, I);
  return false;
}

static char getNMTypeChar(SymbolicFile *Obj, basic_symbol_iterator I) {
  uint32_t Symflags = I->getFlags();
  if ((Symflags & object::SymbolRef::SF_Weak) && !isa<MachOObjectFile>(Obj)) {
    char Ret = isObject(Obj, I) ? 'v' : 'w';
    if (!(Symflags & object::SymbolRef::SF_Undefined))
      Ret = toupper(Ret);
    return Ret;
  }

  if (Symflags & object::SymbolRef::SF_Undefined)
    return 'U';

  if (Symflags & object::SymbolRef::SF_Common)
    return 'C';

  char Ret = '?';
  if (Symflags & object::SymbolRef::SF_Absolute)
    Ret = 'a';
  else if (IRObjectFile *IR = dyn_cast<IRObjectFile>(Obj))
    Ret = getSymbolNMTypeChar(*IR, I);
  else if (COFFObjectFile *COFF = dyn_cast<COFFObjectFile>(Obj))
    Ret = getSymbolNMTypeChar(*COFF, I);
  else if (MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(Obj))
    Ret = getSymbolNMTypeChar(*MachO, I);
  else if (ELF32LEObjectFile *ELF = dyn_cast<ELF32LEObjectFile>(Obj))
    Ret = getSymbolNMTypeChar(*ELF, I);
  else if (ELF64LEObjectFile *ELF = dyn_cast<ELF64LEObjectFile>(Obj))
    Ret = getSymbolNMTypeChar(*ELF, I);
  else if (ELF32BEObjectFile *ELF = dyn_cast<ELF32BEObjectFile>(Obj))
    Ret = getSymbolNMTypeChar(*ELF, I);
  else
    Ret = getSymbolNMTypeChar(*cast<ELF64BEObjectFile>(Obj), I);

  if (Symflags & object::SymbolRef::SF_Global)
    Ret = toupper(Ret);

  return Ret;
}

static void dumpSymbolNamesFromObject(SymbolicFile *Obj) {
  basic_symbol_iterator IBegin = Obj->symbol_begin();
  basic_symbol_iterator IEnd = Obj->symbol_end();
  if (DynamicSyms) {
    if (!Obj->isELF()) {
      error("File format has no dynamic symbol table", Obj->getFileName());
      return;
    }
    std::pair<symbol_iterator, symbol_iterator> IDyn =
        getELFDynamicSymbolIterators(Obj);
    IBegin = IDyn.first;
    IEnd = IDyn.second;
  }
  std::string NameBuffer;
  raw_string_ostream OS(NameBuffer);
  for (basic_symbol_iterator I = IBegin; I != IEnd; ++I) {
    uint32_t SymFlags = I->getFlags();
    if (!DebugSyms && (SymFlags & SymbolRef::SF_FormatSpecific))
      continue;
    if (WithoutAliases) {
      if (IRObjectFile *IR = dyn_cast<IRObjectFile>(Obj)) {
        const GlobalValue &GV = IR->getSymbolGV(I->getRawDataRefImpl());
        if(isa<GlobalAlias>(GV))
          continue;
      }
    }
    NMSymbol S;
    S.Size = UnknownAddressOrSize;
    S.Address = UnknownAddressOrSize;
    if ((PrintSize || SizeSort) && isa<ObjectFile>(Obj)) {
      symbol_iterator SymI = I;
      if (error(SymI->getSize(S.Size)))
        break;
    }
    if (PrintAddress && isa<ObjectFile>(Obj))
      if (error(symbol_iterator(I)->getAddress(S.Address)))
        break;
    S.TypeChar = getNMTypeChar(Obj, I);
    if (error(I->printName(OS)))
      break;
    OS << '\0';
    SymbolList.push_back(S);
  }

  OS.flush();
  const char *P = NameBuffer.c_str();
  for (unsigned I = 0; I < SymbolList.size(); ++I) {
    SymbolList[I].Name = P;
    P += strlen(P) + 1;
  }

  CurrentFilename = Obj->getFileName();
  sortAndPrintSymbolList();
}

static void dumpSymbolNamesFromFile(std::string &Filename) {
  std::unique_ptr<MemoryBuffer> Buffer;
  if (error(MemoryBuffer::getFileOrSTDIN(Filename, Buffer), Filename))
    return;

  LLVMContext &Context = getGlobalContext();
  ErrorOr<Binary *> BinaryOrErr = createBinary(Buffer.release(), &Context);
  if (error(BinaryOrErr.getError(), Filename))
    return;
  std::unique_ptr<Binary> Bin(BinaryOrErr.get());

  if (Archive *A = dyn_cast<Archive>(Bin.get())) {
    if (ArchiveMap) {
      Archive::symbol_iterator I = A->symbol_begin();
      Archive::symbol_iterator E = A->symbol_end();
      if (I != E) {
        outs() << "Archive map\n";
        for (; I != E; ++I) {
          Archive::child_iterator C;
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

    for (Archive::child_iterator I = A->child_begin(), E = A->child_end();
         I != E; ++I) {
      std::unique_ptr<Binary> Child;
      if (I->getAsBinary(Child, &Context))
        continue;
      if (SymbolicFile *O = dyn_cast<SymbolicFile>(Child.get())) {
        outs() << O->getFileName() << ":\n";
        dumpSymbolNamesFromObject(O);
      }
    }
    return;
  }
  if (MachOUniversalBinary *UB = dyn_cast<MachOUniversalBinary>(Bin.get())) {
    for (MachOUniversalBinary::object_iterator I = UB->begin_objects(),
                                               E = UB->end_objects();
         I != E; ++I) {
      std::unique_ptr<ObjectFile> Obj;
      if (!I->getAsObjectFile(Obj)) {
        outs() << Obj->getFileName() << ":\n";
        dumpSymbolNamesFromObject(Obj.get());
      }
    }
    return;
  }
  if (SymbolicFile *O = dyn_cast<SymbolicFile>(Bin.get())) {
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
