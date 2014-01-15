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
  cl::opt<OutputFormatTy>
  OutputFormat("format",
       cl::desc("Specify output format"),
         cl::values(clEnumVal(bsd,   "BSD format"),
                    clEnumVal(sysv,  "System V format"),
                    clEnumVal(posix, "POSIX.2 format"),
                    clEnumValEnd), cl::init(bsd));
  cl::alias OutputFormat2("f", cl::desc("Alias for --format"),
                          cl::aliasopt(OutputFormat));

  cl::list<std::string>
  InputFilenames(cl::Positional, cl::desc("<input bitcode files>"),
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

  cl::opt<bool> PrintFileName("print-file-name",
    cl::desc("Precede each symbol with the object file it came from"));

  cl::alias PrintFileNameA("A", cl::desc("Alias for --print-file-name"),
                                cl::aliasopt(PrintFileName));
  cl::alias PrintFileNameo("o", cl::desc("Alias for --print-file-name"),
                                cl::aliasopt(PrintFileName));

  cl::opt<bool> DebugSyms("debug-syms",
    cl::desc("Show all symbols, even debugger only"));
  cl::alias DebugSymsa("a", cl::desc("Alias for --debug-syms"),
                            cl::aliasopt(DebugSyms));

  cl::opt<bool> NumericSort("numeric-sort",
    cl::desc("Sort symbols by address"));
  cl::alias NumericSortn("n", cl::desc("Alias for --numeric-sort"),
                              cl::aliasopt(NumericSort));
  cl::alias NumericSortv("v", cl::desc("Alias for --numeric-sort"),
                              cl::aliasopt(NumericSort));

  cl::opt<bool> NoSort("no-sort",
    cl::desc("Show symbols in order encountered"));
  cl::alias NoSortp("p", cl::desc("Alias for --no-sort"),
                         cl::aliasopt(NoSort));

  cl::opt<bool> PrintSize("print-size",
    cl::desc("Show symbol size instead of address"));
  cl::alias PrintSizeS("S", cl::desc("Alias for --print-size"),
                            cl::aliasopt(PrintSize));

  cl::opt<bool> SizeSort("size-sort", cl::desc("Sort symbols by size"));

  cl::opt<bool> WithoutAliases("without-aliases", cl::Hidden,
                               cl::desc("Exclude aliases from output"));

  cl::opt<bool> ArchiveMap("print-armap",
    cl::desc("Print the archive map"));
  cl::alias ArchiveMaps("s", cl::desc("Alias for --print-armap"),
                                 cl::aliasopt(ArchiveMap));
  bool PrintAddress = true;

  bool MultipleFiles = false;

  bool HadError = false;

  std::string ToolName;
}


static void error(Twine message, Twine path = Twine()) {
  errs() << ToolName << ": " << path << ": " << message << ".\n";
}

static bool error(error_code ec, Twine path = Twine()) {
  if (ec) {
    error(ec.message(), path);
    HadError = true;
    return true;
  }
  return false;
}

namespace {
  struct NMSymbol {
    uint64_t  Address;
    uint64_t  Size;
    char      TypeChar;
    StringRef Name;
  };

  static bool CompareSymbolAddress(const NMSymbol &a, const NMSymbol &b) {
    if (a.Address < b.Address)
      return true;
    else if (a.Address == b.Address && a.Name < b.Name)
      return true;
    else if (a.Address == b.Address && a.Name == b.Name && a.Size < b.Size)
      return true;
    else
      return false;

  }

  static bool CompareSymbolSize(const NMSymbol &a, const NMSymbol &b) {
    if (a.Size < b.Size)
      return true;
    else if (a.Size == b.Size && a.Name < b.Name)
      return true;
    else if (a.Size == b.Size && a.Name == b.Name && a.Address < b.Address)
      return true;
    else
      return false;
  }

  static bool CompareSymbolName(const NMSymbol &a, const NMSymbol &b) {
    if (a.Name < b.Name)
      return true;
    else if (a.Name == b.Name && a.Size < b.Size)
      return true;
    else if (a.Name == b.Name && a.Size == b.Size && a.Address < b.Address)
      return true;
    else
      return false;
  }

  StringRef CurrentFilename;
  typedef std::vector<NMSymbol> SymbolListT;
  SymbolListT SymbolList;
}

static void SortAndPrintSymbolList() {
  if (!NoSort) {
    if (NumericSort)
      std::sort(SymbolList.begin(), SymbolList.end(), CompareSymbolAddress);
    else if (SizeSort)
      std::sort(SymbolList.begin(), SymbolList.end(), CompareSymbolSize);
    else
      std::sort(SymbolList.begin(), SymbolList.end(), CompareSymbolName);
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

  for (SymbolListT::iterator i = SymbolList.begin(),
                             e = SymbolList.end(); i != e; ++i) {
    if ((i->TypeChar != 'U') && UndefinedOnly)
      continue;
    if ((i->TypeChar == 'U') && DefinedOnly)
      continue;
    if (SizeSort && !PrintAddress && i->Size == UnknownAddressOrSize)
      continue;

    char SymbolAddrStr[10] = "";
    char SymbolSizeStr[10] = "";

    if (OutputFormat == sysv || i->Address == object::UnknownAddressOrSize)
      strcpy(SymbolAddrStr, "        ");
    if (OutputFormat == sysv)
      strcpy(SymbolSizeStr, "        ");

    if (i->Address != object::UnknownAddressOrSize)
      format("%08" PRIx64, i->Address).print(SymbolAddrStr,
                                             sizeof(SymbolAddrStr));
    if (i->Size != object::UnknownAddressOrSize)
      format("%08" PRIx64, i->Size).print(SymbolSizeStr, sizeof(SymbolSizeStr));

    if (OutputFormat == posix) {
      outs() << i->Name << " " << i->TypeChar << " "
             << SymbolAddrStr << SymbolSizeStr << "\n";
    } else if (OutputFormat == bsd) {
      if (PrintAddress)
        outs() << SymbolAddrStr << ' ';
      if (PrintSize) {
        outs() << SymbolSizeStr;
        if (i->Size != object::UnknownAddressOrSize)
          outs() << ' ';
      }
      outs() << i->TypeChar << " " << i->Name  << "\n";
    } else if (OutputFormat == sysv) {
      std::string PaddedName (i->Name);
      while (PaddedName.length () < 20)
        PaddedName += " ";
      outs() << PaddedName << "|" << SymbolAddrStr << "|   "
             << i->TypeChar
             << "  |                  |" << SymbolSizeStr << "|     |\n";
    }
  }

  SymbolList.clear();
}

static char TypeCharForSymbol(GlobalValue &GV) {
  if (GV.isDeclaration())                                  return 'U';
  if (GV.hasLinkOnceLinkage())                             return 'C';
  if (GV.hasCommonLinkage())                               return 'C';
  if (GV.hasWeakLinkage())                                 return 'W';
  if (isa<Function>(GV) && GV.hasInternalLinkage())        return 't';
  if (isa<Function>(GV))                                   return 'T';
  if (isa<GlobalVariable>(GV) && GV.hasInternalLinkage())  return 'd';
  if (isa<GlobalVariable>(GV))                             return 'D';
  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(&GV)) {
    const GlobalValue *AliasedGV = GA->getAliasedGlobal();
    if (isa<Function>(AliasedGV))                          return 'T';
    if (isa<GlobalVariable>(AliasedGV))                    return 'D';
  }
                                                           return '?';
}

static void DumpSymbolNameForGlobalValue(GlobalValue &GV) {
  // Private linkage and available_externally linkage don't exist in symtab.
  if (GV.hasPrivateLinkage() ||
      GV.hasLinkerPrivateLinkage() ||
      GV.hasLinkerPrivateWeakLinkage() ||
      GV.hasAvailableExternallyLinkage())
    return;
  char TypeChar = TypeCharForSymbol(GV);
  if (GV.hasLocalLinkage () && ExternalOnly)
    return;

  NMSymbol s;
  s.Address = object::UnknownAddressOrSize;
  s.Size = object::UnknownAddressOrSize;
  s.TypeChar = TypeChar;
  s.Name     = GV.getName();
  SymbolList.push_back(s);
}

static void DumpSymbolNamesFromModule(Module *M) {
  CurrentFilename = M->getModuleIdentifier();
  std::for_each (M->begin(), M->end(), DumpSymbolNameForGlobalValue);
  std::for_each (M->global_begin(), M->global_end(),
                 DumpSymbolNameForGlobalValue);
  if (!WithoutAliases)
    std::for_each (M->alias_begin(), M->alias_end(),
		   DumpSymbolNameForGlobalValue);

  SortAndPrintSymbolList();
}

template <class ELFT>
error_code getSymbolNMTypeChar(ELFObjectFile<ELFT> &Obj, symbol_iterator I,
                               char &Result) {
  typedef typename ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;

  DataRefImpl Symb = I->getRawDataRefImpl();
  const Elf_Sym *ESym = Obj.getSymbol(Symb);
  const ELFFile<ELFT> &EF = *Obj.getELFFile();
  const Elf_Shdr *ESec = EF.getSection(ESym);

  char ret = '?';

  if (ESec) {
    switch (ESec->sh_type) {
    case ELF::SHT_PROGBITS:
    case ELF::SHT_DYNAMIC:
      switch (ESec->sh_flags) {
      case(ELF::SHF_ALLOC | ELF::SHF_EXECINSTR) :
        ret = 't';
        break;
      case(ELF::SHF_ALLOC | ELF::SHF_WRITE) :
        ret = 'd';
        break;
      case ELF::SHF_ALLOC:
      case(ELF::SHF_ALLOC | ELF::SHF_MERGE) :
      case(ELF::SHF_ALLOC | ELF::SHF_MERGE | ELF::SHF_STRINGS) :
        ret = 'r';
        break;
      }
      break;
    case ELF::SHT_NOBITS:
      ret = 'b';
    }
  }

  switch (EF.getSymbolTableIndex(ESym)) {
  case ELF::SHN_UNDEF:
    if (ret == '?')
      ret = 'U';
    break;
  case ELF::SHN_ABS:
    ret = 'a';
    break;
  case ELF::SHN_COMMON:
    ret = 'c';
    break;
  }

  switch (ESym->getBinding()) {
  case ELF::STB_GLOBAL:
    ret = ::toupper(ret);
    break;
  case ELF::STB_WEAK:
    if (EF.getSymbolTableIndex(ESym) == ELF::SHN_UNDEF)
      ret = 'w';
    else if (ESym->getType() == ELF::STT_OBJECT)
      ret = 'V';
    else
      ret = 'W';
  }

  if (ret == '?' && ESym->getType() == ELF::STT_SECTION) {
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

  Result = ret;
  return object_error::success;
}

static error_code getSymbolNMTypeChar(COFFObjectFile &Obj, symbol_iterator I,
                                      char &Result) {
  const coff_symbol *symb = Obj.getCOFFSymbol(I);
  StringRef name;
  if (error_code ec = I->getName(name))
    return ec;
  char ret = StringSwitch<char>(name)
                 .StartsWith(".debug", 'N')
                 .StartsWith(".sxdata", 'N')
                 .Default('?');

  if (ret != '?') {
    Result = ret;
    return object_error::success;
  }

  uint32_t Characteristics = 0;
  if (symb->SectionNumber > 0) {
    section_iterator SecI = Obj.end_sections();
    if (error_code ec = I->getSection(SecI))
      return ec;
    const coff_section *Section = Obj.getCOFFSection(SecI);
    Characteristics = Section->Characteristics;
  }

  switch (symb->SectionNumber) {
  case COFF::IMAGE_SYM_UNDEFINED:
    // Check storage classes.
    if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL) {
      Result = 'w';
      return object_error::success; // Don't do ::toupper.
    } else if (symb->Value != 0)    // Check for common symbols.
      ret = 'c';
    else
      ret = 'u';
    break;
  case COFF::IMAGE_SYM_ABSOLUTE:
    ret = 'a';
    break;
  case COFF::IMAGE_SYM_DEBUG:
    ret = 'n';
    break;
  default:
    // Check section type.
    if (Characteristics & COFF::IMAGE_SCN_CNT_CODE)
      ret = 't';
    else if (Characteristics & COFF::IMAGE_SCN_MEM_READ &&
             ~Characteristics & COFF::IMAGE_SCN_MEM_WRITE) // Read only.
      ret = 'r';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
      ret = 'd';
    else if (Characteristics & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
      ret = 'b';
    else if (Characteristics & COFF::IMAGE_SCN_LNK_INFO)
      ret = 'i';

    // Check for section symbol.
    else if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_STATIC &&
             symb->Value == 0)
      ret = 's';
  }

  if (symb->StorageClass == COFF::IMAGE_SYM_CLASS_EXTERNAL)
    ret = ::toupper(static_cast<unsigned char>(ret));

  Result = ret;
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

static void DumpSymbolNamesFromObject(ObjectFile *obj) {
  error_code ec;
  symbol_iterator ibegin = obj->begin_symbols();
  symbol_iterator iend = obj->end_symbols();
  if (DynamicSyms) {
    ibegin = obj->begin_dynamic_symbols();
    iend = obj->end_dynamic_symbols();
  }
  for (symbol_iterator i = ibegin; i != iend; i.increment(ec)) {
    if (error(ec)) break;
    uint32_t symflags;
    if (error(i->getFlags(symflags))) break;
    if (!DebugSyms && (symflags & SymbolRef::SF_FormatSpecific))
      continue;
    NMSymbol s;
    s.Size = object::UnknownAddressOrSize;
    s.Address = object::UnknownAddressOrSize;
    if (PrintSize || SizeSort) {
      if (error(i->getSize(s.Size))) break;
    }
    if (PrintAddress)
      if (error(i->getAddress(s.Address))) break;
    s.TypeChar = getNMTypeChar(obj, i);
    if (error(i->getName(s.Name))) break;
    SymbolList.push_back(s);
  }

  CurrentFilename = obj->getFileName();
  SortAndPrintSymbolList();
}

static void DumpSymbolNamesFromFile(std::string &Filename) {
  if (Filename != "-" && !sys::fs::exists(Filename)) {
    errs() << ToolName << ": '" << Filename << "': " << "No such file\n";
    return;
  }

  OwningPtr<MemoryBuffer> Buffer;
  if (error(MemoryBuffer::getFileOrSTDIN(Filename, Buffer), Filename))
    return;

  sys::fs::file_magic magic = sys::fs::identify_magic(Buffer->getBuffer());

  LLVMContext &Context = getGlobalContext();
  if (magic == sys::fs::file_magic::bitcode) {
    ErrorOr<Module *> ModuleOrErr = parseBitcodeFile(Buffer.get(), Context);
    if (error(ModuleOrErr.getError(), Filename)) {
      return;
    } else {
      Module *Result = ModuleOrErr.get();
      DumpSymbolNamesFromModule(Result);
      delete Result;
    }
  } else if (magic == sys::fs::file_magic::archive) {
    OwningPtr<Binary> arch;
    if (error(object::createBinary(Buffer.take(), arch), Filename))
      return;

    if (object::Archive *a = dyn_cast<object::Archive>(arch.get())) {
      if (ArchiveMap) {
        object::Archive::symbol_iterator I = a->begin_symbols();
        object::Archive::symbol_iterator E = a->end_symbols();
        if (I !=E) {
          outs() << "Archive map" << "\n";
          for (; I != E; ++I) {
            object::Archive::child_iterator c;
            StringRef symname;
            StringRef filename;
            if (error(I->getMember(c)))
              return;
            if (error(I->getName(symname)))
              return;
            if (error(c->getName(filename)))
              return;
            outs() << symname << " in " << filename << "\n";
          }
          outs() << "\n";
        }
      }

      for (object::Archive::child_iterator i = a->begin_children(),
                                           e = a->end_children(); i != e; ++i) {
        OwningPtr<Binary> child;
        if (i->getAsBinary(child)) {
          // Try opening it as a bitcode file.
          OwningPtr<MemoryBuffer> buff;
          if (error(i->getMemoryBuffer(buff)))
            return;

          ErrorOr<Module *> ModuleOrErr = parseBitcodeFile(buff.get(), Context);
          if (ModuleOrErr) {
            Module *Result = ModuleOrErr.get();
            DumpSymbolNamesFromModule(Result);
            delete Result;
          }
          continue;
        }
        if (object::ObjectFile *o = dyn_cast<ObjectFile>(child.get())) {
          outs() << o->getFileName() << ":\n";
          DumpSymbolNamesFromObject(o);
        }
      }
    }
  } else if (magic == sys::fs::file_magic::macho_universal_binary) {
    OwningPtr<Binary> Bin;
    if (error(object::createBinary(Buffer.take(), Bin), Filename))
      return;

    object::MachOUniversalBinary *UB =
        cast<object::MachOUniversalBinary>(Bin.get());
    for (object::MachOUniversalBinary::object_iterator
             I = UB->begin_objects(),
             E = UB->end_objects();
         I != E; ++I) {
      OwningPtr<ObjectFile> Obj;
      if (!I->getAsObjectFile(Obj)) {
        outs() << Obj->getFileName() << ":\n";
        DumpSymbolNamesFromObject(Obj.get());
      }
    }
  } else if (magic.is_object()) {
    OwningPtr<Binary> obj;
    if (error(object::createBinary(Buffer.take(), obj), Filename))
      return;
    if (object::ObjectFile *o = dyn_cast<ObjectFile>(obj.get()))
      DumpSymbolNamesFromObject(o);
  } else {
    errs() << ToolName << ": " << Filename << ": "
           << "unrecognizable file type\n";
    HadError = true;
    return;
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm symbol table dumper\n");

  // llvm-nm only reads binary files.
  if (error(sys::ChangeStdinToBinary()))
    return 1;

  ToolName = argv[0];
  if (BSDFormat) OutputFormat = bsd;
  if (POSIXFormat) OutputFormat = posix;

  // The relative order of these is important. If you pass --size-sort it should
  // only print out the size. However, if you pass -S --size-sort, it should
  // print out both the size and address.
  if (SizeSort && !PrintSize) PrintAddress = false;
  if (OutputFormat == sysv || SizeSort) PrintSize = true;

  switch (InputFilenames.size()) {
  case 0: InputFilenames.push_back("-");
  case 1: break;
  default: MultipleFiles = true;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DumpSymbolNamesFromFile);

  if (HadError)
    return 1;

  return 0;
}
