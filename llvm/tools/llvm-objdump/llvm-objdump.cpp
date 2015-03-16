//===-- llvm-objdump.cpp - Object file dumping utility for llvm -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like binutils "objdump", that is, it
// dumps out a plethora of information about an object file depending on the
// flags.
//
// The flags and output of this program should be near identical to those of
// binutils objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCRelocationInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <system_error>

using namespace llvm;
using namespace object;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input object files>"),cl::ZeroOrMore);

cl::opt<bool>
llvm::Disassemble("disassemble",
  cl::desc("Display assembler mnemonics for the machine instructions"));
static cl::alias
Disassembled("d", cl::desc("Alias for --disassemble"),
             cl::aliasopt(Disassemble));

cl::opt<bool>
llvm::Relocations("r", cl::desc("Display the relocation entries in the file"));

cl::opt<bool>
llvm::SectionContents("s", cl::desc("Display the content of each section"));

cl::opt<bool>
llvm::SymbolTable("t", cl::desc("Display the symbol table"));

cl::opt<bool>
llvm::ExportsTrie("exports-trie", cl::desc("Display mach-o exported symbols"));

cl::opt<bool>
llvm::Rebase("rebase", cl::desc("Display mach-o rebasing info"));

cl::opt<bool>
llvm::Bind("bind", cl::desc("Display mach-o binding info"));

cl::opt<bool>
llvm::LazyBind("lazy-bind", cl::desc("Display mach-o lazy binding info"));

cl::opt<bool>
llvm::WeakBind("weak-bind", cl::desc("Display mach-o weak binding info"));

static cl::opt<bool>
MachOOpt("macho", cl::desc("Use MachO specific object file parser"));
static cl::alias
MachOm("m", cl::desc("Alias for --macho"), cl::aliasopt(MachOOpt));

cl::opt<std::string>
llvm::TripleName("triple", cl::desc("Target triple to disassemble for, "
                                    "see -version for available targets"));

cl::opt<std::string>
llvm::MCPU("mcpu",
     cl::desc("Target a specific cpu type (-mcpu=help for details)"),
     cl::value_desc("cpu-name"),
     cl::init(""));

cl::opt<std::string>
llvm::ArchName("arch-name", cl::desc("Target arch to disassemble for, "
                                "see -version for available targets"));

cl::opt<bool>
llvm::SectionHeaders("section-headers", cl::desc("Display summaries of the "
                                                 "headers for each section."));
static cl::alias
SectionHeadersShort("headers", cl::desc("Alias for --section-headers"),
                    cl::aliasopt(SectionHeaders));
static cl::alias
SectionHeadersShorter("h", cl::desc("Alias for --section-headers"),
                      cl::aliasopt(SectionHeaders));

cl::list<std::string>
llvm::MAttrs("mattr",
  cl::CommaSeparated,
  cl::desc("Target specific attributes"),
  cl::value_desc("a1,+a2,-a3,..."));

cl::opt<bool>
llvm::NoShowRawInsn("no-show-raw-insn", cl::desc("When disassembling "
                                                 "instructions, do not print "
                                                 "the instruction bytes."));

cl::opt<bool>
llvm::UnwindInfo("unwind-info", cl::desc("Display unwind information"));

static cl::alias
UnwindInfoShort("u", cl::desc("Alias for --unwind-info"),
                cl::aliasopt(UnwindInfo));

cl::opt<bool>
llvm::PrivateHeaders("private-headers",
                     cl::desc("Display format specific file headers"));

static cl::alias
PrivateHeadersShort("p", cl::desc("Alias for --private-headers"),
                    cl::aliasopt(PrivateHeaders));

static StringRef ToolName;
static int ReturnValue = EXIT_SUCCESS;

bool llvm::error(std::error_code EC) {
  if (!EC)
    return false;

  outs() << ToolName << ": error reading file: " << EC.message() << ".\n";
  outs().flush();
  ReturnValue = EXIT_FAILURE;
  return true;
}

static const Target *getTarget(const ObjectFile *Obj = nullptr) {
  // Figure out the target triple.
  llvm::Triple TheTriple("unknown-unknown-unknown");
  if (TripleName.empty()) {
    if (Obj) {
      TheTriple.setArch(Triple::ArchType(Obj->getArch()));
      // TheTriple defaults to ELF, and COFF doesn't have an environment:
      // the best we can do here is indicate that it is mach-o.
      if (Obj->isMachO())
        TheTriple.setObjectFormat(Triple::MachO);

      if (Obj->isCOFF()) {
        const auto COFFObj = dyn_cast<COFFObjectFile>(Obj);
        if (COFFObj->getArch() == Triple::thumb)
          TheTriple.setTriple("thumbv7-windows");
      }
    }
  } else
    TheTriple.setTriple(Triple::normalize(TripleName));

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(ArchName, TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << ToolName << ": " << Error;
    return nullptr;
  }

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

void llvm::DumpBytes(StringRef bytes) {
  static const char hex_rep[] = "0123456789abcdef";
  // FIXME: The real way to do this is to figure out the longest instruction
  //        and align to that size before printing. I'll fix this when I get
  //        around to outputting relocations.
  // 15 is the longest x86 instruction
  // 3 is for the hex rep of a byte + a space.
  // 1 is for the null terminator.
  enum { OutputSize = (15 * 3) + 1 };
  char output[OutputSize];

  assert(bytes.size() <= 15
    && "DumpBytes only supports instructions of up to 15 bytes");
  memset(output, ' ', sizeof(output));
  unsigned index = 0;
  for (StringRef::iterator i = bytes.begin(),
                           e = bytes.end(); i != e; ++i) {
    output[index] = hex_rep[(*i & 0xF0) >> 4];
    output[index + 1] = hex_rep[*i & 0xF];
    index += 3;
  }

  output[sizeof(output) - 1] = 0;
  outs() << output;
}

bool llvm::RelocAddressLess(RelocationRef a, RelocationRef b) {
  uint64_t a_addr, b_addr;
  if (error(a.getOffset(a_addr))) return false;
  if (error(b.getOffset(b_addr))) return false;
  return a_addr < b_addr;
}

static void DisassembleObject(const ObjectFile *Obj, bool InlineRelocs) {
  const Target *TheTarget = getTarget(Obj);
  // getTarget() will have already issued a diagnostic if necessary, so
  // just bail here if it failed.
  if (!TheTarget)
    return;

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  if (!MRI) {
    errs() << "error: no register info for target " << TripleName << "\n";
    return;
  }

  // Set up disassembler.
  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!AsmInfo) {
    errs() << "error: no assembly info for target " << TripleName << "\n";
    return;
  }

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, FeaturesStr));
  if (!STI) {
    errs() << "error: no subtarget info for target " << TripleName << "\n";
    return;
  }

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "error: no instruction info for target " << TripleName << "\n";
    return;
  }

  std::unique_ptr<const MCObjectFileInfo> MOFI(new MCObjectFileInfo);
  MCContext Ctx(AsmInfo.get(), MRI.get(), MOFI.get());

  std::unique_ptr<MCDisassembler> DisAsm(
    TheTarget->createMCDisassembler(*STI, Ctx));

  if (!DisAsm) {
    errs() << "error: no disassembler for target " << TripleName << "\n";
    return;
  }

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      AsmPrinterVariant, *AsmInfo, *MII, *MRI, *STI));
  if (!IP) {
    errs() << "error: no instruction printer for target " << TripleName
      << '\n';
    return;
  }

  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "\t\t%016" PRIx64 ":  " :
                                                 "\t\t\t%08" PRIx64 ":  ";

  // Create a mapping, RelocSecs = SectionRelocMap[S], where sections
  // in RelocSecs contain the relocations for section S.
  std::error_code EC;
  std::map<SectionRef, SmallVector<SectionRef, 1>> SectionRelocMap;
  for (const SectionRef &Section : Obj->sections()) {
    section_iterator Sec2 = Section.getRelocatedSection();
    if (Sec2 != Obj->section_end())
      SectionRelocMap[*Sec2].push_back(Section);
  }

  for (const SectionRef &Section : Obj->sections()) {
    if (!Section.isText() || Section.isVirtual())
      continue;

    uint64_t SectionAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    // Make a list of all the symbols in this section.
    std::vector<std::pair<uint64_t, StringRef>> Symbols;
    for (const SymbolRef &Symbol : Obj->symbols()) {
      if (Section.containsSymbol(Symbol)) {
        uint64_t Address;
        if (error(Symbol.getAddress(Address)))
          break;
        if (Address == UnknownAddressOrSize)
          continue;
        Address -= SectionAddr;
        if (Address >= SectSize)
          continue;

        StringRef Name;
        if (error(Symbol.getName(Name)))
          break;
        Symbols.push_back(std::make_pair(Address, Name));
      }
    }

    // Sort the symbols by address, just in case they didn't come in that way.
    array_pod_sort(Symbols.begin(), Symbols.end());

    // Make a list of all the relocations for this section.
    std::vector<RelocationRef> Rels;
    if (InlineRelocs) {
      for (const SectionRef &RelocSec : SectionRelocMap[Section]) {
        for (const RelocationRef &Reloc : RelocSec.relocations()) {
          Rels.push_back(Reloc);
        }
      }
    }

    // Sort relocations by address.
    std::sort(Rels.begin(), Rels.end(), RelocAddressLess);

    StringRef SegmentName = "";
    if (const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(Obj)) {
      DataRefImpl DR = Section.getRawDataRefImpl();
      SegmentName = MachO->getSectionFinalSegmentName(DR);
    }
    StringRef name;
    if (error(Section.getName(name)))
      break;
    outs() << "Disassembly of section ";
    if (!SegmentName.empty())
      outs() << SegmentName << ",";
    outs() << name << ':';

    // If the section has no symbols just insert a dummy one and disassemble
    // the whole section.
    if (Symbols.empty())
      Symbols.push_back(std::make_pair(0, name));


    SmallString<40> Comments;
    raw_svector_ostream CommentStream(Comments);

    StringRef BytesStr;
    if (error(Section.getContents(BytesStr)))
      break;
    ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(BytesStr.data()),
                            BytesStr.size());

    uint64_t Size;
    uint64_t Index;

    std::vector<RelocationRef>::const_iterator rel_cur = Rels.begin();
    std::vector<RelocationRef>::const_iterator rel_end = Rels.end();
    // Disassemble symbol by symbol.
    for (unsigned si = 0, se = Symbols.size(); si != se; ++si) {

      uint64_t Start = Symbols[si].first;
      // The end is either the section end or the beginning of the next symbol.
      uint64_t End = (si == se - 1) ? SectSize : Symbols[si + 1].first;
      // If this symbol has the same address as the next symbol, then skip it.
      if (Start == End)
        continue;

      outs() << '\n' << Symbols[si].second << ":\n";

#ifndef NDEBUG
      raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
      raw_ostream &DebugOut = nulls();
#endif

      for (Index = Start; Index < End; Index += Size) {
        MCInst Inst;

        if (DisAsm->getInstruction(Inst, Size, Bytes.slice(Index),
                                   SectionAddr + Index, DebugOut,
                                   CommentStream)) {
          outs() << format("%8" PRIx64 ":", SectionAddr + Index);
          if (!NoShowRawInsn) {
            outs() << "\t";
            DumpBytes(StringRef(
                reinterpret_cast<const char *>(Bytes.data()) + Index, Size));
          }
          IP->printInst(&Inst, outs(), "");
          outs() << CommentStream.str();
          Comments.clear();
          outs() << "\n";
        } else {
          errs() << ToolName << ": warning: invalid instruction encoding\n";
          if (Size == 0)
            Size = 1; // skip illegible bytes
        }

        // Print relocation for instruction.
        while (rel_cur != rel_end) {
          bool hidden = false;
          uint64_t addr;
          SmallString<16> name;
          SmallString<32> val;

          // If this relocation is hidden, skip it.
          if (error(rel_cur->getHidden(hidden))) goto skip_print_rel;
          if (hidden) goto skip_print_rel;

          if (error(rel_cur->getOffset(addr))) goto skip_print_rel;
          // Stop when rel_cur's address is past the current instruction.
          if (addr >= Index + Size) break;
          if (error(rel_cur->getTypeName(name))) goto skip_print_rel;
          if (error(rel_cur->getValueString(val))) goto skip_print_rel;

          outs() << format(Fmt.data(), SectionAddr + addr) << name
                 << "\t" << val << "\n";

        skip_print_rel:
          ++rel_cur;
        }
      }
    }
  }
}

void llvm::PrintRelocations(const ObjectFile *Obj) {
  StringRef Fmt = Obj->getBytesInAddress() > 4 ? "%016" PRIx64 :
                                                 "%08" PRIx64;
  // Regular objdump doesn't print relocations in non-relocatable object
  // files.
  if (!Obj->isRelocatableObject())
    return;

  for (const SectionRef &Section : Obj->sections()) {
    if (Section.relocation_begin() == Section.relocation_end())
      continue;
    StringRef secname;
    if (error(Section.getName(secname)))
      continue;
    outs() << "RELOCATION RECORDS FOR [" << secname << "]:\n";
    for (const RelocationRef &Reloc : Section.relocations()) {
      bool hidden;
      uint64_t address;
      SmallString<32> relocname;
      SmallString<32> valuestr;
      if (error(Reloc.getHidden(hidden)))
        continue;
      if (hidden)
        continue;
      if (error(Reloc.getTypeName(relocname)))
        continue;
      if (error(Reloc.getOffset(address)))
        continue;
      if (error(Reloc.getValueString(valuestr)))
        continue;
      outs() << format(Fmt.data(), address) << " " << relocname << " "
             << valuestr << "\n";
    }
    outs() << "\n";
  }
}

void llvm::PrintSectionHeaders(const ObjectFile *Obj) {
  outs() << "Sections:\n"
            "Idx Name          Size      Address          Type\n";
  unsigned i = 0;
  for (const SectionRef &Section : Obj->sections()) {
    StringRef Name;
    if (error(Section.getName(Name)))
      return;
    uint64_t Address = Section.getAddress();
    uint64_t Size = Section.getSize();
    bool Text = Section.isText();
    bool Data = Section.isData();
    bool BSS = Section.isBSS();
    std::string Type = (std::string(Text ? "TEXT " : "") +
                        (Data ? "DATA " : "") + (BSS ? "BSS" : ""));
    outs() << format("%3d %-13s %08" PRIx64 " %016" PRIx64 " %s\n", i,
                     Name.str().c_str(), Size, Address, Type.c_str());
    ++i;
  }
}

void llvm::PrintSectionContents(const ObjectFile *Obj) {
  std::error_code EC;
  for (const SectionRef &Section : Obj->sections()) {
    StringRef Name;
    StringRef Contents;
    if (error(Section.getName(Name)))
      continue;
    uint64_t BaseAddr = Section.getAddress();
    uint64_t Size = Section.getSize();
    if (!Size)
      continue;

    outs() << "Contents of section " << Name << ":\n";
    if (Section.isBSS()) {
      outs() << format("<skipping contents of bss section at [%04" PRIx64
                       ", %04" PRIx64 ")>\n",
                       BaseAddr, BaseAddr + Size);
      continue;
    }

    if (error(Section.getContents(Contents)))
      continue;

    // Dump out the content as hex and printable ascii characters.
    for (std::size_t addr = 0, end = Contents.size(); addr < end; addr += 16) {
      outs() << format(" %04" PRIx64 " ", BaseAddr + addr);
      // Dump line of hex.
      for (std::size_t i = 0; i < 16; ++i) {
        if (i != 0 && i % 4 == 0)
          outs() << ' ';
        if (addr + i < end)
          outs() << hexdigit((Contents[addr + i] >> 4) & 0xF, true)
                 << hexdigit(Contents[addr + i] & 0xF, true);
        else
          outs() << "  ";
      }
      // Print ascii.
      outs() << "  ";
      for (std::size_t i = 0; i < 16 && addr + i < end; ++i) {
        if (std::isprint(static_cast<unsigned char>(Contents[addr + i]) & 0xFF))
          outs() << Contents[addr + i];
        else
          outs() << ".";
      }
      outs() << "\n";
    }
  }
}

static void PrintCOFFSymbolTable(const COFFObjectFile *coff) {
  for (unsigned SI = 0, SE = coff->getNumberOfSymbols(); SI != SE; ++SI) {
    ErrorOr<COFFSymbolRef> Symbol = coff->getSymbol(SI);
    StringRef Name;
    if (error(Symbol.getError()))
      return;

    if (error(coff->getSymbolName(*Symbol, Name)))
      return;

    outs() << "[" << format("%2d", SI) << "]"
           << "(sec " << format("%2d", int(Symbol->getSectionNumber())) << ")"
           << "(fl 0x00)" // Flag bits, which COFF doesn't have.
           << "(ty " << format("%3x", unsigned(Symbol->getType())) << ")"
           << "(scl " << format("%3x", unsigned(Symbol->getStorageClass())) << ") "
           << "(nx " << unsigned(Symbol->getNumberOfAuxSymbols()) << ") "
           << "0x" << format("%08x", unsigned(Symbol->getValue())) << " "
           << Name << "\n";

    for (unsigned AI = 0, AE = Symbol->getNumberOfAuxSymbols(); AI < AE; ++AI, ++SI) {
      if (Symbol->isSectionDefinition()) {
        const coff_aux_section_definition *asd;
        if (error(coff->getAuxSymbol<coff_aux_section_definition>(SI + 1, asd)))
          return;

        int32_t AuxNumber = asd->getNumber(Symbol->isBigObj());

        outs() << "AUX "
               << format("scnlen 0x%x nreloc %d nlnno %d checksum 0x%x "
                         , unsigned(asd->Length)
                         , unsigned(asd->NumberOfRelocations)
                         , unsigned(asd->NumberOfLinenumbers)
                         , unsigned(asd->CheckSum))
               << format("assoc %d comdat %d\n"
                         , unsigned(AuxNumber)
                         , unsigned(asd->Selection));
      } else if (Symbol->isFileRecord()) {
        const char *FileName;
        if (error(coff->getAuxSymbol<char>(SI + 1, FileName)))
          return;

        StringRef Name(FileName, Symbol->getNumberOfAuxSymbols() *
                                     coff->getSymbolTableEntrySize());
        outs() << "AUX " << Name.rtrim(StringRef("\0", 1))  << '\n';

        SI = SI + Symbol->getNumberOfAuxSymbols();
        break;
      } else {
        outs() << "AUX Unknown\n";
      }
    }
  }
}

void llvm::PrintSymbolTable(const ObjectFile *o) {
  outs() << "SYMBOL TABLE:\n";

  if (const COFFObjectFile *coff = dyn_cast<const COFFObjectFile>(o)) {
    PrintCOFFSymbolTable(coff);
    return;
  }
  for (const SymbolRef &Symbol : o->symbols()) {
    StringRef Name;
    uint64_t Address;
    SymbolRef::Type Type;
    uint64_t Size;
    uint32_t Flags = Symbol.getFlags();
    section_iterator Section = o->section_end();
    if (error(Symbol.getName(Name)))
      continue;
    if (error(Symbol.getAddress(Address)))
      continue;
    if (error(Symbol.getType(Type)))
      continue;
    if (error(Symbol.getSize(Size)))
      continue;
    if (error(Symbol.getSection(Section)))
      continue;

    bool Global = Flags & SymbolRef::SF_Global;
    bool Weak = Flags & SymbolRef::SF_Weak;
    bool Absolute = Flags & SymbolRef::SF_Absolute;
    bool Common = Flags & SymbolRef::SF_Common;

    if (Common) {
      uint32_t Alignment;
      if (error(Symbol.getAlignment(Alignment)))
        Alignment = 0;
      Address = Size;
      Size = Alignment;
    }
    if (Address == UnknownAddressOrSize)
      Address = 0;
    if (Size == UnknownAddressOrSize)
      Size = 0;
    char GlobLoc = ' ';
    if (Type != SymbolRef::ST_Unknown)
      GlobLoc = Global ? 'g' : 'l';
    char Debug = (Type == SymbolRef::ST_Debug || Type == SymbolRef::ST_File)
                 ? 'd' : ' ';
    char FileFunc = ' ';
    if (Type == SymbolRef::ST_File)
      FileFunc = 'f';
    else if (Type == SymbolRef::ST_Function)
      FileFunc = 'F';

    const char *Fmt = o->getBytesInAddress() > 4 ? "%016" PRIx64 :
                                                   "%08" PRIx64;

    outs() << format(Fmt, Address) << " "
           << GlobLoc // Local -> 'l', Global -> 'g', Neither -> ' '
           << (Weak ? 'w' : ' ') // Weak?
           << ' ' // Constructor. Not supported yet.
           << ' ' // Warning. Not supported yet.
           << ' ' // Indirect reference to another symbol.
           << Debug // Debugging (d) or dynamic (D) symbol.
           << FileFunc // Name of function (F), file (f) or object (O).
           << ' ';
    if (Absolute) {
      outs() << "*ABS*";
    } else if (Common) {
      outs() << "*COM*";
    } else if (Section == o->section_end()) {
      outs() << "*UND*";
    } else {
      if (const MachOObjectFile *MachO =
          dyn_cast<const MachOObjectFile>(o)) {
        DataRefImpl DR = Section->getRawDataRefImpl();
        StringRef SegmentName = MachO->getSectionFinalSegmentName(DR);
        outs() << SegmentName << ",";
      }
      StringRef SectionName;
      if (error(Section->getName(SectionName)))
        SectionName = "";
      outs() << SectionName;
    }
    outs() << '\t'
           << format("%08" PRIx64 " ", Size)
           << Name
           << '\n';
  }
}

static void PrintUnwindInfo(const ObjectFile *o) {
  outs() << "Unwind info:\n\n";

  if (const COFFObjectFile *coff = dyn_cast<COFFObjectFile>(o)) {
    printCOFFUnwindInfo(coff);
  } else if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOUnwindInfo(MachO);
  else {
    // TODO: Extract DWARF dump tool to objdump.
    errs() << "This operation is only currently supported "
              "for COFF and MachO object files.\n";
    return;
  }
}

void llvm::printExportsTrie(const ObjectFile *o) {
  outs() << "Exports trie:\n";
  if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOExportsTrie(MachO);
  else {
    errs() << "This operation is only currently supported "
              "for Mach-O executable files.\n";
    return;
  }
}

void llvm::printRebaseTable(const ObjectFile *o) {
  outs() << "Rebase table:\n";
  if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachORebaseTable(MachO);
  else {
    errs() << "This operation is only currently supported "
              "for Mach-O executable files.\n";
    return;
  }
}

void llvm::printBindTable(const ObjectFile *o) {
  outs() << "Bind table:\n";
  if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOBindTable(MachO);
  else {
    errs() << "This operation is only currently supported "
              "for Mach-O executable files.\n";
    return;
  }
}

void llvm::printLazyBindTable(const ObjectFile *o) {
  outs() << "Lazy bind table:\n";
  if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOLazyBindTable(MachO);
  else {
    errs() << "This operation is only currently supported "
              "for Mach-O executable files.\n";
    return;
  }
}

void llvm::printWeakBindTable(const ObjectFile *o) {
  outs() << "Weak bind table:\n";
  if (const MachOObjectFile *MachO = dyn_cast<MachOObjectFile>(o))
    printMachOWeakBindTable(MachO);
  else {
    errs() << "This operation is only currently supported "
              "for Mach-O executable files.\n";
    return;
  }
}

static void printPrivateFileHeader(const ObjectFile *o) {
  if (o->isELF()) {
    printELFFileHeader(o);
  } else if (o->isCOFF()) {
    printCOFFFileHeader(o);
  } else if (o->isMachO()) {
    printMachOFileHeader(o);
  }
}

static void DumpObject(const ObjectFile *o) {
  outs() << '\n';
  outs() << o->getFileName()
         << ":\tfile format " << o->getFileFormatName() << "\n\n";

  if (Disassemble)
    DisassembleObject(o, Relocations);
  if (Relocations && !Disassemble)
    PrintRelocations(o);
  if (SectionHeaders)
    PrintSectionHeaders(o);
  if (SectionContents)
    PrintSectionContents(o);
  if (SymbolTable)
    PrintSymbolTable(o);
  if (UnwindInfo)
    PrintUnwindInfo(o);
  if (PrivateHeaders)
    printPrivateFileHeader(o);
  if (ExportsTrie)
    printExportsTrie(o);
  if (Rebase)
    printRebaseTable(o);
  if (Bind)
    printBindTable(o);
  if (LazyBind)
    printLazyBindTable(o);
  if (WeakBind)
    printWeakBindTable(o);
}

/// @brief Dump each object file in \a a;
static void DumpArchive(const Archive *a) {
  for (Archive::child_iterator i = a->child_begin(), e = a->child_end(); i != e;
       ++i) {
    ErrorOr<std::unique_ptr<Binary>> ChildOrErr = i->getAsBinary();
    if (std::error_code EC = ChildOrErr.getError()) {
      // Ignore non-object files.
      if (EC != object_error::invalid_file_type)
        errs() << ToolName << ": '" << a->getFileName() << "': " << EC.message()
               << ".\n";
      continue;
    }
    if (ObjectFile *o = dyn_cast<ObjectFile>(&*ChildOrErr.get()))
      DumpObject(o);
    else
      errs() << ToolName << ": '" << a->getFileName() << "': "
              << "Unrecognized file type.\n";
  }
}

/// @brief Open file and figure out how to dump it.
static void DumpInput(StringRef file) {
  // If file isn't stdin, check that it exists.
  if (file != "-" && !sys::fs::exists(file)) {
    errs() << ToolName << ": '" << file << "': " << "No such file\n";
    return;
  }

  // If we are using the Mach-O specific object file parser, then let it parse
  // the file and process the command line options.  So the -arch flags can
  // be used to select specific slices, etc.
  if (MachOOpt) {
    ParseInputMachO(file);
    return;
  }

  // Attempt to open the binary.
  ErrorOr<OwningBinary<Binary>> BinaryOrErr = createBinary(file);
  if (std::error_code EC = BinaryOrErr.getError()) {
    errs() << ToolName << ": '" << file << "': " << EC.message() << ".\n";
    return;
  }
  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (Archive *a = dyn_cast<Archive>(&Binary))
    DumpArchive(a);
  else if (ObjectFile *o = dyn_cast<ObjectFile>(&Binary))
    DumpObject(o);
  else
    errs() << ToolName << ": '" << file << "': " << "Unrecognized file type.\n";
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm object file dumper\n");
  TripleName = Triple::normalize(TripleName);

  ToolName = argv[0];

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  if (!Disassemble
      && !Relocations
      && !SectionHeaders
      && !SectionContents
      && !SymbolTable
      && !UnwindInfo
      && !PrivateHeaders
      && !ExportsTrie
      && !Rebase
      && !Bind
      && !LazyBind
      && !WeakBind
      && !(UniversalHeaders && MachOOpt)
      && !(ArchiveHeaders && MachOOpt)
      && !(IndirectSymbols && MachOOpt)
      && !(DataInCode && MachOOpt)
      && !(LinkOptHints && MachOOpt)
      && !(InfoPlist && MachOOpt)
      && !(DylibsUsed && MachOOpt)
      && !(DylibId && MachOOpt)
      && !(DumpSections.size() != 0 && MachOOpt)) {
    cl::PrintHelpMessage();
    return 2;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DumpInput);

  return ReturnValue;
}
