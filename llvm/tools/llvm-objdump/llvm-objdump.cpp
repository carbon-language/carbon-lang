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
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFunction.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCModule.h"
#include "llvm/MC/MCModuleYAML.h"
#include "llvm/MC/MCObjectDisassembler.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectSymbolizer.h"
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
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cctype>
#include <cstring>

using namespace llvm;
using namespace object;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input object files>"),cl::ZeroOrMore);

static cl::opt<bool>
Disassemble("disassemble",
  cl::desc("Display assembler mnemonics for the machine instructions"));
static cl::alias
Disassembled("d", cl::desc("Alias for --disassemble"),
             cl::aliasopt(Disassemble));

static cl::opt<bool>
Relocations("r", cl::desc("Display the relocation entries in the file"));

static cl::opt<bool>
SectionContents("s", cl::desc("Display the content of each section"));

static cl::opt<bool>
SymbolTable("t", cl::desc("Display the symbol table"));

static cl::opt<bool>
MachOOpt("macho", cl::desc("Use MachO specific object file parser"));
static cl::alias
MachOm("m", cl::desc("Alias for --macho"), cl::aliasopt(MachOOpt));

cl::opt<std::string>
llvm::TripleName("triple", cl::desc("Target triple to disassemble for, "
                                    "see -version for available targets"));

cl::opt<std::string>
llvm::ArchName("arch", cl::desc("Target arch to disassemble for, "
                                "see -version for available targets"));

static cl::opt<bool>
SectionHeaders("section-headers", cl::desc("Display summaries of the headers "
                                           "for each section."));
static cl::alias
SectionHeadersShort("headers", cl::desc("Alias for --section-headers"),
                    cl::aliasopt(SectionHeaders));
static cl::alias
SectionHeadersShorter("h", cl::desc("Alias for --section-headers"),
                      cl::aliasopt(SectionHeaders));

static cl::list<std::string>
MAttrs("mattr",
  cl::CommaSeparated,
  cl::desc("Target specific attributes"),
  cl::value_desc("a1,+a2,-a3,..."));

static cl::opt<bool>
NoShowRawInsn("no-show-raw-insn", cl::desc("When disassembling instructions, "
                                           "do not print the instruction bytes."));

static cl::opt<bool>
UnwindInfo("unwind-info", cl::desc("Display unwind information"));

static cl::alias
UnwindInfoShort("u", cl::desc("Alias for --unwind-info"),
                cl::aliasopt(UnwindInfo));

static cl::opt<bool>
PrivateHeaders("private-headers",
               cl::desc("Display format specific file headers"));

static cl::alias
PrivateHeadersShort("p", cl::desc("Alias for --private-headers"),
                    cl::aliasopt(PrivateHeaders));

static cl::opt<bool>
Symbolize("symbolize", cl::desc("When disassembling instructions, "
                                "try to symbolize operands."));

static cl::opt<bool>
CFG("cfg", cl::desc("Create a CFG for every function found in the object"
                      " and write it to a graphviz file"));

// FIXME: Does it make sense to have a dedicated tool for yaml cfg output?
static cl::opt<std::string>
YAMLCFG("yaml-cfg",
        cl::desc("Create a CFG and write it as a YAML MCModule."),
        cl::value_desc("yaml output file"));

static StringRef ToolName;

bool llvm::error(error_code EC) {
  if (!EC)
    return false;

  outs() << ToolName << ": error reading file: " << EC.message() << ".\n";
  outs().flush();
  return true;
}

static const Target *getTarget(const ObjectFile *Obj = NULL) {
  // Figure out the target triple.
  llvm::Triple TheTriple("unknown-unknown-unknown");
  if (TripleName.empty()) {
    if (Obj) {
      TheTriple.setArch(Triple::ArchType(Obj->getArch()));
      // TheTriple defaults to ELF, and COFF doesn't have an environment:
      // the best we can do here is indicate that it is mach-o.
      if (Obj->isMachO())
        TheTriple.setEnvironment(Triple::MachO);
    }
  } else
    TheTriple.setTriple(Triple::normalize(TripleName));

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(ArchName, TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << ToolName << ": " << Error;
    return 0;
  }

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

// Write a graphviz file for the CFG inside an MCFunction.
// FIXME: Use GraphWriter
static void emitDOTFile(const char *FileName, const MCFunction &f,
                        MCInstPrinter *IP) {
  // Start a new dot file.
  std::string Error;
  raw_fd_ostream Out(FileName, Error, sys::fs::F_None);
  if (!Error.empty()) {
    errs() << "llvm-objdump: warning: " << Error << '\n';
    return;
  }

  Out << "digraph \"" << f.getName() << "\" {\n";
  Out << "graph [ rankdir = \"LR\" ];\n";
  for (MCFunction::const_iterator i = f.begin(), e = f.end(); i != e; ++i) {
    // Only print blocks that have predecessors.
    bool hasPreds = (*i)->pred_begin() != (*i)->pred_end();

    if (!hasPreds && i != f.begin())
      continue;

    Out << '"' << (*i)->getInsts()->getBeginAddr() << "\" [ label=\"<a>";
    // Print instructions.
    for (unsigned ii = 0, ie = (*i)->getInsts()->size(); ii != ie;
        ++ii) {
      if (ii != 0) // Not the first line, start a new row.
        Out << '|';
      if (ii + 1 == ie) // Last line, add an end id.
        Out << "<o>";

      // Escape special chars and print the instruction in mnemonic form.
      std::string Str;
      raw_string_ostream OS(Str);
      IP->printInst(&(*i)->getInsts()->at(ii).Inst, OS, "");
      Out << DOT::EscapeString(OS.str());
    }
    Out << "\" shape=\"record\" ];\n";

    // Add edges.
    for (MCBasicBlock::succ_const_iterator si = (*i)->succ_begin(),
        se = (*i)->succ_end(); si != se; ++si)
      Out << (*i)->getInsts()->getBeginAddr() << ":o -> "
          << (*si)->getInsts()->getBeginAddr() << ":a\n";
  }
  Out << "}\n";
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

  OwningPtr<const MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI) {
    errs() << "error: no register info for target " << TripleName << "\n";
    return;
  }

  // Set up disassembler.
  OwningPtr<const MCAsmInfo> AsmInfo(
    TheTarget->createMCAsmInfo(*MRI, TripleName));
  if (!AsmInfo) {
    errs() << "error: no assembly info for target " << TripleName << "\n";
    return;
  }

  OwningPtr<const MCSubtargetInfo> STI(
    TheTarget->createMCSubtargetInfo(TripleName, "", FeaturesStr));
  if (!STI) {
    errs() << "error: no subtarget info for target " << TripleName << "\n";
    return;
  }

  OwningPtr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII) {
    errs() << "error: no instruction info for target " << TripleName << "\n";
    return;
  }

  OwningPtr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI));
  if (!DisAsm) {
    errs() << "error: no disassembler for target " << TripleName << "\n";
    return;
  }

  OwningPtr<const MCObjectFileInfo> MOFI;
  OwningPtr<MCContext> Ctx;

  if (Symbolize) {
    MOFI.reset(new MCObjectFileInfo);
    Ctx.reset(new MCContext(AsmInfo.get(), MRI.get(), MOFI.get()));
    OwningPtr<MCRelocationInfo> RelInfo(
      TheTarget->createMCRelocationInfo(TripleName, *Ctx.get()));
    if (RelInfo) {
      OwningPtr<MCSymbolizer> Symzer(
        MCObjectSymbolizer::createObjectSymbolizer(*Ctx.get(), RelInfo, Obj));
      if (Symzer)
        DisAsm->setSymbolizer(Symzer);
    }
  }

  OwningPtr<const MCInstrAnalysis>
    MIA(TheTarget->createMCInstrAnalysis(MII.get()));

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  OwningPtr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      AsmPrinterVariant, *AsmInfo, *MII, *MRI, *STI));
  if (!IP) {
    errs() << "error: no instruction printer for target " << TripleName
      << '\n';
    return;
  }

  if (CFG || !YAMLCFG.empty()) {
    OwningPtr<MCObjectDisassembler> OD(
      new MCObjectDisassembler(*Obj, *DisAsm, *MIA));
    OwningPtr<MCModule> Mod(OD->buildModule(/* withCFG */ true));
    for (MCModule::const_atom_iterator AI = Mod->atom_begin(),
                                       AE = Mod->atom_end();
                                       AI != AE; ++AI) {
      outs() << "Atom " << (*AI)->getName() << ": \n";
      if (const MCTextAtom *TA = dyn_cast<MCTextAtom>(*AI)) {
        for (MCTextAtom::const_iterator II = TA->begin(), IE = TA->end();
             II != IE;
             ++II) {
          IP->printInst(&II->Inst, outs(), "");
          outs() << "\n";
        }
      }
    }
    if (CFG) {
      for (MCModule::const_func_iterator FI = Mod->func_begin(),
                                         FE = Mod->func_end();
                                         FI != FE; ++FI) {
        static int filenum = 0;
        emitDOTFile((Twine((*FI)->getName()) + "_" +
                     utostr(filenum) + ".dot").str().c_str(),
                      **FI, IP.get());
        ++filenum;
      }
    }
    if (!YAMLCFG.empty()) {
      std::string Error;
      raw_fd_ostream YAMLOut(YAMLCFG.c_str(), Error, sys::fs::F_None);
      if (!Error.empty()) {
        errs() << ToolName << ": warning: " << Error << '\n';
        return;
      }
      mcmodule2yaml(YAMLOut, *Mod, *MII, *MRI);
    }
  }

  // Create a mapping, RelocSecs = SectionRelocMap[S], where sections
  // in RelocSecs contain the relocations for section S.
  error_code EC;
  std::map<SectionRef, SmallVector<SectionRef, 1> > SectionRelocMap;
  for (section_iterator I = Obj->section_begin(), E = Obj->section_end();
       I != E; ++I) {
    section_iterator Sec2 = I->getRelocatedSection();
    if (Sec2 != Obj->section_end())
      SectionRelocMap[*Sec2].push_back(*I);
  }

  for (section_iterator I = Obj->section_begin(), E = Obj->section_end();
       I != E; ++I) {
    bool Text;
    if (error(I->isText(Text)))
      break;
    if (!Text)
      continue;

    uint64_t SectionAddr;
    if (error(I->getAddress(SectionAddr)))
      break;

    // Make a list of all the symbols in this section.
    std::vector<std::pair<uint64_t, StringRef> > Symbols;
    for (symbol_iterator SI = Obj->symbol_begin(), SE = Obj->symbol_end();
         SI != SE; ++SI) {
      bool contains;
      if (!error(I->containsSymbol(*SI, contains)) && contains) {
        uint64_t Address;
        if (error(SI->getAddress(Address)))
          break;
        if (Address == UnknownAddressOrSize)
          continue;
        Address -= SectionAddr;

        StringRef Name;
        if (error(SI->getName(Name)))
          break;
        Symbols.push_back(std::make_pair(Address, Name));
      }
    }

    // Sort the symbols by address, just in case they didn't come in that way.
    array_pod_sort(Symbols.begin(), Symbols.end());

    // Make a list of all the relocations for this section.
    std::vector<RelocationRef> Rels;
    if (InlineRelocs) {
      SmallVectorImpl<SectionRef> *RelocSecs = &SectionRelocMap[*I];
      for (SmallVectorImpl<SectionRef>::iterator RelocSec = RelocSecs->begin(),
                                                 E = RelocSecs->end();
           RelocSec != E; ++RelocSec) {
        for (relocation_iterator RI = RelocSec->relocation_begin(),
                                 RE = RelocSec->relocation_end();
             RI != RE; ++RI)
          Rels.push_back(*RI);
      }
    }

    // Sort relocations by address.
    std::sort(Rels.begin(), Rels.end(), RelocAddressLess);

    StringRef SegmentName = "";
    if (const MachOObjectFile *MachO = dyn_cast<const MachOObjectFile>(Obj)) {
      DataRefImpl DR = I->getRawDataRefImpl();
      SegmentName = MachO->getSectionFinalSegmentName(DR);
    }
    StringRef name;
    if (error(I->getName(name)))
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

    StringRef Bytes;
    if (error(I->getContents(Bytes)))
      break;
    StringRefMemoryObject memoryObject(Bytes, SectionAddr);
    uint64_t Size;
    uint64_t Index;
    uint64_t SectSize;
    if (error(I->getSize(SectSize)))
      break;

    std::vector<RelocationRef>::const_iterator rel_cur = Rels.begin();
    std::vector<RelocationRef>::const_iterator rel_end = Rels.end();
    // Disassemble symbol by symbol.
    for (unsigned si = 0, se = Symbols.size(); si != se; ++si) {
      uint64_t Start = Symbols[si].first;
      uint64_t End;
      // The end is either the size of the section or the beginning of the next
      // symbol.
      if (si == se - 1)
        End = SectSize;
      // Make sure this symbol takes up space.
      else if (Symbols[si + 1].first != Start)
        End = Symbols[si + 1].first - 1;
      else
        // This symbol has the same address as the next symbol. Skip it.
        continue;

      outs() << '\n' << Symbols[si].second << ":\n";

#ifndef NDEBUG
      raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
      raw_ostream &DebugOut = nulls();
#endif

      for (Index = Start; Index < End; Index += Size) {
        MCInst Inst;

        if (DisAsm->getInstruction(Inst, Size, memoryObject,
                                   SectionAddr + Index,
                                   DebugOut, CommentStream)) {
          outs() << format("%8" PRIx64 ":", SectionAddr + Index);
          if (!NoShowRawInsn) {
            outs() << "\t";
            DumpBytes(StringRef(Bytes.data() + Index, Size));
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

          outs() << format("\t\t\t%8" PRIx64 ": ", SectionAddr + addr) << name
                 << "\t" << val << "\n";

        skip_print_rel:
          ++rel_cur;
        }
      }
    }
  }
}

static void PrintRelocations(const ObjectFile *o) {
  for (section_iterator si = o->section_begin(), se = o->section_end();
       si != se; ++si) {
    if (si->relocation_begin() == si->relocation_end())
      continue;
    StringRef secname;
    if (error(si->getName(secname))) continue;
    outs() << "RELOCATION RECORDS FOR [" << secname << "]:\n";
    for (relocation_iterator ri = si->relocation_begin(),
                             re = si->relocation_end();
         ri != re; ++ri) {
      bool hidden;
      uint64_t address;
      SmallString<32> relocname;
      SmallString<32> valuestr;
      if (error(ri->getHidden(hidden))) continue;
      if (hidden) continue;
      if (error(ri->getTypeName(relocname))) continue;
      if (error(ri->getOffset(address))) continue;
      if (error(ri->getValueString(valuestr))) continue;
      outs() << address << " " << relocname << " " << valuestr << "\n";
    }
    outs() << "\n";
  }
}

static void PrintSectionHeaders(const ObjectFile *o) {
  outs() << "Sections:\n"
            "Idx Name          Size      Address          Type\n";
  unsigned i = 0;
  for (section_iterator si = o->section_begin(), se = o->section_end();
       si != se; ++si) {
    StringRef Name;
    if (error(si->getName(Name)))
      return;
    uint64_t Address;
    if (error(si->getAddress(Address))) return;
    uint64_t Size;
    if (error(si->getSize(Size))) return;
    bool Text, Data, BSS;
    if (error(si->isText(Text))) return;
    if (error(si->isData(Data))) return;
    if (error(si->isBSS(BSS))) return;
    std::string Type = (std::string(Text ? "TEXT " : "") +
                        (Data ? "DATA " : "") + (BSS ? "BSS" : ""));
    outs() << format("%3d %-13s %08" PRIx64 " %016" PRIx64 " %s\n",
                     i, Name.str().c_str(), Size, Address, Type.c_str());
    ++i;
  }
}

static void PrintSectionContents(const ObjectFile *o) {
  error_code EC;
  for (section_iterator si = o->section_begin(), se = o->section_end();
       si != se; ++si) {
    StringRef Name;
    StringRef Contents;
    uint64_t BaseAddr;
    bool BSS;
    if (error(si->getName(Name))) continue;
    if (error(si->getContents(Contents))) continue;
    if (error(si->getAddress(BaseAddr))) continue;
    if (error(si->isBSS(BSS))) continue;

    outs() << "Contents of section " << Name << ":\n";
    if (BSS) {
      outs() << format("<skipping contents of bss section at [%04" PRIx64
                       ", %04" PRIx64 ")>\n", BaseAddr,
                       BaseAddr + Contents.size());
      continue;
    }

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
  const coff_file_header *header;
  if (error(coff->getHeader(header))) return;
  int aux_count = 0;
  const coff_symbol *symbol = 0;
  for (int i = 0, e = header->NumberOfSymbols; i != e; ++i) {
    if (aux_count--) {
      // Figure out which type of aux this is.
      if (symbol->StorageClass == COFF::IMAGE_SYM_CLASS_STATIC
          && symbol->Value == 0) { // Section definition.
        const coff_aux_section_definition *asd;
        if (error(coff->getAuxSymbol<coff_aux_section_definition>(i, asd)))
          return;
        outs() << "AUX "
               << format("scnlen 0x%x nreloc %d nlnno %d checksum 0x%x "
                         , unsigned(asd->Length)
                         , unsigned(asd->NumberOfRelocations)
                         , unsigned(asd->NumberOfLinenumbers)
                         , unsigned(asd->CheckSum))
               << format("assoc %d comdat %d\n"
                         , unsigned(asd->Number)
                         , unsigned(asd->Selection));
      } else
        outs() << "AUX Unknown\n";
    } else {
      StringRef name;
      if (error(coff->getSymbol(i, symbol))) return;
      if (error(coff->getSymbolName(symbol, name))) return;
      outs() << "[" << format("%2d", i) << "]"
             << "(sec " << format("%2d", int(symbol->SectionNumber)) << ")"
             << "(fl 0x00)" // Flag bits, which COFF doesn't have.
             << "(ty " << format("%3x", unsigned(symbol->Type)) << ")"
             << "(scl " << format("%3x", unsigned(symbol->StorageClass)) << ") "
             << "(nx " << unsigned(symbol->NumberOfAuxSymbols) << ") "
             << "0x" << format("%08x", unsigned(symbol->Value)) << " "
             << name << "\n";
      aux_count = symbol->NumberOfAuxSymbols;
    }
  }
}

static void PrintSymbolTable(const ObjectFile *o) {
  outs() << "SYMBOL TABLE:\n";

  if (const COFFObjectFile *coff = dyn_cast<const COFFObjectFile>(o))
    PrintCOFFSymbolTable(coff);
  else {
    for (symbol_iterator si = o->symbol_begin(), se = o->symbol_end();
         si != se; ++si) {
      StringRef Name;
      uint64_t Address;
      SymbolRef::Type Type;
      uint64_t Size;
      uint32_t Flags = si->getFlags();
      section_iterator Section = o->section_end();
      if (error(si->getName(Name))) continue;
      if (error(si->getAddress(Address))) continue;
      if (error(si->getType(Type))) continue;
      if (error(si->getSize(Size))) continue;
      if (error(si->getSection(Section))) continue;

      bool Global = Flags & SymbolRef::SF_Global;
      bool Weak = Flags & SymbolRef::SF_Weak;
      bool Absolute = Flags & SymbolRef::SF_Absolute;

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
      if (Absolute)
        outs() << "*ABS*";
      else if (Section == o->section_end())
        outs() << "*UND*";
      else {
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
}

static void PrintUnwindInfo(const ObjectFile *o) {
  outs() << "Unwind info:\n\n";

  if (const COFFObjectFile *coff = dyn_cast<COFFObjectFile>(o)) {
    printCOFFUnwindInfo(coff);
  } else {
    // TODO: Extract DWARF dump tool to objdump.
    errs() << "This operation is only currently supported "
              "for COFF object files.\n";
    return;
  }
}

static void printPrivateFileHeader(const ObjectFile *o) {
  if (o->isELF()) {
    printELFFileHeader(o);
  } else if (o->isCOFF()) {
    printCOFFFileHeader(o);
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
}

/// @brief Dump each object file in \a a;
static void DumpArchive(const Archive *a) {
  for (Archive::child_iterator i = a->child_begin(), e = a->child_end(); i != e;
       ++i) {
    OwningPtr<Binary> child;
    if (error_code EC = i->getAsBinary(child)) {
      // Ignore non-object files.
      if (EC != object_error::invalid_file_type)
        errs() << ToolName << ": '" << a->getFileName() << "': " << EC.message()
               << ".\n";
      continue;
    }
    if (ObjectFile *o = dyn_cast<ObjectFile>(child.get()))
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

  if (MachOOpt && Disassemble) {
    DisassembleInputMachO(file);
    return;
  }

  // Attempt to open the binary.
  ErrorOr<Binary *> BinaryOrErr = createBinary(file);
  if (error_code EC = BinaryOrErr.getError()) {
    errs() << ToolName << ": '" << file << "': " << EC.message() << ".\n";
    return;
  }
  OwningPtr<Binary> binary(BinaryOrErr.get());

  if (Archive *a = dyn_cast<Archive>(binary.get()))
    DumpArchive(a);
  else if (ObjectFile *o = dyn_cast<ObjectFile>(binary.get()))
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
      && !PrivateHeaders) {
    cl::PrintHelpMessage();
    return 2;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DumpInput);

  return 0;
}
