//===-- MachODump.cpp - Object file dumping utility for llvm --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachO-specific dumper for llvm-objdump.
//
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "MCFunction.h"
#include "llvm/Support/MachO.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>
using namespace llvm;
using namespace object;

static cl::opt<bool>
  CFG("cfg", cl::desc("Create a CFG for every symbol in the object file and"
                      "write it to a graphviz file (MachO-only)"));

static const Target *GetTarget(const MachOObject *MachOObj) {
  // Figure out the target triple.
  llvm::Triple TT("unknown-unknown-unknown");
  switch (MachOObj->getHeader().CPUType) {
  case llvm::MachO::CPUTypeI386:
    TT.setArch(Triple::ArchType(Triple::x86));
    break;
  case llvm::MachO::CPUTypeX86_64:
    TT.setArch(Triple::ArchType(Triple::x86_64));
    break;
  case llvm::MachO::CPUTypeARM:
    TT.setArch(Triple::ArchType(Triple::arm));
    break;
  case llvm::MachO::CPUTypePowerPC:
    TT.setArch(Triple::ArchType(Triple::ppc));
    break;
  case llvm::MachO::CPUTypePowerPC64:
    TT.setArch(Triple::ArchType(Triple::ppc64));
    break;
  }

  TripleName = TT.str();

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget)
    return TheTarget;

  errs() << "llvm-objdump: error: unable to get target for '" << TripleName
         << "', see --version and --triple.\n";
  return 0;
}

struct Section {
  char Name[16];
  uint64_t Address;
  uint64_t Size;
  uint32_t Offset;
  uint32_t NumRelocs;
  uint64_t RelocTableOffset;
};

struct Symbol {
  uint64_t Value;
  uint32_t StringIndex;
  uint8_t SectionIndex;
  bool operator<(const Symbol &RHS) const { return Value < RHS.Value; }
};

static void DumpAddress(uint64_t Address, ArrayRef<Section> Sections,
                        MachOObject *MachOObj, raw_ostream &OS) {
  for (unsigned i = 0; i != Sections.size(); ++i) {
    uint64_t addr = Address-Sections[i].Address;
    if (Sections[i].Address <= Address &&
        Sections[i].Address + Sections[i].Size > Address) {
      StringRef bytes = MachOObj->getData(Sections[i].Offset,
                                          Sections[i].Size);
      if (!strcmp(Sections[i].Name, "__cstring"))
        OS << '"' << bytes.substr(addr, bytes.find('\0', addr)) << '"';
      if (!strcmp(Sections[i].Name, "__cfstring"))
        OS << "@\"" << bytes.substr(addr, bytes.find('\0', addr)) << '"';
    }
  }
}

void llvm::DisassembleInputMachO(StringRef Filename) {
  OwningPtr<MemoryBuffer> Buff;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
    errs() << "llvm-objdump: " << Filename << ": " << ec.message() << "\n";
    return;
  }

  OwningPtr<MachOObject> MachOObj(MachOObject::LoadFromBuffer(Buff.take()));

  const Target *TheTarget = GetTarget(MachOObj.get());
  if (!TheTarget) {
    // GetTarget prints out stuff.
    return;
  }
  const MCInstrInfo *InstrInfo = TheTarget->createMCInstrInfo();
  OwningPtr<MCInstrAnalysis>
    InstrAnalysis(TheTarget->createMCInstrAnalysis(InstrInfo));

  // Set up disassembler.
  OwningPtr<const MCAsmInfo> AsmInfo(TheTarget->createMCAsmInfo(TripleName));

  if (!AsmInfo) {
    errs() << "error: no assembly info for target " << TripleName << "\n";
    return;
  }

  OwningPtr<const MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TripleName, "", ""));

  if (!STI) {
    errs() << "error: no subtarget info for target " << TripleName << "\n";
    return;
  }

  OwningPtr<const MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI));
  if (!DisAsm) {
    errs() << "error: no disassembler for target " << TripleName << "\n";
    return;
  }

  int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
  OwningPtr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
        AsmPrinterVariant, *AsmInfo, *STI));
  if (!IP) {
    errs() << "error: no instruction printer for target " << TripleName << '\n';
    return;
  }

  outs() << '\n';
  outs() << Filename << ":\n\n";

  const macho::Header &Header = MachOObj->getHeader();

  const MachOObject::LoadCommandInfo *SymtabLCI = 0;
  for (unsigned i = 0; i != Header.NumLoadCommands; ++i) {
    const MachOObject::LoadCommandInfo &LCI = MachOObj->getLoadCommandInfo(i);
    switch (LCI.Command.Type) {
    case macho::LCT_Symtab:
      SymtabLCI = &LCI;
      break;
    }
  }

  // Read and register the symbol table data.
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLC;
  MachOObj->ReadSymtabLoadCommand(*SymtabLCI, SymtabLC);
  MachOObj->RegisterStringTable(*SymtabLC);

  std::vector<Section> Sections;
  std::vector<Symbol> Symbols;
  std::vector<Symbol> UnsortedSymbols; // FIXME: duplication
  SmallVector<uint64_t, 8> FoundFns;

  for (unsigned i = 0; i != Header.NumLoadCommands; ++i) {
    const MachOObject::LoadCommandInfo &LCI = MachOObj->getLoadCommandInfo(i);
    if (LCI.Command.Type == macho::LCT_Segment) {
      InMemoryStruct<macho::SegmentLoadCommand> SegmentLC;
      MachOObj->ReadSegmentLoadCommand(LCI, SegmentLC);

      for (unsigned SectNum = 0; SectNum != SegmentLC->NumSections; ++SectNum) {
        InMemoryStruct<macho::Section> Sect;
        MachOObj->ReadSection(LCI, SectNum, Sect);

        Section S;
        memcpy(S.Name, Sect->Name, 16);
        S.Address = Sect->Address;
        S.Size = Sect->Size;
        S.Offset = Sect->Offset;
        S.NumRelocs = Sect->NumRelocationTableEntries;
        S.RelocTableOffset = Sect->RelocationTableOffset;
        Sections.push_back(S);

        for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
          InMemoryStruct<macho::SymbolTableEntry> STE;
          MachOObj->ReadSymbolTableEntry(SymtabLC->SymbolTableOffset, i, STE);

          Symbol S;
          S.StringIndex = STE->StringIndex;
          S.SectionIndex = STE->SectionIndex;
          S.Value = STE->Value;
          Symbols.push_back(S);
          UnsortedSymbols.push_back(Symbols.back());
        }
      }
    } else if (LCI.Command.Type == macho::LCT_Segment64) {
      InMemoryStruct<macho::Segment64LoadCommand> Segment64LC;
      MachOObj->ReadSegment64LoadCommand(LCI, Segment64LC);

      for (unsigned SectNum = 0; SectNum != Segment64LC->NumSections; ++SectNum) {
        InMemoryStruct<macho::Section64> Sect64;
        MachOObj->ReadSection64(LCI, SectNum, Sect64);

        Section S;
        memcpy(S.Name, Sect64->Name, 16);
        S.Address = Sect64->Address;
        S.Size = Sect64->Size;
        S.Offset = Sect64->Offset;
        S.NumRelocs = Sect64->NumRelocationTableEntries;
        S.RelocTableOffset = Sect64->RelocationTableOffset;
        Sections.push_back(S);

        for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
          InMemoryStruct<macho::Symbol64TableEntry> STE;
          MachOObj->ReadSymbol64TableEntry(SymtabLC->SymbolTableOffset, i, STE);

          Symbol S;
          S.StringIndex = STE->StringIndex;
          S.SectionIndex = STE->SectionIndex;
          S.Value = STE->Value;
          Symbols.push_back(S);
          UnsortedSymbols.push_back(Symbols.back());
        }
      }
    } else if (LCI.Command.Type == macho::LCT_FunctionStarts) {
      InMemoryStruct<macho::LinkeditDataLoadCommand> LLC;
      MachOObj->ReadLinkeditDataLoadCommand(LCI, LLC);

      MachOObj->ReadULEB128s(LLC->DataOffset, FoundFns);
    }
  }

  std::map<uint64_t, MCFunction*> FunctionMap;

  // Sort the symbols by address, just in case they didn't come in that way.
  array_pod_sort(Symbols.begin(), Symbols.end());

#ifndef NDEBUG
  raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#else
  raw_ostream &DebugOut = nulls();
#endif

  SmallVector<MCFunction, 16> Functions;

  for (unsigned SectIdx = 0; SectIdx != Sections.size(); SectIdx++) {
    if (strcmp(Sections[SectIdx].Name, "__text"))
      continue;

    uint64_t VMAddr = Sections[SectIdx].Address - Sections[SectIdx].Offset;
    for (unsigned i = 0, e = FoundFns.size(); i != e; ++i)
      FunctionMap.insert(std::pair<uint64_t,MCFunction*>(FoundFns[i]+VMAddr,0));

    StringRef Bytes = MachOObj->getData(Sections[SectIdx].Offset,
                                        Sections[SectIdx].Size);
    StringRefMemoryObject memoryObject(Bytes);
    bool symbolTableWorked = false;

    std::vector<std::pair<uint64_t, uint32_t> > Relocs;
    for (unsigned j = 0; j != Sections[SectIdx].NumRelocs; ++j) {
      InMemoryStruct<macho::RelocationEntry> RE;
      MachOObj->ReadRelocationEntry(Sections[SectIdx].RelocTableOffset, j, RE);
      Relocs.push_back(std::make_pair(RE->Word0, RE->Word1 & 0xffffff));
    }
    array_pod_sort(Relocs.begin(), Relocs.end());

    for (unsigned SymIdx = 0; SymIdx != Symbols.size(); SymIdx++) {
      if ((unsigned)Symbols[SymIdx].SectionIndex - 1 != SectIdx)
        continue;

      uint64_t Start = Symbols[SymIdx].Value - Sections[SectIdx].Address;
      uint64_t End = (SymIdx+1 == Symbols.size() ||
          Symbols[SymIdx].SectionIndex != Symbols[SymIdx+1].SectionIndex) ?
          Sections[SectIdx].Size :
          Symbols[SymIdx+1].Value - Sections[SectIdx].Address;
      uint64_t Size;

      if (Start >= End)
        continue;

      symbolTableWorked = true;

      if (!CFG) {
        outs() << MachOObj->getStringAtIndex(Symbols[SymIdx].StringIndex)
          << ":\n";
        for (uint64_t Index = Start; Index < End; Index += Size) {
          MCInst Inst;

          if (DisAsm->getInstruction(Inst, Size, memoryObject, Index,
                                     DebugOut, nulls())) {
            outs() << format("%8llx:\t", Sections[SectIdx].Address + Index);
            DumpBytes(StringRef(Bytes.data() + Index, Size));
            IP->printInst(&Inst, outs(), "");
            outs() << "\n";
          } else {
            errs() << "llvm-objdump: warning: invalid instruction encoding\n";
            if (Size == 0)
              Size = 1; // skip illegible bytes
          }
        }
      } else {
        // Create CFG and use it for disassembly.
        SmallVector<uint64_t, 16> Calls;
        MCFunction f =
          MCFunction::createFunctionFromMC(
              MachOObj->getStringAtIndex(Symbols[SymIdx].StringIndex),
              DisAsm.get(),
              memoryObject, Start, End,
              InstrAnalysis.get(), DebugOut,
              Calls);

        Functions.push_back(f);
        FunctionMap[Start] = &Functions.back();

        for (unsigned i = 0, e = Calls.size(); i != e; ++i)
          FunctionMap.insert(std::pair<uint64_t, MCFunction*>(Calls[i], 0));
      }
    }

    if (CFG) {
      if (!symbolTableWorked) {
        // Create CFG and use it for disassembly.
        SmallVector<uint64_t, 16> Calls;
        MCFunction f =
          MCFunction::createFunctionFromMC("__TEXT", DisAsm.get(),
              memoryObject, 0, Sections[SectIdx].Size,
              InstrAnalysis.get(), DebugOut,
              Calls);

        Functions.push_back(f);
        FunctionMap[Sections[SectIdx].Offset] = &Functions.back();

        for (unsigned i = 0, e = Calls.size(); i != e; ++i)
          FunctionMap.insert(std::pair<uint64_t, MCFunction*>(Calls[i], 0));
      }
      for (std::map<uint64_t, MCFunction*>::iterator mi = FunctionMap.begin(),
           me = FunctionMap.end(); mi != me; ++mi)
        if (mi->second == 0) {
          SmallVector<uint64_t, 16> Calls;
          MCFunction f =
            MCFunction::createFunctionFromMC("unknown", DisAsm.get(),
                                             memoryObject, mi->first,
                                             Sections[SectIdx].Size,
                                             InstrAnalysis.get(), DebugOut,
                                             Calls);
          Functions.push_back(f);
          mi->second = &Functions.back();
          for (unsigned i = 0, e = Calls.size(); i != e; ++i)
            if (FunctionMap.insert(std::pair<uint64_t, MCFunction*>(Calls[i],0))
                                                                        .second)
              mi = FunctionMap.begin();
        }

      DenseSet<uint64_t> PrintedBlocks;
      for (unsigned ffi = 0, ffe = Functions.size(); ffi != ffe; ++ffi) {
        MCFunction &f = Functions[ffi];
        for (MCFunction::iterator fi = f.begin(), fe = f.end(); fi != fe; ++fi){
          if (!PrintedBlocks.insert(fi->first).second)
            continue;
          bool hasPreds = FunctionMap.find(fi->first) != FunctionMap.end();

          // Only print blocks that have predecessors.
          // FIXME: Slow.
          for (MCFunction::iterator pi = f.begin(), pe = f.end(); pi != pe;
              ++pi)
            if (pi->second.contains(fi->first)) {
              hasPreds = true;
              break;
            }

          // Data block.
          if (!hasPreds && fi != f.begin()) {
            uint64_t End = llvm::next(fi) == fe ? Sections[SectIdx].Size :
                                                  llvm::next(fi)->first;
            outs() << "# " << End-fi->first << " bytes of data:\n";
            for (unsigned pos = fi->first; pos != End; ++pos) {
              outs() << format("%8x:\t", Sections[SectIdx].Address + pos);
              DumpBytes(StringRef(Bytes.data() + pos, 1));
              outs() << format("\t.byte 0x%02x\n", (uint8_t)Bytes[pos]);
            }
            continue;
          }

          if (fi->second.contains(fi->first))
            outs() << "# Loop begin:\n";

          for (unsigned ii = 0, ie = fi->second.getInsts().size(); ii != ie;
               ++ii) {
            const MCDecodedInst &Inst = fi->second.getInsts()[ii];
            if (FunctionMap.find(Sections[SectIdx].Address + Inst.Address) !=
                FunctionMap.end())
              outs() << FunctionMap[Sections[SectIdx].Address + Inst.Address]->
                                                             getName() << ":\n";
            outs() << format("%8llx:\t", Sections[SectIdx].Address +
                                         Inst.Address);
            DumpBytes(StringRef(Bytes.data() + Inst.Address, Inst.Size));
            // Simple loops.
            if (fi->second.contains(fi->first))
              outs() << '\t';
            IP->printInst(&Inst.Inst, outs(), "");
            for (unsigned j = 0; j != Relocs.size(); ++j)
              if (Relocs[j].first >= Sections[SectIdx].Address + Inst.Address &&
                  Relocs[j].first < Sections[SectIdx].Address + Inst.Address +
                                    Inst.Size) {
                outs() << "\t# "
                   << MachOObj->getStringAtIndex(
                                  UnsortedSymbols[Relocs[j].second].StringIndex)
                   << ' ';
                DumpAddress(UnsortedSymbols[Relocs[j].second].Value, Sections,
                            MachOObj.get(), outs());
              }
            uint64_t targ = InstrAnalysis->evaluateBranch(Inst.Inst,
                                                          Inst.Address,
                                                          Inst.Size);
            if (targ != -1ULL)
              DumpAddress(targ, Sections, MachOObj.get(), outs());

            outs() << '\n';
          }
        }

        // Start a new dot file.
        std::string Error;
        raw_fd_ostream Out((f.getName().str() + ".dot").c_str(), Error);
        if (!Error.empty()) {
          errs() << "llvm-objdump: warning: " << Error << '\n';
          continue;
        }

        Out << "digraph " << f.getName() << " {\n";
        Out << "graph [ rankdir = \"LR\" ];\n";
        for (MCFunction::iterator i = f.begin(), e = f.end(); i != e; ++i) {
          bool hasPreds = false;
          // Only print blocks that have predecessors.
          // FIXME: Slow.
          for (MCFunction::iterator pi = f.begin(), pe = f.end(); pi != pe;
               ++pi)
            if (pi->second.contains(i->first)) {
              hasPreds = true;
              break;
            }

          if (!hasPreds && i != f.begin())
            continue;

          Out << '"' << i->first << "\" [ label=\"<a>";
          // Print instructions.
          for (unsigned ii = 0, ie = i->second.getInsts().size(); ii != ie;
               ++ii) {
            // Escape special chars and print the instruction in mnemonic form.
            std::string Str;
            raw_string_ostream OS(Str);
            IP->printInst(&i->second.getInsts()[ii].Inst, OS, "");
            Out << DOT::EscapeString(OS.str()) << '|';
          }
          Out << "<o>\" shape=\"record\" ];\n";

          // Add edges.
          for (MCBasicBlock::succ_iterator si = i->second.succ_begin(),
              se = i->second.succ_end(); si != se; ++si)
            Out << i->first << ":o -> " << *si <<":a\n";
        }
        Out << "}\n";
      }
    }
  }
}
