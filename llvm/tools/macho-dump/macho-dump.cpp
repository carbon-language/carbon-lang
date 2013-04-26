//===-- macho-dump.cpp - Mach Object Dumping Tool -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC/Mach-O LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachO.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
ShowSectionData("dump-section-data", cl::desc("Dump the contents of sections"),
                cl::init(false));

///

static const char *ProgramName;

static void Message(const char *Type, const Twine &Msg) {
  errs() << ProgramName << ": " << Type << ": " << Msg << "\n";
}

static int Error(const Twine &Msg) {
  Message("error", Msg);
  return 1;
}

static void Warning(const Twine &Msg) {
  Message("warning", Msg);
}

///

static void DumpSegmentCommandData(StringRef Name,
                                   uint64_t VMAddr, uint64_t VMSize,
                                   uint64_t FileOffset, uint64_t FileSize,
                                   uint32_t MaxProt, uint32_t InitProt,
                                   uint32_t NumSections, uint32_t Flags) {
  outs() << "  ('segment_name', '";
  outs().write_escaped(Name, /*UseHexEscapes=*/true) << "')\n";
  outs() << "  ('vm_addr', " << VMAddr << ")\n";
  outs() << "  ('vm_size', " << VMSize << ")\n";
  outs() << "  ('file_offset', " << FileOffset << ")\n";
  outs() << "  ('file_size', " << FileSize << ")\n";
  outs() << "  ('maxprot', " << MaxProt << ")\n";
  outs() << "  ('initprot', " << InitProt << ")\n";
  outs() << "  ('num_sections', " << NumSections << ")\n";
  outs() << "  ('flags', " << Flags << ")\n";
}

static int DumpSectionData(const MachOObjectFile &Obj, unsigned Index,
                           StringRef Name,
                           StringRef SegmentName, uint64_t Address,
                           uint64_t Size, uint32_t Offset,
                           uint32_t Align, uint32_t RelocationTableOffset,
                           uint32_t NumRelocationTableEntries,
                           uint32_t Flags, uint32_t Reserved1,
                           uint32_t Reserved2, uint64_t Reserved3 = ~0ULL) {
  outs() << "    # Section " << Index << "\n";
  outs() << "   (('section_name', '";
  outs().write_escaped(Name, /*UseHexEscapes=*/true) << "')\n";
  outs() << "    ('segment_name', '";
  outs().write_escaped(SegmentName, /*UseHexEscapes=*/true) << "')\n";
  outs() << "    ('address', " << Address << ")\n";
  outs() << "    ('size', " << Size << ")\n";
  outs() << "    ('offset', " << Offset << ")\n";
  outs() << "    ('alignment', " << Align << ")\n";
  outs() << "    ('reloc_offset', " << RelocationTableOffset << ")\n";
  outs() << "    ('num_reloc', " << NumRelocationTableEntries << ")\n";
  outs() << "    ('flags', " << format("0x%x", Flags) << ")\n";
  outs() << "    ('reserved1', " << Reserved1 << ")\n";
  outs() << "    ('reserved2', " << Reserved2 << ")\n";
  if (Reserved3 != ~0ULL)
    outs() << "    ('reserved3', " << Reserved3 << ")\n";
  outs() << "   ),\n";

  // Dump the relocation entries.
  outs() << "  ('_relocations', [\n";
  unsigned RelNum = 0;
  error_code EC;
  for (relocation_iterator I = Obj.getSectionRelBegin(Index),
         E = Obj.getSectionRelEnd(Index); I != E; I.increment(EC), ++RelNum) {
    macho::RelocationEntry RE = Obj.getRelocation(I->getRawDataRefImpl());
    outs() << "    # Relocation " << RelNum << "\n";
    outs() << "    (('word-0', " << format("0x%x", RE.Word0) << "),\n";
    outs() << "     ('word-1', " << format("0x%x", RE.Word1) << ")),\n";
  }
  outs() << "  ])\n";

  // Dump the section data, if requested.
  if (ShowSectionData) {
    outs() << "  ('_section_data', '";
    StringRef Data = Obj.getData().substr(Offset, Size);
    for (unsigned i = 0; i != Data.size(); ++i) {
      if (i && (i % 4) == 0)
        outs() << ' ';
      outs() << hexdigit((Data[i] >> 4) & 0xF, /*LowerCase=*/true);
      outs() << hexdigit((Data[i] >> 0) & 0xF, /*LowerCase=*/true);
    }
    outs() << "')\n";
  }

  return 0;
}

static int DumpSegmentCommand(const MachOObjectFile &Obj,
                              const MachOObjectFile::LoadCommandInfo &LCI) {
  macho::SegmentLoadCommand SLC = Obj.getSegmentLoadCommand(LCI);

  DumpSegmentCommandData(StringRef(SLC.Name, 16), SLC.VMAddress,
                         SLC.VMSize, SLC.FileOffset, SLC.FileSize,
                         SLC.MaxVMProtection, SLC.InitialVMProtection,
                         SLC.NumSections, SLC.Flags);

  // Dump the sections.
  outs() << "  ('sections', [\n";
  for (unsigned i = 0; i != SLC.NumSections; ++i) {
    macho::Section Sect = Obj.getSection(LCI, i);
    DumpSectionData(Obj, i, StringRef(Sect.Name, 16),
                    StringRef(Sect.SegmentName, 16), Sect.Address,
                    Sect.Size, Sect.Offset, Sect.Align,
                    Sect.RelocationTableOffset,
                    Sect.NumRelocationTableEntries, Sect.Flags,
                    Sect.Reserved1, Sect.Reserved2);
  }
  outs() << "  ])\n";

  return 0;
}

static int DumpSegment64Command(const MachOObjectFile &Obj,
                                const MachOObjectFile::LoadCommandInfo &LCI) {
  macho::Segment64LoadCommand SLC = Obj.getSegment64LoadCommand(LCI);
  DumpSegmentCommandData(StringRef(SLC.Name, 16), SLC.VMAddress,
                          SLC.VMSize, SLC.FileOffset, SLC.FileSize,
                          SLC.MaxVMProtection, SLC.InitialVMProtection,
                          SLC.NumSections, SLC.Flags);

  // Dump the sections.
  outs() << "  ('sections', [\n";
  for (unsigned i = 0; i != SLC.NumSections; ++i) {
    macho::Section64 Sect = Obj.getSection64(LCI, i);

    DumpSectionData(Obj, i, StringRef(Sect.Name, 16),
                    StringRef(Sect.SegmentName, 16), Sect.Address,
                    Sect.Size, Sect.Offset, Sect.Align,
                    Sect.RelocationTableOffset,
                    Sect.NumRelocationTableEntries, Sect.Flags,
                    Sect.Reserved1, Sect.Reserved2,
                    Sect.Reserved3);
  }
  outs() << "  ])\n";

  return 0;
}

static void DumpSymbolTableEntryData(const MachOObjectFile &Obj,
                                     unsigned Index, uint32_t StringIndex,
                                     uint8_t Type, uint8_t SectionIndex,
                                     uint16_t Flags, uint64_t Value,
                                     StringRef StringTable) {
  const char *Name = &StringTable.data()[StringIndex];
  outs() << "    # Symbol " << Index << "\n";
  outs() << "   (('n_strx', " << StringIndex << ")\n";
  outs() << "    ('n_type', " << format("0x%x", Type) << ")\n";
  outs() << "    ('n_sect', " << uint32_t(SectionIndex) << ")\n";
  outs() << "    ('n_desc', " << Flags << ")\n";
  outs() << "    ('n_value', " << Value << ")\n";
  outs() << "    ('_string', '" << Name << "')\n";
  outs() << "   ),\n";
}

static int DumpSymtabCommand(const MachOObjectFile &Obj) {
  macho::SymtabLoadCommand SLC = Obj.getSymtabLoadCommand();

  outs() << "  ('symoff', " << SLC.SymbolTableOffset << ")\n";
  outs() << "  ('nsyms', " << SLC.NumSymbolTableEntries << ")\n";
  outs() << "  ('stroff', " << SLC.StringTableOffset << ")\n";
  outs() << "  ('strsize', " << SLC.StringTableSize << ")\n";

  // Dump the string data.
  outs() << "  ('_string_data', '";
  StringRef StringTable = Obj.getStringTableData();
  outs().write_escaped(StringTable,
                       /*UseHexEscapes=*/true) << "')\n";

  // Dump the symbol table.
  outs() << "  ('_symbols', [\n";
  error_code EC;
  unsigned SymNum = 0;
  for (symbol_iterator I = Obj.begin_symbols(), E = Obj.end_symbols(); I != E;
       I.increment(EC), ++SymNum) {
    DataRefImpl DRI = I->getRawDataRefImpl();
    if (Obj.is64Bit()) {
      macho::Symbol64TableEntry STE = Obj.getSymbol64TableEntry(DRI);
      DumpSymbolTableEntryData(Obj, SymNum, STE.StringIndex, STE.Type,
                               STE.SectionIndex, STE.Flags, STE.Value,
                               StringTable);
    } else {
      macho::SymbolTableEntry STE = Obj.getSymbolTableEntry(DRI);
      DumpSymbolTableEntryData(Obj, SymNum, STE.StringIndex, STE.Type,
                               STE.SectionIndex, STE.Flags, STE.Value,
                               StringTable);
    }
  }
  outs() << "  ])\n";

  return 0;
}

static int DumpDysymtabCommand(const MachOObjectFile &Obj) {
  macho::DysymtabLoadCommand DLC = Obj.getDysymtabLoadCommand();

  outs() << "  ('ilocalsym', " << DLC.LocalSymbolsIndex << ")\n";
  outs() << "  ('nlocalsym', " << DLC.NumLocalSymbols << ")\n";
  outs() << "  ('iextdefsym', " << DLC.ExternalSymbolsIndex << ")\n";
  outs() << "  ('nextdefsym', " << DLC.NumExternalSymbols << ")\n";
  outs() << "  ('iundefsym', " << DLC.UndefinedSymbolsIndex << ")\n";
  outs() << "  ('nundefsym', " << DLC.NumUndefinedSymbols << ")\n";
  outs() << "  ('tocoff', " << DLC.TOCOffset << ")\n";
  outs() << "  ('ntoc', " << DLC.NumTOCEntries << ")\n";
  outs() << "  ('modtaboff', " << DLC.ModuleTableOffset << ")\n";
  outs() << "  ('nmodtab', " << DLC.NumModuleTableEntries << ")\n";
  outs() << "  ('extrefsymoff', " << DLC.ReferenceSymbolTableOffset << ")\n";
  outs() << "  ('nextrefsyms', "
         << DLC.NumReferencedSymbolTableEntries << ")\n";
  outs() << "  ('indirectsymoff', " << DLC.IndirectSymbolTableOffset << ")\n";
  outs() << "  ('nindirectsyms', "
         << DLC.NumIndirectSymbolTableEntries << ")\n";
  outs() << "  ('extreloff', " << DLC.ExternalRelocationTableOffset << ")\n";
  outs() << "  ('nextrel', " << DLC.NumExternalRelocationTableEntries << ")\n";
  outs() << "  ('locreloff', " << DLC.LocalRelocationTableOffset << ")\n";
  outs() << "  ('nlocrel', " << DLC.NumLocalRelocationTableEntries << ")\n";

  // Dump the indirect symbol table.
  outs() << "  ('_indirect_symbols', [\n";
  for (unsigned i = 0; i != DLC.NumIndirectSymbolTableEntries; ++i) {
    macho::IndirectSymbolTableEntry ISTE =
      Obj.getIndirectSymbolTableEntry(DLC, i);
    outs() << "    # Indirect Symbol " << i << "\n";
    outs() << "    (('symbol_index', "
           << format("0x%x", ISTE.Index) << "),),\n";
  }
  outs() << "  ])\n";

  return 0;
}

static int
DumpLinkeditDataCommand(const MachOObjectFile &Obj,
                        const MachOObjectFile::LoadCommandInfo &LCI) {
  macho::LinkeditDataLoadCommand LLC = Obj.getLinkeditDataLoadCommand(LCI);
  outs() << "  ('dataoff', " << LLC.DataOffset << ")\n"
         << "  ('datasize', " << LLC.DataSize << ")\n"
         << "  ('_addresses', [\n";

  SmallVector<uint64_t, 8> Addresses;
  Obj.ReadULEB128s(LLC.DataOffset, Addresses);
  for (unsigned i = 0, e = Addresses.size(); i != e; ++i)
    outs() << "    # Address " << i << '\n'
           << "    ('address', " << format("0x%x", Addresses[i]) << "),\n";

  outs() << "  ])\n";

  return 0;
}

static int
DumpDataInCodeDataCommand(const MachOObjectFile &Obj,
                          const MachOObjectFile::LoadCommandInfo &LCI) {
  macho::LinkeditDataLoadCommand LLC = Obj.getLinkeditDataLoadCommand(LCI);
  outs() << "  ('dataoff', " << LLC.DataOffset << ")\n"
         << "  ('datasize', " << LLC.DataSize << ")\n"
         << "  ('_data_regions', [\n";

  unsigned NumRegions = LLC.DataSize / 8;
  for (unsigned i = 0; i < NumRegions; ++i) {
    macho::DataInCodeTableEntry DICE =
      Obj.getDataInCodeTableEntry(LLC.DataOffset, i);
    outs() << "    # DICE " << i << "\n"
           << "    ('offset', " << DICE.Offset << ")\n"
           << "    ('length', " << DICE.Length << ")\n"
           << "    ('kind', " << DICE.Kind << ")\n";
  }

  outs() <<"  ])\n";

  return 0;
}

static int
DumpLinkerOptionsCommand(const MachOObjectFile &Obj,
                         const MachOObjectFile::LoadCommandInfo &LCI) {
  macho::LinkerOptionsLoadCommand LOLC = Obj.getLinkerOptionsLoadCommand(LCI);
   outs() << "  ('count', " << LOLC.Count << ")\n"
          << "  ('_strings', [\n";

   uint64_t DataSize = LOLC.Size - sizeof(macho::LinkerOptionsLoadCommand);
   const char *P = LCI.Ptr + sizeof(macho::LinkerOptionsLoadCommand);
   StringRef Data(P, DataSize);
   for (unsigned i = 0; i != LOLC.Count; ++i) {
     std::pair<StringRef,StringRef> Split = Data.split('\0');
     outs() << "\t\"";
     outs().write_escaped(Split.first);
     outs() << "\",\n";
     Data = Split.second;
   }
   outs() <<"  ])\n";

  return 0;
}

static int DumpLoadCommand(const MachOObjectFile &Obj,
                           MachOObjectFile::LoadCommandInfo &LCI) {
  switch (LCI.C.Type) {
  case macho::LCT_Segment:
    return DumpSegmentCommand(Obj, LCI);
  case macho::LCT_Segment64:
    return DumpSegment64Command(Obj, LCI);
  case macho::LCT_Symtab:
    return DumpSymtabCommand(Obj);
  case macho::LCT_Dysymtab:
    return DumpDysymtabCommand(Obj);
  case macho::LCT_CodeSignature:
  case macho::LCT_SegmentSplitInfo:
  case macho::LCT_FunctionStarts:
    return DumpLinkeditDataCommand(Obj, LCI);
  case macho::LCT_DataInCode:
    return DumpDataInCodeDataCommand(Obj, LCI);
  case macho::LCT_LinkerOptions:
    return DumpLinkerOptionsCommand(Obj, LCI);
  default:
    Warning("unknown load command: " + Twine(LCI.C.Type));
    return 0;
  }
}


static int DumpLoadCommand(const MachOObjectFile &Obj, unsigned Index,
                           MachOObjectFile::LoadCommandInfo &LCI) {
  outs() << "  # Load Command " << Index << "\n"
         << " (('command', " << LCI.C.Type << ")\n"
         << "  ('size', " << LCI.C.Size << ")\n";
  int Res = DumpLoadCommand(Obj, LCI);
  outs() << " ),\n";
  return Res;
}

static void printHeader(const MachOObjectFile *Obj,
                        const macho::Header &Header) {
  outs() << "('cputype', " << Header.CPUType << ")\n";
  outs() << "('cpusubtype', " << Header.CPUSubtype << ")\n";
  outs() << "('filetype', " << Header.FileType << ")\n";
  outs() << "('num_load_commands', " << Header.NumLoadCommands << ")\n";
  outs() << "('load_commands_size', " << Header.SizeOfLoadCommands << ")\n";
  outs() << "('flag', " << Header.Flags << ")\n";

  // Print extended header if 64-bit.
  if (Obj->is64Bit()) {
    macho::Header64Ext Header64Ext = Obj->getHeader64Ext();
    outs() << "('reserved', " << Header64Ext.Reserved << ")\n";
  }
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm Mach-O dumping tool\n");

  OwningPtr<Binary> Binary;
  if (error_code EC = createBinary(InputFile, Binary))
    return Error("unable to read input: '" + EC.message() + "'");

  const MachOObjectFile *InputObject = dyn_cast<MachOObjectFile>(Binary.get());
  if (!InputObject)
    return Error("Not a MachO object");

  // Print the header
  macho::Header Header = InputObject->getHeader();
  printHeader(InputObject, Header);

  // Print the load commands.
  int Res = 0;
  MachOObjectFile::LoadCommandInfo Command =
    InputObject->getFirstLoadCommandInfo();
  outs() << "('load_commands', [\n";
  for (unsigned i = 0; ; ++i) {
    if (DumpLoadCommand(*InputObject, i, Command))
      break;

    if (i == Header.NumLoadCommands - 1)
      break;
    Command = InputObject->getNextLoadCommandInfo(Command);
  }
  outs() << "])\n";

  return Res;
}
