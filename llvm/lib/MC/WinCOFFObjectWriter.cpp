//===-- llvm/MC/WinCOFFObjectWriter.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of a Win32 COFF object file writer.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "WinCOFFObjectWriter"

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCSectionCOFF.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdio>

using namespace llvm;

namespace {
typedef llvm::SmallString<COFF::NameSize> name;

enum AuxiliaryType {
  ATFunctionDefinition,
  ATbfAndefSymbol,
  ATWeakExternal,
  ATFile,
  ATSectionDefinition
};

struct AuxSymbol {
  AuxiliaryType   AuxType;
  COFF::Auxiliary Aux;
};

class COFFSymbol {
public:
  COFF::symbol Data;

  typedef llvm::SmallVector<AuxSymbol, 1> AuxiliarySymbols;

  name             Name;
  size_t           Index;
  AuxiliarySymbols Aux;
  COFFSymbol      *Other;

  MCSymbolData const *MCData;

  COFFSymbol(llvm::StringRef name, size_t index);
  size_t size() const;
  void set_name_offset(uint32_t Offset);
};

// This class contains staging data for a COFF relocation entry.
struct COFFRelocation {
  COFF::relocation Data;
  COFFSymbol          *Symb;

  COFFRelocation() : Symb(NULL) {}
  static size_t size() { return COFF::RelocationSize; }
};

typedef std::vector<COFFRelocation> relocations;

class COFFSection {
public:
  COFF::section Header;

  std::string          Name;
  size_t               Number;
  MCSectionData const *MCData;
  COFFSymbol              *Symb;
  relocations          Relocations;

  COFFSection(llvm::StringRef name, size_t Index);
  static size_t size();
};

// This class holds the COFF string table.
class StringTable {
  typedef llvm::StringMap<size_t> map;
  map Map;

  void update_length();
public:
  std::vector<char> Data;

  StringTable();
  size_t size() const;
  size_t insert(llvm::StringRef String);
};

class WinCOFFObjectWriter : public MCObjectWriter {
public:

  typedef std::vector<COFFSymbol*>  symbols;
  typedef std::vector<COFFSection*> sections;

  typedef StringMap<COFFSymbol *>  name_symbol_map;
  typedef StringMap<COFFSection *> name_section_map;

  typedef DenseMap<MCSymbolData const *, COFFSymbol *>   symbol_map;
  typedef DenseMap<MCSectionData const *, COFFSection *> section_map;

  // Root level file contents.
  COFF::header Header;
  sections     Sections;
  symbols      Symbols;
  StringTable  Strings;

  // Maps used during object file creation.
  section_map SectionMap;
  symbol_map  SymbolMap;

  WinCOFFObjectWriter(raw_ostream &OS);
  ~WinCOFFObjectWriter();

  COFFSymbol *createSymbol(llvm::StringRef Name);
  COFFSection *createSection(llvm::StringRef Name);

  void InitCOFFEntity(COFFSymbol &Symbol);
  void InitCOFFEntity(COFFSection &Section);

  template <typename object_t, typename list_t>
  object_t *createCOFFEntity(llvm::StringRef Name, list_t &List);

  void DefineSection(MCSectionData const &SectionData);
  void DefineSymbol(MCSymbolData const &SymbolData, MCAssembler &Assembler);

  bool ExportSection(COFFSection *S);
  bool ExportSymbol(MCSymbolData const &SymbolData, MCAssembler &Asm);

  // Entity writing methods.

  void WriteFileHeader(const COFF::header &Header);
  void WriteSymbol(const COFFSymbol *S);
  void WriteAuxiliarySymbols(const COFFSymbol::AuxiliarySymbols &S);
  void WriteSectionHeader(const COFF::section &S);
  void WriteRelocation(const COFF::relocation &R);

  // MCObjectWriter interface implementation.

  void ExecutePostLayoutBinding(MCAssembler &Asm);

  void RecordRelocation(const MCAssembler &Asm,
                        const MCAsmLayout &Layout,
                        const MCFragment *Fragment,
                        const MCFixup &Fixup,
                        MCValue Target,
                        uint64_t &FixedValue);

  void WriteObject(const MCAssembler &Asm, const MCAsmLayout &Layout);
};
}

static inline void write_uint32_le(void *Data, uint32_t const &Value) {
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Data);
  Ptr[0] = (Value & 0x000000FF) >>  0;
  Ptr[1] = (Value & 0x0000FF00) >>  8;
  Ptr[2] = (Value & 0x00FF0000) >> 16;
  Ptr[3] = (Value & 0xFF000000) >> 24;
}

static inline void write_uint16_le(void *Data, uint16_t const &Value) {
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Data);
  Ptr[0] = (Value & 0x00FF) >> 0;
  Ptr[1] = (Value & 0xFF00) >> 8;
}

static inline void write_uint8_le(void *Data, uint8_t const &Value) {
  uint8_t *Ptr = reinterpret_cast<uint8_t *>(Data);
  Ptr[0] = (Value & 0xFF) >> 0;
}

//------------------------------------------------------------------------------
// Symbol class implementation

COFFSymbol::COFFSymbol(llvm::StringRef name, size_t index)
      : Name(name.begin(), name.end()), Index(-1)
      , Other(NULL), MCData(NULL) {
  memset(&Data, 0, sizeof(Data));
}

size_t COFFSymbol::size() const {
  return COFF::SymbolSize + (Data.NumberOfAuxSymbols * COFF::SymbolSize);
}

// In the case that the name does not fit within 8 bytes, the offset
// into the string table is stored in the last 4 bytes instead, leaving
// the first 4 bytes as 0.
void COFFSymbol::set_name_offset(uint32_t Offset) {
  write_uint32_le(Data.Name + 0, 0);
  write_uint32_le(Data.Name + 4, Offset);
}

//------------------------------------------------------------------------------
// Section class implementation

COFFSection::COFFSection(llvm::StringRef name, size_t Index)
       : Name(name), Number(Index + 1)
       , MCData(NULL), Symb(NULL) {
  memset(&Header, 0, sizeof(Header));
}

size_t COFFSection::size() {
  return COFF::SectionSize;
}

//------------------------------------------------------------------------------
// StringTable class implementation

/// Write the length of the string table into Data.
/// The length of the string table includes uint32 length header.
void StringTable::update_length() {
  write_uint32_le(&Data.front(), Data.size());
}

StringTable::StringTable() {
  // The string table data begins with the length of the entire string table
  // including the length header. Allocate space for this header.
  Data.resize(4);
}

size_t StringTable::size() const {
  return Data.size();
}

/// Add String to the table iff it is not already there.
/// @returns the index into the string table where the string is now located.
size_t StringTable::insert(llvm::StringRef String) {
  map::iterator i = Map.find(String);

  if (i != Map.end())
    return i->second;

  size_t Offset = Data.size();

  // Insert string data into string table.
  Data.insert(Data.end(), String.begin(), String.end());
  Data.push_back('\0');

  // Put a reference to it in the map.
  Map[String] = Offset;

  // Update the internal length field.
  update_length();

  return Offset;
}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter class implementation

WinCOFFObjectWriter::WinCOFFObjectWriter(raw_ostream &OS)
                                : MCObjectWriter(OS, true) {
  memset(&Header, 0, sizeof(Header));
  // TODO: Move magic constant out to COFF.h
  Header.Machine = 0x14C; // x86
}

WinCOFFObjectWriter::~WinCOFFObjectWriter() {
  for (symbols::iterator I = Symbols.begin(), E = Symbols.end(); I != E; ++I)
    delete *I;
  for (sections::iterator I = Sections.begin(), E = Sections.end(); I != E; ++I)
    delete *I;
}

COFFSymbol *WinCOFFObjectWriter::createSymbol(llvm::StringRef Name) {
  return createCOFFEntity<COFFSymbol>(Name, Symbols);
}

COFFSection *WinCOFFObjectWriter::createSection(llvm::StringRef Name) {
  return createCOFFEntity<COFFSection>(Name, Sections);
}

/// This function initializes a symbol by entering its name into the string
/// table if it is too long to fit in the symbol table header.
void WinCOFFObjectWriter::InitCOFFEntity(COFFSymbol &S) {
  if (S.Name.size() > COFF::NameSize) {
    size_t StringTableEntry = Strings.insert(S.Name.c_str());

    S.set_name_offset(StringTableEntry);
  } else
    memcpy(S.Data.Name, S.Name.c_str(), S.Name.size());
}

/// This function initializes a section by entering its name into the string
/// table if it is too long to fit in the section table header.
void WinCOFFObjectWriter::InitCOFFEntity(COFFSection &S) {
  if (S.Name.size() > COFF::NameSize) {
    size_t StringTableEntry = Strings.insert(S.Name.c_str());

    // FIXME: Why is this number 999999? This number is never mentioned in the
    // spec. I'm assuming this is due to the printed value needing to fit into
    // the S.Header.Name field. In which case why not 9999999 (7 9's instead of
    // 6)? The spec does not state if this entry should be null terminated in
    // this case, and thus this seems to be the best way to do it. I think I
    // just solved my own FIXME...
    if (StringTableEntry > 999999)
      report_fatal_error("COFF string table is greater than 999999 bytes.");

    sprintf(S.Header.Name, "/%d", (unsigned)StringTableEntry);
  } else
    memcpy(S.Header.Name, S.Name.c_str(), S.Name.size());
}

/// A template used to lookup or create a symbol/section, and initialize it if
/// needed.
template <typename object_t, typename list_t>
object_t *WinCOFFObjectWriter::createCOFFEntity(llvm::StringRef Name,
                                                list_t &List) {
  object_t *Object = new object_t(Name, List.size());

  InitCOFFEntity(*Object);

  List.push_back(Object);

  return Object;
}

/// This function takes a section data object from the assembler
/// and creates the associated COFF section staging object.
void WinCOFFObjectWriter::DefineSection(MCSectionData const &SectionData) {
  // FIXME: Not sure how to verify this (at least in a debug build).
  MCSectionCOFF const &Sec =
    static_cast<MCSectionCOFF const &>(SectionData.getSection());

  COFFSection *coff_section = createSection(Sec.getSectionName());
  COFFSymbol  *coff_symbol = createSymbol(Sec.getSectionName());

  coff_section->Symb = coff_symbol;
  coff_symbol->Data.StorageClass = COFF::IMAGE_SYM_CLASS_STATIC;
  coff_symbol->Data.SectionNumber = coff_section->Number;

  // In this case the auxiliary symbol is a Section Definition.
  coff_symbol->Aux.resize(1);
  memset(&coff_symbol->Aux[0], 0, sizeof(coff_symbol->Aux[0]));
  coff_symbol->Aux[0].AuxType = ATSectionDefinition;
  coff_symbol->Aux[0].Aux.SectionDefinition.Number = coff_section->Number;
  coff_symbol->Aux[0].Aux.SectionDefinition.Selection = Sec.getSelection();

  coff_section->Header.Characteristics = Sec.getCharacteristics();

  uint32_t &Characteristics = coff_section->Header.Characteristics;
  switch (SectionData.getAlignment()) {
  case 1:    Characteristics |= COFF::IMAGE_SCN_ALIGN_1BYTES;    break;
  case 2:    Characteristics |= COFF::IMAGE_SCN_ALIGN_2BYTES;    break;
  case 4:    Characteristics |= COFF::IMAGE_SCN_ALIGN_4BYTES;    break;
  case 8:    Characteristics |= COFF::IMAGE_SCN_ALIGN_8BYTES;    break;
  case 16:   Characteristics |= COFF::IMAGE_SCN_ALIGN_16BYTES;   break;
  case 32:   Characteristics |= COFF::IMAGE_SCN_ALIGN_32BYTES;   break;
  case 64:   Characteristics |= COFF::IMAGE_SCN_ALIGN_64BYTES;   break;
  case 128:  Characteristics |= COFF::IMAGE_SCN_ALIGN_128BYTES;  break;
  case 256:  Characteristics |= COFF::IMAGE_SCN_ALIGN_256BYTES;  break;
  case 512:  Characteristics |= COFF::IMAGE_SCN_ALIGN_512BYTES;  break;
  case 1024: Characteristics |= COFF::IMAGE_SCN_ALIGN_1024BYTES; break;
  case 2048: Characteristics |= COFF::IMAGE_SCN_ALIGN_2048BYTES; break;
  case 4096: Characteristics |= COFF::IMAGE_SCN_ALIGN_4096BYTES; break;
  case 8192: Characteristics |= COFF::IMAGE_SCN_ALIGN_8192BYTES; break;
  default:
    llvm_unreachable("unsupported section alignment");
  }

  // Bind internal COFF section to MC section.
  coff_section->MCData = &SectionData;
  SectionMap[&SectionData] = coff_section;
}

/// This function takes a section data object from the assembler
/// and creates the associated COFF symbol staging object.
void WinCOFFObjectWriter::DefineSymbol(MCSymbolData const &SymbolData,
                                        MCAssembler &Assembler) {
  COFFSymbol *coff_symbol = createSymbol(SymbolData.getSymbol().getName());

  coff_symbol->Data.Type         = (SymbolData.getFlags() & 0x0000FFFF) >>  0;
  coff_symbol->Data.StorageClass = (SymbolData.getFlags() & 0x00FF0000) >> 16;

  // If no storage class was specified in the streamer, define it here.
  if (coff_symbol->Data.StorageClass == 0) {
    bool external = SymbolData.isExternal() || (SymbolData.Fragment == NULL);

    coff_symbol->Data.StorageClass =
      external ? COFF::IMAGE_SYM_CLASS_EXTERNAL : COFF::IMAGE_SYM_CLASS_STATIC;
  }

  if (SymbolData.getFlags() & COFF::SF_WeakReference) {
    coff_symbol->Data.StorageClass = COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL;

    const MCExpr *Value = SymbolData.getSymbol().getVariableValue();

    // FIXME: This assert message isn't very good.
    assert(Value->getKind() == MCExpr::SymbolRef &&
           "Value must be a SymbolRef!");

    const MCSymbolRefExpr *SymbolRef =
      static_cast<const MCSymbolRefExpr *>(Value);

    const MCSymbolData &OtherSymbolData =
      Assembler.getSymbolData(SymbolRef->getSymbol());

    // FIXME: This assert message isn't very good.
    assert(SymbolMap.find(&OtherSymbolData) != SymbolMap.end() &&
           "OtherSymbolData must be in the symbol map!");

    coff_symbol->Other = SymbolMap[&OtherSymbolData];

    // Setup the Weak External auxiliary symbol.
    coff_symbol->Aux.resize(1);
    memset(&coff_symbol->Aux[0], 0, sizeof(coff_symbol->Aux[0]));
    coff_symbol->Aux[0].AuxType = ATWeakExternal;
    coff_symbol->Aux[0].Aux.WeakExternal.TagIndex = 0;
    coff_symbol->Aux[0].Aux.WeakExternal.Characteristics =
                                        COFF::IMAGE_WEAK_EXTERN_SEARCH_LIBRARY;
  }

  // Bind internal COFF symbol to MC symbol.
  coff_symbol->MCData = &SymbolData;
  SymbolMap[&SymbolData] = coff_symbol;
}

bool WinCOFFObjectWriter::ExportSection(COFFSection *S) {
  return (S->Header.Characteristics
         & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA) == 0;
}

bool WinCOFFObjectWriter::ExportSymbol(MCSymbolData const &SymbolData,
                                       MCAssembler &Asm) {
  // This doesn't seem to be right. Strings referred to from the .data section
  // need symbols so they can be linked to code in the .text section right?

  // return Asm.isSymbolLinkerVisible (&SymbolData);

  // For now, all symbols are exported, the linker will sort it out for us.
  return true;
}

//------------------------------------------------------------------------------
// entity writing methods

void WinCOFFObjectWriter::WriteFileHeader(const COFF::header &Header) {
  WriteLE16(Header.Machine);
  WriteLE16(Header.NumberOfSections);
  WriteLE32(Header.TimeDateStamp);
  WriteLE32(Header.PointerToSymbolTable);
  WriteLE32(Header.NumberOfSymbols);
  WriteLE16(Header.SizeOfOptionalHeader);
  WriteLE16(Header.Characteristics);
}

void WinCOFFObjectWriter::WriteSymbol(const COFFSymbol *S) {
  WriteBytes(StringRef(S->Data.Name, COFF::NameSize));
  WriteLE32(S->Data.Value);
  WriteLE16(S->Data.SectionNumber);
  WriteLE16(S->Data.Type);
  Write8(S->Data.StorageClass);
  Write8(S->Data.NumberOfAuxSymbols);
  WriteAuxiliarySymbols(S->Aux);
}

void WinCOFFObjectWriter::WriteAuxiliarySymbols(
                                        const COFFSymbol::AuxiliarySymbols &S) {
  for(COFFSymbol::AuxiliarySymbols::const_iterator i = S.begin(), e = S.end();
      i != e; ++i) {
    switch(i->AuxType) {
    case ATFunctionDefinition:
      WriteLE32(i->Aux.FunctionDefinition.TagIndex);
      WriteLE32(i->Aux.FunctionDefinition.TotalSize);
      WriteLE32(i->Aux.FunctionDefinition.PointerToLinenumber);
      WriteLE32(i->Aux.FunctionDefinition.PointerToNextFunction);
      WriteZeros(sizeof(i->Aux.FunctionDefinition.unused));
      break;
    case ATbfAndefSymbol:
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused1));
      WriteLE16(i->Aux.bfAndefSymbol.Linenumber);
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused2));
      WriteLE32(i->Aux.bfAndefSymbol.PointerToNextFunction);
      WriteZeros(sizeof(i->Aux.bfAndefSymbol.unused3));
      break;
    case ATWeakExternal:
      WriteLE32(i->Aux.WeakExternal.TagIndex);
      WriteLE32(i->Aux.WeakExternal.Characteristics);
      WriteZeros(sizeof(i->Aux.WeakExternal.unused));
      break;
    case ATFile:
      WriteBytes(StringRef(reinterpret_cast<const char *>(i->Aux.File.FileName),
                 sizeof(i->Aux.File.FileName)));
      break;
    case ATSectionDefinition:
      WriteLE32(i->Aux.SectionDefinition.Length);
      WriteLE16(i->Aux.SectionDefinition.NumberOfRelocations);
      WriteLE16(i->Aux.SectionDefinition.NumberOfLinenumbers);
      WriteLE32(i->Aux.SectionDefinition.CheckSum);
      WriteLE16(i->Aux.SectionDefinition.Number);
      Write8(i->Aux.SectionDefinition.Selection);
      WriteZeros(sizeof(i->Aux.SectionDefinition.unused));
      break;
    }
  }
}

void WinCOFFObjectWriter::WriteSectionHeader(const COFF::section &S) {
  WriteBytes(StringRef(S.Name, COFF::NameSize));

  WriteLE32(S.VirtualSize);
  WriteLE32(S.VirtualAddress);
  WriteLE32(S.SizeOfRawData);
  WriteLE32(S.PointerToRawData);
  WriteLE32(S.PointerToRelocations);
  WriteLE32(S.PointerToLineNumbers);
  WriteLE16(S.NumberOfRelocations);
  WriteLE16(S.NumberOfLineNumbers);
  WriteLE32(S.Characteristics);
}

void WinCOFFObjectWriter::WriteRelocation(const COFF::relocation &R) {
  WriteLE32(R.VirtualAddress);
  WriteLE32(R.SymbolTableIndex);
  WriteLE16(R.Type);
}

////////////////////////////////////////////////////////////////////////////////
// MCObjectWriter interface implementations

void WinCOFFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm) {
  // "Define" each section & symbol. This creates section & symbol
  // entries in the staging area and gives them their final indexes.

  for (MCAssembler::const_iterator i = Asm.begin(), e = Asm.end(); i != e; i++)
    DefineSection(*i);

  for (MCAssembler::const_symbol_iterator i = Asm.symbol_begin(),
                                          e = Asm.symbol_end(); i != e; i++) {
    if (ExportSymbol(*i, Asm))
      DefineSymbol(*i, Asm);
  }
}

void WinCOFFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           uint64_t &FixedValue) {
  assert(Target.getSymA() != NULL && "Relocation must reference a symbol!");
  assert(Target.getSymB() == NULL &&
         "Relocation must reference only one symbol!");

  MCSectionData const *SectionData = Fragment->getParent();
  MCSymbolData const *SymbolData =
                              &Asm.getSymbolData(Target.getSymA()->getSymbol());

  assert(SectionMap.find(SectionData) != SectionMap.end() &&
         "Section must already have been defined in ExecutePostLayoutBinding!");
  assert(SymbolMap.find(SymbolData) != SymbolMap.end() &&
         "Symbol must already have been defined in ExecutePostLayoutBinding!");

  COFFSection *coff_section = SectionMap[SectionData];
  COFFSymbol *coff_symbol = SymbolMap[SymbolData];

  FixedValue = Target.getConstant();

  COFFRelocation Reloc;

  Reloc.Data.SymbolTableIndex = 0;
  Reloc.Data.VirtualAddress = Layout.getFragmentOffset(Fragment);
  Reloc.Symb = coff_symbol;

  Reloc.Data.VirtualAddress += Fixup.getOffset();

  switch (Fixup.getKind()) {
  case FirstTargetFixupKind: // reloc_pcrel_4byte
    Reloc.Data.Type = COFF::IMAGE_REL_I386_REL32;
    FixedValue += 4;
    break;
  case FK_Data_4:
    Reloc.Data.Type = COFF::IMAGE_REL_I386_DIR32;
    break;
  default:
    llvm_unreachable("unsupported relocation type");
  }

  coff_section->Relocations.push_back(Reloc);
}

void WinCOFFObjectWriter::WriteObject(const MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
  // Assign symbol and section indexes and offsets.

  Header.NumberOfSymbols = 0;

  for (symbols::iterator i = Symbols.begin(), e = Symbols.end(); i != e; i++) {
    COFFSymbol *coff_symbol = *i;
    MCSymbolData const *SymbolData = coff_symbol->MCData;

    coff_symbol->Index = Header.NumberOfSymbols++;

    // Update section number & offset for symbols that have them.
    if ((SymbolData != NULL) && (SymbolData->Fragment != NULL)) {
      COFFSection *coff_section = SectionMap[SymbolData->Fragment->getParent()];

      coff_symbol->Data.SectionNumber = coff_section->Number;
      coff_symbol->Data.Value = Layout.getFragmentOffset(SymbolData->Fragment);
    }

    // Update auxiliary symbol info.
    coff_symbol->Data.NumberOfAuxSymbols = coff_symbol->Aux.size();
    Header.NumberOfSymbols += coff_symbol->Data.NumberOfAuxSymbols;
  }

  // Fixup weak external references.
  for (symbols::iterator i = Symbols.begin(), e = Symbols.end(); i != e; i++) {
    COFFSymbol *symb = *i;

    if (symb->Other != NULL) {
      assert(symb->Aux.size() == 1 &&
             "Symbol must contain one aux symbol!");
      assert(symb->Aux[0].AuxType == ATWeakExternal &&
             "Symbol's aux symbol must be a Weak External!");
      symb->Aux[0].Aux.WeakExternal.TagIndex = symb->Other->Index;
    }
  }

  // Assign file offsets to COFF object file structures.

  unsigned offset = 0;

  offset += COFF::HeaderSize;
  offset += COFF::SectionSize * Asm.size();

  Header.NumberOfSections = Sections.size();

  for (MCAssembler::const_iterator i = Asm.begin(),
                                   e = Asm.end();
                                   i != e; i++) {
    COFFSection *Sec = SectionMap[i];

    Sec->Header.SizeOfRawData = Layout.getSectionFileSize(i);

    if (ExportSection(Sec)) {
      Sec->Header.PointerToRawData = offset;

      offset += Sec->Header.SizeOfRawData;
    }

    if (Sec->Relocations.size() > 0) {
      Sec->Header.NumberOfRelocations = Sec->Relocations.size();
      Sec->Header.PointerToRelocations = offset;

      offset += COFF::RelocationSize * Sec->Relocations.size();

      for (relocations::iterator cr = Sec->Relocations.begin(),
                                 er = Sec->Relocations.end();
                                 cr != er; cr++) {
        (*cr).Data.SymbolTableIndex = (*cr).Symb->Index;
      }
    }

    assert(Sec->Symb->Aux.size() == 1 && "Section's symbol must have one aux!");
    AuxSymbol &Aux = Sec->Symb->Aux[0];
    assert(Aux.AuxType == ATSectionDefinition &&
           "Section's symbol's aux symbol must be a Section Definition!");
    Aux.Aux.SectionDefinition.Length = Sec->Header.SizeOfRawData;
    Aux.Aux.SectionDefinition.NumberOfRelocations =
                                                Sec->Header.NumberOfRelocations;
    Aux.Aux.SectionDefinition.NumberOfLinenumbers =
                                                Sec->Header.NumberOfLineNumbers;
  }

  Header.PointerToSymbolTable = offset;

  // Write it all to disk...
  WriteFileHeader(Header);

  {
    sections::iterator i, ie;
    MCAssembler::const_iterator j, je;

    for (i = Sections.begin(), ie = Sections.end(); i != ie; i++)
      WriteSectionHeader((*i)->Header);

    for (i = Sections.begin(), ie = Sections.end(),
         j = Asm.begin(), je = Asm.end();
         (i != ie) && (j != je); i++, j++) {
      if ((*i)->Header.PointerToRawData != 0) {
        assert(OS.tell() == (*i)->Header.PointerToRawData &&
               "Section::PointerToRawData is insane!");

        Asm.WriteSectionData(j, Layout, this);
      }

      if ((*i)->Relocations.size() > 0) {
        assert(OS.tell() == (*i)->Header.PointerToRelocations &&
               "Section::PointerToRelocations is insane!");

        for (relocations::const_iterator k = (*i)->Relocations.begin(),
                                               ke = (*i)->Relocations.end();
                                               k != ke; k++) {
          WriteRelocation(k->Data);
        }
      } else
        assert((*i)->Header.PointerToRelocations == 0 &&
               "Section::PointerToRelocations is insane!");
    }
  }

  assert(OS.tell() == Header.PointerToSymbolTable &&
         "Header::PointerToSymbolTable is insane!");

  for (symbols::iterator i = Symbols.begin(), e = Symbols.end(); i != e; i++)
    WriteSymbol(*i);

  OS.write((char const *)&Strings.Data.front(), Strings.Data.size());
}

//------------------------------------------------------------------------------
// WinCOFFObjectWriter factory function

namespace llvm {
  MCObjectWriter *createWinCOFFObjectWriter(raw_ostream &OS) {
    return new WinCOFFObjectWriter(OS);
  }
}
