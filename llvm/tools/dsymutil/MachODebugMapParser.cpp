//===- tools/dsymutil/MachODebugMapParser.cpp - Parse STABS debug maps ----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "BinaryHolder.h"
#include "DebugMap.h"
#include "dsymutil.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace llvm;
using namespace llvm::dsymutil;
using namespace llvm::object;

class MachODebugMapParser {
public:
  MachODebugMapParser(StringRef BinaryPath, StringRef PathPrefix = "",
                      bool Verbose = false)
      : BinaryPath(BinaryPath), PathPrefix(PathPrefix),
        MainBinaryHolder(Verbose), CurrentObjectHolder(Verbose),
        CurrentDebugMapObject(nullptr) {}

  /// \brief Parses and returns the DebugMap of the input binary.
  /// \returns an error in case the provided BinaryPath doesn't exist
  /// or isn't of a supported type.
  ErrorOr<std::unique_ptr<DebugMap>> parse();

private:
  std::string BinaryPath;
  std::string PathPrefix;

  /// Owns the MemoryBuffer for the main binary.
  BinaryHolder MainBinaryHolder;
  /// Map of the binary symbol addresses.
  StringMap<uint64_t> MainBinarySymbolAddresses;
  StringRef MainBinaryStrings;
  /// The constructed DebugMap.
  std::unique_ptr<DebugMap> Result;

  /// Owns the MemoryBuffer for the currently handled object file.
  BinaryHolder CurrentObjectHolder;
  /// Map of the currently processed object file symbol addresses.
  StringMap<uint64_t> CurrentObjectAddresses;
  /// Element of the debug map corresponfing to the current object file.
  DebugMapObject *CurrentDebugMapObject;

  /// Holds function info while function scope processing.
  const char *CurrentFunctionName;
  uint64_t CurrentFunctionAddress;

  void switchToNewDebugMapObject(StringRef Filename);
  void resetParserState();
  uint64_t getMainBinarySymbolAddress(StringRef Name);
  void loadMainBinarySymbols();
  void loadCurrentObjectFileSymbols();
  void handleStabSymbolTableEntry(uint32_t StringIndex, uint8_t Type,
                                  uint8_t SectionIndex, uint16_t Flags,
                                  uint64_t Value);

  template <typename STEType> void handleStabDebugMapEntry(const STEType &STE) {
    handleStabSymbolTableEntry(STE.n_strx, STE.n_type, STE.n_sect, STE.n_desc,
                               STE.n_value);
  }
};

static void Warning(const Twine &Msg) { errs() << "warning: " + Msg + "\n"; }
}

/// Reset the parser state coresponding to the current object
/// file. This is to be called after an object file is finished
/// processing.
void MachODebugMapParser::resetParserState() {
  CurrentObjectAddresses.clear();
  CurrentDebugMapObject = nullptr;
}

/// Create a new DebugMapObject. This function resets the state of the
/// parser that was referring to the last object file and sets
/// everything up to add symbols to the new one.
void MachODebugMapParser::switchToNewDebugMapObject(StringRef Filename) {
  resetParserState();

  SmallString<80> Path(PathPrefix);
  sys::path::append(Path, Filename);

  auto MachOOrError = CurrentObjectHolder.GetFileAs<MachOObjectFile>(Path);
  if (auto Error = MachOOrError.getError()) {
    Warning(Twine("cannot open debug object \"") + Path.str() + "\": " +
            Error.message() + "\n");
    return;
  }

  loadCurrentObjectFileSymbols();
  CurrentDebugMapObject = &Result->addDebugMapObject(Path);
}

static Triple getTriple(const object::MachOObjectFile &Obj) {
  Triple TheTriple("unknown-unknown-unknown");
  TheTriple.setArch(Triple::ArchType(Obj.getArch()));
  TheTriple.setObjectFormat(Triple::MachO);
  return TheTriple;
}

/// This main parsing routine tries to open the main binary and if
/// successful iterates over the STAB entries. The real parsing is
/// done in handleStabSymbolTableEntry.
ErrorOr<std::unique_ptr<DebugMap>> MachODebugMapParser::parse() {
  auto MainBinOrError = MainBinaryHolder.GetFileAs<MachOObjectFile>(BinaryPath);
  if (auto Error = MainBinOrError.getError())
    return Error;

  const MachOObjectFile &MainBinary = *MainBinOrError;
  loadMainBinarySymbols();
  Result = make_unique<DebugMap>(getTriple(MainBinary));
  MainBinaryStrings = MainBinary.getStringTableData();
  for (const SymbolRef &Symbol : MainBinary.symbols()) {
    const DataRefImpl &DRI = Symbol.getRawDataRefImpl();
    if (MainBinary.is64Bit())
      handleStabDebugMapEntry(MainBinary.getSymbol64TableEntry(DRI));
    else
      handleStabDebugMapEntry(MainBinary.getSymbolTableEntry(DRI));
  }

  resetParserState();
  return std::move(Result);
}

/// Interpret the STAB entries to fill the DebugMap.
void MachODebugMapParser::handleStabSymbolTableEntry(uint32_t StringIndex,
                                                     uint8_t Type,
                                                     uint8_t SectionIndex,
                                                     uint16_t Flags,
                                                     uint64_t Value) {
  if (!(Type & MachO::N_STAB))
    return;

  const char *Name = &MainBinaryStrings.data()[StringIndex];

  // An N_OSO entry represents the start of a new object file description.
  if (Type == MachO::N_OSO)
    return switchToNewDebugMapObject(Name);

  // If the last N_OSO object file wasn't found,
  // CurrentDebugMapObject will be null. Do not update anything
  // until we find the next valid N_OSO entry.
  if (!CurrentDebugMapObject)
    return;

  uint32_t Size = 0;
  switch (Type) {
  case MachO::N_GSYM:
    // This is a global variable. We need to query the main binary
    // symbol table to find its address as it might not be in the
    // debug map (for common symbols).
    Value = getMainBinarySymbolAddress(Name);
    break;
  case MachO::N_FUN:
    // Functions are scopes in STABS. They have an end marker that
    // contains the function size.
    if (Name[0] == '\0') {
      Size = Value;
      Value = CurrentFunctionAddress;
      Name = CurrentFunctionName;
      break;
    } else {
      CurrentFunctionName = Name;
      CurrentFunctionAddress = Value;
      return;
    }
  case MachO::N_STSYM:
    break;
  default:
    return;
  }

  auto ObjectSymIt = CurrentObjectAddresses.find(Name);
  if (ObjectSymIt == CurrentObjectAddresses.end())
    return Warning("could not find object file symbol for symbol " +
                   Twine(Name));
  if (!CurrentDebugMapObject->addSymbol(Name, ObjectSymIt->getValue(), Value,
                                        Size))
    return Warning(Twine("failed to insert symbol '") + Name +
                   "' in the debug map.");
}

/// Load the current object file symbols into CurrentObjectAddresses.
void MachODebugMapParser::loadCurrentObjectFileSymbols() {
  CurrentObjectAddresses.clear();

  for (auto Sym : CurrentObjectHolder.Get().symbols()) {
    uint64_t Addr = Sym.getValue();
    ErrorOr<StringRef> Name = Sym.getName();
    if (!Name)
      continue;
    CurrentObjectAddresses[*Name] = Addr;
  }
}

/// Lookup a symbol address in the main binary symbol table. The
/// parser only needs to query common symbols, thus not every symbol's
/// address is available through this function.
uint64_t MachODebugMapParser::getMainBinarySymbolAddress(StringRef Name) {
  auto Sym = MainBinarySymbolAddresses.find(Name);
  if (Sym == MainBinarySymbolAddresses.end())
    return 0;
  return Sym->second;
}

/// Load the interesting main binary symbols' addresses into
/// MainBinarySymbolAddresses.
void MachODebugMapParser::loadMainBinarySymbols() {
  const MachOObjectFile &MainBinary = MainBinaryHolder.GetAs<MachOObjectFile>();
  section_iterator Section = MainBinary.section_end();
  for (const auto &Sym : MainBinary.symbols()) {
    SymbolRef::Type Type = Sym.getType();
    // Skip undefined and STAB entries.
    if ((Type & SymbolRef::ST_Debug) || (Type & SymbolRef::ST_Unknown))
      continue;
    // The only symbols of interest are the global variables. These
    // are the only ones that need to be queried because the address
    // of common data won't be described in the debug map. All other
    // addresses should be fetched for the debug map.
    if (!(Sym.getFlags() & SymbolRef::SF_Global) || Sym.getSection(Section) ||
        Section == MainBinary.section_end() || Section->isText())
      continue;
    uint64_t Addr = Sym.getValue();
    ErrorOr<StringRef> NameOrErr = Sym.getName();
    if (!NameOrErr)
      continue;
    StringRef Name = *NameOrErr;
    if (Name.size() == 0 || Name[0] == '\0')
      continue;
    MainBinarySymbolAddresses[Name] = Addr;
  }
}

namespace llvm {
namespace dsymutil {
llvm::ErrorOr<std::unique_ptr<DebugMap>> parseDebugMap(StringRef InputFile,
                                                       StringRef PrependPath,
                                                       bool Verbose,
                                                       bool InputIsYAML) {
  if (!InputIsYAML) {
    MachODebugMapParser Parser(InputFile, PrependPath, Verbose);
    return Parser.parse();
  } else {
    return DebugMap::parseYAMLDebugMap(InputFile, PrependPath, Verbose);
  }
}
}
}
