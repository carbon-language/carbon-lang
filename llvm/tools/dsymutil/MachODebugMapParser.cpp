//===- tools/dsymutil/MachODebugMapParser.cpp - Parse STABS debug maps ----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachODebugMapParser.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::object;

namespace llvm {

static void Warning(const Twine &Msg) { errs() << "warning: " + Msg + "\n"; }

static ErrorOr<OwningBinary<MachOObjectFile>> createMachOBinary(StringRef file) {
  ErrorOr<OwningBinary<Binary>> BinaryOrErr = createBinary(file);
  if (BinaryOrErr.getError())
    return BinaryOrErr.getError();

  std::unique_ptr<Binary> Bin;
  std::unique_ptr<MemoryBuffer> Buf;
  std::tie(Bin, Buf) = BinaryOrErr->takeBinary();
  if (!isa<MachOObjectFile>(Bin.get()))
    return make_error_code(object_error::invalid_file_type);

  std::unique_ptr<MachOObjectFile> MachOFile(cast<MachOObjectFile>(Bin.release()));
  return OwningBinary<MachOObjectFile>(std::move(MachOFile), std::move(Buf));
}

/// Reset the parser state coresponding to the current object
/// file. This is to be called after an object file is finished
/// processing.
void MachODebugMapParser::resetParserState() {
  CurrentObjectFile = OwningBinary<object::MachOObjectFile>();
  CurrentObjectAddresses.clear();
  CurrentDebugMapObject = nullptr;
}

/// Create a new DebugMapObject. This function resets the state of the
/// parser that was referring to the last object file and sets
/// everything up to add symbols to the new one.
void MachODebugMapParser::switchToNewDebugMapObject(StringRef Filename) {
  resetParserState();

  std::string Path = Filename;
  if (!PathPrefix.empty())
    Path = PathPrefix + sys::path::get_separator().data() + Path;

  auto MachOOrError = createMachOBinary(Path);
  if (auto Error = MachOOrError.getError()) {
    Warning(Twine("cannot open debug object \"") + Path + "\": "
            + Error.message() + "\n");
    return;
  }

  CurrentObjectFile = std::move(*MachOOrError);
  loadCurrentObjectFileSymbols();
  CurrentDebugMapObject = &Result->addDebugMapObject(Path);
}

/// This main parsing routine tries to open the main binary and if
/// successful iterates over the STAB entries. The real parsing is
/// done in handleStabSymbolTableEntry.
ErrorOr<std::unique_ptr<DebugMap>> MachODebugMapParser::parse() {
  auto MainBinaryOrError = createMachOBinary(BinaryPath);
  if (MainBinaryOrError.getError())
    return MainBinaryOrError.getError();

  MainOwningBinary = std::move(*MainBinaryOrError);
  Result = make_unique<DebugMap>();
  const auto &MainBinary = *MainOwningBinary.getBinary();
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

  const MachOObjectFile &MachOBinary = *MainOwningBinary.getBinary();
  const char *Name = &MachOBinary.getStringTableData().data()[StringIndex];

  // An N_OSO entry represents the start of a new object file description.
  if (Type == MachO::N_OSO)
    return switchToNewDebugMapObject(Name);

  // If the last N_OSO object file wasn't found,
  // CurrentDebugMapObject will be null. Do not update anything
  // until we find the next valid N_OSO entry.
  if (!CurrentDebugMapObject)
    return;

  switch (Type) {
  case MachO::N_GSYM:
    // This is a global variable. We need to query the main binary
    // symbol table to find its address as it might not be in the
    // debug map (for common symbols).
    Value = getMainBinarySymbolAddress(Name);
    if (Value == UnknownAddressOrSize)
      return;
    break;
  case MachO::N_FUN:
    // Functions are scopes in STABS. They have an end marker that we
    // need to ignore.
    if (Name[0] == '\0')
      return;
    break;
  case MachO::N_STSYM:
    break;
  default:
    return;
  }

  auto ObjectSymIt = CurrentObjectAddresses.find(Name);
  if (ObjectSymIt == CurrentObjectAddresses.end())
    return Warning("could not find object file symbol for symbol " +
                   Twine(Name));
  if (!CurrentDebugMapObject->addSymbol(Name, ObjectSymIt->getValue(), Value))
    return Warning(Twine("failed to insert symbol '") + Name + "' in the debug map.");
}

/// Load the current object file symbols into CurrentObjectAddresses.
void MachODebugMapParser::loadCurrentObjectFileSymbols() {
  CurrentObjectAddresses.clear();
  const auto &Binary = *CurrentObjectFile.getBinary();

  for (auto Sym : Binary.symbols()) {
    StringRef Name;
    uint64_t Addr;
    if (Sym.getAddress(Addr) || Addr == UnknownAddressOrSize ||
        Sym.getName(Name))
      continue;
    CurrentObjectAddresses[Name] = Addr;
  }
}

/// Lookup a symbol address in the main binary symbol table. The
/// parser only needs to query common symbols, thus not every symbol's
/// address is available through this function.
uint64_t MachODebugMapParser::getMainBinarySymbolAddress(StringRef Name) {
  if (MainBinarySymbolAddresses.empty())
    loadMainBinarySymbols();

  auto Sym = MainBinarySymbolAddresses.find(Name);
  if (Sym == MainBinarySymbolAddresses.end())
    return UnknownAddressOrSize;
  return Sym->second;
}

/// Load the interesting main binary symbols' addresses into
/// MainBinarySymbolAddresses.
void MachODebugMapParser::loadMainBinarySymbols() {
  const MachOObjectFile &Binary = *MainOwningBinary.getBinary();
  section_iterator Section = Binary.section_end();
  for (const auto &Sym : Binary.symbols()) {
    SymbolRef::Type Type;
    // Skip undefined and STAB entries.
    if (Sym.getType(Type) || (Type & SymbolRef::ST_Debug) ||
        (Type & SymbolRef::ST_Unknown))
      continue;
    StringRef Name;
    uint64_t Addr;
    // The only symbols of interest are the global variables. These
    // are the only ones that need to be queried because the address
    // of common data won't be described in the debug map. All other
    // addresses should be fetched for the debug map.
    if (Sym.getAddress(Addr) || Addr == UnknownAddressOrSize ||
        !(Sym.getFlags() & SymbolRef::SF_Global) ||
        Sym.getSection(Section) || Section->isText() || Sym.getName(Name) ||
        Name.size() == 0 || Name[0] == '\0')
      continue;
    MainBinarySymbolAddresses[Name] = Addr;
  }
}

}
