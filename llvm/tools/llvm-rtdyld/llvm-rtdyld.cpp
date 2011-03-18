//===-- llvm-rtdyld.cpp - MCJIT Testing Tool ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC-JIT LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

enum ActionType {
  AC_Execute
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::init(AC_Execute),
       cl::values(clEnumValN(AC_Execute, "execute",
                             "Load, link, and execute the inputs."),
                  clEnumValEnd));

/* *** */

static const char *ProgramName;

static void Message(const char *Type, const Twine &Msg) {
  errs() << ProgramName << ": " << Type << ": " << Msg << "\n";
}

static int Error(const Twine &Msg) {
  Message("error", Msg);
  return 1;
}

/* *** */

static int ExecuteInput() {
  // Load the input memory buffer.
  OwningPtr<MemoryBuffer> InputBuffer;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFile, InputBuffer))
    return Error("unable to read input: '" + ec.message() + "'");

  // Load the Mach-O wrapper object.
  std::string ErrorStr;
  OwningPtr<MachOObject> Obj(
    MachOObject::LoadFromBuffer(InputBuffer.take(), &ErrorStr));
  if (!Obj)
    return Error("unable to load object: '" + ErrorStr + "'");

  // Validate that the load commands match what we expect.
  const MachOObject::LoadCommandInfo *SegmentLCI = 0, *SymtabLCI = 0,
    *DysymtabLCI = 0;
  for (unsigned i = 0; i != Obj->getHeader().NumLoadCommands; ++i) {
    const MachOObject::LoadCommandInfo &LCI = Obj->getLoadCommandInfo(i);
    switch (LCI.Command.Type) {
    case macho::LCT_Segment:
    case macho::LCT_Segment64:
      if (SegmentLCI)
        return Error("unexpected input object (multiple segments)");
      SegmentLCI = &LCI;
      break;
    case macho::LCT_Symtab:
      if (SymtabLCI)
        return Error("unexpected input object (multiple symbol tables)");
      SymtabLCI = &LCI;
      break;
    case macho::LCT_Dysymtab:
      if (DysymtabLCI)
        return Error("unexpected input object (multiple symbol tables)");
      DysymtabLCI = &LCI;
      break;
    default:
      return Error("unexpected input object (unexpected load command");
    }
  }

  if (!SymtabLCI)
    return Error("no symbol table found in object");
  if (!SegmentLCI)
    return Error("no symbol table found in object");

  // Read and register the symbol table data.
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLC;
  Obj->ReadSymtabLoadCommand(*SymtabLCI, SymtabLC);
  if (!SymtabLC)
    return Error("unable to load symbol table load command");
  Obj->RegisterStringTable(*SymtabLC);

  // Read the dynamic link-edit information, if present (not present in static
  // objects).
  if (DysymtabLCI) {
    InMemoryStruct<macho::DysymtabLoadCommand> DysymtabLC;
    Obj->ReadDysymtabLoadCommand(*DysymtabLCI, DysymtabLC);
    if (!DysymtabLC)
      return Error("unable to load dynamic link-exit load command");

    // FIXME: We don't support anything interesting yet.
    if (DysymtabLC->LocalSymbolsIndex != 0)
      return Error("NOT YET IMPLEMENTED: local symbol entries");
    if (DysymtabLC->ExternalSymbolsIndex != 0)
      return Error("NOT YET IMPLEMENTED: non-external symbol entries");
    if (DysymtabLC->UndefinedSymbolsIndex != SymtabLC->NumSymbolTableEntries)
      return Error("NOT YET IMPLEMENTED: undefined symbol entries");
  }

  // Load the segment load command.
  if (SegmentLCI->Command.Type != macho::LCT_Segment64)
    return Error("Segment32 not yet implemented!");
  InMemoryStruct<macho::Segment64LoadCommand> Segment64LC;
  Obj->ReadSegment64LoadCommand(*SegmentLCI, Segment64LC);
  if (!Segment64LC)
    return Error("unable to load segment load command");

  // Map the segment into memory.
  sys::MemoryBlock Data = sys::Memory::AllocateRWX(Segment64LC->VMSize,
                                                   0, &ErrorStr);
  if (!Data.base())
    return Error("unable to allocate memory block: '" + ErrorStr + "'");
  memcpy(Data.base(), Obj->getData(Segment64LC->FileOffset,
                                   Segment64LC->FileSize).data(),
         Segment64LC->FileSize);
  memset((char*)Data.base() + Segment64LC->FileSize, 0,
         Segment64LC->VMSize - Segment64LC->FileSize);

  // Bind the section indices to address.
  void **SectionBases = new void*[Segment64LC->NumSections];
  for (unsigned i = 0; i != Segment64LC->NumSections; ++i) {
    InMemoryStruct<macho::Section64> Sect;
    Obj->ReadSection64(*SegmentLCI, i, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(i) + "'");

    // FIXME: We don't support relocations yet.
    if (Sect->NumRelocationTableEntries != 0)
      return Error("not yet implemented: relocations!");

    // FIXME: Improve check.
    if (Sect->Flags != 0x80000400)
      return Error("unsupported section type!");

    SectionBases[i] = (char*) Data.base() + Sect->Address;
  }

  // Bind all the symbols to address.
  StringMap<void*> SymbolTable;
  for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
    InMemoryStruct<macho::Symbol64TableEntry> STE;
    Obj->ReadSymbol64TableEntry(SymtabLC->SymbolTableOffset, i, STE);
    if (!STE)
      return Error("unable to read symbol: '" + Twine(i) + "'");
    if (STE->SectionIndex == 0)
      return Error("unexpected undefined symbol!");

    unsigned Index = STE->SectionIndex - 1;
    if (Index >= Segment64LC->NumSections)
      return Error("invalid section index for symbol: '" + Twine() + "'");

    // Get the symbol name.
    StringRef Name = Obj->getStringAtIndex(STE->StringIndex);

    // Get the section base address.
    void *SectionBase = SectionBases[Index];

    // Get the symbol address.
    void *Address = (char*) SectionBase + STE->Value;

    // FIXME: Check the symbol type and flags.
    if (STE->Type != 0xF)
      return Error("unexpected symbol type!");
    if (STE->Flags != 0x0)
      return Error("unexpected symbol type!");

    SymbolTable[Name] = Address;
  }

  // Get the address of "_main".
  StringMap<void*>::iterator it = SymbolTable.find("_main");
  if (it == SymbolTable.end())
    return Error("no definition for '_main'");

  // Invalidate the instruction cache.
  sys::Memory::InvalidateInstructionCache(Data.base(), Data.size());

  // Make sure the memory is executable.
  if (!sys::Memory::setExecutable(Data, &ErrorStr))
    return Error("unable to mark function executable: '" + ErrorStr + "'");

  // Dispatch to _main().
  void *MainAddress = it->second;
  errs() << "loaded '_main' at: " << MainAddress << "\n";

  int (*Main)(int, const char**) =
    (int(*)(int,const char**)) uintptr_t(MainAddress);
  const char **Argv = new const char*[2];
  Argv[0] = InputFile.c_str();
  Argv[1] = 0;
  return Main(1, Argv);
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm MC-JIT tool\n");

  switch (Action) {
  default:
  case AC_Execute:
    return ExecuteInput();
  }

  return 0;
}
