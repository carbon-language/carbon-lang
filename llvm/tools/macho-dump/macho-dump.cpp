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

#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFile(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<bool>
DumpSectionData("dump-section-data", cl::desc("Dump the contents of sections"),
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

static int DumpHeader(MachOObject &Obj) {
  // Read the header.
  const macho::Header &Hdr = Obj.getHeader();
  outs() << "('cputype', " << Hdr.CPUType << ")\n";
  outs() << "('cpusubtype', " << Hdr.CPUSubtype << ")\n";
  outs() << "('filetype', " << Hdr.FileType << ")\n";
  outs() << "('num_load_commands', " << Hdr.NumLoadCommands << ")\n";
  outs() << "('load_commands_size', " << Hdr.SizeOfLoadCommands << ")\n";
  outs() << "('flag', " << Hdr.Flags << ")\n";

  // Print extended header if 64-bit.
  if (Obj.is64Bit()) {
    const macho::Header64Ext &Hdr64 = Obj.getHeader64Ext();
    outs() << "('reserved', " << Hdr64.Reserved << ")\n";
  }

  return 0;
}

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

static int DumpSegmentCommand(MachOObject &Obj,
                               const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::SegmentLoadCommand> SLC;
  Obj.ReadSegmentLoadCommand(LCI, SLC);
  if (!SLC)
    return Error("unable to read segment load command");

  DumpSegmentCommandData(StringRef(SLC->Name, 16), SLC->VMAddress, SLC->VMSize,
                         SLC->FileOffset, SLC->FileSize,
                         SLC->MaxVMProtection, SLC->InitialVMProtection,
                         SLC->NumSections, SLC->Flags);

  return 0;
}
static int DumpSegment64Command(MachOObject &Obj,
                               const MachOObject::LoadCommandInfo &LCI) {
  InMemoryStruct<macho::Segment64LoadCommand> SLC;
  Obj.ReadSegment64LoadCommand(LCI, SLC);
  if (!SLC)
    return Error("unable to read segment load command");

  DumpSegmentCommandData(StringRef(SLC->Name, 16), SLC->VMAddress, SLC->VMSize,
                         SLC->FileOffset, SLC->FileSize,
                         SLC->MaxVMProtection, SLC->InitialVMProtection,
                         SLC->NumSections, SLC->Flags);

  return 0;
}

static int DumpLoadCommand(MachOObject &Obj, unsigned Index) {
  const MachOObject::LoadCommandInfo &LCI = Obj.getLoadCommandInfo(Index);
  int Res = 0;

  outs() << "  # Load Command " << Index << "\n"
         << " (('command', " << LCI.Command.Type << ")\n"
         << "  ('size', " << LCI.Command.Size << ")\n";
  switch (LCI.Command.Type) {
  case macho::LCT_Segment:
    Res = DumpSegmentCommand(Obj, LCI);
    break;
  case macho::LCT_Segment64:
    Res = DumpSegment64Command(Obj, LCI);
    break;
  default:
    Warning("unknown load command: " + Twine(LCI.Command.Type));
    break;
  }
  outs() << " ),\n";

  return Res;
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm Mach-O dumping tool\n");

  // Load the input file.
  std::string ErrorStr;
  OwningPtr<MemoryBuffer> InputBuffer(
    MemoryBuffer::getFileOrSTDIN(InputFile, &ErrorStr));
  if (!InputBuffer)
    return Error("unable to read input: '" + ErrorStr + "'");

  // Construct the Mach-O wrapper object.
  OwningPtr<MachOObject> InputObject(
    MachOObject::LoadFromBuffer(InputBuffer.take(), &ErrorStr));
  if (!InputObject)
    return Error("unable to load object: '" + ErrorStr + "'");

  if (int Res = DumpHeader(*InputObject))
    return Res;

  // Print the load commands.
  int Res = 0;
  outs() << "('load_commands', [\n";
  for (unsigned i = 0; i != InputObject->getHeader().NumLoadCommands; ++i)
    if ((Res = DumpLoadCommand(*InputObject, i)))
      break;
  outs() << "])\n";

  return Res;
}
