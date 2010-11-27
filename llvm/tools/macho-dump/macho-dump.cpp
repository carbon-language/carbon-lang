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

static int DumpLoadCommand(MachOObject &Obj, unsigned Index) {
  const MachOObject::LoadCommandInfo &LCI = Obj.getLoadCommandInfo(Index);

  outs() << "  # Load Command " << Index << "\n"
         << " (('command', " << LCI.Command.Type << ")\n"
         << "  ('size', " << LCI.Command.Size << ")\n";
  switch (LCI.Command.Type) {
  default:
    Warning("unknown load command: " + Twine(LCI.Command.Type));
    break;
  }
  outs() << " ),\n";

  return 0;
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
  outs() << "('load_commands', [\n";
  for (unsigned i = 0; i != InputObject->getHeader().NumLoadCommands; ++i)
    DumpLoadCommand(*InputObject, i);
  outs() << "])\n";

  return 0;
}
