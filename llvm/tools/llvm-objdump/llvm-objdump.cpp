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
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"
#include <algorithm>
#include <cstring>
using namespace llvm;
using namespace object;

namespace {
  cl::list<std::string>
  InputFilenames(cl::Positional, cl::desc("<input object files>"),
                 cl::ZeroOrMore);

  cl::opt<bool>
  Disassemble("disassemble",
    cl::desc("Display assembler mnemonics for the machine instructions"));
  cl::alias
  Disassembled("d", cl::desc("Alias for --disassemble"),
               cl::aliasopt(Disassemble));

  cl::opt<std::string>
  TripleName("triple", cl::desc("Target triple to disassemble for, "
                                "see -version for available targets"));

  cl::opt<std::string>
  ArchName("arch", cl::desc("Target arch to disassemble for, "
                            "see -version for available targets"));

  StringRef ToolName;

  bool error(error_code ec) {
    if (!ec) return false;

    outs() << ToolName << ": error reading file: " << ec.message() << ".\n";
    outs().flush();
    return true;
  }
}

static const Target *GetTarget(const ObjectFile *Obj = NULL) {
  // Figure out the target triple.
  llvm::Triple TT("unknown-unknown-unknown");
  if (TripleName.empty()) {
    if (Obj)
      TT.setArch(Triple::ArchType(Obj->getArch()));
  } else
    TT.setTriple(Triple::normalize(TripleName));

  if (!ArchName.empty())
    TT.setArchName(ArchName);

  TripleName = TT.str();

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (TheTarget)
    return TheTarget;

  errs() << ToolName << ": error: unable to get target for '" << TripleName
         << "', see --version and --triple.\n";
  return 0;
}

namespace {
class StringRefMemoryObject : public MemoryObject {
private:
  StringRef Bytes;
public:
  StringRefMemoryObject(StringRef bytes) : Bytes(bytes) {}

  uint64_t getBase() const { return 0; }
  uint64_t getExtent() const { return Bytes.size(); }

  int readByte(uint64_t Addr, uint8_t *Byte) const {
    if (Addr > getExtent())
      return -1;
    *Byte = Bytes[Addr];
    return 0;
  }
};
}

static void DumpBytes(StringRef bytes) {
  static char hex_rep[] = "0123456789abcdef";
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

static void DisassembleInput(const StringRef &Filename) {
  OwningPtr<MemoryBuffer> Buff;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
    errs() << ToolName << ": " << Filename << ": " << ec.message() << "\n";
    return;
  }

  OwningPtr<ObjectFile> Obj(ObjectFile::createObjectFile(Buff.take()));

  const Target *TheTarget = GetTarget(Obj.get());
  if (!TheTarget) {
    // GetTarget prints out stuff.
    return;
  }

  outs() << '\n';
  outs() << Filename
         << ":\tfile format " << Obj->getFileFormatName() << "\n\n\n";

  error_code ec;
  for (ObjectFile::section_iterator i = Obj->begin_sections(),
                                    e = Obj->end_sections();
                                    i != e; i.increment(ec)) {
    if (error(ec)) break;
    bool text;
    if (error(i->isText(text))) break;
    if (!text) continue;

    StringRef name;
    if (error(i->getName(name))) break;
    outs() << "Disassembly of section " << name << ":\n\n";

    // Set up disassembler.
    OwningPtr<const MCAsmInfo> AsmInfo(TheTarget->createAsmInfo(TripleName));

    if (!AsmInfo) {
      errs() << "error: no assembly info for target " << TripleName << "\n";
      return;
    }

    OwningPtr<const MCDisassembler> DisAsm(TheTarget->createMCDisassembler());
    if (!DisAsm) {
      errs() << "error: no disassembler for target " << TripleName << "\n";
      return;
    }

    int AsmPrinterVariant = AsmInfo->getAssemblerDialect();
    OwningPtr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
                                  AsmPrinterVariant, *AsmInfo));
    if (!IP) {
      errs() << "error: no instruction printer for target " << TripleName << '\n';
      return;
    }

    StringRef Bytes;
    if (error(i->getContents(Bytes))) break;
    StringRefMemoryObject memoryObject(Bytes);
    uint64_t Size;
    uint64_t Index;

    for (Index = 0; Index < Bytes.size(); Index += Size) {
      MCInst Inst;

#     ifndef NDEBUG
      raw_ostream &DebugOut = DebugFlag ? dbgs() : nulls();
#     else
      raw_ostream &DebugOut = nulls();
#     endif

      if (DisAsm->getInstruction(Inst, Size, memoryObject, Index, DebugOut)) {
        uint64_t addr;
        if (error(i->getAddress(addr))) break;
        outs() << format("%8x:\t", addr + Index);
        DumpBytes(StringRef(Bytes.data() + Index, Size));
        IP->printInst(&Inst, outs());
        outs() << "\n";
      } else {
        errs() << ToolName << ": warning: invalid instruction encoding\n";
        if (Size == 0)
          Size = 1; // skip illegible bytes
      }
    }
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  // FIXME: We shouldn't need to initialize the Target(Machine)s.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, "llvm object file dumper\n");
  TripleName = Triple::normalize(TripleName);

  ToolName = argv[0];

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  // -d is the only flag that is currently implemented, so just print help if
  // it is not set.
  if (!Disassemble) {
    cl::PrintHelpMessage();
    return 2;
  }

  std::for_each(InputFilenames.begin(), InputFilenames.end(),
                DisassembleInput);

  return 0;
}
