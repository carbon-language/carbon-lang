//===-- llvm-dwarfdump.cpp - Debug info dumping utility for llvm ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like "dwarfdump".
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocVisitor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <algorithm>
#include <cstring>
#include <list>
#include <string>

using namespace llvm;
using namespace object;

static cl::list<std::string>
InputFilenames(cl::Positional, cl::desc("<input object files>"),
               cl::ZeroOrMore);

static cl::opt<unsigned long long>
Address("address", cl::init(-1ULL),
        cl::desc("Print line information for a given address"));

static cl::opt<bool>
PrintFunctions("functions", cl::init(false),
               cl::desc("Print function names as well as line information "
                        "for a given address"));

static cl::opt<bool>
PrintInlining("inlining", cl::init(false),
              cl::desc("Print all inlined frames for a given address"));

static cl::opt<DIDumpType>
DumpType("debug-dump", cl::init(DIDT_All),
  cl::desc("Dump of debug sections:"),
  cl::values(
        clEnumValN(DIDT_All, "all", "Dump all debug sections"),
        clEnumValN(DIDT_Abbrev, "abbrev", ".debug_abbrev"),
        clEnumValN(DIDT_AbbrevDwo, "abbrev.dwo", ".debug_abbrev.dwo"),
        clEnumValN(DIDT_Aranges, "aranges", ".debug_aranges"),
        clEnumValN(DIDT_Info, "info", ".debug_info"),
        clEnumValN(DIDT_InfoDwo, "info.dwo", ".debug_info.dwo"),
        clEnumValN(DIDT_Types, "types", ".debug_types"),
        clEnumValN(DIDT_TypesDwo, "types.dwo", ".debug_types.dwo"),
        clEnumValN(DIDT_Line, "line", ".debug_line"),
        clEnumValN(DIDT_LineDwo, "line.dwo", ".debug_line.dwo"),
        clEnumValN(DIDT_Loc, "loc", ".debug_loc"),
        clEnumValN(DIDT_Frames, "frames", ".debug_frame"),
        clEnumValN(DIDT_Ranges, "ranges", ".debug_ranges"),
        clEnumValN(DIDT_Pubnames, "pubnames", ".debug_pubnames"),
        clEnumValN(DIDT_Pubtypes, "pubtypes", ".debug_pubtypes"),
        clEnumValN(DIDT_GnuPubnames, "gnu_pubnames", ".debug_gnu_pubnames"),
        clEnumValN(DIDT_GnuPubtypes, "gnu_pubtypes", ".debug_gnu_pubtypes"),
        clEnumValN(DIDT_Str, "str", ".debug_str"),
        clEnumValN(DIDT_StrDwo, "str.dwo", ".debug_str.dwo"),
        clEnumValN(DIDT_StrOffsetsDwo, "str_offsets.dwo", ".debug_str_offsets.dwo"),
        clEnumValEnd));

static void PrintDILineInfo(DILineInfo dli) {
  if (PrintFunctions)
    outs() << (dli.getFunctionName() ? dli.getFunctionName() : "<unknown>")
           << "\n";
  outs() << (dli.getFileName() ? dli.getFileName() : "<unknown>") << ':'
         << dli.getLine() << ':' << dli.getColumn() << '\n';
}

static void DumpInput(const StringRef &Filename) {
  std::unique_ptr<MemoryBuffer> Buff;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
    errs() << Filename << ": " << ec.message() << "\n";
    return;
  }

  ErrorOr<ObjectFile*> ObjOrErr(ObjectFile::createObjectFile(Buff.release()));
  if (error_code EC = ObjOrErr.getError()) {
    errs() << Filename << ": " << EC.message() << '\n';
    return;
  }
  std::unique_ptr<ObjectFile> Obj(ObjOrErr.get());

  std::unique_ptr<DIContext> DICtx(DIContext::getDWARFContext(Obj.get()));

  if (Address == -1ULL) {
    outs() << Filename
           << ":\tfile format " << Obj->getFileFormatName() << "\n\n";
    // Dump the complete DWARF structure.
    DICtx->dump(outs(), DumpType);
  } else {
    // Print line info for the specified address.
    int SpecFlags = DILineInfoSpecifier::FileLineInfo |
                    DILineInfoSpecifier::AbsoluteFilePath;
    if (PrintFunctions)
      SpecFlags |= DILineInfoSpecifier::FunctionName;
    if (PrintInlining) {
      DIInliningInfo InliningInfo =
        DICtx->getInliningInfoForAddress(Address, SpecFlags);
      uint32_t n = InliningInfo.getNumberOfFrames();
      if (n == 0) {
        // Print one empty debug line info in any case.
        PrintDILineInfo(DILineInfo());
      } else {
        for (uint32_t i = 0; i < n; i++) {
          DILineInfo dli = InliningInfo.getFrame(i);
          PrintDILineInfo(dli);
        }
      }
    } else {
      DILineInfo dli = DICtx->getLineInfoForAddress(Address, SpecFlags);
      PrintDILineInfo(dli);
    }
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm dwarf dumper\n");

  // Defaults to a.out if no filenames specified.
  if (InputFilenames.size() == 0)
    InputFilenames.push_back("a.out");

  std::for_each(InputFilenames.begin(), InputFilenames.end(), DumpInput);

  return 0;
}
