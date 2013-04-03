//===- llvm-readobj.cpp - Dump contents of an Object File -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a tool similar to readelf, except it works on multiple object file
// formats. The main purpose of this tool is to provide detailed output suitable
// for FileCheck.
//
// Flags should be similar to readelf where supported, but the output format
// does not need to be identical. The point is to not make users learn yet
// another set of flags.
//
// Output should be specialized for each format where appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm-readobj.h"

#include "Error.h"
#include "ObjDumper.h"
#include "StreamWriter.h"

#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/system_error.h"

#include <string>


using namespace llvm;
using namespace llvm::object;

namespace opts {
  cl::list<std::string> InputFilenames(cl::Positional,
    cl::desc("<input object files>"),
    cl::ZeroOrMore);

  // -file-headers, -h
  cl::opt<bool> FileHeaders("file-headers",
    cl::desc("Display file headers "));
  cl::alias FileHeadersShort("h",
    cl::desc("Alias for --file-headers"),
    cl::aliasopt(FileHeaders));

  // -sections, -s
  cl::opt<bool> Sections("sections",
    cl::desc("Display all sections."));
  cl::alias SectionsShort("s",
    cl::desc("Alias for --sections"),
    cl::aliasopt(Sections));

  // -section-relocations, -sr
  cl::opt<bool> SectionRelocations("section-relocations",
    cl::desc("Display relocations for each section shown."));
  cl::alias SectionRelocationsShort("sr",
    cl::desc("Alias for --section-relocations"),
    cl::aliasopt(SectionRelocations));

  // -section-symbols, -st
  cl::opt<bool> SectionSymbols("section-symbols",
    cl::desc("Display symbols for each section shown."));
  cl::alias SectionSymbolsShort("st",
    cl::desc("Alias for --section-symbols"),
    cl::aliasopt(SectionSymbols));

  // -section-data, -sd
  cl::opt<bool> SectionData("section-data",
    cl::desc("Display section data for each section shown."));
  cl::alias SectionDataShort("sd",
    cl::desc("Alias for --section-data"),
    cl::aliasopt(SectionData));

  // -relocations, -r
  cl::opt<bool> Relocations("relocations",
    cl::desc("Display the relocation entries in the file"));
  cl::alias RelocationsShort("r",
    cl::desc("Alias for --relocations"),
    cl::aliasopt(Relocations));

  // -symbols, -t
  cl::opt<bool> Symbols("symbols",
    cl::desc("Display the symbol table"));
  cl::alias SymbolsShort("t",
    cl::desc("Alias for --symbols"),
    cl::aliasopt(Symbols));

  // -dyn-symbols, -dt
  cl::opt<bool> DynamicSymbols("dyn-symbols",
    cl::desc("Display the dynamic symbol table"));
  cl::alias DynamicSymbolsShort("dt",
    cl::desc("Alias for --dyn-symbols"),
    cl::aliasopt(DynamicSymbols));

  // -unwind, -u
  cl::opt<bool> UnwindInfo("unwind",
    cl::desc("Display unwind information"));
  cl::alias UnwindInfoShort("u",
    cl::desc("Alias for --unwind"),
    cl::aliasopt(UnwindInfo));

  // -dynamic-table
  cl::opt<bool> DynamicTable("dynamic-table",
    cl::desc("Display the ELF .dynamic section table"));

  // -needed-libs
  cl::opt<bool> NeededLibraries("needed-libs",
    cl::desc("Display the needed libraries"));
} // namespace opts

namespace llvm {

bool error(error_code EC) {
  if (!EC)
    return false;

  outs() << "\nError reading file: " << EC.message() << ".\n";
  outs().flush();
  return true;
}

bool relocAddressLess(RelocationRef a, RelocationRef b) {
  uint64_t a_addr, b_addr;
  if (error(a.getAddress(a_addr))) return false;
  if (error(b.getAddress(b_addr))) return false;
  return a_addr < b_addr;
}

} // namespace llvm


static void reportError(StringRef Input, error_code EC) {
  if (Input == "-")
    Input = "<stdin>";

  errs() << Input << ": " << EC.message() << "\n";
  errs().flush();
}

static void reportError(StringRef Input, StringRef Message) {
  if (Input == "-")
    Input = "<stdin>";

  errs() << Input << ": " << Message << "\n";
}

/// @brief Creates an format-specific object file dumper.
static error_code createDumper(const ObjectFile *Obj,
                               StreamWriter &Writer,
                               OwningPtr<ObjDumper> &Result) {
  if (!Obj)
    return readobj_error::unsupported_file_format;

  if (Obj->isCOFF())
    return createCOFFDumper(Obj, Writer, Result);
  if (Obj->isELF())
    return createELFDumper(Obj, Writer, Result);
  if (Obj->isMachO())
    return createMachODumper(Obj, Writer, Result);

  return readobj_error::unsupported_obj_file_format;
}


/// @brief Dumps the specified object file.
static void dumpObject(const ObjectFile *Obj) {
  StreamWriter Writer(outs());
  OwningPtr<ObjDumper> Dumper;
  if (error_code EC = createDumper(Obj, Writer, Dumper)) {
    reportError(Obj->getFileName(), EC);
    return;
  }

  outs() << '\n';
  outs() << "File: " << Obj->getFileName() << "\n";
  outs() << "Format: " << Obj->getFileFormatName() << "\n";
  outs() << "Arch: "
         << Triple::getArchTypeName((llvm::Triple::ArchType)Obj->getArch())
         << "\n";
  outs() << "AddressSize: " << (8*Obj->getBytesInAddress()) << "bit\n";
  if (Obj->isELF())
    outs() << "LoadName: " << Obj->getLoadName() << "\n";

  if (opts::FileHeaders)
    Dumper->printFileHeaders();
  if (opts::Sections)
    Dumper->printSections();
  if (opts::Relocations)
    Dumper->printRelocations();
  if (opts::Symbols)
    Dumper->printSymbols();
  if (opts::DynamicSymbols)
    Dumper->printDynamicSymbols();
  if (opts::UnwindInfo)
    Dumper->printUnwindInfo();
  if (opts::DynamicTable)
    Dumper->printDynamicTable();
  if (opts::NeededLibraries)
    Dumper->printNeededLibraries();
}


/// @brief Dumps each object file in \a Arc;
static void dumpArchive(const Archive *Arc) {
  for (Archive::child_iterator ArcI = Arc->begin_children(),
                               ArcE = Arc->end_children();
                               ArcI != ArcE; ++ArcI) {
    OwningPtr<Binary> child;
    if (error_code EC = ArcI->getAsBinary(child)) {
      // Ignore non-object files.
      if (EC != object_error::invalid_file_type)
        reportError(Arc->getFileName(), EC.message());
      continue;
    }

    if (ObjectFile *Obj = dyn_cast<ObjectFile>(child.get()))
      dumpObject(Obj);
    else
      reportError(Arc->getFileName(), readobj_error::unrecognized_file_format);
  }
}


/// @brief Opens \a File and dumps it.
static void dumpInput(StringRef File) {
  // If file isn't stdin, check that it exists.
  if (File != "-" && !sys::fs::exists(File)) {
    reportError(File, readobj_error::file_not_found);
    return;
  }

  // Attempt to open the binary.
  OwningPtr<Binary> Binary;
  if (error_code EC = createBinary(File, Binary)) {
    reportError(File, EC);
    return;
  }

  if (Archive *Arc = dyn_cast<Archive>(Binary.get()))
    dumpArchive(Arc);
  else if (ObjectFile *Obj = dyn_cast<ObjectFile>(Binary.get()))
    dumpObject(Obj);
  else
    reportError(File, readobj_error::unrecognized_file_format);
}


int main(int argc, const char *argv[]) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;

  // Initialize targets.
  llvm::InitializeAllTargetInfos();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "LLVM Object Reader\n");

  // Default to stdin if no filename is specified.
  if (opts::InputFilenames.size() == 0)
    opts::InputFilenames.push_back("-");

  std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                dumpInput);

  return 0;
}
