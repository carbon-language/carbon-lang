//===- llvm-vtabledump.cpp - Dump vtables in an Object File -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Dumps VTables resident in object files and archives.  Note, it currently only
// supports MS-ABI style object files.
//
//===----------------------------------------------------------------------===//

#include "llvm-vtabledump.h"
#include "Error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include <map>
#include <string>
#include <system_error>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support;

namespace opts {
cl::list<std::string> InputFilenames(cl::Positional,
                                     cl::desc("<input object files>"),
                                     cl::ZeroOrMore);
} // namespace opts

static int ReturnValue = EXIT_SUCCESS;

namespace llvm {

bool error(std::error_code EC) {
  if (!EC)
    return false;

  ReturnValue = EXIT_FAILURE;
  outs() << "\nError reading file: " << EC.message() << ".\n";
  outs().flush();
  return true;
}

} // namespace llvm

static void reportError(StringRef Input, StringRef Message) {
  if (Input == "-")
    Input = "<stdin>";

  errs() << Input << ": " << Message << "\n";
  errs().flush();
  ReturnValue = EXIT_FAILURE;
}

static void reportError(StringRef Input, std::error_code EC) {
  reportError(Input, EC.message());
}

static void dumpVTables(const ObjectFile *Obj) {
  std::map<std::pair<StringRef, uint64_t>, StringRef> VFTableEntries;
  std::map<StringRef, ArrayRef<aligned_little32_t>> VBTables;
  for (const object::SymbolRef &Sym : Obj->symbols()) {
    StringRef SymName;
    if (error(Sym.getName(SymName)))
      return;
    // VFTables in the MS-ABI start with '??_7' and are contained within their
    // own COMDAT section.  We then determine the contents of the VFTable by
    // looking at each relocation in the section.
    if (SymName.startswith("??_7")) {
      object::section_iterator SecI(Obj->section_begin());
      if (error(Sym.getSection(SecI)))
        return;
      if (SecI == Obj->section_end())
        continue;
      // Each relocation either names a virtual method or a thunk.  We note the
      // offset into the section and the symbol used for the relocation.
      for (const object::RelocationRef &Reloc : SecI->relocations()) {
        const object::symbol_iterator RelocSymI = Reloc.getSymbol();
        if (RelocSymI == Obj->symbol_end())
          continue;
        StringRef RelocSymName;
        if (error(RelocSymI->getName(RelocSymName)))
          return;
        uint64_t Offset;
        if (error(Reloc.getOffset(Offset)))
          return;
        VFTableEntries[std::make_pair(SymName, Offset)] = RelocSymName;
      }
    }
    // VBTables in the MS-ABI start with '??_8' and are filled with 32-bit
    // offsets of virtual bases.
    else if (SymName.startswith("??_8")) {
      object::section_iterator SecI(Obj->section_begin());
      if (error(Sym.getSection(SecI)))
        return;
      if (SecI == Obj->section_end())
        continue;
      StringRef SecContents;
      if (error(SecI->getContents(SecContents)))
        return;

      ArrayRef<aligned_little32_t> VBTableData(
          reinterpret_cast<const aligned_little32_t *>(SecContents.data()),
          SecContents.size() / sizeof(aligned_little32_t));
      VBTables[SymName] = VBTableData;
    }
  }
  for (
      const std::pair<std::pair<StringRef, uint64_t>, StringRef> &VFTableEntry :
      VFTableEntries) {
    StringRef VFTableName = VFTableEntry.first.first;
    uint64_t Offset = VFTableEntry.first.second;
    StringRef SymName = VFTableEntry.second;
    outs() << VFTableName << '[' << Offset << "]: " << SymName << '\n';
  }
  for (const std::pair<StringRef, ArrayRef<aligned_little32_t>> &VBTable :
       VBTables) {
    StringRef VBTableName = VBTable.first;
    uint32_t Idx = 0;
    for (aligned_little32_t Offset : VBTable.second) {
      outs() << VBTableName << '[' << Idx << "]: " << Offset << '\n';
      Idx += sizeof(Offset);
    }
  }
}

static void dumpArchive(const Archive *Arc) {
  for (Archive::child_iterator ArcI = Arc->child_begin(),
                               ArcE = Arc->child_end();
       ArcI != ArcE; ++ArcI) {
    ErrorOr<std::unique_ptr<Binary>> ChildOrErr = ArcI->getAsBinary();
    if (std::error_code EC = ChildOrErr.getError()) {
      // Ignore non-object files.
      if (EC != object_error::invalid_file_type)
        reportError(Arc->getFileName(), EC.message());
      continue;
    }

    if (ObjectFile *Obj = dyn_cast<ObjectFile>(&*ChildOrErr.get()))
      dumpVTables(Obj);
    else
      reportError(Arc->getFileName(),
                  vtabledump_error::unrecognized_file_format);
  }
}

static void dumpInput(StringRef File) {
  // If file isn't stdin, check that it exists.
  if (File != "-" && !sys::fs::exists(File)) {
    reportError(File, vtabledump_error::file_not_found);
    return;
  }

  // Attempt to open the binary.
  ErrorOr<std::unique_ptr<Binary>> BinaryOrErr = createBinary(File);
  if (std::error_code EC = BinaryOrErr.getError()) {
    reportError(File, EC);
    return;
  }
  Binary &Binary = *BinaryOrErr.get();

  if (Archive *Arc = dyn_cast<Archive>(&Binary))
    dumpArchive(Arc);
  else if (ObjectFile *Obj = dyn_cast<ObjectFile>(&Binary))
    dumpVTables(Obj);
  else
    reportError(File, vtabledump_error::unrecognized_file_format);
}

int main(int argc, const char *argv[]) {
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;

  // Initialize targets.
  llvm::InitializeAllTargetInfos();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "LLVM VTable Dumper\n");

  // Default to stdin if no filename is specified.
  if (opts::InputFilenames.size() == 0)
    opts::InputFilenames.push_back("-");

  std::for_each(opts::InputFilenames.begin(), opts::InputFilenames.end(),
                dumpInput);

  return ReturnValue;
}
