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

static bool collectRelocatedSymbols(const ObjectFile *Obj,
                                    object::section_iterator SecI, StringRef *I,
                                    StringRef *E) {
  for (const object::RelocationRef &Reloc : SecI->relocations()) {
    if (I == E)
      break;
    const object::symbol_iterator RelocSymI = Reloc.getSymbol();
    if (RelocSymI == Obj->symbol_end())
      continue;
    StringRef RelocSymName;
    if (error(RelocSymI->getName(RelocSymName)))
      return true;
    *I = RelocSymName;
    ++I;
  }
  return false;
}

static bool collectRelocationOffsets(
    const ObjectFile *Obj, object::section_iterator SecI, StringRef SymName,
    std::map<std::pair<StringRef, uint64_t>, StringRef> &Collection) {
  for (const object::RelocationRef &Reloc : SecI->relocations()) {
    const object::symbol_iterator RelocSymI = Reloc.getSymbol();
    if (RelocSymI == Obj->symbol_end())
      continue;
    StringRef RelocSymName;
    if (error(RelocSymI->getName(RelocSymName)))
      return true;
    uint64_t Offset;
    if (error(Reloc.getOffset(Offset)))
      return true;
    Collection[std::make_pair(SymName, Offset)] = RelocSymName;
  }
  return false;
}

static void dumpVTables(const ObjectFile *Obj) {
  struct CompleteObjectLocator {
    StringRef Symbols[2];
    ArrayRef<little32_t> Data;
  };
  struct ClassHierarchyDescriptor {
    StringRef Symbols[1];
    ArrayRef<little32_t> Data;
  };
  struct BaseClassDescriptor {
    StringRef Symbols[2];
    ArrayRef<little32_t> Data;
  };
  struct TypeDescriptor {
    StringRef Symbols[1];
    uint64_t AlwaysZero;
    StringRef MangledName;
  };
  std::map<std::pair<StringRef, uint64_t>, StringRef> VFTableEntries;
  std::map<StringRef, ArrayRef<little32_t>> VBTables;
  std::map<StringRef, CompleteObjectLocator> COLs;
  std::map<StringRef, ClassHierarchyDescriptor> CHDs;
  std::map<std::pair<StringRef, uint64_t>, StringRef> BCAEntries;
  std::map<StringRef, BaseClassDescriptor> BCDs;
  std::map<StringRef, TypeDescriptor> TDs;
  for (const object::SymbolRef &Sym : Obj->symbols()) {
    StringRef SymName;
    if (error(Sym.getName(SymName)))
      return;
    object::section_iterator SecI(Obj->section_begin());
    if (error(Sym.getSection(SecI)))
      return;
    // Skip external symbols.
    if (SecI == Obj->section_end())
      continue;
    bool IsBSS, IsVirtual;
    if (error(SecI->isBSS(IsBSS)) || error(SecI->isVirtual(IsVirtual)))
      break;
    // Skip virtual or BSS sections.
    if (IsBSS || IsVirtual)
      continue;
    StringRef SecContents;
    if (error(SecI->getContents(SecContents)))
      return;
    // VFTables in the MS-ABI start with '??_7' and are contained within their
    // own COMDAT section.  We then determine the contents of the VFTable by
    // looking at each relocation in the section.
    if (SymName.startswith("??_7")) {
      // Each relocation either names a virtual method or a thunk.  We note the
      // offset into the section and the symbol used for the relocation.
      collectRelocationOffsets(Obj, SecI, SymName, VFTableEntries);
    }
    // VBTables in the MS-ABI start with '??_8' and are filled with 32-bit
    // offsets of virtual bases.
    else if (SymName.startswith("??_8")) {
      ArrayRef<little32_t> VBTableData(
          reinterpret_cast<const little32_t *>(SecContents.data()),
          SecContents.size() / sizeof(little32_t));
      VBTables[SymName] = VBTableData;
    }
    // Complete object locators in the MS-ABI start with '??_R4'
    else if (SymName.startswith("??_R4")) {
      CompleteObjectLocator COL;
      COL.Data = ArrayRef<little32_t>(
          reinterpret_cast<const little32_t *>(SecContents.data()), 3);
      StringRef *I = std::begin(COL.Symbols), *E = std::end(COL.Symbols);
      if (collectRelocatedSymbols(Obj, SecI, I, E))
        return;
      COLs[SymName] = COL;
    }
    // Class hierarchy descriptors in the MS-ABI start with '??_R3'
    else if (SymName.startswith("??_R3")) {
      ClassHierarchyDescriptor CHD;
      CHD.Data = ArrayRef<little32_t>(
          reinterpret_cast<const little32_t *>(SecContents.data()), 3);
      StringRef *I = std::begin(CHD.Symbols), *E = std::end(CHD.Symbols);
      if (collectRelocatedSymbols(Obj, SecI, I, E))
        return;
      CHDs[SymName] = CHD;
    }
    // Class hierarchy descriptors in the MS-ABI start with '??_R2'
    else if (SymName.startswith("??_R2")) {
      // Each relocation names a base class descriptor.  We note the offset into
      // the section and the symbol used for the relocation.
      collectRelocationOffsets(Obj, SecI, SymName, BCAEntries);
    }
    // Base class descriptors in the MS-ABI start with '??_R1'
    else if (SymName.startswith("??_R1")) {
      BaseClassDescriptor BCD;
      BCD.Data = ArrayRef<little32_t>(
          reinterpret_cast<const little32_t *>(SecContents.data()) + 1,
          5);
      StringRef *I = std::begin(BCD.Symbols), *E = std::end(BCD.Symbols);
      if (collectRelocatedSymbols(Obj, SecI, I, E))
        return;
      BCDs[SymName] = BCD;
    }
    // Type descriptors in the MS-ABI start with '??_R0'
    else if (SymName.startswith("??_R0")) {
      uint8_t BytesInAddress = Obj->getBytesInAddress();
      const char *DataPtr =
          SecContents.drop_front(Obj->getBytesInAddress()).data();
      TypeDescriptor TD;
      if (BytesInAddress == 8)
        TD.AlwaysZero = *reinterpret_cast<const little64_t *>(DataPtr);
      else
        TD.AlwaysZero = *reinterpret_cast<const little32_t *>(DataPtr);
      TD.MangledName = SecContents.drop_front(Obj->getBytesInAddress() * 2);
      StringRef *I = std::begin(TD.Symbols), *E = std::end(TD.Symbols);
      if (collectRelocatedSymbols(Obj, SecI, I, E))
        return;
      TDs[SymName] = TD;
    }
  }
  for (const std::pair<std::pair<StringRef, uint64_t>, StringRef> &VFTableEntry :
       VFTableEntries) {
    StringRef VFTableName = VFTableEntry.first.first;
    uint64_t Offset = VFTableEntry.first.second;
    StringRef SymName = VFTableEntry.second;
    outs() << VFTableName << '[' << Offset << "]: " << SymName << '\n';
  }
  for (const std::pair<StringRef, ArrayRef<little32_t>> &VBTable :
       VBTables) {
    StringRef VBTableName = VBTable.first;
    uint32_t Idx = 0;
    for (little32_t Offset : VBTable.second) {
      outs() << VBTableName << '[' << Idx << "]: " << Offset << '\n';
      Idx += sizeof(Offset);
    }
  }
  for (const std::pair<StringRef, CompleteObjectLocator> &COLPair : COLs) {
    StringRef COLName = COLPair.first;
    const CompleteObjectLocator &COL = COLPair.second;
    outs() << COLName << "[IsImageRelative]: " << COL.Data[0] << '\n';
    outs() << COLName << "[OffsetToTop]: " << COL.Data[1] << '\n';
    outs() << COLName << "[VFPtrOffset]: " << COL.Data[2] << '\n';
    outs() << COLName << "[TypeDescriptor]: " << COL.Symbols[0] << '\n';
    outs() << COLName << "[ClassHierarchyDescriptor]: " << COL.Symbols[1] << '\n';
  }
  for (const std::pair<StringRef, ClassHierarchyDescriptor> &CHDPair : CHDs) {
    StringRef CHDName = CHDPair.first;
    const ClassHierarchyDescriptor &CHD = CHDPair.second;
    outs() << CHDName << "[AlwaysZero]: " << CHD.Data[0] << '\n';
    outs() << CHDName << "[Flags]: " << CHD.Data[1] << '\n';
    outs() << CHDName << "[NumClasses]: " << CHD.Data[2] << '\n';
    outs() << CHDName << "[BaseClassArray]: " << CHD.Symbols[0] << '\n';
  }
  for (const std::pair<std::pair<StringRef, uint64_t>, StringRef> &BCAEntry :
       BCAEntries) {
    StringRef BCAName = BCAEntry.first.first;
    uint64_t Offset = BCAEntry.first.second;
    StringRef SymName = BCAEntry.second;
    outs() << BCAName << '[' << Offset << "]: " << SymName << '\n';
  }
  for (const std::pair<StringRef, BaseClassDescriptor> &BCDPair : BCDs) {
    StringRef BCDName = BCDPair.first;
    const BaseClassDescriptor &BCD = BCDPair.second;
    outs() << BCDName << "[TypeDescriptor]: " << BCD.Symbols[0] << '\n';
    outs() << BCDName << "[NumBases]: " << BCD.Data[0] << '\n';
    outs() << BCDName << "[OffsetInVBase]: " << BCD.Data[1] << '\n';
    outs() << BCDName << "[VBPtrOffset]: " << BCD.Data[2] << '\n';
    outs() << BCDName << "[OffsetInVBTable]: " << BCD.Data[3] << '\n';
    outs() << BCDName << "[Flags]: " << BCD.Data[4] << '\n';
    outs() << BCDName << "[ClassHierarchyDescriptor]: " << BCD.Symbols[1] << '\n';
  }
  for (const std::pair<StringRef, TypeDescriptor> &TDPair : TDs) {
    StringRef TDName = TDPair.first;
    const TypeDescriptor &TD = TDPair.second;
    outs() << TDName << "[VFPtr]: " << TD.Symbols[0] << '\n';
    outs() << TDName << "[AlwaysZero]: " << TD.AlwaysZero << '\n';
    outs() << TDName << "[MangledName]: ";
    outs().write_escaped(TD.MangledName.rtrim(StringRef("\0", 1)),
                         /*UseHexEscapes=*/true)
        << '\n';
  }
}

static void dumpArchive(const Archive *Arc) {
  for (const Archive::Child &ArcC : Arc->children()) {
    ErrorOr<std::unique_ptr<Binary>> ChildOrErr = ArcC.getAsBinary();
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
  ErrorOr<OwningBinary<Binary>> BinaryOrErr = createBinary(File);
  if (std::error_code EC = BinaryOrErr.getError()) {
    reportError(File, EC);
    return;
  }
  Binary &Binary = *BinaryOrErr.get().getBinary();

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
