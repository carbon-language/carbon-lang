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

#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input object>"), cl::init(""));

static void dumpSymbolHeader() {
  outs() << format("  %-32s", (const char *)"Name")
         << format("  %-4s", (const char *)"Type")
         << format("  %-4s", (const char *)"Section")
         << format("  %-16s", (const char *)"Address")
         << format("  %-16s", (const char *)"Size")
         << format("  %-16s", (const char *)"FileOffset")
         << format("  %-26s", (const char *)"Flags") << "\n";
}

static void dumpSectionHeader() {
  outs() << format("  %-24s", (const char*)"Name")
         << format("  %-16s", (const char*)"Address")
         << format("  %-16s", (const char*)"Size")
         << format("  %-8s", (const char*)"Align")
         << format("  %-26s", (const char*)"Flags")
         << "\n";
}

static const char *getTypeStr(SymbolRef::Type Type) {
  switch (Type) {
  case SymbolRef::ST_Unknown: return "?";
  case SymbolRef::ST_Data: return "DATA";
  case SymbolRef::ST_Debug: return "DBG";
  case SymbolRef::ST_File: return "FILE";
  case SymbolRef::ST_Function: return "FUNC";
  case SymbolRef::ST_Other: return "-";
  }
  return "INV";
}

static std::string getSymbolFlagStr(uint32_t Flags) {
  std::string result;
  if (Flags & SymbolRef::SF_Undefined)
    result += "undef,";
  if (Flags & SymbolRef::SF_Global)
    result += "global,";
  if (Flags & SymbolRef::SF_Weak)
    result += "weak,";
  if (Flags & SymbolRef::SF_Absolute)
    result += "absolute,";
  if (Flags & SymbolRef::SF_ThreadLocal)
    result += "threadlocal,";
  if (Flags & SymbolRef::SF_Common)
    result += "common,";
  if (Flags & SymbolRef::SF_FormatSpecific)
    result += "formatspecific,";

  // Remove trailing comma
  if (result.size() > 0) {
    result.erase(result.size() - 1);
  }
  return result;
}

static void checkError(error_code ec, const char *msg) {
  if (ec)
    report_fatal_error(std::string(msg) + ": " + ec.message());
}

static std::string getSectionFlagStr(const SectionRef &Section) {
  const struct {
    error_code (SectionRef::*MemF)(bool &) const;
    const char *FlagStr, *ErrorStr;
  } Work[] =
      {{ &SectionRef::isText, "text,", "Section.isText() failed" },
       { &SectionRef::isData, "data,", "Section.isData() failed" },
       { &SectionRef::isBSS, "bss,", "Section.isBSS() failed"  },
       { &SectionRef::isRequiredForExecution, "required,",
         "Section.isRequiredForExecution() failed" },
       { &SectionRef::isVirtual, "virtual,", "Section.isVirtual() failed" },
       { &SectionRef::isZeroInit, "zeroinit,", "Section.isZeroInit() failed" },
       { &SectionRef::isReadOnlyData, "rodata,",
         "Section.isReadOnlyData() failed" }};

  std::string result;
  for (uint32_t I = 0; I < sizeof(Work)/sizeof(*Work); ++I) {
    bool B;
    checkError((Section.*Work[I].MemF)(B), Work[I].ErrorStr);
    if (B)
      result += Work[I].FlagStr;
  }

  // Remove trailing comma
  if (result.size() > 0) {
    result.erase(result.size() - 1);
  }
  return result;
}

static void
dumpSymbol(const SymbolRef &Sym, const ObjectFile *obj, bool IsDynamic) {
  StringRef Name;
  SymbolRef::Type Type;
  uint32_t Flags;
  uint64_t Address;
  uint64_t Size;
  uint64_t FileOffset;
  checkError(Sym.getName(Name), "SymbolRef.getName() failed");
  checkError(Sym.getAddress(Address), "SymbolRef.getAddress() failed");
  checkError(Sym.getSize(Size), "SymbolRef.getSize() failed");
  checkError(Sym.getFileOffset(FileOffset),
             "SymbolRef.getFileOffset() failed");
  checkError(Sym.getType(Type), "SymbolRef.getType() failed");
  checkError(Sym.getFlags(Flags), "SymbolRef.getFlags() failed");
  std::string FullName = Name;

  llvm::object::section_iterator symSection(obj->begin_sections());
  Sym.getSection(symSection);
  StringRef sectionName;

  if (symSection != obj->end_sections())
    checkError(symSection->getName(sectionName),
               "SectionRef::getName() failed");

  // If this is a dynamic symbol from an ELF object, append
  // the symbol's version to the name.
  if (IsDynamic && obj->isELF()) {
    StringRef Version;
    bool IsDefault;
    GetELFSymbolVersion(obj, Sym, Version, IsDefault);
    if (!Version.empty()) {
      FullName += (IsDefault ? "@@" : "@");
      FullName += Version;
    }
  }

  // format() can't handle StringRefs
  outs() << format("  %-32s", FullName.c_str())
         << format("  %-4s", getTypeStr(Type))
         << format("  %-32s", std::string(sectionName).c_str())
         << format("  %16" PRIx64, Address) << format("  %16" PRIx64, Size)
         << format("  %16" PRIx64, FileOffset) << "  "
         << getSymbolFlagStr(Flags) << "\n";
}

static void dumpStaticSymbol(const SymbolRef &Sym, const ObjectFile *obj) {
  return dumpSymbol(Sym, obj, false);
}

static void dumpDynamicSymbol(const SymbolRef &Sym, const ObjectFile *obj) {
  return dumpSymbol(Sym, obj, true);
}

static void dumpSection(const SectionRef &Section, const ObjectFile *obj) {
  StringRef Name;
  checkError(Section.getName(Name), "SectionRef::getName() failed");
  uint64_t Addr, Size, Align;
  checkError(Section.getAddress(Addr), "SectionRef::getAddress() failed");
  checkError(Section.getSize(Size), "SectionRef::getSize() failed");
  checkError(Section.getAlignment(Align), "SectionRef::getAlignment() failed");
  outs() << format("  %-24s", std::string(Name).c_str())
         << format("  %16" PRIx64, Addr)
         << format("  %16" PRIx64, Size)
         << format("  %8" PRIx64, Align)
         << "  " << getSectionFlagStr(Section)
         << "\n";
}

static void dumpLibrary(const LibraryRef &lib, const ObjectFile *obj) {
  StringRef path;
  lib.getPath(path);
  outs() << "  " << path << "\n";
}

template<typename Iterator, typename Func>
static void dump(const ObjectFile *obj, Func f, Iterator begin, Iterator end,
                 const char *errStr) {
  error_code ec;
  uint32_t count = 0;
  Iterator it = begin, ie = end;
  while (it != ie) {
    f(*it, obj);
    it.increment(ec);
    if (ec)
      report_fatal_error(errStr);
    ++count;
  }
  outs() << "  Total: " << count << "\n\n";
}

static void dumpHeaders(const ObjectFile *obj) {
  outs() << "File Format : " << obj->getFileFormatName() << "\n";
  outs() << "Arch        : "
         << Triple::getArchTypeName((llvm::Triple::ArchType)obj->getArch())
         << "\n";
  outs() << "Address Size: " << (8*obj->getBytesInAddress()) << " bits\n";
  outs() << "Load Name   : " << obj->getLoadName() << "\n";
  outs() << "\n";
}

int main(int argc, char** argv) {
  error_code ec;
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "LLVM Object Reader\n");

  if (InputFilename.empty()) {
    errs() << "Please specify an input filename\n";
    return 1;
  }

  // Open the object file
  OwningPtr<MemoryBuffer> File;
  if (MemoryBuffer::getFile(InputFilename, File)) {
    errs() << InputFilename << ": Open failed\n";
    return 1;
  }

  OwningPtr<ObjectFile> o(ObjectFile::createObjectFile(File.take()));
  ObjectFile *obj = o.get();
  if (!obj) {
    errs() << InputFilename << ": Object type not recognized\n";
  }

  dumpHeaders(obj);

  outs() << "Symbols:\n";
  dumpSymbolHeader();
  dump(obj, dumpStaticSymbol, obj->begin_symbols(), obj->end_symbols(),
       "Symbol iteration failed");

  outs() << "Dynamic Symbols:\n";
  dumpSymbolHeader();
  dump(obj, dumpDynamicSymbol, obj->begin_dynamic_symbols(),
       obj->end_dynamic_symbols(), "Symbol iteration failed");

  outs() << "Sections:\n";
  dumpSectionHeader();
  dump(obj, &dumpSection, obj->begin_sections(), obj->end_sections(),
       "Section iteration failed");

  if (obj->isELF()) {
    if (ErrorOr<void> e = dumpELFDynamicTable(obj, outs()))
      ;
    else
      errs() << "InputFilename" << ": " << error_code(e).message() << "\n";
  }

  outs() << "Libraries needed:\n";
  dump(obj, &dumpLibrary, obj->begin_libraries_needed(),
       obj->end_libraries_needed(), "Needed libraries iteration failed");

  return 0;
}

