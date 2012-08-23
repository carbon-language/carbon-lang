//===-- llvm-symbolizer.cpp - Simple addr2line-like symbolizer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility works much like "addr2line". It is able of transforming
// tuples (module name, module offset) to code locations (function name,
// file, line number, column number). It is targeted for compiler-rt tools
// (especially AddressSanitizer and ThreadSanitizer) that can use it
// to symbolize stack traces in their error reports.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstring>
#include <iostream>  // NOLINT
#include <map>
#include <string>

using namespace llvm;
using namespace object;
using std::string;

static cl::opt<bool>
UseSymbolTable("use-symbol-table", cl::init(true),
               cl::desc("Prefer names in symbol table to names "
                        "in debug info"));

static cl::opt<bool>
PrintFunctions("functions", cl::init(true),
               cl::desc("Print function names as well as line "
                        "information for a given address"));

static cl::opt<bool>
SubprocessMode("subprocess-mode", cl::init(false),
               cl::desc("Is run as a subprocess (format of the output "
                        "differs a bit)"));

static StringRef ToolInvocationPath;

static bool error(error_code ec) {
  if (!ec) return false;
  errs() << ToolInvocationPath << ": error reading file: "
         << ec.message() << ".\n";
  return true;
}

namespace {
class ModuleInfo {
  OwningPtr<ObjectFile> Module;
  OwningPtr<DIContext> DebugInfoContext;
 public:
  ModuleInfo(ObjectFile *obj, DIContext *di_ctx)
      : Module(obj), DebugInfoContext(di_ctx) {}
  DILineInfo symbolizeCode(uint64_t module_offset) const {
    DILineInfo dli;
    if (DebugInfoContext) {
      uint32_t flags = llvm::DILineInfoSpecifier::FileLineInfo |
                       llvm::DILineInfoSpecifier::AbsoluteFilePath;
      if (PrintFunctions)
        flags |= llvm::DILineInfoSpecifier::FunctionName;
      dli = DebugInfoContext->getLineInfoForAddress(
          module_offset, flags);
    }
    // Override function name from symbol table if necessary.
    if (PrintFunctions && UseSymbolTable) {
      string filename = dli.getFileName();
      string function = dli.getFunctionName();
      if (getFunctionNameFromSymbolTable(module_offset, function)) {
        dli = DILineInfo(StringRef(filename), StringRef(function),
                         dli.getLine(), dli.getColumn());
      }
    }
    return dli;
  }
 private:
  bool getFunctionNameFromSymbolTable(size_t address,
                                      string &function_name) const {
    assert(Module);
    error_code ec;
    for (symbol_iterator si = Module->begin_symbols(),
                         se = Module->end_symbols();
                         si != se; si.increment(ec)) {
      if (error(ec)) return false;
      uint64_t Address;
      uint64_t Size;
      if (error(si->getAddress(Address))) continue;
      if (error(si->getSize(Size))) continue;
      // FIXME: If a function has alias, there are two entries in symbol table
      // with same address size. Make sure we choose the correct one.
      if (Address <= address && address < Address + Size) {
        StringRef Name;
        if (error(si->getName(Name))) continue;
        function_name = Name.str();
        return true;
      }
    }
    return false;
  }
};

typedef std::map<string, ModuleInfo*> ModuleMapTy;
typedef ModuleMapTy::iterator ModuleMapIter;
typedef ModuleMapTy::const_iterator ModuleMapConstIter;
}  // namespace

static ModuleMapTy modules;

static bool isFullNameOfDwarfSection(const StringRef &full_name,
                                     const StringRef &short_name) {
  static const char kDwarfPrefix[] = "__DWARF,";
  StringRef name = full_name;
  // Skip "__DWARF," prefix.
  if (name.startswith(kDwarfPrefix))
    name = name.substr(strlen(kDwarfPrefix));
  // Skip . and _ prefixes.
  name = name.substr(name.find_first_not_of("._"));
  return (name == short_name);
}

// Returns true if the object endianness is known.
static bool getObjectEndianness(const ObjectFile *obj,
                                bool &is_little_endian) {
  // FIXME: Implement this when libLLVMObject allows to do it easily.
  is_little_endian = true;
  return true;
}

static ModuleInfo *getOrCreateModuleInfo(const string &module_name) {
  ModuleMapIter I = modules.find(module_name);
  if (I != modules.end())
    return I->second;

  OwningPtr<MemoryBuffer> Buff;
  MemoryBuffer::getFile(module_name, Buff);
  ObjectFile *obj = ObjectFile::createObjectFile(Buff.take());
  assert(obj);

  DIContext *di_context = 0;
  bool IsLittleEndian;
  if (getObjectEndianness(obj, IsLittleEndian)) {
    StringRef DebugInfoSection;
    StringRef DebugAbbrevSection;
    StringRef DebugLineSection;
    StringRef DebugArangesSection;
    StringRef DebugStringSection;
    error_code ec;
    for (section_iterator i = obj->begin_sections(),
                          e = obj->end_sections();
                          i != e; i.increment(ec)) {
      if (error(ec)) break;
      StringRef name;
      if (error(i->getName(name))) continue;
      StringRef data;
      if (error(i->getContents(data))) continue;
      if (isFullNameOfDwarfSection(name, "debug_info"))
        DebugInfoSection = data;
      else if (isFullNameOfDwarfSection(name, "debug_abbrev"))
        DebugAbbrevSection = data;
      else if (isFullNameOfDwarfSection(name, "debug_line"))
        DebugLineSection = data;
      // Don't use debug_aranges for now, as address ranges contained
      // there may not cover all instructions in the module
      // else if (isFullNameOfDwarfSection(name, "debug_aranges"))
      //   DebugArangesSection = data;
      else if (isFullNameOfDwarfSection(name, "debug_str"))
        DebugStringSection = data;
    }

    di_context = DIContext::getDWARFContext(
        IsLittleEndian, DebugInfoSection, DebugAbbrevSection,
        DebugArangesSection, DebugLineSection, DebugStringSection);
    assert(di_context);
  }

  ModuleInfo *module_info = new ModuleInfo(obj, di_context);
  modules.insert(make_pair(module_name, module_info));
  return module_info;
}

static void symbolize(const string &module_name,
                      const string &module_offset_str) {
  // FIXME: check that module_name points to valid file.
  ModuleInfo *module_info = getOrCreateModuleInfo(module_name);
  DILineInfo line_info;
  uint64_t module_offset;
  if (!StringRef(module_offset_str).getAsInteger(0, module_offset)) {
    line_info = module_info->symbolizeCode(module_offset);
  }
  // By default, DILineInfo contains "<invalid>" for function/filename it
  // cannot fetch. We replace it to "??" to make our output closer to addr2line.
  static const string kDILineInfoBadString = "<invalid>";
  static const string kSymbolizerBadString = "??";

  if (PrintFunctions) {
    string function_name = line_info.getFunctionName();
    if (function_name == kDILineInfoBadString)
      function_name = kSymbolizerBadString;
    outs() << function_name << "\n";
  }
  string filename = line_info.getFileName();
  if (filename == kDILineInfoBadString)
    filename = kSymbolizerBadString;
  outs() << filename <<
         ":" << line_info.getLine() <<
         ":" << line_info.getColumn() <<
         "\n";
  if (SubprocessMode) {
    outs() << "\n";  // Print extra empty line to mark the end of output.
  }
  outs().flush();
}

int main(int argc, char **argv) {
  // Print stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm symbolizer for compiler-rt\n");
  ToolInvocationPath = argv[0];

  string module_name;
  string module_offset_str;
  while (std::cin >> module_name >> module_offset_str) {
    symbolize(module_name, module_offset_str);
  }
  return 0;
}
