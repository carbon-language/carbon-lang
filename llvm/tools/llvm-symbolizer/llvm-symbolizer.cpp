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
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

using namespace llvm;
using namespace object;

static cl::opt<bool>
UseSymbolTable("use-symbol-table", cl::init(true),
               cl::desc("Prefer names in symbol table to names "
                        "in debug info"));

static cl::opt<bool>
PrintFunctions("functions", cl::init(true),
               cl::desc("Print function names as well as line "
                        "information for a given address"));

static cl::opt<bool>
PrintInlining("inlining", cl::init(true),
              cl::desc("Print all inlined frames for a given address"));

static cl::opt<bool>
Demangle("demangle", cl::init(true),
         cl::desc("Demangle function names"));

static StringRef ToolInvocationPath;

static bool error(error_code ec) {
  if (!ec) return false;
  errs() << ToolInvocationPath << ": error reading file: "
         << ec.message() << ".\n";
  return true;
}

static uint32_t getDILineInfoSpecifierFlags() {
  uint32_t Flags = llvm::DILineInfoSpecifier::FileLineInfo |
                   llvm::DILineInfoSpecifier::AbsoluteFilePath;
  if (PrintFunctions)
    Flags |= llvm::DILineInfoSpecifier::FunctionName;
  return Flags;
}

static void patchFunctionNameInDILineInfo(const std::string &NewFunctionName,
                                          DILineInfo &LineInfo) {
  std::string FileName = LineInfo.getFileName();
  LineInfo = DILineInfo(StringRef(FileName), StringRef(NewFunctionName),
                        LineInfo.getLine(), LineInfo.getColumn());
}

namespace {
class ModuleInfo {
  OwningPtr<ObjectFile> Module;
  OwningPtr<DIContext> DebugInfoContext;
 public:
  ModuleInfo(ObjectFile *Obj, DIContext *DICtx)
      : Module(Obj), DebugInfoContext(DICtx) {}

  DILineInfo symbolizeCode(uint64_t ModuleOffset) const {
    DILineInfo LineInfo;
    if (DebugInfoContext) {
      LineInfo = DebugInfoContext->getLineInfoForAddress(
          ModuleOffset, getDILineInfoSpecifierFlags());
    }
    // Override function name from symbol table if necessary.
    if (PrintFunctions && UseSymbolTable) {
      std::string Function;
      if (getFunctionNameFromSymbolTable(ModuleOffset, Function)) {
        patchFunctionNameInDILineInfo(Function, LineInfo);
      }
    }
    return LineInfo;
  }

  DIInliningInfo symbolizeInlinedCode(uint64_t ModuleOffset) const {
    DIInliningInfo InlinedContext;
    if (DebugInfoContext) {
      InlinedContext = DebugInfoContext->getInliningInfoForAddress(
          ModuleOffset, getDILineInfoSpecifierFlags());
    }
    // Make sure there is at least one frame in context.
    if (InlinedContext.getNumberOfFrames() == 0) {
      InlinedContext.addFrame(DILineInfo());
    }
    // Override the function name in lower frame with name from symbol table.
    if (PrintFunctions && UseSymbolTable) {
      DIInliningInfo PatchedInlinedContext;
      for (uint32_t i = 0, n = InlinedContext.getNumberOfFrames();
           i != n; i++) {
        DILineInfo LineInfo = InlinedContext.getFrame(i);
        if (i == n - 1) {
          std::string Function;
          if (getFunctionNameFromSymbolTable(ModuleOffset, Function)) {
            patchFunctionNameInDILineInfo(Function, LineInfo);
          }
        }
        PatchedInlinedContext.addFrame(LineInfo);
      }
      InlinedContext = PatchedInlinedContext;
    }
    return InlinedContext;
  }

 private:
  bool getFunctionNameFromSymbolTable(uint64_t Address,
                                      std::string &FunctionName) const {
    assert(Module);
    error_code ec;
    for (symbol_iterator si = Module->begin_symbols(),
                         se = Module->end_symbols();
                         si != se; si.increment(ec)) {
      if (error(ec)) return false;
      uint64_t SymbolAddress;
      uint64_t SymbolSize;
      SymbolRef::Type SymbolType;
      if (error(si->getAddress(SymbolAddress)) ||
          SymbolAddress == UnknownAddressOrSize) continue;
      if (error(si->getSize(SymbolSize)) ||
          SymbolSize == UnknownAddressOrSize) continue;
      if (error(si->getType(SymbolType))) continue;
      // FIXME: If a function has alias, there are two entries in symbol table
      // with same address size. Make sure we choose the correct one.
      if (SymbolAddress <= Address && Address < SymbolAddress + SymbolSize &&
          SymbolType == SymbolRef::ST_Function) {
        StringRef Name;
        if (error(si->getName(Name))) continue;
        FunctionName = Name.str();
        return true;
      }
    }
    return false;
  }
};

typedef std::map<std::string, ModuleInfo*> ModuleMapTy;
typedef ModuleMapTy::iterator ModuleMapIter;
}  // namespace

static ModuleMapTy Modules;

static bool isFullNameOfDwarfSection(const StringRef &FullName,
                                     const StringRef &ShortName) {
  static const char kDwarfPrefix[] = "__DWARF,";
  StringRef Name = FullName;
  // Skip "__DWARF," prefix.
  if (Name.startswith(kDwarfPrefix))
    Name = Name.substr(strlen(kDwarfPrefix));
  // Skip . and _ prefixes.
  Name = Name.substr(Name.find_first_not_of("._"));
  return (Name == ShortName);
}

// Returns true if the object endianness is known.
static bool getObjectEndianness(const ObjectFile *Obj,
                                bool &IsLittleEndian) {
  // FIXME: Implement this when libLLVMObject allows to do it easily.
  IsLittleEndian = true;
  return true;
}

static void getDebugInfoSections(const ObjectFile *Obj,
                                 StringRef &DebugInfoSection,
                                 StringRef &DebugAbbrevSection,
                                 StringRef &DebugLineSection,
                                 StringRef &DebugArangesSection, 
                                 StringRef &DebugStringSection, 
                                 StringRef &DebugRangesSection) {
  if (Obj == 0)
    return;
  error_code ec;
  for (section_iterator i = Obj->begin_sections(),
                        e = Obj->end_sections();
                        i != e; i.increment(ec)) {
    if (error(ec)) break;
    StringRef Name;
    if (error(i->getName(Name))) continue;
    StringRef Data;
    if (error(i->getContents(Data))) continue;
    if (isFullNameOfDwarfSection(Name, "debug_info"))
      DebugInfoSection = Data;
    else if (isFullNameOfDwarfSection(Name, "debug_abbrev"))
      DebugAbbrevSection = Data;
    else if (isFullNameOfDwarfSection(Name, "debug_line"))
      DebugLineSection = Data;
    // Don't use debug_aranges for now, as address ranges contained
    // there may not cover all instructions in the module
    // else if (isFullNameOfDwarfSection(Name, "debug_aranges"))
    //   DebugArangesSection = Data;
    else if (isFullNameOfDwarfSection(Name, "debug_str"))
      DebugStringSection = Data;
    else if (isFullNameOfDwarfSection(Name, "debug_ranges"))
      DebugRangesSection = Data;
  }
}

static ObjectFile *getObjectFile(const std::string &Path) {
  OwningPtr<MemoryBuffer> Buff;
  MemoryBuffer::getFile(Path, Buff);
  return ObjectFile::createObjectFile(Buff.take());
}

static std::string getDarwinDWARFResourceForModule(const std::string &Path) {
  StringRef Basename = sys::path::filename(Path);
  const std::string &DSymDirectory = Path + ".dSYM";
  SmallString<16> ResourceName = StringRef(DSymDirectory);
  sys::path::append(ResourceName, "Contents", "Resources", "DWARF");
  sys::path::append(ResourceName, Basename);
  return ResourceName.str();
}

static ModuleInfo *getOrCreateModuleInfo(const std::string &ModuleName) {
  ModuleMapIter I = Modules.find(ModuleName);
  if (I != Modules.end())
    return I->second;

  ObjectFile *Obj = getObjectFile(ModuleName);
  if (Obj == 0) {
    // Module name doesn't point to a valid object file.
    Modules.insert(make_pair(ModuleName, (ModuleInfo*)0));
    return 0;
  }

  DIContext *Context = 0;
  bool IsLittleEndian;
  if (getObjectEndianness(Obj, IsLittleEndian)) {
    StringRef DebugInfo;
    StringRef DebugAbbrev;
    StringRef DebugLine;
    StringRef DebugAranges;
    StringRef DebugString;
    StringRef DebugRanges;
    getDebugInfoSections(Obj, DebugInfo, DebugAbbrev, DebugLine,
                         DebugAranges, DebugString, DebugRanges);
    
    // On Darwin we may find DWARF in separate object file in
    // resource directory.
    if (isa<MachOObjectFile>(Obj)) {
      const std::string &ResourceName = getDarwinDWARFResourceForModule(
          ModuleName);
      ObjectFile *ResourceObj = getObjectFile(ResourceName);
      if (ResourceObj != 0)
        getDebugInfoSections(ResourceObj, DebugInfo, DebugAbbrev, DebugLine,
                             DebugAranges, DebugString, DebugRanges);
    }

    Context = DIContext::getDWARFContext(
        IsLittleEndian, DebugInfo, DebugAbbrev,
        DebugAranges, DebugLine, DebugString,
        DebugRanges);
    assert(Context);
  }

  ModuleInfo *Info = new ModuleInfo(Obj, Context);
  Modules.insert(make_pair(ModuleName, Info));
  return Info;
}

#if !defined(_MSC_VER)
// Assume that __cxa_demangle is provided by libcxxabi (except for Windows).
extern "C" char *__cxa_demangle(const char *mangled_name, char *output_buffer,
                                size_t *length, int *status);
#endif

static void printDILineInfo(DILineInfo LineInfo) {
  // By default, DILineInfo contains "<invalid>" for function/filename it
  // cannot fetch. We replace it to "??" to make our output closer to addr2line.
  static const std::string kDILineInfoBadString = "<invalid>";
  static const std::string kSymbolizerBadString = "??";
  if (PrintFunctions) {
    std::string FunctionName = LineInfo.getFunctionName();
    if (FunctionName == kDILineInfoBadString)
      FunctionName = kSymbolizerBadString;
#if !defined(_MSC_VER)
    if (Demangle) {
      int status = 0;
      char *DemangledName = __cxa_demangle(
          FunctionName.c_str(), 0, 0, &status);
      if (status == 0) {
        FunctionName = DemangledName;
        free(DemangledName);
      }
    }
#endif
    outs() << FunctionName << "\n";
  }
  std::string Filename = LineInfo.getFileName();
  if (Filename == kDILineInfoBadString)
    Filename = kSymbolizerBadString;
  outs() << Filename <<
         ":" << LineInfo.getLine() <<
         ":" << LineInfo.getColumn() <<
         "\n";
}

static void symbolize(std::string ModuleName, std::string ModuleOffsetStr) {
  ModuleInfo *Info = getOrCreateModuleInfo(ModuleName);
  uint64_t Offset = 0;
  if (Info == 0 ||
      StringRef(ModuleOffsetStr).getAsInteger(0, Offset)) {
    printDILineInfo(DILineInfo());
  } else if (PrintInlining) {
    DIInliningInfo InlinedContext = Info->symbolizeInlinedCode(Offset);
    uint32_t FramesNum = InlinedContext.getNumberOfFrames();
    assert(FramesNum > 0);
    for (uint32_t i = 0; i < FramesNum; i++) {
      DILineInfo LineInfo = InlinedContext.getFrame(i);
      printDILineInfo(LineInfo);
    }
  } else {
    DILineInfo LineInfo = Info->symbolizeCode(Offset);
    printDILineInfo(LineInfo);
  }

  outs() << "\n";  // Print extra empty line to mark the end of output.
  outs().flush();
}

static bool parseModuleNameAndOffset(std::string &ModuleName,
                                     std::string &ModuleOffsetStr) {
  static const int kMaxInputStringLength = 1024;
  static const char kDelimiters[] = " \n";
  char InputString[kMaxInputStringLength];
  if (!fgets(InputString, sizeof(InputString), stdin))
    return false;
  ModuleName = "";
  ModuleOffsetStr = "";
  // FIXME: Handle case when filename is given in quotes.
  if (char *FilePath = strtok(InputString, kDelimiters)) {
    ModuleName = FilePath;
    if (char *OffsetStr = strtok((char*)0, kDelimiters))
      ModuleOffsetStr = OffsetStr;
  }
  return true;
}

int main(int argc, char **argv) {
  // Print stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm symbolizer for compiler-rt\n");
  ToolInvocationPath = argv[0];

  std::string ModuleName;
  std::string ModuleOffsetStr;
  while (parseModuleNameAndOffset(ModuleName, ModuleOffsetStr)) {
    symbolize(ModuleName, ModuleOffsetStr);
  }
  return 0;
}
