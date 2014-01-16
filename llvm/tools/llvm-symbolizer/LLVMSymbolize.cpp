//===-- LLVMSymbolize.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation for LLVM symbolization library.
//
//===----------------------------------------------------------------------===//

#include "LLVMSymbolize.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <sstream>
#include <stdlib.h>

namespace llvm {
namespace symbolize {

static bool error(error_code ec) {
  if (!ec)
    return false;
  errs() << "LLVMSymbolizer: error reading file: " << ec.message() << ".\n";
  return true;
}

static uint32_t
getDILineInfoSpecifierFlags(const LLVMSymbolizer::Options &Opts) {
  uint32_t Flags = llvm::DILineInfoSpecifier::FileLineInfo |
                   llvm::DILineInfoSpecifier::AbsoluteFilePath;
  if (Opts.PrintFunctions)
    Flags |= llvm::DILineInfoSpecifier::FunctionName;
  return Flags;
}

static void patchFunctionNameInDILineInfo(const std::string &NewFunctionName,
                                          DILineInfo &LineInfo) {
  std::string FileName = LineInfo.getFileName();
  LineInfo = DILineInfo(StringRef(FileName), StringRef(NewFunctionName),
                        LineInfo.getLine(), LineInfo.getColumn());
}

ModuleInfo::ModuleInfo(ObjectFile *Obj, DIContext *DICtx)
    : Module(Obj), DebugInfoContext(DICtx) {
  error_code ec;
  for (symbol_iterator si = Module->begin_symbols(), se = Module->end_symbols();
       si != se; si.increment(ec)) {
    if (error(ec))
      return;
    SymbolRef::Type SymbolType;
    if (error(si->getType(SymbolType)))
      continue;
    if (SymbolType != SymbolRef::ST_Function &&
        SymbolType != SymbolRef::ST_Data)
      continue;
    uint64_t SymbolAddress;
    if (error(si->getAddress(SymbolAddress)) ||
        SymbolAddress == UnknownAddressOrSize)
      continue;
    uint64_t SymbolSize;
    // Getting symbol size is linear for Mach-O files, so assume that symbol
    // occupies the memory range up to the following symbol.
    if (isa<MachOObjectFile>(Obj))
      SymbolSize = 0;
    else if (error(si->getSize(SymbolSize)) ||
             SymbolSize == UnknownAddressOrSize)
      continue;
    StringRef SymbolName;
    if (error(si->getName(SymbolName)))
      continue;
    // Mach-O symbol table names have leading underscore, skip it.
    if (Module->isMachO() && SymbolName.size() > 0 && SymbolName[0] == '_')
      SymbolName = SymbolName.drop_front();
    // FIXME: If a function has alias, there are two entries in symbol table
    // with same address size. Make sure we choose the correct one.
    SymbolMapTy &M = SymbolType == SymbolRef::ST_Function ? Functions : Objects;
    SymbolDesc SD = { SymbolAddress, SymbolSize };
    M.insert(std::make_pair(SD, SymbolName));
  }
}

bool ModuleInfo::getNameFromSymbolTable(SymbolRef::Type Type, uint64_t Address,
                                        std::string &Name, uint64_t &Addr,
                                        uint64_t &Size) const {
  const SymbolMapTy &M = Type == SymbolRef::ST_Function ? Functions : Objects;
  if (M.empty())
    return false;
  SymbolDesc SD = { Address, Address };
  SymbolMapTy::const_iterator it = M.upper_bound(SD);
  if (it == M.begin())
    return false;
  --it;
  if (it->first.Size != 0 && it->first.Addr + it->first.Size <= Address)
    return false;
  Name = it->second.str();
  Addr = it->first.Addr;
  Size = it->first.Size;
  return true;
}

DILineInfo ModuleInfo::symbolizeCode(
    uint64_t ModuleOffset, const LLVMSymbolizer::Options &Opts) const {
  DILineInfo LineInfo;
  if (DebugInfoContext) {
    LineInfo = DebugInfoContext->getLineInfoForAddress(
        ModuleOffset, getDILineInfoSpecifierFlags(Opts));
  }
  // Override function name from symbol table if necessary.
  if (Opts.PrintFunctions && Opts.UseSymbolTable) {
    std::string FunctionName;
    uint64_t Start, Size;
    if (getNameFromSymbolTable(SymbolRef::ST_Function, ModuleOffset,
                               FunctionName, Start, Size)) {
      patchFunctionNameInDILineInfo(FunctionName, LineInfo);
    }
  }
  return LineInfo;
}

DIInliningInfo ModuleInfo::symbolizeInlinedCode(
    uint64_t ModuleOffset, const LLVMSymbolizer::Options &Opts) const {
  DIInliningInfo InlinedContext;
  if (DebugInfoContext) {
    InlinedContext = DebugInfoContext->getInliningInfoForAddress(
        ModuleOffset, getDILineInfoSpecifierFlags(Opts));
  }
  // Make sure there is at least one frame in context.
  if (InlinedContext.getNumberOfFrames() == 0) {
    InlinedContext.addFrame(DILineInfo());
  }
  // Override the function name in lower frame with name from symbol table.
  if (Opts.PrintFunctions && Opts.UseSymbolTable) {
    DIInliningInfo PatchedInlinedContext;
    for (uint32_t i = 0, n = InlinedContext.getNumberOfFrames(); i < n; i++) {
      DILineInfo LineInfo = InlinedContext.getFrame(i);
      if (i == n - 1) {
        std::string FunctionName;
        uint64_t Start, Size;
        if (getNameFromSymbolTable(SymbolRef::ST_Function, ModuleOffset,
                                   FunctionName, Start, Size)) {
          patchFunctionNameInDILineInfo(FunctionName, LineInfo);
        }
      }
      PatchedInlinedContext.addFrame(LineInfo);
    }
    InlinedContext = PatchedInlinedContext;
  }
  return InlinedContext;
}

bool ModuleInfo::symbolizeData(uint64_t ModuleOffset, std::string &Name,
                               uint64_t &Start, uint64_t &Size) const {
  return getNameFromSymbolTable(SymbolRef::ST_Data, ModuleOffset, Name, Start,
                                Size);
}

const char LLVMSymbolizer::kBadString[] = "??";

std::string LLVMSymbolizer::symbolizeCode(const std::string &ModuleName,
                                          uint64_t ModuleOffset) {
  ModuleInfo *Info = getOrCreateModuleInfo(ModuleName);
  if (Info == 0)
    return printDILineInfo(DILineInfo());
  if (Opts.PrintInlining) {
    DIInliningInfo InlinedContext =
        Info->symbolizeInlinedCode(ModuleOffset, Opts);
    uint32_t FramesNum = InlinedContext.getNumberOfFrames();
    assert(FramesNum > 0);
    std::string Result;
    for (uint32_t i = 0; i < FramesNum; i++) {
      DILineInfo LineInfo = InlinedContext.getFrame(i);
      Result += printDILineInfo(LineInfo);
    }
    return Result;
  }
  DILineInfo LineInfo = Info->symbolizeCode(ModuleOffset, Opts);
  return printDILineInfo(LineInfo);
}

std::string LLVMSymbolizer::symbolizeData(const std::string &ModuleName,
                                          uint64_t ModuleOffset) {
  std::string Name = kBadString;
  uint64_t Start = 0;
  uint64_t Size = 0;
  if (Opts.UseSymbolTable) {
    if (ModuleInfo *Info = getOrCreateModuleInfo(ModuleName)) {
      if (Info->symbolizeData(ModuleOffset, Name, Start, Size) && Opts.Demangle)
        Name = DemangleName(Name);
    }
  }
  std::stringstream ss;
  ss << Name << "\n" << Start << " " << Size << "\n";
  return ss.str();
}

void LLVMSymbolizer::flush() {
  DeleteContainerSeconds(Modules);
  DeleteContainerPointers(ParsedBinariesAndObjects);
  BinaryForPath.clear();
  ObjectFileForArch.clear();
}

static std::string getDarwinDWARFResourceForPath(const std::string &Path) {
  StringRef Basename = sys::path::filename(Path);
  const std::string &DSymDirectory = Path + ".dSYM";
  SmallString<16> ResourceName = StringRef(DSymDirectory);
  sys::path::append(ResourceName, "Contents", "Resources", "DWARF");
  sys::path::append(ResourceName, Basename);
  return ResourceName.str();
}

static bool checkFileCRC(StringRef Path, uint32_t CRCHash) {
  OwningPtr<MemoryBuffer> MB;
  if (MemoryBuffer::getFileOrSTDIN(Path, MB))
    return false;
  return !zlib::isAvailable() || CRCHash == zlib::crc32(MB->getBuffer());
}

static bool findDebugBinary(const std::string &OrigPath,
                            const std::string &DebuglinkName, uint32_t CRCHash,
                            std::string &Result) {
  std::string OrigRealPath = OrigPath;
#if defined(HAVE_REALPATH)
  if (char *RP = realpath(OrigPath.c_str(), NULL)) {
    OrigRealPath = RP;
    free(RP);
  }
#endif
  SmallString<16> OrigDir(OrigRealPath);
  llvm::sys::path::remove_filename(OrigDir);
  SmallString<16> DebugPath = OrigDir;
  // Try /path/to/original_binary/debuglink_name
  llvm::sys::path::append(DebugPath, DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = DebugPath.str();
    return true;
  }
  // Try /path/to/original_binary/.debug/debuglink_name
  DebugPath = OrigRealPath;
  llvm::sys::path::append(DebugPath, ".debug", DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = DebugPath.str();
    return true;
  }
  // Try /usr/lib/debug/path/to/original_binary/debuglink_name
  DebugPath = "/usr/lib/debug";
  llvm::sys::path::append(DebugPath, llvm::sys::path::relative_path(OrigDir),
                          DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = DebugPath.str();
    return true;
  }
  return false;
}

static bool getGNUDebuglinkContents(const Binary *Bin, std::string &DebugName,
                                    uint32_t &CRCHash) {
  const ObjectFile *Obj = dyn_cast<ObjectFile>(Bin);
  if (!Obj)
    return false;
  error_code EC;
  for (section_iterator I = Obj->begin_sections(), E = Obj->end_sections();
       I != E; I.increment(EC)) {
    StringRef Name;
    I->getName(Name);
    Name = Name.substr(Name.find_first_not_of("._"));
    if (Name == "gnu_debuglink") {
      StringRef Data;
      I->getContents(Data);
      DataExtractor DE(Data, Obj->isLittleEndian(), 0);
      uint32_t Offset = 0;
      if (const char *DebugNameStr = DE.getCStr(&Offset)) {
        // 4-byte align the offset.
        Offset = (Offset + 3) & ~0x3;
        if (DE.isValidOffsetForDataOfSize(Offset, 4)) {
          DebugName = DebugNameStr;
          CRCHash = DE.getU32(&Offset);
          return true;
        }
      }
      break;
    }
  }
  return false;
}

LLVMSymbolizer::BinaryPair
LLVMSymbolizer::getOrCreateBinary(const std::string &Path) {
  BinaryMapTy::iterator I = BinaryForPath.find(Path);
  if (I != BinaryForPath.end())
    return I->second;
  Binary *Bin = 0;
  Binary *DbgBin = 0;
  ErrorOr<Binary *> BinaryOrErr = createBinary(Path);
  if (!error(BinaryOrErr.getError())) {
    OwningPtr<Binary> ParsedBinary(BinaryOrErr.get());
    // Check if it's a universal binary.
    Bin = ParsedBinary.take();
    ParsedBinariesAndObjects.push_back(Bin);
    if (Bin->isMachO() || Bin->isMachOUniversalBinary()) {
      // On Darwin we may find DWARF in separate object file in
      // resource directory.
      const std::string &ResourcePath =
          getDarwinDWARFResourceForPath(Path);
      BinaryOrErr = createBinary(ResourcePath);
      error_code EC = BinaryOrErr.getError();
      if (EC != errc::no_such_file_or_directory && !error(EC)) {
        DbgBin = BinaryOrErr.get();
        ParsedBinariesAndObjects.push_back(DbgBin);
      }
    }
    // Try to locate the debug binary using .gnu_debuglink section.
    if (DbgBin == 0) {
      std::string DebuglinkName;
      uint32_t CRCHash;
      std::string DebugBinaryPath;
      if (getGNUDebuglinkContents(Bin, DebuglinkName, CRCHash) &&
          findDebugBinary(Path, DebuglinkName, CRCHash, DebugBinaryPath)) {
        BinaryOrErr = createBinary(DebugBinaryPath);
        if (!error(BinaryOrErr.getError())) {
          DbgBin = BinaryOrErr.get();
          ParsedBinariesAndObjects.push_back(DbgBin);
        }
      }
    }
  }
  if (DbgBin == 0)
    DbgBin = Bin;
  BinaryPair Res = std::make_pair(Bin, DbgBin);
  BinaryForPath[Path] = Res;
  return Res;
}

ObjectFile *
LLVMSymbolizer::getObjectFileFromBinary(Binary *Bin, const std::string &ArchName) {
  if (Bin == 0)
    return 0;
  ObjectFile *Res = 0;
  if (MachOUniversalBinary *UB = dyn_cast<MachOUniversalBinary>(Bin)) {
    ObjectFileForArchMapTy::iterator I = ObjectFileForArch.find(
        std::make_pair(UB, ArchName));
    if (I != ObjectFileForArch.end())
      return I->second;
    OwningPtr<ObjectFile> ParsedObj;
    if (!UB->getObjectForArch(Triple(ArchName).getArch(), ParsedObj)) {
      Res = ParsedObj.take();
      ParsedBinariesAndObjects.push_back(Res);
    }
    ObjectFileForArch[std::make_pair(UB, ArchName)] = Res;
  } else if (Bin->isObject()) {
    Res = cast<ObjectFile>(Bin);
  }
  return Res;
}

ModuleInfo *
LLVMSymbolizer::getOrCreateModuleInfo(const std::string &ModuleName) {
  ModuleMapTy::iterator I = Modules.find(ModuleName);
  if (I != Modules.end())
    return I->second;
  std::string BinaryName = ModuleName;
  std::string ArchName = Opts.DefaultArch;
  size_t ColonPos = ModuleName.find_last_of(':');
  // Verify that substring after colon form a valid arch name.
  if (ColonPos != std::string::npos) {
    std::string ArchStr = ModuleName.substr(ColonPos + 1);
    if (Triple(ArchStr).getArch() != Triple::UnknownArch) {
      BinaryName = ModuleName.substr(0, ColonPos);
      ArchName = ArchStr;
    }
  }
  BinaryPair Binaries = getOrCreateBinary(BinaryName);
  ObjectFile *Obj = getObjectFileFromBinary(Binaries.first, ArchName);
  ObjectFile *DbgObj = getObjectFileFromBinary(Binaries.second, ArchName);

  if (Obj == 0) {
    // Failed to find valid object file.
    Modules.insert(make_pair(ModuleName, (ModuleInfo *)0));
    return 0;
  }
  DIContext *Context = DIContext::getDWARFContext(DbgObj);
  assert(Context);
  ModuleInfo *Info = new ModuleInfo(Obj, Context);
  Modules.insert(make_pair(ModuleName, Info));
  return Info;
}

std::string LLVMSymbolizer::printDILineInfo(DILineInfo LineInfo) const {
  // By default, DILineInfo contains "<invalid>" for function/filename it
  // cannot fetch. We replace it to "??" to make our output closer to addr2line.
  static const std::string kDILineInfoBadString = "<invalid>";
  std::stringstream Result;
  if (Opts.PrintFunctions) {
    std::string FunctionName = LineInfo.getFunctionName();
    if (FunctionName == kDILineInfoBadString)
      FunctionName = kBadString;
    else if (Opts.Demangle)
      FunctionName = DemangleName(FunctionName);
    Result << FunctionName << "\n";
  }
  std::string Filename = LineInfo.getFileName();
  if (Filename == kDILineInfoBadString)
    Filename = kBadString;
  Result << Filename << ":" << LineInfo.getLine() << ":" << LineInfo.getColumn()
         << "\n";
  return Result.str();
}

#if !defined(_MSC_VER)
// Assume that __cxa_demangle is provided by libcxxabi (except for Windows).
extern "C" char *__cxa_demangle(const char *mangled_name, char *output_buffer,
                                size_t *length, int *status);
#endif

std::string LLVMSymbolizer::DemangleName(const std::string &Name) {
#if !defined(_MSC_VER)
  // We can spoil names of symbols with C linkage, so use an heuristic
  // approach to check if the name should be demangled.
  if (Name.substr(0, 2) != "_Z")
    return Name;
  int status = 0;
  char *DemangledName = __cxa_demangle(Name.c_str(), 0, 0, &status);
  if (status != 0)
    return Name;
  std::string Result = DemangledName;
  free(DemangledName);
  return Result;
#else
  return Name;
#endif
}

} // namespace symbolize
} // namespace llvm
