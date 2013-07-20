//===- lib/ReaderWriter/MachO/MachOTargetInfo.cpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/MachOTargetInfo.h"
#include "GOTPass.hpp"
#include "StubsPass.hpp"
#include "ReferenceKinds.h"
#include "MachOFormat.hpp"

#include "lld/Core/PassManager.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"
#include "lld/Passes/LayoutPass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"

using lld::mach_o::KindHandler;


namespace lld {


MachOTargetInfo::PackedVersion::PackedVersion(StringRef str) {
  if (parse(str, *this))
    llvm_unreachable("bad version string");
}

/// Construct 32-bit PackedVersion from string "X.Y.Z" where
/// bits are xxxx.yy.zz.  Largest number is 65535.255.255
bool MachOTargetInfo::PackedVersion::parse(StringRef str, 
                                    MachOTargetInfo::PackedVersion &result) {
  result._value = 0;

  if (str.empty()) 
    return false;
  
  SmallVector<StringRef, 3> parts;
  llvm::SplitString(str, parts, ".");
  
  unsigned long long num;
  if (llvm::getAsUnsignedInteger(parts[0], 10, num))
    return true;
  if (num > 65535)
    return true;
  result._value = num << 16;
  
  if (parts.size() > 1) {
    if (llvm::getAsUnsignedInteger(parts[1], 10, num))
      return true;
    if (num > 255)
      return true;
    result._value |= (num << 8);
  }
  
  if (parts.size() > 2) {
    if (llvm::getAsUnsignedInteger(parts[2], 10, num))
      return true;
    if (num > 255)
      return true;
    result._value |= num;
  }
  
  return false;
}

bool MachOTargetInfo::PackedVersion::operator<(
                                              const PackedVersion &rhs) const {
  return _value < rhs._value;
}

bool MachOTargetInfo::PackedVersion::operator>=(
                                              const PackedVersion &rhs) const { 
  return _value >= rhs._value;
}

bool MachOTargetInfo::PackedVersion::operator==(
                                              const PackedVersion &rhs) const {
  return _value == rhs._value;
}

struct ArchInfo {
  StringRef               archName;
  MachOTargetInfo::Arch   arch;
  uint32_t                cputype;
  uint32_t                cpusubtype;
};

static ArchInfo archInfos[] = {
  { "x86_64", MachOTargetInfo::arch_x86_64, mach_o::CPU_TYPE_X86_64, 
                                            mach_o::CPU_SUBTYPE_X86_64_ALL },
  { "i386",   MachOTargetInfo::arch_x86,    mach_o::CPU_TYPE_I386,   
                                            mach_o::CPU_SUBTYPE_X86_ALL },
  { "armv6",  MachOTargetInfo::arch_armv6,  mach_o::CPU_TYPE_ARM,   
                                            mach_o::CPU_SUBTYPE_ARM_V6 },
  { "armv7",  MachOTargetInfo::arch_armv7,  mach_o::CPU_TYPE_ARM,   
                                            mach_o::CPU_SUBTYPE_ARM_V7 },
  { "armv7s", MachOTargetInfo::arch_armv7s, mach_o::CPU_TYPE_ARM,   
                                            mach_o::CPU_SUBTYPE_ARM_V7S },
  { StringRef(),  MachOTargetInfo::arch_unknown, 0, 0 }

};

MachOTargetInfo::Arch 
MachOTargetInfo::archFromCpuType(uint32_t cputype, uint32_t cpusubtype) {
  for (ArchInfo *info = archInfos; !info->archName.empty(); ++info) {
    if ( (info->cputype == cputype) && (info->cpusubtype == cpusubtype)) {
      return info->arch;
    }
  }
  return arch_unknown;
}

MachOTargetInfo::Arch MachOTargetInfo::archFromName(StringRef archName) {
  for (ArchInfo *info = archInfos; !info->archName.empty(); ++info) {
    if (info->archName.equals(archName)) {
      return info->arch;
    }
  }
  return arch_unknown;
}

uint32_t MachOTargetInfo::cpuTypeFromArch(Arch arch) { 
  assert(arch != arch_unknown);
  for (ArchInfo *info = archInfos; !info->archName.empty(); ++info) {
    if (info->arch == arch) {
      return info->cputype;
    }
  }
  llvm_unreachable("Unknown arch type");
}

uint32_t MachOTargetInfo::cpuSubtypeFromArch(Arch arch) {
  assert(arch != arch_unknown);
  for (ArchInfo *info = archInfos; !info->archName.empty(); ++info) {
    if (info->arch == arch) {
      return info->cpusubtype;
    }
  }
  llvm_unreachable("Unknown arch type");
}


MachOTargetInfo::MachOTargetInfo() 
  : _outputFileType(mach_o::MH_EXECUTE)
  , _outputFileTypeStatic(false)
  , _doNothing(false)
  , _arch(arch_unknown)
  , _os(OS::macOSX)
  , _osMinVersion("0.0")
  , _pageZeroSize(0x1000)
  , _kindHandler(nullptr) { 
}

 
MachOTargetInfo::~MachOTargetInfo() {
}

uint32_t MachOTargetInfo::getCPUType() const {
  return cpuTypeFromArch(_arch);
}

uint32_t MachOTargetInfo::getCPUSubType() const {
  return cpuSubtypeFromArch(_arch);
}


bool MachOTargetInfo::outputTypeHasEntry() const {
  switch (_outputFileType) {
  case mach_o::MH_EXECUTE:
  case mach_o::MH_DYLINKER:
  case mach_o::MH_PRELOAD:
    return true;
  default:
    return false;
  }
}


bool MachOTargetInfo::minOS(StringRef mac, StringRef iOS) const  {
  switch (_os) {
  case OS::macOSX:
    return (_osMinVersion >= PackedVersion(mac));
  case OS::iOS:
  case OS::iOS_simulator:
    return (_osMinVersion >= PackedVersion(iOS));
  }
  llvm_unreachable("target not configured for iOS or MacOSX");
}

bool MachOTargetInfo::addEntryPointLoadCommand() const {
  if ((_outputFileType == mach_o::MH_EXECUTE) && !_outputFileTypeStatic) {
    return minOS("10.8", "6.0");
  }
  return false;
}

bool MachOTargetInfo::addUnixThreadLoadCommand() const {
  switch (_outputFileType) {
  case mach_o::MH_EXECUTE:
    if (_outputFileTypeStatic)
      return true;
    else
      return !minOS("10.8", "6.0");
    break;
  case mach_o::MH_DYLINKER:
  case mach_o::MH_PRELOAD:
    return true;
  default:
    return false;
  }
}

bool MachOTargetInfo::validateImpl(raw_ostream &diagnostics) {
  if (_inputFiles.empty()) {
    diagnostics << "no object files specified\n";
    return true;
  }

  if ((_outputFileType == mach_o::MH_EXECUTE) && _entrySymbolName.empty()) {
    if (_outputFileTypeStatic) {
      _entrySymbolName = "start";
    }
    else {
      // If targeting newer OS, use _main
      if (addEntryPointLoadCommand())
        _entrySymbolName = "_main";

      // If targeting older OS, use start (in crt1.o)
      if (addUnixThreadLoadCommand())
        _entrySymbolName = "start";
    }
  }

  return false;
}

bool MachOTargetInfo::setOS(OS os, StringRef minOSVersion) {
  _os = os;
  return PackedVersion::parse(minOSVersion, _osMinVersion);
}

void MachOTargetInfo::addPasses(PassManager &pm) const {
  pm.add(std::unique_ptr<Pass>(new mach_o::GOTPass));
  pm.add(std::unique_ptr<Pass>(new mach_o::StubsPass(*this)));
  pm.add(std::unique_ptr<Pass>(new LayoutPass()));
}

error_code MachOTargetInfo::parseFile(std::unique_ptr<MemoryBuffer> &mb,
                          std::vector<std::unique_ptr<File>> &result) const {
//  if (!_machoReader)
//    _machoReader = createReaderMachO(*this);
//  error_code ec = _machoReader->parseFile(mb,result);
//  if (ec) {
    return _yamlReader->parseFile(mb, result);
//  }

  return error_code::success();
}


Writer &MachOTargetInfo::writer() const {
  if (!_writer) {
    _writer = createWriterMachO(*this);
  }
  return *_writer;
}

KindHandler &MachOTargetInfo::kindHandler() const {
  if (!_kindHandler)
    _kindHandler = KindHandler::create(_arch);
  return *_kindHandler;
}

ErrorOr<Reference::Kind> 
MachOTargetInfo::relocKindFromString(StringRef str) const {
  return kindHandler().stringToKind(str);
 }

ErrorOr<std::string> 
MachOTargetInfo::stringFromRelocKind(Reference::Kind kind) const {
  return std::string(kindHandler().kindToString(kind));
}


} // end namespace lld
