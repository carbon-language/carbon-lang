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


MachOTargetInfo::MachOTargetInfo() 
  : _outputFileType(mach_o::MH_EXECUTE)
  , _outputFileTypeStatic(false)
  , _arch(arch_unknown)
  , _os(OS::macOSX)
  , _osMinVersion("0.0")
  , _pageZeroSize(0x1000)
  , _kindHandler(nullptr) { 
}

 
MachOTargetInfo::~MachOTargetInfo() {
}

uint32_t MachOTargetInfo::getCPUType() const {
  switch (_arch) {
  case MachOTargetInfo::arch_x86:
    return mach_o::CPU_TYPE_I386;
  case MachOTargetInfo::arch_x86_64:
    return mach_o::CPU_TYPE_X86_64;
  case MachOTargetInfo::arch_armv6:
  case MachOTargetInfo::arch_armv7:
  case MachOTargetInfo::arch_armv7s:
    return mach_o::CPU_TYPE_ARM;
  case MachOTargetInfo::arch_unknown:
    llvm_unreachable("Unknown arch type");
  }
}

uint32_t MachOTargetInfo::getCPUSubType() const {
  switch (_arch) {
  case MachOTargetInfo::arch_x86:
    return mach_o::CPU_SUBTYPE_X86_ALL;
  case MachOTargetInfo::arch_x86_64:
    return mach_o::CPU_SUBTYPE_X86_64_ALL;
  case MachOTargetInfo::arch_armv6:
    return mach_o::CPU_SUBTYPE_ARM_V6;
  case MachOTargetInfo::arch_armv7:
    return mach_o::CPU_SUBTYPE_ARM_V7;
  case MachOTargetInfo::arch_armv7s:
    return mach_o::CPU_SUBTYPE_ARM_V7S;
  case MachOTargetInfo::arch_unknown:
    llvm_unreachable("Unknown arch type");
  }
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

bool MachOTargetInfo::validate(raw_ostream &diagnostics) {
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

error_code MachOTargetInfo::parseFile(
    std::unique_ptr<MemoryBuffer> mb,
    std::vector<std::unique_ptr<File>> &result) const {
//  if (!_machoReader)
//    _machoReader = createReaderMachO(*this);
//  error_code ec = _machoReader->parseFile(mb,result);
//  if (ec) {
    if (!_yamlReader)
      _yamlReader = createReaderYAML(*this);
    return _yamlReader->parseFile(std::move(mb), result);
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
