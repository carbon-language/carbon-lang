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

#include "lld/Core/LinkerOptions.h"
#include "lld/Core/PassManager.h"
#include "lld/Passes/LayoutPass.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/MachO.h"

namespace lld {
uint32_t MachOTargetInfo::getCPUType() const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
    return llvm::MachO::CPUTypeI386;
  case llvm::Triple::x86_64:
    return llvm::MachO::CPUTypeX86_64;
  case llvm::Triple::arm:
    return llvm::MachO::CPUTypeARM;
  default:
    llvm_unreachable("Unknown arch type");
  }
}

uint32_t MachOTargetInfo::getCPUSubType() const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
    return llvm::MachO::CPUSubType_I386_ALL;
  case llvm::Triple::x86_64:
    return llvm::MachO::CPUSubType_X86_64_ALL;
  case llvm::Triple::arm:
    return llvm::MachO::CPUSubType_ARM_ALL;
  default:
    llvm_unreachable("Unknown arch type");
  }
}

bool MachOTargetInfo::addEntryPointLoadCommand() const {
  switch (_options._outputKind) {
  case OutputKind::Executable:
    return true;
  default:
    return false;
  }
}

bool MachOTargetInfo::addUnixThreadLoadCommand() const {
  switch (_options._outputKind) {
  case OutputKind::Executable:
    return true;
  default:
    return false;
  }
}

class GenericMachOTargetInfo LLVM_FINAL : public MachOTargetInfo {
public:
  GenericMachOTargetInfo(const LinkerOptions &lo) : MachOTargetInfo(lo) {}

  virtual uint64_t getPageSize() const { return 0x1000; }
  virtual uint64_t getPageZeroSize() const { return getPageSize(); }

  virtual StringRef getEntry() const {
    if (!_options._entrySymbol.empty())
      return _options._entrySymbol;
    return "_main";
  }

  virtual void addPasses(PassManager &pm) const {
    pm.add(std::unique_ptr<Pass>(new mach_o::GOTPass));
    pm.add(std::unique_ptr<Pass>(new mach_o::StubsPass(*this)));
    pm.add(std::unique_ptr<Pass>(new LayoutPass()));
  }
};

std::unique_ptr<MachOTargetInfo>
MachOTargetInfo::create(const LinkerOptions &lo) {
  return std::unique_ptr<MachOTargetInfo>(new GenericMachOTargetInfo(lo));
}
} // end namespace lld
