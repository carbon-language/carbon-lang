//===- lib/ReaderWriter/MachO/WriterOptionsMachO.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/WriterMachO.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/system_error.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "MachOFormat.hpp"

namespace lld {

WriterOptionsMachO::WriterOptionsMachO() 
 : _outputkind(outputDynamicExecutable),
   _architecture(arch_x86),
   _pageZeroSize(0x1000),
   _noTextRelocations(true) {
}

WriterOptionsMachO::~WriterOptionsMachO() {
}

StringRef WriterOptionsMachO::archName() const {
  switch ( _architecture ) {
    case arch_x86_64:
      return StringRef("x86_64");
    case arch_x86:
       return StringRef("i386");
    case arch_armv6:
       return StringRef("armv6");
    case arch_armv7:
       return StringRef("armv7");
  }
  assert(0 && "unknown arch");
  return StringRef("???");
} 

uint32_t WriterOptionsMachO::cpuType() const {
  switch ( _architecture ) {
    case arch_x86_64:
       return mach_o::CPU_TYPE_X86_64;
    case arch_x86:
       return mach_o::CPU_TYPE_I386;
    case arch_armv6:
    case arch_armv7:
       return mach_o::CPU_TYPE_ARM;
  }
  assert(0 && "unknown arch");
  return 0;
}

uint32_t WriterOptionsMachO::cpuSubtype() const {
  switch ( _architecture ) {
    case arch_x86_64:
       return mach_o::CPU_SUBTYPE_X86_64_ALL;
    case arch_x86:
       return mach_o::CPU_SUBTYPE_X86_ALL;
    case arch_armv6:
       return mach_o::CPU_SUBTYPE_ARM_V6;
    case arch_armv7:
       return mach_o::CPU_SUBTYPE_ARM_V7;
  }
  assert(0 && "unknown arch");
  return 0;
}

uint64_t WriterOptionsMachO::pageZeroSize() const { 
  switch ( _outputkind ) {
    case outputDynamicExecutable:
      return _pageZeroSize; 
    case outputDylib:
    case outputBundle:
    case outputObjectFile:
      assert(_pageZeroSize == 0);
      return 0;
  }
  assert(0 && "unknown outputkind");
  return 0;
}

bool WriterOptionsMachO::addEntryPointLoadCommand() const {
  switch ( _outputkind ) {
    case outputDynamicExecutable:
      // Only main executables have an entry point
      return false; 
    case outputDylib:
    case outputBundle:
    case outputObjectFile:
      return false;
  }
  assert(0 && "unknown outputkind");
  return false;
}

bool WriterOptionsMachO::addUnixThreadLoadCommand() const {
  switch ( _outputkind ) {
    case outputDynamicExecutable:
      // Only main executables have an entry point
      return true; 
    case outputDylib:
    case outputBundle:
    case outputObjectFile:
      return false;
  }
  assert(0 && "unknown outputkind");
  return false;
}

StringRef WriterOptionsMachO::entryPointName() const {
  switch ( _outputkind ) {
    case outputDynamicExecutable:
      // Only main executables have an entry point
      if ( ! _customEntryPointName.empty() ) {
        return _customEntryPointName;
      }
      else {
        if ( true || this->addEntryPointLoadCommand() ) 
          return StringRef("_main");
        else
          return StringRef("start"); 
      }
      break;
    case outputDylib:
    case outputBundle:
    case outputObjectFile:
      return StringRef();
  }
  assert(0 && "unknown outputkind");
  return StringRef();
}


} // namespace lld

