//===--- Targets.cpp - Implement -arch option and targets -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the -arch command line option and creates a TargetInfo
// that represents them.
//
//===----------------------------------------------------------------------===//

#include "clang.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;
using namespace clang;

/// Note: a hard coded list of targets is clearly silly, these should be
/// dynamicly registered and loadable with "-load".
enum SupportedTargets {
  target_ppc, target_ppc64,
  target_i386, target_x86_64,
  target_linux_i386
};

static cl::list<SupportedTargets>
Archs("arch", cl::desc("Architectures to compile for"),
      cl::values(clEnumValN(target_ppc,       "ppc",   "32-bit Darwin PowerPC"),
                 clEnumValN(target_ppc64,     "ppc64", "64-bit Darwin PowerPC"),
                 clEnumValN(target_i386,      "i386",  "32-bit Darwin X86"),
                 clEnumValN(target_x86_64,    "x86_64","64-bit Darwin X86"),
                 clEnumValN(target_linux_i386,"linux", "Linux i386"),
                 clEnumValEnd));

//===----------------------------------------------------------------------===//
//  Common code shared among targets.
//===----------------------------------------------------------------------===//

namespace {
class DarwinTargetInfo : public TargetInfoImpl {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    Defines.push_back("__APPLE__");
  }

};
} // end anonymous namespace.


/// getPowerPCDefines - Return a set of the PowerPC-specific #defines that are
/// not tied to a specific subtarget.
static void getPowerPCDefines(std::vector<std::string> &Defines, bool is64Bit) {
  Defines.push_back("__ppc__");

}

/// getX86Defines - Return a set of the X86-specific #defines that are
/// not tied to a specific subtarget.
static void getX86Defines(std::vector<std::string> &Defines, bool is64Bit) {
  Defines.push_back("__i386__");

}

//===----------------------------------------------------------------------===//
// Specific target implementations.
//===----------------------------------------------------------------------===//


namespace {
class DarwinPPCTargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getPowerPCDefines(Defines, false);
  }
};
} // end anonymous namespace.

namespace {
class DarwinPPC64TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getPowerPCDefines(Defines, true);
  }
};
} // end anonymous namespace.

namespace {
class DarwinI386TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getX86Defines(Defines, false);
  }
};
} // end anonymous namespace.

namespace {
class DarwinX86_64TargetInfo : public DarwinTargetInfo {
public:
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    DarwinTargetInfo::getTargetDefines(Defines);
    getX86Defines(Defines, true);
  }
};
} // end anonymous namespace.

namespace {
class LinuxTargetInfo : public DarwinTargetInfo {
public:
  LinuxTargetInfo() {
    // Note: I have no idea if this is right, just for testing.
    WCharWidth = 2;
  }
  
  virtual void getTargetDefines(std::vector<std::string> &Defines) const {
    // TODO: linux-specific stuff.
    getX86Defines(Defines, false);
  }
};
} // end anonymous namespace.


//===----------------------------------------------------------------------===//
// Driver code
//===----------------------------------------------------------------------===//

/// CreateTarget - Create the TargetInfoImpl object for the specified target
/// enum value.
static TargetInfoImpl *CreateTarget(SupportedTargets T) {
  switch (T) {
  default: assert(0 && "Unknown target!");
  case target_ppc:        return new DarwinPPCTargetInfo();
  case target_ppc64:      return new DarwinPPC64TargetInfo();
  case target_i386:       return new DarwinI386TargetInfo();
  case target_x86_64:     return new DarwinX86_64TargetInfo();
  case target_linux_i386: return new LinuxTargetInfo();
  }
}

/// CreateTargetInfo - Return the set of target info objects as specified by
/// the -arch command line option.
TargetInfo *clang::CreateTargetInfo(Diagnostic &Diags) {
  // If the user didn't specify at least one architecture, auto-sense the
  // current host.  TODO: This is a hack. :)
  if (Archs.empty()) {
#ifndef __APPLE__
    // Assume non-apple = linux.
    Archs.push_back(target_linux_i386);
#elif (defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)) && \
      defined(__ppc64__)
    Archs.push_back(target_ppc64);
#elif defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
    Archs.push_back(target_ppc);
#elif defined(__x86_64__)
    Archs.push_back(target_x86_64);
#elif defined(__i386__) || defined(i386) || defined(_M_IX86)
    Archs.push_back(target_i386);
#else
    // Don't know what this is!
    return 0;
#endif
  }

  // Create the primary target and target info.
  TargetInfo *TI = new TargetInfo(CreateTarget(Archs[0]), &Diags);
  
  // Add all secondary targets.
  for (unsigned i = 1, e = Archs.size(); i != e; ++i)
    TI->AddSecondaryTarget(CreateTarget(Archs[i]));
  return TI;
}
