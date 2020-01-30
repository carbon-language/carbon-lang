//===--- MachORewriteInstance.cpp - Instance of a rewriting process. ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "MachORewriteInstance.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "Utils.h"
#include "llvm/Support/Timer.h"

namespace opts {

using namespace llvm;
extern cl::opt<bool> PrintSections;

} // namespace opts

namespace llvm {
namespace bolt {

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

MachORewriteInstance::MachORewriteInstance(object::MachOObjectFile *InputFile,
                                           DataReader &DR)
    : InputFile(InputFile),
      BC(BinaryContext::createBinaryContext(
          InputFile, DR,
          DWARFContext::create(*InputFile, nullptr,
                               DWARFContext::defaultErrorHandler, "", false))) {
}

void MachORewriteInstance::readSpecialSections() {
  for (const auto &Section : InputFile->sections()) {
    StringRef SectionName;
    check_error(Section.getName(SectionName), "cannot get section name");
    // Only register sections with names.
    if (!SectionName.empty()) {
      BC->registerSection(Section);
      DEBUG(dbgs() << "BOLT-DEBUG: registering section " << SectionName
                   << " @ 0x" << Twine::utohexstr(Section.getAddress()) << ":0x"
                   << Twine::utohexstr(Section.getAddress() + Section.getSize())
                   << "\n");
    }
  }

  if (opts::PrintSections) {
    outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(outs());
  }
}

void MachORewriteInstance::run() {
  readSpecialSections();
}

MachORewriteInstance::~MachORewriteInstance() {}

} // namespace bolt
} // namespace llvm
