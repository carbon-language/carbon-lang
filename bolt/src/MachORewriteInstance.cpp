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
extern cl::opt<bool> PrintDisasm;
extern cl::opt<bool> PrintCFG;
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

void MachORewriteInstance::discoverFileObjects() {
  std::vector<SymbolRef> FunctionSymbols;
  for (const SymbolRef &S : InputFile->symbols()) {
    SymbolRef::Type Type = cantFail(S.getType(), "cannot get symbol type");
    if (Type == SymbolRef::ST_Function)
      FunctionSymbols.push_back(S);
  }
  if (FunctionSymbols.empty())
    return;
  std::stable_sort(FunctionSymbols.begin(), FunctionSymbols.end(),
                   [](const SymbolRef &LHS, const SymbolRef &RHS) {
                     return LHS.getValue() < RHS.getValue();
                   });
  for (size_t Index = 0; Index < FunctionSymbols.size(); ++Index) {
    const uint64_t Address = FunctionSymbols[Index].getValue();
    auto Section = BC->getSectionForAddress(Address);
    // TODO: It happens for some symbols (e.g. __mh_execute_header).
    // Add proper logic to handle them correctly.
    if (!Section) {
      errs() << "BOLT-WARNING: no section found for address " << Address
             << "\n";
      continue;
    }
    StringRef SymbolName =
        cantFail(FunctionSymbols[Index].getName(), "cannot get symbol name");
    section_iterator S = cantFail(FunctionSymbols[Index].getSection());
    uint64_t EndAddress = S->getAddress() + S->getSize();
    if (Index + 1 < FunctionSymbols.size() &&
        S == cantFail(FunctionSymbols[Index + 1].getSection()))
      EndAddress = FunctionSymbols[Index + 1].getValue();
    const uint64_t SymbolSize = EndAddress - Address;
    // TODO: Add proper logic to handle nonunique names.
    assert(BC->getBinaryFunctions().find(Address) ==
           BC->getBinaryFunctions().end());
    BC->createBinaryFunction(SymbolName.str(), *Section, Address, SymbolSize,
                             /* IsSimple */ true);
  }
}

void MachORewriteInstance::disassembleFunctions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    auto FunctionData = Function.getData();
    if (!FunctionData) {
      errs() << "BOLT-ERROR: corresponding section is non-executable or "
             << "empty for function " << Function << '\n';
      continue;
    }

    // Treat zero-sized functions as non-simple ones.
    if (Function.getSize() == 0) {
      Function.setSimple(false);
      continue;
    }

    // Offset of the function in the file.
    const auto *FileBegin =
        reinterpret_cast<const uint8_t *>(InputFile->getData().data());
    Function.setFileOffset(FunctionData->begin() - FileBegin);

    Function.disassemble();
    if (opts::PrintDisasm)
      Function.print(outs(), "after disassembly", true);
    if (!Function.buildCFG(/*AllocId*/0)) {
      errs() << "BOLT-WARNING: failed to build CFG for the function "
             << Function << "\n";
    }
  }
}

void MachORewriteInstance::postProcessFunctions() {
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (Function.empty())
      continue;
    Function.postProcessCFG();
    if (opts::PrintCFG)
      Function.print(outs(), "after building cfg", true);
  }
}

void MachORewriteInstance::run() {
  readSpecialSections();
  discoverFileObjects();
  disassembleFunctions();
  postProcessFunctions();
}

MachORewriteInstance::~MachORewriteInstance() {}

} // namespace bolt
} // namespace llvm
