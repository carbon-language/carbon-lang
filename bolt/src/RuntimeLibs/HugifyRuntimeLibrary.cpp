//===------HugifyRuntimeLibrary.cpp - The Hugify Runtime Library ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HugifyRuntimeLibrary.h"
#include "BinaryFunction.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<bool> HotText;

cl::opt<bool>
    Hugify("hugify",
           cl::desc("Automatically put hot code on 2MB page(s) (hugify) at "
                    "runtime. No manual call to hugify is needed in the binary "
                    "(which is what --hot-text relies on)."),
           cl::ZeroOrMore, cl::cat(BoltOptCategory));

static cl::opt<std::string> RuntimeHugifyLib(
    "runtime-hugify-lib",
    cl::desc("specify file name of the runtime hugify library"), cl::ZeroOrMore,
    cl::init("libbolt_rt_hugify.a"), cl::cat(BoltOptCategory));

} // namespace opts

void HugifyRuntimeLibrary::adjustCommandLineOptions(
    const BinaryContext &BC) const {
  if (opts::HotText) {
    errs()
        << "BOLT-ERROR: -hot-text should be applied to binaries with "
           "pre-compiled manual hugify support, while -hugify will add hugify "
           "support automatcally. These two options cannot both be present.\n";
    exit(1);
  }
  // After the check, we set HotText to be true because automated hugify support
  // relies on it.
  opts::HotText = true;
  if (!BC.StartFunctionAddress) {
    errs() << "BOLT-ERROR: hugify runtime libraries require a known entry "
              "point of "
              "the input binary\n";
    exit(1);
  }
}

void HugifyRuntimeLibrary::emitBinary(BinaryContext &BC, MCStreamer &Streamer) {
  const auto *StartFunction =
      BC.getBinaryFunctionAtAddress(*(BC.StartFunctionAddress));
  if (!StartFunction) {
    errs() << "BOLT-ERROR: failed to locate function at binary start address\n";
    exit(1);
  }

  const auto Flags = BinarySection::getFlags(/*IsReadOnly=*/false,
                                             /*IsText=*/false,
                                             /*IsAllocatable=*/true);
  auto *Section =
      BC.Ctx->getELFSection(".bolt.hugify.entries", ELF::SHT_PROGBITS, Flags);

  // __bolt_hugify_init_ptr stores the poiter the hugify library needs to
  // jump to after finishing the init code.
  MCSymbol *InitPtr = BC.Ctx->getOrCreateSymbol("__bolt_hugify_init_ptr");

  Section->setAlignment(BC.RegularPageSize);
  Streamer.SwitchSection(Section);

  Streamer.EmitLabel(InitPtr);
  Streamer.EmitSymbolAttribute(InitPtr, MCSymbolAttr::MCSA_Global);
  Streamer.EmitValue(
      MCSymbolRefExpr::create(StartFunction->getSymbol(), *(BC.Ctx)),
      /*Size=*/8);
}

void HugifyRuntimeLibrary::link(BinaryContext &BC, StringRef ToolPath,
                                orc::ExecutionSession &ES,
                                orc::RTDyldObjectLinkingLayer &OLT) {
  auto LibPath = getLibPath(ToolPath, opts::RuntimeHugifyLib);
  loadLibraryToOLT(LibPath, ES, OLT);

  assert(!RuntimeStartAddress &&
         "We don't currently support linking multiple runtime libraries");
  RuntimeStartAddress =
      cantFail(OLT.findSymbol("__bolt_hugify_self", false).getAddress());
  if (!RuntimeStartAddress) {
    errs() << "BOLT-ERROR: instrumentation library does not define "
              "__bolt_hugify_self: "
           << LibPath << "\n";
    exit(1);
  }
}
