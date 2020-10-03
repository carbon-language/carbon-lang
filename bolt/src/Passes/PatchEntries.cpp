//===--- Passes/PatchEntries.cpp - pass for patching function entries -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass for patching original function entry points.
//
//===----------------------------------------------------------------------===//

#include "PatchEntries.h"
#include "NameResolver.h"
#include "llvm/Support/Options.h"

namespace opts {

extern llvm::cl::OptionCategory BoltCategory;

extern llvm::cl::opt<unsigned> Verbosity;

llvm::cl::opt<bool>
ForcePatch("force-patch",
  llvm::cl::desc("force patching of original entry points"),
  llvm::cl::init(false),
  llvm::cl::Hidden,
  llvm::cl::ZeroOrMore,
  llvm::cl::cat(BoltCategory));

}

namespace llvm {
namespace bolt {

void PatchEntries::runOnFunctions(BinaryContext &BC) {
  if (!opts::ForcePatch) {
    // Mark the binary for patching if we did not create external references
    // for original code in any of functions we are not going to emit.
    bool NeedsPatching = false;
    for (auto &BFI : BC.getBinaryFunctions()) {
      BinaryFunction &Function = BFI.second;
      if (!BC.shouldEmit(Function) && !Function.hasExternalRefRelocations()) {
        NeedsPatching = true;
        break;
      }
    }

    if (!NeedsPatching)
      return;
  }

  if (opts::Verbosity >= 1) {
    outs() << "BOLT-INFO: patching entries in original code\n";
  }

  for (auto &BFI : BC.getBinaryFunctions()) {
    auto &Function = BFI.second;

    // Patch original code only for functions that will be emitted.
    if (!BC.shouldEmit(Function))
      continue;

    // Check if we can skip patching the function.
    if (!opts::ForcePatch && !Function.hasEHRanges() &&
        Function.getSize() < PatchThreshold) {
      continue;
    }

    uint64_t NextValidByte = 0; // offset of the byte past the last patch
    bool Success = Function.forEachEntryPoint([&](uint64_t Offset,
                                                  const MCSymbol *Symbol) {
      if (Offset < NextValidByte) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: unable to patch entry point in " << Function
                 << " at offset 0x" << Twine::utohexstr(Offset) << '\n';
        }
        return false;
      }

      BinaryFunction *PatchFunction =
          BC.createInjectedBinaryFunction(
              NameResolver::append(Symbol->getName(), ".org.0"));
      PatchFunction->setOutputAddress(Function.getAddress() + Offset);
      PatchFunction->setFileOffset(Function.getFileOffset() + Offset);
      PatchFunction->setOriginSection(Function.getOriginSection());

      MCInst TailCallInst;
      BC.MIB->createTailCall(TailCallInst, Symbol, BC.Ctx.get());
      PatchFunction->addBasicBlock(0)->addInstruction(TailCallInst);

      uint64_t HotSize, ColdSize;
      std::tie(HotSize, ColdSize) = BC.calculateEmittedSize(*PatchFunction);
      assert(!ColdSize && "unexpected cold code");
      NextValidByte = Offset + HotSize;
      if (NextValidByte > Function.getMaxSize()) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: function " << Function << " too small to patch "
                    "its entry point\n";
        }
        return false;
      }

      return true;
    });

    if (!Success) {
      // If the original function entries cannot be patched, then we cannot
      // safely emit new function body.
      errs() << "BOLT-WARNING: failed to patch entries in " << Function
             << ". The function will not be optimized.\n";
      Function.setIgnored();
      continue;
    }

    Function.setIsPatched(true);
  }
}

} // end namespace bolt
} // end namespace llvm
