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
#include "llvm/Support/Options.h"

namespace opts {
extern llvm::cl::opt<unsigned> Verbosity;
}

namespace llvm {
namespace bolt {

void PatchEntries::runOnFunctions(BinaryContext &BC) {
  if (!BC.HasRelocations)
    return;

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

  if (opts::Verbosity >= 1) {
    outs() << "BOLT-INFO: patching entries in original code\n";
  }

  static auto Emitter = BC.createIndependentMCCodeEmitter();

  for (auto &BFI : BC.getBinaryFunctions()) {
    auto &Function = BFI.second;

    // Patch original code only for functions that are emitted.
    if (!BC.shouldEmit(Function))
      continue;

    // Check if we can skip patching the function.
    if (!Function.hasEHRanges() && Function.getSize() < PatchThreshold) {
      continue;
    }

    // List of patches for function entries. We either successfully patch
    // all entries or, if we cannot patch any, do no patch the rest and
    // mark the function as ignorable.
    std::vector<InstructionPatch> PendingPatches;

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

      MCInst TailCallInst;
      BC.MIB->createTailCall(TailCallInst, Symbol, BC.Ctx.get());

      PendingPatches.push_back(InstructionPatch());
      InstructionPatch &InstPatch = PendingPatches.back();
      InstPatch.Offset =
        Function.getAddress() - Function.getSection().getAddress() + Offset;

      SmallVector<MCFixup, 4> Fixups;
      raw_svector_ostream VecOS(InstPatch.Code);

      Emitter.MCE->encodeInstruction(TailCallInst, VecOS, Fixups, *BC.STI);

      NextValidByte = Offset + InstPatch.Code.size();
      if (NextValidByte > Function.getMaxSize()) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: function " << Function << " too small to patch "
                    "its entry point\n";
        }
        return false;
      }

      assert(Fixups.size() == 1 && "unexpected fixup size");
      Optional<Relocation> Rel = BC.MIB->createRelocation(Fixups[0], *BC.MAB);
      assert(Rel && "unable to convert fixup to relocation");

      InstPatch.Rel = *Rel;
      InstPatch.Rel.Offset += InstPatch.Offset;

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

    // Apply all recorded patches.
    for (auto &Patch : PendingPatches) {
      Function.getSection().addPatch(Patch.Offset, Patch.Code);
      Function.getSection().addPendingRelocation(Patch.Rel);
    }
    Function.setIsPatched(true);
  }
}

} // end namespace bolt
} // end namespace llvm
