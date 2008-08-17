//===-- OcamlGCMetadataPrinter.cpp - Ocaml frametable emitter -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements gc metadata printing for the llvm.gc* intrinsics
// compatible with Objective Caml 3.10.0, which uses a liveness-accurate static
// stack map.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/Collectors.h"
#include "llvm/CodeGen/Collector.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN OcamlGCMetadataPrinter : public GCMetadataPrinter {
  public:
    void beginAssembly(std::ostream &OS, AsmPrinter &AP,
                       const TargetAsmInfo &TAI);
    void finishAssembly(std::ostream &OS, AsmPrinter &AP,
                        const TargetAsmInfo &TAI);
  };
}

static GCMetadataPrinterRegistry::Add<OcamlGCMetadataPrinter>
X("ocaml", "ocaml 3.10-compatible collector");

static void EmitCamlGlobal(const Module &M, std::ostream &OS, AsmPrinter &AP,
                           const TargetAsmInfo &TAI, const char *Id) {
  const std::string &MId = M.getModuleIdentifier();

  std::string Mangled;
  Mangled += TAI.getGlobalPrefix();
  Mangled += "caml";
  size_t Letter = Mangled.size();
  Mangled.append(MId.begin(), std::find(MId.begin(), MId.end(), '.'));
  Mangled += "__";
  Mangled += Id;

  // Capitalize the first letter of the module name.
  Mangled[Letter] = toupper(Mangled[Letter]);

  if (const char *GlobalDirective = TAI.getGlobalDirective())
    OS << GlobalDirective << Mangled << "\n";
  OS << Mangled << ":\n";
}

void OcamlGCMetadataPrinter::beginAssembly(std::ostream &OS, AsmPrinter &AP,
                                           const TargetAsmInfo &TAI) {
  AP.SwitchToTextSection(TAI.getTextSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "code_begin");

  AP.SwitchToDataSection(TAI.getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "data_begin");
}

/// emitAssembly - Print the frametable. The ocaml frametable format is thus:
///
///   extern "C" struct align(sizeof(intptr_t)) {
///     uint16_t NumDescriptors;
///     struct align(sizeof(intptr_t)) {
///       void *ReturnAddress;
///       uint16_t FrameSize;
///       uint16_t NumLiveOffsets;
///       uint16_t LiveOffsets[NumLiveOffsets];
///     } Descriptors[NumDescriptors];
///   } caml${module}__frametable;
///
/// Note that this precludes programs from stack frames larger than 64K
/// (FrameSize and LiveOffsets would overflow). FrameTablePrinter will abort if
/// either condition is detected in a function which uses the collector.
///
void OcamlGCMetadataPrinter::finishAssembly(std::ostream &OS, AsmPrinter &AP,
                                            const TargetAsmInfo &TAI) {
  const char *AddressDirective;
  int AddressAlignLog;
  if (AP.TM.getTargetData()->getPointerSize() == sizeof(int32_t)) {
    AddressDirective = TAI.getData32bitsDirective();
    AddressAlignLog = 2;
  } else {
    AddressDirective = TAI.getData64bitsDirective();
    AddressAlignLog = 3;
  }

  AP.SwitchToTextSection(TAI.getTextSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "code_end");

  AP.SwitchToDataSection(TAI.getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "data_end");

  OS << AddressDirective << 0; // FIXME: Why does ocaml emit this??
  AP.EOL();

  AP.SwitchToDataSection(TAI.getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, TAI, "frametable");

  for (iterator FI = begin(), FE = end(); FI != FE; ++FI) {
    CollectorMetadata &MD = **FI;

    OS << "\t" << TAI.getCommentString() << " live roots for "
       << MD.getFunction().getNameStart() << "\n";

    for (CollectorMetadata::iterator PI = MD.begin(),
                                     PE = MD.end(); PI != PE; ++PI) {

      uint64_t FrameSize = MD.getFrameSize();
      if (FrameSize >= 1<<16) {
        cerr << "Function '" << MD.getFunction().getNameStart()
             << "' is too large for the ocaml collector! "
             << "Frame size " << FrameSize << " >= 65536.\n";
        abort(); // Very rude!
      }

      size_t LiveCount = MD.live_size(PI);
      if (LiveCount >= 1<<16) {
        cerr << "Function '" << MD.getFunction().getNameStart()
             << "' is too large for the ocaml collector! "
             << "Live root count " << LiveCount << " >= 65536.\n";
        abort(); // Very rude!
      }

      OS << AddressDirective
         << TAI.getPrivateGlobalPrefix() << "label" << PI->Num;
      AP.EOL("call return address");

      AP.EmitInt16(FrameSize);
      AP.EOL("stack frame size");

      AP.EmitInt16(LiveCount);
      AP.EOL("live root count");

      for (CollectorMetadata::live_iterator LI = MD.live_begin(PI),
                                            LE = MD.live_end(PI);
                                            LI != LE; ++LI) {
        assert(LI->StackOffset < 1<<16 &&
               "GC root stack offset is outside of fixed stack frame and out "
               "of range for Ocaml collector!");

        OS << "\t.word\t" << LI->StackOffset;
        AP.EOL("stack offset");
      }

      AP.EmitAlignment(AddressAlignLog);
    }
  }
}
