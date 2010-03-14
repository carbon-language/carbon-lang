//===-- OcamlGCPrinter.cpp - Ocaml frametable emitter ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements printing the assembly code for an Ocaml frametable.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GCs.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

namespace {

  class OcamlGCMetadataPrinter : public GCMetadataPrinter {
  public:
    void beginAssembly(raw_ostream &OS, AsmPrinter &AP,
                       const MCAsmInfo &MAI);

    void finishAssembly(raw_ostream &OS, AsmPrinter &AP,
                        const MCAsmInfo &MAI);
  };

}

static GCMetadataPrinterRegistry::Add<OcamlGCMetadataPrinter>
Y("ocaml", "ocaml 3.10-compatible collector");

void llvm::linkOcamlGCPrinter() { }

static void EmitCamlGlobal(const Module &M, raw_ostream &OS, AsmPrinter &AP,
                           const MCAsmInfo &MAI, const char *Id) {
  const std::string &MId = M.getModuleIdentifier();

  std::string Mangled;
  Mangled += MAI.getGlobalPrefix();
  Mangled += "caml";
  size_t Letter = Mangled.size();
  Mangled.append(MId.begin(), std::find(MId.begin(), MId.end(), '.'));
  Mangled += "__";
  Mangled += Id;

  // Capitalize the first letter of the module name.
  Mangled[Letter] = toupper(Mangled[Letter]);

  if (const char *GlobalDirective = MAI.getGlobalDirective())
    OS << GlobalDirective << Mangled << "\n";
  OS << Mangled << ":\n";
}

void OcamlGCMetadataPrinter::beginAssembly(raw_ostream &OS, AsmPrinter &AP,
                                           const MCAsmInfo &MAI) {
  AP.OutStreamer.SwitchSection(AP.getObjFileLowering().getTextSection());
  EmitCamlGlobal(getModule(), OS, AP, MAI, "code_begin");

  AP.OutStreamer.SwitchSection(AP.getObjFileLowering().getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, MAI, "data_begin");
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
/// either condition is detected in a function which uses the GC.
///
void OcamlGCMetadataPrinter::finishAssembly(raw_ostream &OS, AsmPrinter &AP,
                                            const MCAsmInfo &MAI) {
  const char *AddressDirective;
  int AddressAlignLog;
  if (AP.TM.getTargetData()->getPointerSize() == sizeof(int32_t)) {
    AddressDirective = MAI.getData32bitsDirective();
    AddressAlignLog = 2;
  } else {
    AddressDirective = MAI.getData64bitsDirective();
    AddressAlignLog = 3;
  }

  AP.OutStreamer.SwitchSection(AP.getObjFileLowering().getTextSection());
  EmitCamlGlobal(getModule(), OS, AP, MAI, "code_end");

  AP.OutStreamer.SwitchSection(AP.getObjFileLowering().getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, MAI, "data_end");

  OS << AddressDirective << 0 << '\n'; // FIXME: Why does ocaml emit this??

  AP.OutStreamer.SwitchSection(AP.getObjFileLowering().getDataSection());
  EmitCamlGlobal(getModule(), OS, AP, MAI, "frametable");

  for (iterator I = begin(), IE = end(); I != IE; ++I) {
    GCFunctionInfo &FI = **I;

    uint64_t FrameSize = FI.getFrameSize();
    if (FrameSize >= 1<<16) {
      std::string msg;
      raw_string_ostream Msg(msg);
      Msg << "Function '" << FI.getFunction().getName()
           << "' is too large for the ocaml GC! "
           << "Frame size " << FrameSize << " >= 65536.\n";
      Msg << "(" << uintptr_t(&FI) << ")";
      llvm_report_error(Msg.str()); // Very rude!
    }

    OS << "\t" << MAI.getCommentString() << " live roots for "
       << FI.getFunction().getName() << "\n";

    for (GCFunctionInfo::iterator J = FI.begin(), JE = FI.end(); J != JE; ++J) {
      size_t LiveCount = FI.live_size(J);
      if (LiveCount >= 1<<16) {
        std::string msg;
        raw_string_ostream Msg(msg);
        Msg << "Function '" << FI.getFunction().getName()
             << "' is too large for the ocaml GC! "
             << "Live root count " << LiveCount << " >= 65536.";
        llvm_report_error(Msg.str()); // Very rude!
      }

      OS << AddressDirective << J->Label->getName() << '\n';

      AP.EmitInt16(FrameSize);

      AP.EmitInt16(LiveCount);

      for (GCFunctionInfo::live_iterator K = FI.live_begin(J),
                                         KE = FI.live_end(J); K != KE; ++K) {
        assert(K->StackOffset < 1<<16 &&
               "GC root stack offset is outside of fixed stack frame and out "
               "of range for ocaml GC!");

        AP.EmitInt32(K->StackOffset);
      }

      AP.EmitAlignment(AddressAlignLog);
    }
  }
}
