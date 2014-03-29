//===-- llvm/MC/MCLinkerOptimizationHint.cpp ----- LOH handling -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCLinkerOptimizationHint.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/Support/LEB128.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

void MCLOHDirective::Emit_impl(raw_ostream &OutStream,
                               const MachObjectWriter &ObjWriter,
                               const MCAsmLayout &Layout) const {
  const MCAssembler &Asm = Layout.getAssembler();
  encodeULEB128(Kind, OutStream);
  encodeULEB128(Args.size(), OutStream);
  for (LOHArgs::const_iterator It = Args.begin(), EndIt = Args.end();
       It != EndIt; ++It)
    encodeULEB128(ObjWriter.getSymbolAddress(&Asm.getSymbolData(**It), Layout),
                  OutStream);
}
