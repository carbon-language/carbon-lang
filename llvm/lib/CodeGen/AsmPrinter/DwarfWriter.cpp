//===-- llvm/CodeGen/DwarfWriter.cpp - Dwarf Framework --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf info into asm files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DwarfWriter.h"
#include "DwarfDebug.h"
#include "DwarfException.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

using namespace llvm;

static RegisterPass<DwarfWriter>
X("dwarfwriter", "DWARF Information Writer");
char DwarfWriter::ID = 0;

//===----------------------------------------------------------------------===//
/// DwarfWriter Implementation
///

DwarfWriter::DwarfWriter()
  : ImmutablePass(&ID), DD(0), DE(0) {}

DwarfWriter::~DwarfWriter() {
  delete DE;
  delete DD;
}

/// BeginModule - Emit all Dwarf sections that should come prior to the
/// content.
void DwarfWriter::BeginModule(Module *M,
                              MachineModuleInfo *MMI,
                              raw_ostream &OS, AsmPrinter *A,
                              const MCAsmInfo *T) {
  DE = new DwarfException(OS, A, T);
  DD = new DwarfDebug(OS, A, T);
  DE->BeginModule(M, MMI);
  DD->BeginModule(M, MMI);
}

/// EndModule - Emit all Dwarf sections that should come after the content.
///
void DwarfWriter::EndModule() {
  DE->EndModule();
  DD->EndModule();
  delete DD; DD = 0;
  delete DE; DE = 0;
}

/// BeginFunction - Gather pre-function debug information.  Assumes being
/// emitted immediately after the function entry point.
void DwarfWriter::BeginFunction(MachineFunction *MF) {
  DE->BeginFunction(MF);
  DD->BeginFunction(MF);
}

/// EndFunction - Gather and emit post-function debug information.
///
void DwarfWriter::EndFunction(MachineFunction *MF) {
  DD->EndFunction(MF);
  DE->EndFunction();

  if (MachineModuleInfo *MMI = DD->getMMI() ? DD->getMMI() : DE->getMMI())
    // Clear function debug information.
    MMI->EndFunction();
}

/// RecordSourceLine - Records location information and associates it with a 
/// label. Returns a unique label ID used to generate a label and provide
/// correspondence to the source line list.
unsigned DwarfWriter::RecordSourceLine(unsigned Line, unsigned Col, 
                                       MDNode *Scope) {
  return DD->RecordSourceLine(Line, Col, Scope);
}

/// getRecordSourceLineCount - Count source lines.
unsigned DwarfWriter::getRecordSourceLineCount() {
  return DD->getRecordSourceLineCount();
}

/// ShouldEmitDwarfDebug - Returns true if Dwarf debugging declarations should
/// be emitted.
bool DwarfWriter::ShouldEmitDwarfDebug() const {
  return DD && DD->ShouldEmitDwarfDebug();
}

void DwarfWriter::BeginScope(const MachineInstr *MI, unsigned L) {
  DD->BeginScope(MI, L);
}
void DwarfWriter::EndScope(const MachineInstr *MI) {
  DD->EndScope(MI);
}
