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
                              const TargetAsmInfo *T) {
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
                                       DICompileUnit CU) {
  return DD->RecordSourceLine(Line, Col, CU);
}

/// RecordRegionStart - Indicate the start of a region.
unsigned DwarfWriter::RecordRegionStart(GlobalVariable *V) {
  return DD->RecordRegionStart(V);
}

/// RecordRegionEnd - Indicate the end of a region.
unsigned DwarfWriter::RecordRegionEnd(GlobalVariable *V) {
  return DD->RecordRegionEnd(V);
}

/// getRecordSourceLineCount - Count source lines.
unsigned DwarfWriter::getRecordSourceLineCount() {
  return DD->getRecordSourceLineCount();
}

/// RecordVariable - Indicate the declaration of  a local variable.
///
void DwarfWriter::RecordVariable(GlobalVariable *GV, unsigned FrameIndex) {
  DD->RecordVariable(GV, FrameIndex);
}

/// ShouldEmitDwarfDebug - Returns true if Dwarf debugging declarations should
/// be emitted.
bool DwarfWriter::ShouldEmitDwarfDebug() const {
  return DD && DD->ShouldEmitDwarfDebug();
}

//// RecordInlinedFnStart - Global variable GV is inlined at the location marked
//// by LabelID label.
unsigned DwarfWriter::RecordInlinedFnStart(DISubprogram SP, DICompileUnit CU,
                                           unsigned Line, unsigned Col) {
  return DD->RecordInlinedFnStart(SP, CU, Line, Col);
}

/// RecordInlinedFnEnd - Indicate the end of inlined subroutine.
unsigned DwarfWriter::RecordInlinedFnEnd(DISubprogram SP) {
  return DD->RecordInlinedFnEnd(SP);
}

