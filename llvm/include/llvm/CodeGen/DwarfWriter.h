//===-- llvm/CodeGen/DwarfWriter.h - Dwarf Framework ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Dwarf debug and exception info into
// asm files.  For Details on the Dwarf 3 specfication see DWARF Debugging
// Information Format V.3 reference manual http://dwarf.freestandards.org ,
//
// The role of the Dwarf Writer class is to extract information from the
// MachineModuleInfo object, organize it in Dwarf form and then emit it into asm
// the current asm file using data and high level Dwarf directives.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFWRITER_H
#define LLVM_CODEGEN_DWARFWRITER_H

#include "llvm/Pass.h"

namespace llvm {

class AsmPrinter;
class DwarfDebug;
class DwarfException;
class MachineModuleInfo;
class MachineFunction;
class MachineInstr;
class Value;
class Module;
class GlobalVariable;
class TargetAsmInfo;
class raw_ostream;
class Instruction;
class DISubprogram;
class DIVariable;

//===----------------------------------------------------------------------===//
// DwarfWriter - Emits Dwarf debug and exception handling directives.
//

class DwarfWriter : public ImmutablePass {
private:
  /// DD - Provides the DwarfWriter debug implementation.
  ///
  DwarfDebug *DD;

  /// DE - Provides the DwarfWriter exception implementation.
  ///
  DwarfException *DE;

public:
  static char ID; // Pass identification, replacement for typeid

  DwarfWriter();
  virtual ~DwarfWriter();

  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  
  /// BeginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void BeginModule(Module *M, MachineModuleInfo *MMI, raw_ostream &OS,
                   AsmPrinter *A, const TargetAsmInfo *T);
  
  /// EndModule - Emit all Dwarf sections that should come after the content.
  ///
  void EndModule();
  
  /// BeginFunction - Gather pre-function debug information.  Assumes being 
  /// emitted immediately after the function entry point.
  void BeginFunction(MachineFunction *MF);
  
  /// EndFunction - Gather and emit post-function debug information.
  ///
  void EndFunction(MachineFunction *MF);

  /// ValidDebugInfo - Return true if V represents valid debug info value.
  bool ValidDebugInfo(Value *V, bool FastISel);

  /// RecordSourceLine - Register a source line with debug info. Returns a
  /// unique label ID used to generate a label and provide correspondence to
  /// the source line list.
  unsigned RecordSourceLine(unsigned Line, unsigned Col, unsigned Src);

  /// getOrCreateSourceID - Look up the source id with the given directory and
  /// source file names. If none currently exists, create a new id and insert it
  /// in the SourceIds map. This can update DirectoryIds and SourceFileIds maps
  /// as well.
  unsigned getOrCreateSourceID(const std::string &DirName,
                               const std::string &FileName);

  /// RecordRegionStart - Indicate the start of a region.
  unsigned RecordRegionStart(GlobalVariable *V);

  /// RecordRegionEnd - Indicate the end of a region.
  unsigned RecordRegionEnd(GlobalVariable *V);

  /// getRecordSourceLineCount - Count source lines.
  unsigned getRecordSourceLineCount();

  /// RecordVariable - Indicate the declaration of  a local variable.
  ///
  void RecordVariable(GlobalVariable *GV, unsigned FrameIndex, 
                      const MachineInstr *MI);

  /// ShouldEmitDwarfDebug - Returns true if Dwarf debugging declarations should
  /// be emitted.
  bool ShouldEmitDwarfDebug() const;

  //// RecordInlinedFnStart - Indicate the start of a inlined function.
  void RecordInlinedFnStart(Instruction *I, DISubprogram &SP, unsigned LabelID,
                            unsigned Src, unsigned Line, unsigned Col);

  /// RecordInlinedFnEnd - Indicate the end of inlined subroutine.
  unsigned RecordInlinedFnEnd(DISubprogram &SP);

  /// RecordVariableScope - Record scope for the variable declared by
  /// DeclareMI. DeclareMI must describe TargetInstrInfo::DECLARE.
  void RecordVariableScope(DIVariable &DV, const MachineInstr *DeclareMI);
};


} // end llvm namespace

#endif
