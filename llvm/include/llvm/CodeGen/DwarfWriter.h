//===-- llvm/CodeGen/DwarfWriter.h - Dwarf Framework ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Dwarf debug info into asm files.  For
// Details on the Dwarf 3 specfication see DWARF Debugging Information Format
// V.3 reference manual http://dwarf.freestandards.org ,
//
// The role of the Dwarf Writer class is to extract debug information from the
// MachineModuleInfo object, organize it in Dwarf form and then emit it into asm
// the current asm file using data and high level Dwarf directives.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFWRITER_H
#define LLVM_CODEGEN_DWARFWRITER_H

#include <iosfwd>

namespace llvm {

class AsmPrinter;
class Dwarf;
class MachineModuleInfo;
class MachineFunction;
class Module;
class TargetAsmInfo;

//===----------------------------------------------------------------------===//
// DwarfWriter - Emits Dwarf debug and exception handling directives.
//

class DwarfWriter {
private:
  /// DM - Provides the DwarfWriter implementation.
  ///
  Dwarf *DW;
  
public:
  
  DwarfWriter(std::ostream &OS, AsmPrinter *A, const TargetAsmInfo *T);
  virtual ~DwarfWriter();
  
  /// SetModuleInfo - Set machine module info when it's known that pass manager
  /// has created it.  Set by the target AsmPrinter.
  void SetModuleInfo(MachineModuleInfo *MMI);

  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  
  /// BeginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void BeginModule(Module *M);
  
  /// EndModule - Emit all Dwarf sections that should come after the content.
  ///
  void EndModule();
  
  /// BeginFunction - Gather pre-function debug information.  Assumes being 
  /// emitted immediately after the function entry point.
  void BeginFunction(MachineFunction *MF);
  
  /// EndFunction - Gather and emit post-function debug information.
  ///
  void EndFunction();
};


} // end llvm namespace

#endif
