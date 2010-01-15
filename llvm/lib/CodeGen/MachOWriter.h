//=== MachOWriter.h - Target-independent Mach-O writer support --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOWriter class.
//
//===----------------------------------------------------------------------===//

#ifndef MACHOWRITER_H
#define MACHOWRITER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class GlobalVariable;
  class Mangler;
  class MCCodeEmitter;
  class MCContext;
  class MCStreamer;
  
  /// MachOWriter - This class implements the common target-independent code for
  /// writing Mach-O files.  Targets should derive a class from this to
  /// parameterize the output format.
  ///
  class MachOWriter : public MachineFunctionPass {
    static char ID;

  protected:
    /// Output stream to send the resultant object file to.
    ///
    formatted_raw_ostream &O;

    /// Target machine description.
    ///
    TargetMachine &TM;

    /// Target Asm Printer information.
    ///
    const MCAsmInfo *MAI;
    
    /// MCCE - The MCCodeEmitter object that we are exposing to emit machine
    /// code for functions to the .o file.
    MCCodeEmitter *MCCE;
    
    /// OutContext - This is the context for the output file that we are
    /// streaming.  This owns all of the global MC-related objects for the
    /// generated translation unit.
    MCContext &OutContext;
    
    /// OutStreamer - This is the MCStreamer object for the file we are
    /// generating.  This contains the transient state for the current
    /// translation unit that we are generating (such as the current section
    /// etc).
    MCStreamer &OutStreamer;
    
    /// Name-mangler for global names.
    ///
    Mangler *Mang;
    
    /// doInitialization - Emit the file header and all of the global variables
    /// for the module to the Mach-O file.
    bool doInitialization(Module &M);

    /// doFinalization - Now that the module has been completely processed, emit
    /// the Mach-O file to 'O'.
    bool doFinalization(Module &M);

    bool runOnMachineFunction(MachineFunction &MF);
    
  public:
    explicit MachOWriter(formatted_raw_ostream &O, TargetMachine &TM,
                         const MCAsmInfo *T, MCCodeEmitter *MCE);
    
    virtual ~MachOWriter();
    
    virtual const char *getPassName() const {
      return "Mach-O Writer";
    }
  };
}

#endif
