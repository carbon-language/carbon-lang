//===-- X86/Printer.cpp - Convert X86 code to human readable rep. ---------===//
//
// This file contains a printer that converts from our internal representation
// of LLVM code to a nice human readable form that is suitable for debuggging.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include <iostream>

/// X86PrintCode - Print out the specified machine code function to the
/// specified stream.  This function should work regardless of whether or not
/// the function is in SSA form or not, although when in SSA form, we obviously
/// don't care about being consumable by an assembler.
///
void X86PrintCode(const MFunction *MF, std::ostream &O) {
  O << "x86 printing not implemented yet!\n";

  // This should use the X86InstructionInfo::print method to print assembly for
  // each instruction
}
