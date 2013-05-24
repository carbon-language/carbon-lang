//===-- llvm/MC/MCObjectDisassembler.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCObjectDisassembler class, which
// can be used to construct an MCModule and an MC CFG from an ObjectFile.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTDISASSEMBLER_H
#define LLVM_MC_MCOBJECTDISASSEMBLER_H

namespace llvm {

namespace object {
  class ObjectFile;
}

class MCBasicBlock;
class MCDisassembler;
class MCFunction;
class MCInstrAnalysis;
class MCModule;

/// \brief Disassemble an ObjectFile to an MCModule and MCFunctions.
/// This class builds on MCDisassembler to disassemble whole sections, creating
/// MCAtom (MCTextAtom for disassembled sections and MCDataAtom for raw data).
/// It can also be used to create a control flow graph consisting of MCFunctions
/// and MCBasicBlocks.
class MCObjectDisassembler {
  const object::ObjectFile &Obj;
  const MCDisassembler &Dis;
  const MCInstrAnalysis &MIA;

public:
  MCObjectDisassembler(const object::ObjectFile &Obj,
                       const MCDisassembler &Dis,
                       const MCInstrAnalysis &MIA);

  /// \brief Build an MCModule, creating atoms and optionally functions.
  /// \param withCFG Also build a CFG by adding MCFunctions to the Module.
  /// If withCFG is false, the MCModule built only contains atoms, representing
  /// what was found in the object file. If withCFG is true, MCFunctions are
  /// created, containing MCBasicBlocks. All text atoms are split to form basic
  /// block atoms, which then each back an MCBasicBlock.
  MCModule *buildModule(bool withCFG = false);

private:
  /// \brief Fill \p Module by creating an atom for each section.
  /// This could be made much smarter, using information like symbols, but also
  /// format-specific features, like mach-o function_start or data_in_code LCs.
  void buildSectionAtoms(MCModule *Module);

  /// \brief Enrich \p Module with a CFG consisting of MCFunctions.
  /// \param Module An MCModule returned by buildModule, with no CFG.
  /// NOTE: Each MCBasicBlock in a MCFunction is backed by a single MCTextAtom.
  /// When the CFG is built, contiguous instructions that were previously in a
  /// single MCTextAtom will be split in multiple basic block atoms.
  void buildCFG(MCModule *Module);
};

}

#endif
