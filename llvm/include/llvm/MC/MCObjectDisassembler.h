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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryObject.h"
#include <vector>

namespace llvm {

namespace object {
  class ObjectFile;
  class MachOObjectFile;
}

class MCBasicBlock;
class MCDisassembler;
class MCFunction;
class MCInstrAnalysis;
class MCModule;
class MCObjectSymbolizer;

/// \brief Disassemble an ObjectFile to an MCModule and MCFunctions.
/// This class builds on MCDisassembler to disassemble whole sections, creating
/// MCAtom (MCTextAtom for disassembled sections and MCDataAtom for raw data).
/// It can also be used to create a control flow graph consisting of MCFunctions
/// and MCBasicBlocks.
class MCObjectDisassembler {
public:
  MCObjectDisassembler(const object::ObjectFile &Obj,
                       const MCDisassembler &Dis,
                       const MCInstrAnalysis &MIA);
  virtual ~MCObjectDisassembler() {}

  /// \brief Build an MCModule, creating atoms and optionally functions.
  /// \param withCFG Also build a CFG by adding MCFunctions to the Module.
  /// If withCFG is false, the MCModule built only contains atoms, representing
  /// what was found in the object file. If withCFG is true, MCFunctions are
  /// created, containing MCBasicBlocks. All text atoms are split to form basic
  /// block atoms, which then each back an MCBasicBlock.
  MCModule *buildModule(bool withCFG = false);

  MCModule *buildEmptyModule();

  typedef std::vector<uint64_t> AddressSetTy;
  /// \name Create a new MCFunction.
  MCFunction *createFunction(MCModule *Module, uint64_t BeginAddr,
                             AddressSetTy &CallTargets,
                             AddressSetTy &TailCallTargets);

  /// \brief Set the region on which to fallback if disassembly was requested
  /// somewhere not accessible in the object file.
  /// This is used for dynamic disassembly (see RawMemoryObject).
  void setFallbackRegion(OwningPtr<MemoryObject> &Region) {
    FallbackRegion.reset(Region.take());
  }

  /// \brief Set the symbolizer to use to get information on external functions.
  /// Note that this isn't used to do instruction-level symbolization (that is,
  /// plugged into MCDisassembler), but to symbolize function call targets.
  void setSymbolizer(MCObjectSymbolizer *ObjectSymbolizer) {
    MOS = ObjectSymbolizer;
  }

  /// \brief Get the effective address of the entrypoint, or 0 if there is none.
  virtual uint64_t getEntrypoint();

  /// \name Get the addresses of static constructors/destructors in the object.
  /// The caller is expected to know how to interpret the addresses;
  /// for example, Mach-O init functions expect 5 arguments, not for ELF.
  /// The addresses are original object file load addresses, not effective.
  /// @{
  virtual ArrayRef<uint64_t> getStaticInitFunctions();
  virtual ArrayRef<uint64_t> getStaticExitFunctions();
  /// @}

  /// \name Translation between effective and objectfile load address.
  /// @{
  /// \brief Compute the effective load address, from an objectfile virtual
  /// address. This is implemented in a format-specific way, to take into
  /// account things like PIE/ASLR when doing dynamic disassembly.
  /// For example, on Mach-O this would be done by adding the VM addr slide,
  /// on glibc ELF by keeping a map between segment load addresses, filled
  /// using dl_iterate_phdr, etc..
  /// In most static situations and in the default impl., this returns \p Addr.
  virtual uint64_t getEffectiveLoadAddr(uint64_t Addr);

  /// \brief Compute the original load address, as specified in the objectfile.
  /// This is the inverse of getEffectiveLoadAddr.
  virtual uint64_t getOriginalLoadAddr(uint64_t EffectiveAddr);
  /// @}

protected:
  const object::ObjectFile &Obj;
  const MCDisassembler &Dis;
  const MCInstrAnalysis &MIA;
  MCObjectSymbolizer *MOS;

  /// \brief The fallback memory region, outside the object file.
  OwningPtr<MemoryObject> FallbackRegion;

  /// \brief Return a memory region suitable for reading starting at \p Addr.
  /// In most cases, this returns a StringRefMemoryObject backed by the
  /// containing section. When no section was found, this returns the
  /// FallbackRegion, if it is suitable.
  /// If it is not, or if there is no fallback region, this returns 0.
  MemoryObject *getRegionFor(uint64_t Addr);

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

  MCBasicBlock *getBBAt(MCModule *Module, MCFunction *MCFN, uint64_t BeginAddr,
                        AddressSetTy &CallTargets,
                        AddressSetTy &TailCallTargets);
};

class MCMachOObjectDisassembler : public MCObjectDisassembler {
  const object::MachOObjectFile &MOOF;

  uint64_t VMAddrSlide;
  uint64_t HeaderLoadAddress;

  // __DATA;__mod_init_func support.
  llvm::StringRef ModInitContents;
  // __DATA;__mod_exit_func support.
  llvm::StringRef ModExitContents;

public:
  /// \brief Construct a Mach-O specific object disassembler.
  /// \param VMAddrSlide The virtual address slide applied by dyld.
  /// \param HeaderLoadAddress The load address of the mach_header for this
  /// object.
  MCMachOObjectDisassembler(const object::MachOObjectFile &MOOF,
                            const MCDisassembler &Dis,
                            const MCInstrAnalysis &MIA, uint64_t VMAddrSlide,
                            uint64_t HeaderLoadAddress);

protected:
  uint64_t getEffectiveLoadAddr(uint64_t Addr) override;
  uint64_t getOriginalLoadAddr(uint64_t EffectiveAddr) override;
  uint64_t getEntrypoint() override;

  ArrayRef<uint64_t> getStaticInitFunctions() override;
  ArrayRef<uint64_t> getStaticExitFunctions() override;
};

}

#endif
