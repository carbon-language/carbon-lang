//===- llvm/CodeGen/DwarfCompileUnit.cpp - Dwarf Compile Units ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for constructing a dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#include "DwarfCompileUnit.h"
#include "AddressPool.h"
#include "DwarfExpression.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iterator>
#include <string>
#include <utility>

using namespace llvm;

static dwarf::Tag GetCompileUnitType(UnitKind Kind, DwarfDebug *DW) {

  //  According to DWARF Debugging Information Format Version 5,
  //  3.1.2 Skeleton Compilation Unit Entries:
  //  "When generating a split DWARF object file (see Section 7.3.2
  //  on page 187), the compilation unit in the .debug_info section
  //  is a "skeleton" compilation unit with the tag DW_TAG_skeleton_unit"
  if (DW->getDwarfVersion() >= 5 && Kind == UnitKind::Skeleton)
    return dwarf::DW_TAG_skeleton_unit;

  return dwarf::DW_TAG_compile_unit;
}

DwarfCompileUnit::DwarfCompileUnit(unsigned UID, const DICompileUnit *Node,
                                   AsmPrinter *A, DwarfDebug *DW,
                                   DwarfFile *DWU, UnitKind Kind)
    : DwarfUnit(GetCompileUnitType(Kind, DW), Node, A, DW, DWU), UniqueID(UID) {
  insertDIE(Node, &getUnitDie());
  MacroLabelBegin = Asm->createTempSymbol("cu_macro_begin");
}

/// addLabelAddress - Add a dwarf label attribute data and value using
/// DW_FORM_addr or DW_FORM_GNU_addr_index.
void DwarfCompileUnit::addLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                                       const MCSymbol *Label) {
  // Don't use the address pool in non-fission or in the skeleton unit itself.
  if ((!DD->useSplitDwarf() || !Skeleton) && DD->getDwarfVersion() < 5)
    return addLocalLabelAddress(Die, Attribute, Label);

  if (Label)
    DD->addArangeLabel(SymbolCU(this, Label));

  bool UseAddrOffsetFormOrExpressions =
      DD->useAddrOffsetForm() || DD->useAddrOffsetExpressions();

  const MCSymbol *Base = nullptr;
  if (Label->isInSection() && UseAddrOffsetFormOrExpressions)
    Base = DD->getSectionLabel(&Label->getSection());

  if (!Base || Base == Label) {
    unsigned idx = DD->getAddressPool().getIndex(Label);
    addAttribute(Die, Attribute,
                 DD->getDwarfVersion() >= 5 ? dwarf::DW_FORM_addrx
                                            : dwarf::DW_FORM_GNU_addr_index,
                 DIEInteger(idx));
    return;
  }

  // Could be extended to work with DWARFv4 Split DWARF if that's important for
  // someone. In that case DW_FORM_data would be used.
  assert(DD->getDwarfVersion() >= 5 &&
         "Addr+offset expressions are only valuable when using debug_addr (to "
         "reduce relocations) available in DWARFv5 or higher");
  if (DD->useAddrOffsetExpressions()) {
    auto *Loc = new (DIEValueAllocator) DIEBlock();
    addPoolOpAddress(*Loc, Label);
    addBlock(Die, Attribute, dwarf::DW_FORM_exprloc, Loc);
  } else
    addAttribute(Die, Attribute, dwarf::DW_FORM_LLVM_addrx_offset,
                 new (DIEValueAllocator) DIEAddrOffset(
                     DD->getAddressPool().getIndex(Base), Label, Base));
}

void DwarfCompileUnit::addLocalLabelAddress(DIE &Die,
                                            dwarf::Attribute Attribute,
                                            const MCSymbol *Label) {
  if (Label)
    DD->addArangeLabel(SymbolCU(this, Label));

  if (Label)
    addAttribute(Die, Attribute, dwarf::DW_FORM_addr, DIELabel(Label));
  else
    addAttribute(Die, Attribute, dwarf::DW_FORM_addr, DIEInteger(0));
}

unsigned DwarfCompileUnit::getOrCreateSourceID(const DIFile *File) {
  // If we print assembly, we can't separate .file entries according to
  // compile units. Thus all files will belong to the default compile unit.

  // FIXME: add a better feature test than hasRawTextSupport. Even better,
  // extend .file to support this.
  unsigned CUID = Asm->OutStreamer->hasRawTextSupport() ? 0 : getUniqueID();
  if (!File)
    return Asm->OutStreamer->emitDwarfFileDirective(0, "", "", None, None,
                                                    CUID);
  return Asm->OutStreamer->emitDwarfFileDirective(
      0, File->getDirectory(), File->getFilename(), DD->getMD5AsBytes(File),
      File->getSource(), CUID);
}

DIE *DwarfCompileUnit::getOrCreateGlobalVariableDIE(
    const DIGlobalVariable *GV, ArrayRef<GlobalExpr> GlobalExprs) {
  // Check for pre-existence.
  if (DIE *Die = getDIE(GV))
    return Die;

  assert(GV);

  auto *GVContext = GV->getScope();
  const DIType *GTy = GV->getType();

  auto *CB = GVContext ? dyn_cast<DICommonBlock>(GVContext) : nullptr;
  DIE *ContextDIE = CB ? getOrCreateCommonBlock(CB, GlobalExprs)
    : getOrCreateContextDIE(GVContext);

  // Add to map.
  DIE *VariableDIE = &createAndAddDIE(GV->getTag(), *ContextDIE, GV);
  DIScope *DeclContext;
  if (auto *SDMDecl = GV->getStaticDataMemberDeclaration()) {
    DeclContext = SDMDecl->getScope();
    assert(SDMDecl->isStaticMember() && "Expected static member decl");
    assert(GV->isDefinition());
    // We need the declaration DIE that is in the static member's class.
    DIE *VariableSpecDIE = getOrCreateStaticMemberDIE(SDMDecl);
    addDIEEntry(*VariableDIE, dwarf::DW_AT_specification, *VariableSpecDIE);
    // If the global variable's type is different from the one in the class
    // member type, assume that it's more specific and also emit it.
    if (GTy != SDMDecl->getBaseType())
      addType(*VariableDIE, GTy);
  } else {
    DeclContext = GV->getScope();
    // Add name and type.
    addString(*VariableDIE, dwarf::DW_AT_name, GV->getDisplayName());
    if (GTy)
      addType(*VariableDIE, GTy);

    // Add scoping info.
    if (!GV->isLocalToUnit())
      addFlag(*VariableDIE, dwarf::DW_AT_external);

    // Add line number info.
    addSourceLine(*VariableDIE, GV);
  }

  if (!GV->isDefinition())
    addFlag(*VariableDIE, dwarf::DW_AT_declaration);
  else
    addGlobalName(GV->getName(), *VariableDIE, DeclContext);

  addAnnotation(*VariableDIE, GV->getAnnotations());

  if (uint32_t AlignInBytes = GV->getAlignInBytes())
    addUInt(*VariableDIE, dwarf::DW_AT_alignment, dwarf::DW_FORM_udata,
            AlignInBytes);

  if (MDTuple *TP = GV->getTemplateParams())
    addTemplateParams(*VariableDIE, DINodeArray(TP));

  // Add location.
  addLocationAttribute(VariableDIE, GV, GlobalExprs);

  return VariableDIE;
}

void DwarfCompileUnit::addLocationAttribute(
    DIE *VariableDIE, const DIGlobalVariable *GV, ArrayRef<GlobalExpr> GlobalExprs) {
  bool addToAccelTable = false;
  DIELoc *Loc = nullptr;
  Optional<unsigned> NVPTXAddressSpace;
  std::unique_ptr<DIEDwarfExpression> DwarfExpr;
  for (const auto &GE : GlobalExprs) {
    const GlobalVariable *Global = GE.Var;
    const DIExpression *Expr = GE.Expr;

    // For compatibility with DWARF 3 and earlier,
    // DW_AT_location(DW_OP_constu, X, DW_OP_stack_value) or
    // DW_AT_location(DW_OP_consts, X, DW_OP_stack_value) becomes
    // DW_AT_const_value(X).
    if (GlobalExprs.size() == 1 && Expr && Expr->isConstant()) {
      addToAccelTable = true;
      addConstantValue(
          *VariableDIE,
          DIExpression::SignedOrUnsignedConstant::UnsignedConstant ==
              *Expr->isConstant(),
          Expr->getElement(1));
      break;
    }

    // We cannot describe the location of dllimport'd variables: the
    // computation of their address requires loads from the IAT.
    if (Global && Global->hasDLLImportStorageClass())
      continue;

    // Nothing to describe without address or constant.
    if (!Global && (!Expr || !Expr->isConstant()))
      continue;

    if (Global && Global->isThreadLocal() &&
        !Asm->getObjFileLowering().supportDebugThreadLocalLocation())
      continue;

    if (!Loc) {
      addToAccelTable = true;
      Loc = new (DIEValueAllocator) DIELoc;
      DwarfExpr = std::make_unique<DIEDwarfExpression>(*Asm, *this, *Loc);
    }

    if (Expr) {
      // According to
      // https://docs.nvidia.com/cuda/archive/10.0/ptx-writers-guide-to-interoperability/index.html#cuda-specific-dwarf
      // cuda-gdb requires DW_AT_address_class for all variables to be able to
      // correctly interpret address space of the variable address.
      // Decode DW_OP_constu <DWARF Address Space> DW_OP_swap DW_OP_xderef
      // sequence for the NVPTX + gdb target.
      unsigned LocalNVPTXAddressSpace;
      if (Asm->TM.getTargetTriple().isNVPTX() && DD->tuneForGDB()) {
        const DIExpression *NewExpr =
            DIExpression::extractAddressClass(Expr, LocalNVPTXAddressSpace);
        if (NewExpr != Expr) {
          Expr = NewExpr;
          NVPTXAddressSpace = LocalNVPTXAddressSpace;
        }
      }
      DwarfExpr->addFragmentOffset(Expr);
    }

    if (Global) {
      const MCSymbol *Sym = Asm->getSymbol(Global);
      // 16-bit platforms like MSP430 and AVR take this path, so sink this
      // assert to platforms that use it.
      auto GetPointerSizedFormAndOp = [this]() {
        unsigned PointerSize = Asm->getDataLayout().getPointerSize();
        assert((PointerSize == 4 || PointerSize == 8) &&
               "Add support for other sizes if necessary");
        struct FormAndOp {
          dwarf::Form Form;
          dwarf::LocationAtom Op;
        };
        return PointerSize == 4
                   ? FormAndOp{dwarf::DW_FORM_data4, dwarf::DW_OP_const4u}
                   : FormAndOp{dwarf::DW_FORM_data8, dwarf::DW_OP_const8u};
      };
      if (Global->isThreadLocal()) {
        if (Asm->TM.useEmulatedTLS()) {
          // TODO: add debug info for emulated thread local mode.
        } else {
          // FIXME: Make this work with -gsplit-dwarf.
          // Based on GCC's support for TLS:
          if (!DD->useSplitDwarf()) {
            auto FormAndOp = GetPointerSizedFormAndOp();
            // 1) Start with a constNu of the appropriate pointer size
            addUInt(*Loc, dwarf::DW_FORM_data1, FormAndOp.Op);
            // 2) containing the (relocated) offset of the TLS variable
            //    within the module's TLS block.
            addExpr(*Loc, FormAndOp.Form,
                    Asm->getObjFileLowering().getDebugThreadLocalSymbol(Sym));
          } else {
            addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_GNU_const_index);
            addUInt(*Loc, dwarf::DW_FORM_udata,
                    DD->getAddressPool().getIndex(Sym, /* TLS */ true));
          }
          // 3) followed by an OP to make the debugger do a TLS lookup.
          addUInt(*Loc, dwarf::DW_FORM_data1,
                  DD->useGNUTLSOpcode() ? dwarf::DW_OP_GNU_push_tls_address
                                        : dwarf::DW_OP_form_tls_address);
        }
      } else if (Asm->TM.getRelocationModel() == Reloc::RWPI ||
                 Asm->TM.getRelocationModel() == Reloc::ROPI_RWPI) {
        auto FormAndOp = GetPointerSizedFormAndOp();
        // Constant
        addUInt(*Loc, dwarf::DW_FORM_data1, FormAndOp.Op);
        // Relocation offset
        addExpr(*Loc, FormAndOp.Form,
                Asm->getObjFileLowering().getIndirectSymViaRWPI(Sym));
        // Base register
        Register BaseReg = Asm->getObjFileLowering().getStaticBase();
        BaseReg = Asm->TM.getMCRegisterInfo()->getDwarfRegNum(BaseReg, false);
        addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + BaseReg);
        // Offset from base register
        addSInt(*Loc, dwarf::DW_FORM_sdata, 0);
        // Operation
        addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);
      } else {
        DD->addArangeLabel(SymbolCU(this, Sym));
        addOpAddress(*Loc, Sym);
      }
    }
    // Global variables attached to symbols are memory locations.
    // It would be better if this were unconditional, but malformed input that
    // mixes non-fragments and fragments for the same variable is too expensive
    // to detect in the verifier.
    if (DwarfExpr->isUnknownLocation())
      DwarfExpr->setMemoryLocationKind();
    DwarfExpr->addExpression(Expr);
  }
  if (Asm->TM.getTargetTriple().isNVPTX() && DD->tuneForGDB()) {
    // According to
    // https://docs.nvidia.com/cuda/archive/10.0/ptx-writers-guide-to-interoperability/index.html#cuda-specific-dwarf
    // cuda-gdb requires DW_AT_address_class for all variables to be able to
    // correctly interpret address space of the variable address.
    const unsigned NVPTX_ADDR_global_space = 5;
    addUInt(*VariableDIE, dwarf::DW_AT_address_class, dwarf::DW_FORM_data1,
            NVPTXAddressSpace ? *NVPTXAddressSpace : NVPTX_ADDR_global_space);
  }
  if (Loc)
    addBlock(*VariableDIE, dwarf::DW_AT_location, DwarfExpr->finalize());

  if (DD->useAllLinkageNames())
    addLinkageName(*VariableDIE, GV->getLinkageName());

  if (addToAccelTable) {
    DD->addAccelName(*CUNode, GV->getName(), *VariableDIE);

    // If the linkage name is different than the name, go ahead and output
    // that as well into the name table.
    if (GV->getLinkageName() != "" && GV->getName() != GV->getLinkageName() &&
        DD->useAllLinkageNames())
      DD->addAccelName(*CUNode, GV->getLinkageName(), *VariableDIE);
  }
}

DIE *DwarfCompileUnit::getOrCreateCommonBlock(
    const DICommonBlock *CB, ArrayRef<GlobalExpr> GlobalExprs) {
  // Check for pre-existence.
  if (DIE *NDie = getDIE(CB))
    return NDie;
  DIE *ContextDIE = getOrCreateContextDIE(CB->getScope());
  DIE &NDie = createAndAddDIE(dwarf::DW_TAG_common_block, *ContextDIE, CB);
  StringRef Name = CB->getName().empty() ? "_BLNK_" : CB->getName();
  addString(NDie, dwarf::DW_AT_name, Name);
  addGlobalName(Name, NDie, CB->getScope());
  if (CB->getFile())
    addSourceLine(NDie, CB->getLineNo(), CB->getFile());
  if (DIGlobalVariable *V = CB->getDecl())
    getCU().addLocationAttribute(&NDie, V, GlobalExprs);
  return &NDie;
}

void DwarfCompileUnit::addRange(RangeSpan Range) {
  DD->insertSectionLabel(Range.Begin);

  auto *PrevCU = DD->getPrevCU();
  bool SameAsPrevCU = this == PrevCU;
  DD->setPrevCU(this);
  // If we have no current ranges just add the range and return, otherwise,
  // check the current section and CU against the previous section and CU we
  // emitted into and the subprogram was contained within. If these are the
  // same then extend our current range, otherwise add this as a new range.
  if (CURanges.empty() || !SameAsPrevCU ||
      (&CURanges.back().End->getSection() !=
       &Range.End->getSection())) {
    // Before a new range is added, always terminate the prior line table.
    if (PrevCU)
      DD->terminateLineTable(PrevCU);
    CURanges.push_back(Range);
    return;
  }

  CURanges.back().End = Range.End;
}

void DwarfCompileUnit::initStmtList() {
  if (CUNode->isDebugDirectivesOnly())
    return;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  if (DD->useSectionsAsReferences()) {
    LineTableStartSym = TLOF.getDwarfLineSection()->getBeginSymbol();
  } else {
    LineTableStartSym =
        Asm->OutStreamer->getDwarfLineTableSymbol(getUniqueID());
  }

  // DW_AT_stmt_list is a offset of line number information for this
  // compile unit in debug_line section. For split dwarf this is
  // left in the skeleton CU and so not included.
  // The line table entries are not always emitted in assembly, so it
  // is not okay to use line_table_start here.
      addSectionLabel(getUnitDie(), dwarf::DW_AT_stmt_list, LineTableStartSym,
                      TLOF.getDwarfLineSection()->getBeginSymbol());
}

void DwarfCompileUnit::applyStmtList(DIE &D) {
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  addSectionLabel(D, dwarf::DW_AT_stmt_list, LineTableStartSym,
                  TLOF.getDwarfLineSection()->getBeginSymbol());
}

void DwarfCompileUnit::attachLowHighPC(DIE &D, const MCSymbol *Begin,
                                       const MCSymbol *End) {
  assert(Begin && "Begin label should not be null!");
  assert(End && "End label should not be null!");
  assert(Begin->isDefined() && "Invalid starting label");
  assert(End->isDefined() && "Invalid end label");

  addLabelAddress(D, dwarf::DW_AT_low_pc, Begin);
  if (DD->getDwarfVersion() < 4)
    addLabelAddress(D, dwarf::DW_AT_high_pc, End);
  else
    addLabelDelta(D, dwarf::DW_AT_high_pc, End, Begin);
}

// Find DIE for the given subprogram and attach appropriate DW_AT_low_pc
// and DW_AT_high_pc attributes. If there are global variables in this
// scope then create and insert DIEs for these variables.
DIE &DwarfCompileUnit::updateSubprogramScopeDIE(const DISubprogram *SP) {
  DIE *SPDie = getOrCreateSubprogramDIE(SP, includeMinimalInlineScopes());

  SmallVector<RangeSpan, 2> BB_List;
  // If basic block sections are on, ranges for each basic block section has
  // to be emitted separately.
  for (const auto &R : Asm->MBBSectionRanges)
    BB_List.push_back({R.second.BeginLabel, R.second.EndLabel});

  attachRangesOrLowHighPC(*SPDie, BB_List);

  if (DD->useAppleExtensionAttributes() &&
      !DD->getCurrentFunction()->getTarget().Options.DisableFramePointerElim(
          *DD->getCurrentFunction()))
    addFlag(*SPDie, dwarf::DW_AT_APPLE_omit_frame_ptr);

  // Only include DW_AT_frame_base in full debug info
  if (!includeMinimalInlineScopes()) {
    const TargetFrameLowering *TFI = Asm->MF->getSubtarget().getFrameLowering();
    TargetFrameLowering::DwarfFrameBase FrameBase =
        TFI->getDwarfFrameBase(*Asm->MF);
    switch (FrameBase.Kind) {
    case TargetFrameLowering::DwarfFrameBase::Register: {
      if (Register::isPhysicalRegister(FrameBase.Location.Reg)) {
        MachineLocation Location(FrameBase.Location.Reg);
        addAddress(*SPDie, dwarf::DW_AT_frame_base, Location);
      }
      break;
    }
    case TargetFrameLowering::DwarfFrameBase::CFA: {
      DIELoc *Loc = new (DIEValueAllocator) DIELoc;
      addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_call_frame_cfa);
      addBlock(*SPDie, dwarf::DW_AT_frame_base, Loc);
      break;
    }
    case TargetFrameLowering::DwarfFrameBase::WasmFrameBase: {
      // FIXME: duplicated from Target/WebAssembly/WebAssembly.h
      // don't want to depend on target specific headers in this code?
      const unsigned TI_GLOBAL_RELOC = 3;
      if (FrameBase.Location.WasmLoc.Kind == TI_GLOBAL_RELOC) {
        // These need to be relocatable.
        assert(FrameBase.Location.WasmLoc.Index == 0);  // Only SP so far.
        auto SPSym = cast<MCSymbolWasm>(
          Asm->GetExternalSymbolSymbol("__stack_pointer"));
        // FIXME: this repeats what WebAssemblyMCInstLower::
        // GetExternalSymbolSymbol does, since if there's no code that
        // refers to this symbol, we have to set it here.
        SPSym->setType(wasm::WASM_SYMBOL_TYPE_GLOBAL);
        SPSym->setGlobalType(wasm::WasmGlobalType{
            uint8_t(Asm->getSubtargetInfo().getTargetTriple().getArch() ==
                            Triple::wasm64
                        ? wasm::WASM_TYPE_I64
                        : wasm::WASM_TYPE_I32),
            true});
        DIELoc *Loc = new (DIEValueAllocator) DIELoc;
        addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_WASM_location);
        addSInt(*Loc, dwarf::DW_FORM_sdata, TI_GLOBAL_RELOC);
        if (!isDwoUnit()) {
          addLabel(*Loc, dwarf::DW_FORM_data4, SPSym);
        } else {
          // FIXME: when writing dwo, we need to avoid relocations. Probably
          // the "right" solution is to treat globals the way func and data
          // symbols are (with entries in .debug_addr).
          // For now, since we only ever use index 0, this should work as-is.       
          addUInt(*Loc, dwarf::DW_FORM_data4, FrameBase.Location.WasmLoc.Index);
        }
        addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_stack_value);
        addBlock(*SPDie, dwarf::DW_AT_frame_base, Loc);
      } else {
        DIELoc *Loc = new (DIEValueAllocator) DIELoc;
        DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
        DIExpressionCursor Cursor({});
        DwarfExpr.addWasmLocation(FrameBase.Location.WasmLoc.Kind,
            FrameBase.Location.WasmLoc.Index);
        DwarfExpr.addExpression(std::move(Cursor));
        addBlock(*SPDie, dwarf::DW_AT_frame_base, DwarfExpr.finalize());
      }
      break;
    }
    }
  }

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_subprogram nodes.
  DD->addSubprogramNames(*CUNode, SP, *SPDie);

  return *SPDie;
}

// Construct a DIE for this scope.
void DwarfCompileUnit::constructScopeDIE(LexicalScope *Scope,
                                         DIE &ParentScopeDIE) {
  if (!Scope || !Scope->getScopeNode())
    return;

  auto *DS = Scope->getScopeNode();

  assert((Scope->getInlinedAt() || !isa<DISubprogram>(DS)) &&
         "Only handle inlined subprograms here, use "
         "constructSubprogramScopeDIE for non-inlined "
         "subprograms");

  // Emit inlined subprograms.
  if (Scope->getParent() && isa<DISubprogram>(DS)) {
    DIE *ScopeDIE = constructInlinedScopeDIE(Scope);
    if (!ScopeDIE)
      return;

    ParentScopeDIE.addChild(ScopeDIE);
    createAndAddScopeChildren(Scope, *ScopeDIE);
    return;
  }

  // Early exit when we know the scope DIE is going to be null.
  if (DD->isLexicalScopeDIENull(Scope))
    return;

  // Emit lexical blocks.
  DIE *ScopeDIE = constructLexicalScopeDIE(Scope);
  assert(ScopeDIE && "Scope DIE should not be null.");

  ParentScopeDIE.addChild(ScopeDIE);
  createAndAddScopeChildren(Scope, *ScopeDIE);
}

void DwarfCompileUnit::addScopeRangeList(DIE &ScopeDIE,
                                         SmallVector<RangeSpan, 2> Range) {

  HasRangeLists = true;

  // Add the range list to the set of ranges to be emitted.
  auto IndexAndList =
      (DD->getDwarfVersion() < 5 && Skeleton ? Skeleton->DU : DU)
          ->addRange(*(Skeleton ? Skeleton : this), std::move(Range));

  uint32_t Index = IndexAndList.first;
  auto &List = *IndexAndList.second;

  // Under fission, ranges are specified by constant offsets relative to the
  // CU's DW_AT_GNU_ranges_base.
  // FIXME: For DWARF v5, do not generate the DW_AT_ranges attribute under
  // fission until we support the forms using the .debug_addr section
  // (DW_RLE_startx_endx etc.).
  if (DD->getDwarfVersion() >= 5)
    addUInt(ScopeDIE, dwarf::DW_AT_ranges, dwarf::DW_FORM_rnglistx, Index);
  else {
    const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
    const MCSymbol *RangeSectionSym =
        TLOF.getDwarfRangesSection()->getBeginSymbol();
    if (isDwoUnit())
      addSectionDelta(ScopeDIE, dwarf::DW_AT_ranges, List.Label,
                      RangeSectionSym);
    else
      addSectionLabel(ScopeDIE, dwarf::DW_AT_ranges, List.Label,
                      RangeSectionSym);
  }
}

void DwarfCompileUnit::attachRangesOrLowHighPC(
    DIE &Die, SmallVector<RangeSpan, 2> Ranges) {
  assert(!Ranges.empty());
  if (!DD->useRangesSection() ||
      (Ranges.size() == 1 &&
       (!DD->alwaysUseRanges() ||
        DD->getSectionLabel(&Ranges.front().Begin->getSection()) ==
            Ranges.front().Begin))) {
    const RangeSpan &Front = Ranges.front();
    const RangeSpan &Back = Ranges.back();
    attachLowHighPC(Die, Front.Begin, Back.End);
  } else
    addScopeRangeList(Die, std::move(Ranges));
}

void DwarfCompileUnit::attachRangesOrLowHighPC(
    DIE &Die, const SmallVectorImpl<InsnRange> &Ranges) {
  SmallVector<RangeSpan, 2> List;
  List.reserve(Ranges.size());
  for (const InsnRange &R : Ranges) {
    auto *BeginLabel = DD->getLabelBeforeInsn(R.first);
    auto *EndLabel = DD->getLabelAfterInsn(R.second);

    const auto *BeginMBB = R.first->getParent();
    const auto *EndMBB = R.second->getParent();

    const auto *MBB = BeginMBB;
    // Basic block sections allows basic block subsets to be placed in unique
    // sections. For each section, the begin and end label must be added to the
    // list. If there is more than one range, debug ranges must be used.
    // Otherwise, low/high PC can be used.
    // FIXME: Debug Info Emission depends on block order and this assumes that
    // the order of blocks will be frozen beyond this point.
    do {
      if (MBB->sameSection(EndMBB) || MBB->isEndSection()) {
        auto MBBSectionRange = Asm->MBBSectionRanges[MBB->getSectionIDNum()];
        List.push_back(
            {MBB->sameSection(BeginMBB) ? BeginLabel
                                        : MBBSectionRange.BeginLabel,
             MBB->sameSection(EndMBB) ? EndLabel : MBBSectionRange.EndLabel});
      }
      if (MBB->sameSection(EndMBB))
        break;
      MBB = MBB->getNextNode();
    } while (true);
  }
  attachRangesOrLowHighPC(Die, std::move(List));
}

// This scope represents inlined body of a function. Construct DIE to
// represent this concrete inlined copy of the function.
DIE *DwarfCompileUnit::constructInlinedScopeDIE(LexicalScope *Scope) {
  assert(Scope->getScopeNode());
  auto *DS = Scope->getScopeNode();
  auto *InlinedSP = getDISubprogram(DS);
  // Find the subprogram's DwarfCompileUnit in the SPMap in case the subprogram
  // was inlined from another compile unit.
  DIE *OriginDIE = getAbstractSPDies()[InlinedSP];
  assert(OriginDIE && "Unable to find original DIE for an inlined subprogram.");

  auto ScopeDIE = DIE::get(DIEValueAllocator, dwarf::DW_TAG_inlined_subroutine);
  addDIEEntry(*ScopeDIE, dwarf::DW_AT_abstract_origin, *OriginDIE);

  attachRangesOrLowHighPC(*ScopeDIE, Scope->getRanges());

  // Add the call site information to the DIE.
  const DILocation *IA = Scope->getInlinedAt();
  addUInt(*ScopeDIE, dwarf::DW_AT_call_file, None,
          getOrCreateSourceID(IA->getFile()));
  addUInt(*ScopeDIE, dwarf::DW_AT_call_line, None, IA->getLine());
  if (IA->getColumn())
    addUInt(*ScopeDIE, dwarf::DW_AT_call_column, None, IA->getColumn());
  if (IA->getDiscriminator() && DD->getDwarfVersion() >= 4)
    addUInt(*ScopeDIE, dwarf::DW_AT_GNU_discriminator, None,
            IA->getDiscriminator());

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_inlined_subprogram nodes.
  DD->addSubprogramNames(*CUNode, InlinedSP, *ScopeDIE);

  return ScopeDIE;
}

// Construct new DW_TAG_lexical_block for this scope and attach
// DW_AT_low_pc/DW_AT_high_pc labels.
DIE *DwarfCompileUnit::constructLexicalScopeDIE(LexicalScope *Scope) {
  if (DD->isLexicalScopeDIENull(Scope))
    return nullptr;

  auto ScopeDIE = DIE::get(DIEValueAllocator, dwarf::DW_TAG_lexical_block);
  if (Scope->isAbstractScope())
    return ScopeDIE;

  attachRangesOrLowHighPC(*ScopeDIE, Scope->getRanges());

  return ScopeDIE;
}

/// constructVariableDIE - Construct a DIE for the given DbgVariable.
DIE *DwarfCompileUnit::constructVariableDIE(DbgVariable &DV, bool Abstract) {
  auto D = constructVariableDIEImpl(DV, Abstract);
  DV.setDIE(*D);
  return D;
}

DIE *DwarfCompileUnit::constructLabelDIE(DbgLabel &DL,
                                         const LexicalScope &Scope) {
  auto LabelDie = DIE::get(DIEValueAllocator, DL.getTag());
  insertDIE(DL.getLabel(), LabelDie);
  DL.setDIE(*LabelDie);

  if (Scope.isAbstractScope())
    applyLabelAttributes(DL, *LabelDie);

  return LabelDie;
}

DIE *DwarfCompileUnit::constructVariableDIEImpl(const DbgVariable &DV,
                                                bool Abstract) {
  // Define variable debug information entry.
  auto VariableDie = DIE::get(DIEValueAllocator, DV.getTag());
  insertDIE(DV.getVariable(), VariableDie);

  if (Abstract) {
    applyVariableAttributes(DV, *VariableDie);
    return VariableDie;
  }

  // Add variable address.

  unsigned Index = DV.getDebugLocListIndex();
  if (Index != ~0U) {
    addLocationList(*VariableDie, dwarf::DW_AT_location, Index);
    auto TagOffset = DV.getDebugLocListTagOffset();
    if (TagOffset)
      addUInt(*VariableDie, dwarf::DW_AT_LLVM_tag_offset, dwarf::DW_FORM_data1,
              *TagOffset);
    return VariableDie;
  }

  // Check if variable has a single location description.
  if (auto *DVal = DV.getValueLoc()) {
    if (!DVal->isVariadic()) {
      const DbgValueLocEntry *Entry = DVal->getLocEntries().begin();
      if (Entry->isLocation()) {
        addVariableAddress(DV, *VariableDie, Entry->getLoc());
      } else if (Entry->isInt()) {
        auto *Expr = DV.getSingleExpression();
        if (Expr && Expr->getNumElements()) {
          DIELoc *Loc = new (DIEValueAllocator) DIELoc;
          DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
          // If there is an expression, emit raw unsigned bytes.
          DwarfExpr.addFragmentOffset(Expr);
          DwarfExpr.addUnsignedConstant(Entry->getInt());
          DwarfExpr.addExpression(Expr);
          addBlock(*VariableDie, dwarf::DW_AT_location, DwarfExpr.finalize());
          if (DwarfExpr.TagOffset)
            addUInt(*VariableDie, dwarf::DW_AT_LLVM_tag_offset,
                    dwarf::DW_FORM_data1, *DwarfExpr.TagOffset);
        } else
          addConstantValue(*VariableDie, Entry->getInt(), DV.getType());
      } else if (Entry->isConstantFP()) {
        addConstantFPValue(*VariableDie, Entry->getConstantFP());
      } else if (Entry->isConstantInt()) {
        addConstantValue(*VariableDie, Entry->getConstantInt(), DV.getType());
      } else if (Entry->isTargetIndexLocation()) {
        DIELoc *Loc = new (DIEValueAllocator) DIELoc;
        DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
        const DIBasicType *BT = dyn_cast<DIBasicType>(
            static_cast<const Metadata *>(DV.getVariable()->getType()));
        DwarfDebug::emitDebugLocValue(*Asm, BT, *DVal, DwarfExpr);
        addBlock(*VariableDie, dwarf::DW_AT_location, DwarfExpr.finalize());
      }
      return VariableDie;
    }
    // If any of the location entries are registers with the value 0, then the
    // location is undefined.
    if (any_of(DVal->getLocEntries(), [](const DbgValueLocEntry &Entry) {
          return Entry.isLocation() && !Entry.getLoc().getReg();
        }))
      return VariableDie;
    const DIExpression *Expr = DV.getSingleExpression();
    assert(Expr && "Variadic Debug Value must have an Expression.");
    DIELoc *Loc = new (DIEValueAllocator) DIELoc;
    DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
    DwarfExpr.addFragmentOffset(Expr);
    DIExpressionCursor Cursor(Expr);
    const TargetRegisterInfo &TRI = *Asm->MF->getSubtarget().getRegisterInfo();

    auto AddEntry = [&](const DbgValueLocEntry &Entry,
                        DIExpressionCursor &Cursor) {
      if (Entry.isLocation()) {
        if (!DwarfExpr.addMachineRegExpression(TRI, Cursor,
                                               Entry.getLoc().getReg()))
          return false;
      } else if (Entry.isInt()) {
        // If there is an expression, emit raw unsigned bytes.
        DwarfExpr.addUnsignedConstant(Entry.getInt());
      } else if (Entry.isConstantFP()) {
        // DwarfExpression does not support arguments wider than 64 bits
        // (see PR52584).
        // TODO: Consider chunking expressions containing overly wide
        // arguments into separate pointer-sized fragment expressions.
        APInt RawBytes = Entry.getConstantFP()->getValueAPF().bitcastToAPInt();
        if (RawBytes.getBitWidth() > 64)
          return false;
        DwarfExpr.addUnsignedConstant(RawBytes.getZExtValue());
      } else if (Entry.isConstantInt()) {
        APInt RawBytes = Entry.getConstantInt()->getValue();
        if (RawBytes.getBitWidth() > 64)
          return false;
        DwarfExpr.addUnsignedConstant(RawBytes.getZExtValue());
      } else if (Entry.isTargetIndexLocation()) {
        TargetIndexLocation Loc = Entry.getTargetIndexLocation();
        // TODO TargetIndexLocation is a target-independent. Currently only the
        // WebAssembly-specific encoding is supported.
        assert(Asm->TM.getTargetTriple().isWasm());
        DwarfExpr.addWasmLocation(Loc.Index, static_cast<uint64_t>(Loc.Offset));
      } else {
        llvm_unreachable("Unsupported Entry type.");
      }
      return true;
    };

    if (!DwarfExpr.addExpression(
            std::move(Cursor),
            [&](unsigned Idx, DIExpressionCursor &Cursor) -> bool {
              return AddEntry(DVal->getLocEntries()[Idx], Cursor);
            }))
      return VariableDie;

    // Now attach the location information to the DIE.
    addBlock(*VariableDie, dwarf::DW_AT_location, DwarfExpr.finalize());
    if (DwarfExpr.TagOffset)
      addUInt(*VariableDie, dwarf::DW_AT_LLVM_tag_offset, dwarf::DW_FORM_data1,
              *DwarfExpr.TagOffset);

    return VariableDie;
  }

  // .. else use frame index.
  if (!DV.hasFrameIndexExprs())
    return VariableDie;

  Optional<unsigned> NVPTXAddressSpace;
  DIELoc *Loc = new (DIEValueAllocator) DIELoc;
  DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
  for (auto &Fragment : DV.getFrameIndexExprs()) {
    Register FrameReg;
    const DIExpression *Expr = Fragment.Expr;
    const TargetFrameLowering *TFI = Asm->MF->getSubtarget().getFrameLowering();
    StackOffset Offset =
        TFI->getFrameIndexReference(*Asm->MF, Fragment.FI, FrameReg);
    DwarfExpr.addFragmentOffset(Expr);

    auto *TRI = Asm->MF->getSubtarget().getRegisterInfo();
    SmallVector<uint64_t, 8> Ops;
    TRI->getOffsetOpcodes(Offset, Ops);

    // According to
    // https://docs.nvidia.com/cuda/archive/10.0/ptx-writers-guide-to-interoperability/index.html#cuda-specific-dwarf
    // cuda-gdb requires DW_AT_address_class for all variables to be able to
    // correctly interpret address space of the variable address.
    // Decode DW_OP_constu <DWARF Address Space> DW_OP_swap DW_OP_xderef
    // sequence for the NVPTX + gdb target.
    unsigned LocalNVPTXAddressSpace;
    if (Asm->TM.getTargetTriple().isNVPTX() && DD->tuneForGDB()) {
      const DIExpression *NewExpr =
          DIExpression::extractAddressClass(Expr, LocalNVPTXAddressSpace);
      if (NewExpr != Expr) {
        Expr = NewExpr;
        NVPTXAddressSpace = LocalNVPTXAddressSpace;
      }
    }
    if (Expr)
      Ops.append(Expr->elements_begin(), Expr->elements_end());
    DIExpressionCursor Cursor(Ops);
    DwarfExpr.setMemoryLocationKind();
    if (const MCSymbol *FrameSymbol = Asm->getFunctionFrameSymbol())
      addOpAddress(*Loc, FrameSymbol);
    else
      DwarfExpr.addMachineRegExpression(
          *Asm->MF->getSubtarget().getRegisterInfo(), Cursor, FrameReg);
    DwarfExpr.addExpression(std::move(Cursor));
  }
  if (Asm->TM.getTargetTriple().isNVPTX() && DD->tuneForGDB()) {
    // According to
    // https://docs.nvidia.com/cuda/archive/10.0/ptx-writers-guide-to-interoperability/index.html#cuda-specific-dwarf
    // cuda-gdb requires DW_AT_address_class for all variables to be able to
    // correctly interpret address space of the variable address.
    const unsigned NVPTX_ADDR_local_space = 6;
    addUInt(*VariableDie, dwarf::DW_AT_address_class, dwarf::DW_FORM_data1,
            NVPTXAddressSpace ? *NVPTXAddressSpace : NVPTX_ADDR_local_space);
  }
  addBlock(*VariableDie, dwarf::DW_AT_location, DwarfExpr.finalize());
  if (DwarfExpr.TagOffset)
    addUInt(*VariableDie, dwarf::DW_AT_LLVM_tag_offset, dwarf::DW_FORM_data1,
            *DwarfExpr.TagOffset);

  return VariableDie;
}

DIE *DwarfCompileUnit::constructVariableDIE(DbgVariable &DV,
                                            const LexicalScope &Scope,
                                            DIE *&ObjectPointer) {
  auto Var = constructVariableDIE(DV, Scope.isAbstractScope());
  if (DV.isObjectPointer())
    ObjectPointer = Var;
  return Var;
}

/// Return all DIVariables that appear in count: expressions.
static SmallVector<const DIVariable *, 2> dependencies(DbgVariable *Var) {
  SmallVector<const DIVariable *, 2> Result;
  auto *Array = dyn_cast<DICompositeType>(Var->getType());
  if (!Array || Array->getTag() != dwarf::DW_TAG_array_type)
    return Result;
  if (auto *DLVar = Array->getDataLocation())
    Result.push_back(DLVar);
  if (auto *AsVar = Array->getAssociated())
    Result.push_back(AsVar);
  if (auto *AlVar = Array->getAllocated())
    Result.push_back(AlVar);
  for (auto *El : Array->getElements()) {
    if (auto *Subrange = dyn_cast<DISubrange>(El)) {
      if (auto Count = Subrange->getCount())
        if (auto *Dependency = Count.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
      if (auto LB = Subrange->getLowerBound())
        if (auto *Dependency = LB.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
      if (auto UB = Subrange->getUpperBound())
        if (auto *Dependency = UB.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
      if (auto ST = Subrange->getStride())
        if (auto *Dependency = ST.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
    } else if (auto *GenericSubrange = dyn_cast<DIGenericSubrange>(El)) {
      if (auto Count = GenericSubrange->getCount())
        if (auto *Dependency = Count.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
      if (auto LB = GenericSubrange->getLowerBound())
        if (auto *Dependency = LB.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
      if (auto UB = GenericSubrange->getUpperBound())
        if (auto *Dependency = UB.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
      if (auto ST = GenericSubrange->getStride())
        if (auto *Dependency = ST.dyn_cast<DIVariable *>())
          Result.push_back(Dependency);
    }
  }
  return Result;
}

/// Sort local variables so that variables appearing inside of helper
/// expressions come first.
static SmallVector<DbgVariable *, 8>
sortLocalVars(SmallVectorImpl<DbgVariable *> &Input) {
  SmallVector<DbgVariable *, 8> Result;
  SmallVector<PointerIntPair<DbgVariable *, 1>, 8> WorkList;
  // Map back from a DIVariable to its containing DbgVariable.
  SmallDenseMap<const DILocalVariable *, DbgVariable *> DbgVar;
  // Set of DbgVariables in Result.
  SmallDenseSet<DbgVariable *, 8> Visited;
  // For cycle detection.
  SmallDenseSet<DbgVariable *, 8> Visiting;

  // Initialize the worklist and the DIVariable lookup table.
  for (auto Var : reverse(Input)) {
    DbgVar.insert({Var->getVariable(), Var});
    WorkList.push_back({Var, 0});
  }

  // Perform a stable topological sort by doing a DFS.
  while (!WorkList.empty()) {
    auto Item = WorkList.back();
    DbgVariable *Var = Item.getPointer();
    bool visitedAllDependencies = Item.getInt();
    WorkList.pop_back();

    assert(Var);

    // Already handled.
    if (Visited.count(Var))
      continue;

    // Add to Result if all dependencies are visited.
    if (visitedAllDependencies) {
      Visited.insert(Var);
      Result.push_back(Var);
      continue;
    }

    // Detect cycles.
    auto Res = Visiting.insert(Var);
    if (!Res.second) {
      assert(false && "dependency cycle in local variables");
      return Result;
    }

    // Push dependencies and this node onto the worklist, so that this node is
    // visited again after all of its dependencies are handled.
    WorkList.push_back({Var, 1});
    for (auto *Dependency : dependencies(Var)) {
      // Don't add dependency if it is in a different lexical scope or a global.
      if (const auto *Dep = dyn_cast<const DILocalVariable>(Dependency))
        if (DbgVariable *Var = DbgVar.lookup(Dep))
          WorkList.push_back({Var, 0});
    }
  }
  return Result;
}

DIE &DwarfCompileUnit::constructSubprogramScopeDIE(const DISubprogram *Sub,
                                                   LexicalScope *Scope) {
  DIE &ScopeDIE = updateSubprogramScopeDIE(Sub);

  if (Scope) {
    assert(!Scope->getInlinedAt());
    assert(!Scope->isAbstractScope());
    // Collect lexical scope children first.
    // ObjectPointer might be a local (non-argument) local variable if it's a
    // block's synthetic this pointer.
    if (DIE *ObjectPointer = createAndAddScopeChildren(Scope, ScopeDIE))
      addDIEEntry(ScopeDIE, dwarf::DW_AT_object_pointer, *ObjectPointer);
  }

  // If this is a variadic function, add an unspecified parameter.
  DITypeRefArray FnArgs = Sub->getType()->getTypeArray();

  // If we have a single element of null, it is a function that returns void.
  // If we have more than one elements and the last one is null, it is a
  // variadic function.
  if (FnArgs.size() > 1 && !FnArgs[FnArgs.size() - 1] &&
      !includeMinimalInlineScopes())
    ScopeDIE.addChild(
        DIE::get(DIEValueAllocator, dwarf::DW_TAG_unspecified_parameters));

  return ScopeDIE;
}

DIE *DwarfCompileUnit::createAndAddScopeChildren(LexicalScope *Scope,
                                                 DIE &ScopeDIE) {
  DIE *ObjectPointer = nullptr;

  // Emit function arguments (order is significant).
  auto Vars = DU->getScopeVariables().lookup(Scope);
  for (auto &DV : Vars.Args)
    ScopeDIE.addChild(constructVariableDIE(*DV.second, *Scope, ObjectPointer));

  // Emit local variables.
  auto Locals = sortLocalVars(Vars.Locals);
  for (DbgVariable *DV : Locals)
    ScopeDIE.addChild(constructVariableDIE(*DV, *Scope, ObjectPointer));

  // Emit imported entities (skipped in gmlt-like data).
  if (!includeMinimalInlineScopes()) {
    for (const auto *IE : ImportedEntities[Scope->getScopeNode()])
      ScopeDIE.addChild(constructImportedEntityDIE(cast<DIImportedEntity>(IE)));
  }

  // Emit labels.
  for (DbgLabel *DL : DU->getScopeLabels().lookup(Scope))
    ScopeDIE.addChild(constructLabelDIE(*DL, *Scope));

  // Emit inner lexical scopes.
  auto needToEmitLexicalScope = [this](LexicalScope *LS) {
    if (isa<DISubprogram>(LS->getScopeNode()))
      return true;
    auto Vars = DU->getScopeVariables().lookup(LS);
    if (!Vars.Args.empty() || !Vars.Locals.empty())
      return true;
    if (!includeMinimalInlineScopes() &&
        !ImportedEntities[LS->getScopeNode()].empty())
      return true;
    return false;
  };
  for (LexicalScope *LS : Scope->getChildren()) {
    // If the lexical block doesn't have non-scope children, skip
    // its emission and put its children directly to the parent scope.
    if (needToEmitLexicalScope(LS))
      constructScopeDIE(LS, ScopeDIE);
    else
      createAndAddScopeChildren(LS, ScopeDIE);
  }

  return ObjectPointer;
}

void DwarfCompileUnit::constructAbstractSubprogramScopeDIE(
    LexicalScope *Scope) {
  DIE *&AbsDef = getAbstractSPDies()[Scope->getScopeNode()];
  if (AbsDef)
    return;

  auto *SP = cast<DISubprogram>(Scope->getScopeNode());

  DIE *ContextDIE;
  DwarfCompileUnit *ContextCU = this;

  if (includeMinimalInlineScopes())
    ContextDIE = &getUnitDie();
  // Some of this is duplicated from DwarfUnit::getOrCreateSubprogramDIE, with
  // the important distinction that the debug node is not associated with the
  // DIE (since the debug node will be associated with the concrete DIE, if
  // any). It could be refactored to some common utility function.
  else if (auto *SPDecl = SP->getDeclaration()) {
    ContextDIE = &getUnitDie();
    getOrCreateSubprogramDIE(SPDecl);
  } else {
    ContextDIE = getOrCreateContextDIE(SP->getScope());
    // The scope may be shared with a subprogram that has already been
    // constructed in another CU, in which case we need to construct this
    // subprogram in the same CU.
    ContextCU = DD->lookupCU(ContextDIE->getUnitDie());
  }

  // Passing null as the associated node because the abstract definition
  // shouldn't be found by lookup.
  AbsDef = &ContextCU->createAndAddDIE(dwarf::DW_TAG_subprogram, *ContextDIE, nullptr);
  ContextCU->applySubprogramAttributesToDefinition(SP, *AbsDef);
  ContextCU->addSInt(*AbsDef, dwarf::DW_AT_inline,
                     DD->getDwarfVersion() <= 4 ? Optional<dwarf::Form>()
                                                : dwarf::DW_FORM_implicit_const,
                     dwarf::DW_INL_inlined);
  if (DIE *ObjectPointer = ContextCU->createAndAddScopeChildren(Scope, *AbsDef))
    ContextCU->addDIEEntry(*AbsDef, dwarf::DW_AT_object_pointer, *ObjectPointer);
}

bool DwarfCompileUnit::useGNUAnalogForDwarf5Feature() const {
  return DD->getDwarfVersion() == 4 && !DD->tuneForLLDB();
}

dwarf::Tag DwarfCompileUnit::getDwarf5OrGNUTag(dwarf::Tag Tag) const {
  if (!useGNUAnalogForDwarf5Feature())
    return Tag;
  switch (Tag) {
  case dwarf::DW_TAG_call_site:
    return dwarf::DW_TAG_GNU_call_site;
  case dwarf::DW_TAG_call_site_parameter:
    return dwarf::DW_TAG_GNU_call_site_parameter;
  default:
    llvm_unreachable("DWARF5 tag with no GNU analog");
  }
}

dwarf::Attribute
DwarfCompileUnit::getDwarf5OrGNUAttr(dwarf::Attribute Attr) const {
  if (!useGNUAnalogForDwarf5Feature())
    return Attr;
  switch (Attr) {
  case dwarf::DW_AT_call_all_calls:
    return dwarf::DW_AT_GNU_all_call_sites;
  case dwarf::DW_AT_call_target:
    return dwarf::DW_AT_GNU_call_site_target;
  case dwarf::DW_AT_call_origin:
    return dwarf::DW_AT_abstract_origin;
  case dwarf::DW_AT_call_return_pc:
    return dwarf::DW_AT_low_pc;
  case dwarf::DW_AT_call_value:
    return dwarf::DW_AT_GNU_call_site_value;
  case dwarf::DW_AT_call_tail_call:
    return dwarf::DW_AT_GNU_tail_call;
  default:
    llvm_unreachable("DWARF5 attribute with no GNU analog");
  }
}

dwarf::LocationAtom
DwarfCompileUnit::getDwarf5OrGNULocationAtom(dwarf::LocationAtom Loc) const {
  if (!useGNUAnalogForDwarf5Feature())
    return Loc;
  switch (Loc) {
  case dwarf::DW_OP_entry_value:
    return dwarf::DW_OP_GNU_entry_value;
  default:
    llvm_unreachable("DWARF5 location atom with no GNU analog");
  }
}

DIE &DwarfCompileUnit::constructCallSiteEntryDIE(DIE &ScopeDIE,
                                                 const DISubprogram *CalleeSP,
                                                 bool IsTail,
                                                 const MCSymbol *PCAddr,
                                                 const MCSymbol *CallAddr,
                                                 unsigned CallReg) {
  // Insert a call site entry DIE within ScopeDIE.
  DIE &CallSiteDIE = createAndAddDIE(getDwarf5OrGNUTag(dwarf::DW_TAG_call_site),
                                     ScopeDIE, nullptr);

  if (CallReg) {
    // Indirect call.
    addAddress(CallSiteDIE, getDwarf5OrGNUAttr(dwarf::DW_AT_call_target),
               MachineLocation(CallReg));
  } else {
    DIE *CalleeDIE = getOrCreateSubprogramDIE(CalleeSP);
    assert(CalleeDIE && "Could not create DIE for call site entry origin");
    addDIEEntry(CallSiteDIE, getDwarf5OrGNUAttr(dwarf::DW_AT_call_origin),
                *CalleeDIE);
  }

  if (IsTail) {
    // Attach DW_AT_call_tail_call to tail calls for standards compliance.
    addFlag(CallSiteDIE, getDwarf5OrGNUAttr(dwarf::DW_AT_call_tail_call));

    // Attach the address of the branch instruction to allow the debugger to
    // show where the tail call occurred. This attribute has no GNU analog.
    //
    // GDB works backwards from non-standard usage of DW_AT_low_pc (in DWARF4
    // mode -- equivalently, in DWARF5 mode, DW_AT_call_return_pc) at tail-call
    // site entries to figure out the PC of tail-calling branch instructions.
    // This means it doesn't need the compiler to emit DW_AT_call_pc, so we
    // don't emit it here.
    //
    // There's no need to tie non-GDB debuggers to this non-standardness, as it
    // adds unnecessary complexity to the debugger. For non-GDB debuggers, emit
    // the standard DW_AT_call_pc info.
    if (!useGNUAnalogForDwarf5Feature())
      addLabelAddress(CallSiteDIE, dwarf::DW_AT_call_pc, CallAddr);
  }

  // Attach the return PC to allow the debugger to disambiguate call paths
  // from one function to another.
  //
  // The return PC is only really needed when the call /isn't/ a tail call, but
  // GDB expects it in DWARF4 mode, even for tail calls (see the comment above
  // the DW_AT_call_pc emission logic for an explanation).
  if (!IsTail || useGNUAnalogForDwarf5Feature()) {
    assert(PCAddr && "Missing return PC information for a call");
    addLabelAddress(CallSiteDIE,
                    getDwarf5OrGNUAttr(dwarf::DW_AT_call_return_pc), PCAddr);
  }

  return CallSiteDIE;
}

void DwarfCompileUnit::constructCallSiteParmEntryDIEs(
    DIE &CallSiteDIE, SmallVector<DbgCallSiteParam, 4> &Params) {
  for (const auto &Param : Params) {
    unsigned Register = Param.getRegister();
    auto CallSiteDieParam =
        DIE::get(DIEValueAllocator,
                 getDwarf5OrGNUTag(dwarf::DW_TAG_call_site_parameter));
    insertDIE(CallSiteDieParam);
    addAddress(*CallSiteDieParam, dwarf::DW_AT_location,
               MachineLocation(Register));

    DIELoc *Loc = new (DIEValueAllocator) DIELoc;
    DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
    DwarfExpr.setCallSiteParamValueFlag();

    DwarfDebug::emitDebugLocValue(*Asm, nullptr, Param.getValue(), DwarfExpr);

    addBlock(*CallSiteDieParam, getDwarf5OrGNUAttr(dwarf::DW_AT_call_value),
             DwarfExpr.finalize());

    CallSiteDIE.addChild(CallSiteDieParam);
  }
}

DIE *DwarfCompileUnit::constructImportedEntityDIE(
    const DIImportedEntity *Module) {
  DIE *IMDie = DIE::get(DIEValueAllocator, (dwarf::Tag)Module->getTag());
  insertDIE(Module, IMDie);
  DIE *EntityDie;
  auto *Entity = Module->getEntity();
  if (auto *NS = dyn_cast<DINamespace>(Entity))
    EntityDie = getOrCreateNameSpace(NS);
  else if (auto *M = dyn_cast<DIModule>(Entity))
    EntityDie = getOrCreateModule(M);
  else if (auto *SP = dyn_cast<DISubprogram>(Entity))
    EntityDie = getOrCreateSubprogramDIE(SP);
  else if (auto *T = dyn_cast<DIType>(Entity))
    EntityDie = getOrCreateTypeDIE(T);
  else if (auto *GV = dyn_cast<DIGlobalVariable>(Entity))
    EntityDie = getOrCreateGlobalVariableDIE(GV, {});
  else
    EntityDie = getDIE(Entity);
  assert(EntityDie);
  addSourceLine(*IMDie, Module->getLine(), Module->getFile());
  addDIEEntry(*IMDie, dwarf::DW_AT_import, *EntityDie);
  StringRef Name = Module->getName();
  if (!Name.empty())
    addString(*IMDie, dwarf::DW_AT_name, Name);

  // This is for imported module with renamed entities (such as variables and
  // subprograms).
  DINodeArray Elements = Module->getElements();
  for (const auto *Element : Elements) {
    if (!Element)
      continue;
    IMDie->addChild(
        constructImportedEntityDIE(cast<DIImportedEntity>(Element)));
  }

  return IMDie;
}

void DwarfCompileUnit::finishSubprogramDefinition(const DISubprogram *SP) {
  DIE *D = getDIE(SP);
  if (DIE *AbsSPDIE = getAbstractSPDies().lookup(SP)) {
    if (D)
      // If this subprogram has an abstract definition, reference that
      addDIEEntry(*D, dwarf::DW_AT_abstract_origin, *AbsSPDIE);
  } else {
    assert(D || includeMinimalInlineScopes());
    if (D)
      // And attach the attributes
      applySubprogramAttributesToDefinition(SP, *D);
  }
}

void DwarfCompileUnit::finishEntityDefinition(const DbgEntity *Entity) {
  DbgEntity *AbsEntity = getExistingAbstractEntity(Entity->getEntity());

  auto *Die = Entity->getDIE();
  /// Label may be used to generate DW_AT_low_pc, so put it outside
  /// if/else block.
  const DbgLabel *Label = nullptr;
  if (AbsEntity && AbsEntity->getDIE()) {
    addDIEEntry(*Die, dwarf::DW_AT_abstract_origin, *AbsEntity->getDIE());
    Label = dyn_cast<const DbgLabel>(Entity);
  } else {
    if (const DbgVariable *Var = dyn_cast<const DbgVariable>(Entity))
      applyVariableAttributes(*Var, *Die);
    else if ((Label = dyn_cast<const DbgLabel>(Entity)))
      applyLabelAttributes(*Label, *Die);
    else
      llvm_unreachable("DbgEntity must be DbgVariable or DbgLabel.");
  }

  if (Label)
    if (const auto *Sym = Label->getSymbol())
      addLabelAddress(*Die, dwarf::DW_AT_low_pc, Sym);
}

DbgEntity *DwarfCompileUnit::getExistingAbstractEntity(const DINode *Node) {
  auto &AbstractEntities = getAbstractEntities();
  auto I = AbstractEntities.find(Node);
  if (I != AbstractEntities.end())
    return I->second.get();
  return nullptr;
}

void DwarfCompileUnit::createAbstractEntity(const DINode *Node,
                                            LexicalScope *Scope) {
  assert(Scope && Scope->isAbstractScope());
  auto &Entity = getAbstractEntities()[Node];
  if (isa<const DILocalVariable>(Node)) {
    Entity = std::make_unique<DbgVariable>(
                        cast<const DILocalVariable>(Node), nullptr /* IA */);;
    DU->addScopeVariable(Scope, cast<DbgVariable>(Entity.get()));
  } else if (isa<const DILabel>(Node)) {
    Entity = std::make_unique<DbgLabel>(
                        cast<const DILabel>(Node), nullptr /* IA */);
    DU->addScopeLabel(Scope, cast<DbgLabel>(Entity.get()));
  }
}

void DwarfCompileUnit::emitHeader(bool UseOffsets) {
  // Don't bother labeling the .dwo unit, as its offset isn't used.
  if (!Skeleton && !DD->useSectionsAsReferences()) {
    LabelBegin = Asm->createTempSymbol("cu_begin");
    Asm->OutStreamer->emitLabel(LabelBegin);
  }

  dwarf::UnitType UT = Skeleton ? dwarf::DW_UT_split_compile
                                : DD->useSplitDwarf() ? dwarf::DW_UT_skeleton
                                                      : dwarf::DW_UT_compile;
  DwarfUnit::emitCommonHeader(UseOffsets, UT);
  if (DD->getDwarfVersion() >= 5 && UT != dwarf::DW_UT_compile)
    Asm->emitInt64(getDWOId());
}

bool DwarfCompileUnit::hasDwarfPubSections() const {
  switch (CUNode->getNameTableKind()) {
  case DICompileUnit::DebugNameTableKind::None:
    return false;
    // Opting in to GNU Pubnames/types overrides the default to ensure these are
    // generated for things like Gold's gdb_index generation.
  case DICompileUnit::DebugNameTableKind::GNU:
    return true;
  case DICompileUnit::DebugNameTableKind::Default:
    return DD->tuneForGDB() && !includeMinimalInlineScopes() &&
           !CUNode->isDebugDirectivesOnly() &&
           DD->getAccelTableKind() != AccelTableKind::Apple &&
           DD->getDwarfVersion() < 5;
  }
  llvm_unreachable("Unhandled DICompileUnit::DebugNameTableKind enum");
}

/// addGlobalName - Add a new global name to the compile unit.
void DwarfCompileUnit::addGlobalName(StringRef Name, const DIE &Die,
                                     const DIScope *Context) {
  if (!hasDwarfPubSections())
    return;
  std::string FullName = getParentContextString(Context) + Name.str();
  GlobalNames[FullName] = &Die;
}

void DwarfCompileUnit::addGlobalNameForTypeUnit(StringRef Name,
                                                const DIScope *Context) {
  if (!hasDwarfPubSections())
    return;
  std::string FullName = getParentContextString(Context) + Name.str();
  // Insert, allowing the entry to remain as-is if it's already present
  // This way the CU-level type DIE is preferred over the "can't describe this
  // type as a unit offset because it's not really in the CU at all, it's only
  // in a type unit"
  GlobalNames.insert(std::make_pair(std::move(FullName), &getUnitDie()));
}

/// Add a new global type to the unit.
void DwarfCompileUnit::addGlobalType(const DIType *Ty, const DIE &Die,
                                     const DIScope *Context) {
  if (!hasDwarfPubSections())
    return;
  std::string FullName = getParentContextString(Context) + Ty->getName().str();
  GlobalTypes[FullName] = &Die;
}

void DwarfCompileUnit::addGlobalTypeUnitType(const DIType *Ty,
                                             const DIScope *Context) {
  if (!hasDwarfPubSections())
    return;
  std::string FullName = getParentContextString(Context) + Ty->getName().str();
  // Insert, allowing the entry to remain as-is if it's already present
  // This way the CU-level type DIE is preferred over the "can't describe this
  // type as a unit offset because it's not really in the CU at all, it's only
  // in a type unit"
  GlobalTypes.insert(std::make_pair(std::move(FullName), &getUnitDie()));
}

void DwarfCompileUnit::addVariableAddress(const DbgVariable &DV, DIE &Die,
                                          MachineLocation Location) {
  if (DV.hasComplexAddress())
    addComplexAddress(DV, Die, dwarf::DW_AT_location, Location);
  else
    addAddress(Die, dwarf::DW_AT_location, Location);
}

/// Add an address attribute to a die based on the location provided.
void DwarfCompileUnit::addAddress(DIE &Die, dwarf::Attribute Attribute,
                                  const MachineLocation &Location) {
  DIELoc *Loc = new (DIEValueAllocator) DIELoc;
  DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
  if (Location.isIndirect())
    DwarfExpr.setMemoryLocationKind();

  DIExpressionCursor Cursor({});
  const TargetRegisterInfo &TRI = *Asm->MF->getSubtarget().getRegisterInfo();
  if (!DwarfExpr.addMachineRegExpression(TRI, Cursor, Location.getReg()))
    return;
  DwarfExpr.addExpression(std::move(Cursor));

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, DwarfExpr.finalize());

  if (DwarfExpr.TagOffset)
    addUInt(Die, dwarf::DW_AT_LLVM_tag_offset, dwarf::DW_FORM_data1,
            *DwarfExpr.TagOffset);
}

/// Start with the address based on the location provided, and generate the
/// DWARF information necessary to find the actual variable given the extra
/// address information encoded in the DbgVariable, starting from the starting
/// location.  Add the DWARF information to the die.
void DwarfCompileUnit::addComplexAddress(const DbgVariable &DV, DIE &Die,
                                         dwarf::Attribute Attribute,
                                         const MachineLocation &Location) {
  DIELoc *Loc = new (DIEValueAllocator) DIELoc;
  DIEDwarfExpression DwarfExpr(*Asm, *this, *Loc);
  const DIExpression *DIExpr = DV.getSingleExpression();
  DwarfExpr.addFragmentOffset(DIExpr);
  DwarfExpr.setLocation(Location, DIExpr);

  DIExpressionCursor Cursor(DIExpr);

  if (DIExpr->isEntryValue())
    DwarfExpr.beginEntryValueExpression(Cursor);

  const TargetRegisterInfo &TRI = *Asm->MF->getSubtarget().getRegisterInfo();
  if (!DwarfExpr.addMachineRegExpression(TRI, Cursor, Location.getReg()))
    return;
  DwarfExpr.addExpression(std::move(Cursor));

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, DwarfExpr.finalize());

  if (DwarfExpr.TagOffset)
    addUInt(Die, dwarf::DW_AT_LLVM_tag_offset, dwarf::DW_FORM_data1,
            *DwarfExpr.TagOffset);
}

/// Add a Dwarf loclistptr attribute data and value.
void DwarfCompileUnit::addLocationList(DIE &Die, dwarf::Attribute Attribute,
                                       unsigned Index) {
  dwarf::Form Form = (DD->getDwarfVersion() >= 5)
                         ? dwarf::DW_FORM_loclistx
                         : DD->getDwarfSectionOffsetForm();
  addAttribute(Die, Attribute, Form, DIELocList(Index));
}

void DwarfCompileUnit::applyVariableAttributes(const DbgVariable &Var,
                                               DIE &VariableDie) {
  StringRef Name = Var.getName();
  if (!Name.empty())
    addString(VariableDie, dwarf::DW_AT_name, Name);
  const auto *DIVar = Var.getVariable();
  if (DIVar) {
    if (uint32_t AlignInBytes = DIVar->getAlignInBytes())
      addUInt(VariableDie, dwarf::DW_AT_alignment, dwarf::DW_FORM_udata,
              AlignInBytes);
    addAnnotation(VariableDie, DIVar->getAnnotations());
  }

  addSourceLine(VariableDie, DIVar);
  addType(VariableDie, Var.getType());
  if (Var.isArtificial())
    addFlag(VariableDie, dwarf::DW_AT_artificial);
}

void DwarfCompileUnit::applyLabelAttributes(const DbgLabel &Label,
                                            DIE &LabelDie) {
  StringRef Name = Label.getName();
  if (!Name.empty())
    addString(LabelDie, dwarf::DW_AT_name, Name);
  const auto *DILabel = Label.getLabel();
  addSourceLine(LabelDie, DILabel);
}

/// Add a Dwarf expression attribute data and value.
void DwarfCompileUnit::addExpr(DIELoc &Die, dwarf::Form Form,
                               const MCExpr *Expr) {
  addAttribute(Die, (dwarf::Attribute)0, Form, DIEExpr(Expr));
}

void DwarfCompileUnit::applySubprogramAttributesToDefinition(
    const DISubprogram *SP, DIE &SPDie) {
  auto *SPDecl = SP->getDeclaration();
  auto *Context = SPDecl ? SPDecl->getScope() : SP->getScope();
  applySubprogramAttributes(SP, SPDie, includeMinimalInlineScopes());
  addGlobalName(SP->getName(), SPDie, Context);
}

bool DwarfCompileUnit::isDwoUnit() const {
  return DD->useSplitDwarf() && Skeleton;
}

void DwarfCompileUnit::finishNonUnitTypeDIE(DIE& D, const DICompositeType *CTy) {
  constructTypeDIE(D, CTy);
}

bool DwarfCompileUnit::includeMinimalInlineScopes() const {
  return getCUNode()->getEmissionKind() == DICompileUnit::LineTablesOnly ||
         (DD->useSplitDwarf() && !Skeleton);
}

void DwarfCompileUnit::addAddrTableBase() {
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  MCSymbol *Label = DD->getAddressPool().getLabel();
  addSectionLabel(getUnitDie(),
                  DD->getDwarfVersion() >= 5 ? dwarf::DW_AT_addr_base
                                             : dwarf::DW_AT_GNU_addr_base,
                  Label, TLOF.getDwarfAddrSection()->getBeginSymbol());
}

void DwarfCompileUnit::addBaseTypeRef(DIEValueList &Die, int64_t Idx) {
  addAttribute(Die, (dwarf::Attribute)0, dwarf::DW_FORM_udata,
               new (DIEValueAllocator) DIEBaseTypeRef(this, Idx));
}

void DwarfCompileUnit::createBaseTypeDIEs() {
  // Insert the base_type DIEs directly after the CU so that their offsets will
  // fit in the fixed size ULEB128 used inside the location expressions.
  // Maintain order by iterating backwards and inserting to the front of CU
  // child list.
  for (auto &Btr : reverse(ExprRefedBaseTypes)) {
    DIE &Die = getUnitDie().addChildFront(
      DIE::get(DIEValueAllocator, dwarf::DW_TAG_base_type));
    SmallString<32> Str;
    addString(Die, dwarf::DW_AT_name,
              Twine(dwarf::AttributeEncodingString(Btr.Encoding) +
                    "_" + Twine(Btr.BitSize)).toStringRef(Str));
    addUInt(Die, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1, Btr.Encoding);
    addUInt(Die, dwarf::DW_AT_byte_size, None, Btr.BitSize / 8);

    Btr.Die = &Die;
  }
}
