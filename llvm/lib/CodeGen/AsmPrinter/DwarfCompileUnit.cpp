#include "DwarfCompileUnit.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {

DwarfCompileUnit::DwarfCompileUnit(unsigned UID, DICompileUnit Node,
                                   AsmPrinter *A, DwarfDebug *DW,
                                   DwarfFile *DWU)
    : DwarfUnit(UID, dwarf::DW_TAG_compile_unit, Node, A, DW, DWU) {
  insertDIE(Node, &getUnitDie());
}

/// addLabelAddress - Add a dwarf label attribute data and value using
/// DW_FORM_addr or DW_FORM_GNU_addr_index.
///
void DwarfCompileUnit::addLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                                       const MCSymbol *Label) {

  // Don't use the address pool in non-fission or in the skeleton unit itself.
  // FIXME: Once GDB supports this, it's probably worthwhile using the address
  // pool from the skeleton - maybe even in non-fission (possibly fewer
  // relocations by sharing them in the pool, but we have other ideas about how
  // to reduce the number of relocations as well/instead).
  if (!DD->useSplitDwarf() || !Skeleton)
    return addLocalLabelAddress(Die, Attribute, Label);

  if (Label)
    DD->addArangeLabel(SymbolCU(this, Label));

  unsigned idx = DD->getAddressPool().getIndex(Label);
  DIEValue *Value = new (DIEValueAllocator) DIEInteger(idx);
  Die.addValue(Attribute, dwarf::DW_FORM_GNU_addr_index, Value);
}

void DwarfCompileUnit::addLocalLabelAddress(DIE &Die,
                                            dwarf::Attribute Attribute,
                                            const MCSymbol *Label) {
  if (Label)
    DD->addArangeLabel(SymbolCU(this, Label));

  Die.addValue(Attribute, dwarf::DW_FORM_addr,
               Label ? (DIEValue *)new (DIEValueAllocator) DIELabel(Label)
                     : new (DIEValueAllocator) DIEInteger(0));
}

unsigned DwarfCompileUnit::getOrCreateSourceID(StringRef FileName,
                                               StringRef DirName) {
  // If we print assembly, we can't separate .file entries according to
  // compile units. Thus all files will belong to the default compile unit.

  // FIXME: add a better feature test than hasRawTextSupport. Even better,
  // extend .file to support this.
  return Asm->OutStreamer.EmitDwarfFileDirective(
      0, DirName, FileName,
      Asm->OutStreamer.hasRawTextSupport() ? 0 : getUniqueID());
}

// Return const expression if value is a GEP to access merged global
// constant. e.g.
// i8* getelementptr ({ i8, i8, i8, i8 }* @_MergedGlobals, i32 0, i32 0)
static const ConstantExpr *getMergedGlobalExpr(const Value *V) {
  const ConstantExpr *CE = dyn_cast_or_null<ConstantExpr>(V);
  if (!CE || CE->getNumOperands() != 3 ||
      CE->getOpcode() != Instruction::GetElementPtr)
    return nullptr;

  // First operand points to a global struct.
  Value *Ptr = CE->getOperand(0);
  if (!isa<GlobalValue>(Ptr) ||
      !isa<StructType>(cast<PointerType>(Ptr->getType())->getElementType()))
    return nullptr;

  // Second operand is zero.
  const ConstantInt *CI = dyn_cast_or_null<ConstantInt>(CE->getOperand(1));
  if (!CI || !CI->isZero())
    return nullptr;

  // Third operand is offset.
  if (!isa<ConstantInt>(CE->getOperand(2)))
    return nullptr;

  return CE;
}

/// getOrCreateGlobalVariableDIE - get or create global variable DIE.
DIE *DwarfCompileUnit::getOrCreateGlobalVariableDIE(DIGlobalVariable GV) {
  // Check for pre-existence.
  if (DIE *Die = getDIE(GV))
    return Die;

  assert(GV.isGlobalVariable());

  DIScope GVContext = GV.getContext();
  DIType GTy = DD->resolve(GV.getType());

  // If this is a static data member definition, some attributes belong
  // to the declaration DIE.
  DIE *VariableDIE = nullptr;
  bool IsStaticMember = false;
  DIDerivedType SDMDecl = GV.getStaticDataMemberDeclaration();
  if (SDMDecl.Verify()) {
    assert(SDMDecl.isStaticMember() && "Expected static member decl");
    // We need the declaration DIE that is in the static member's class.
    VariableDIE = getOrCreateStaticMemberDIE(SDMDecl);
    IsStaticMember = true;
  }

  // If this is not a static data member definition, create the variable
  // DIE and add the initial set of attributes to it.
  if (!VariableDIE) {
    // Construct the context before querying for the existence of the DIE in
    // case such construction creates the DIE.
    DIE *ContextDIE = getOrCreateContextDIE(GVContext);

    // Add to map.
    VariableDIE = &createAndAddDIE(GV.getTag(), *ContextDIE, GV);

    // Add name and type.
    addString(*VariableDIE, dwarf::DW_AT_name, GV.getDisplayName());
    addType(*VariableDIE, GTy);

    // Add scoping info.
    if (!GV.isLocalToUnit())
      addFlag(*VariableDIE, dwarf::DW_AT_external);

    // Add line number info.
    addSourceLine(*VariableDIE, GV);
  }

  // Add location.
  bool addToAccelTable = false;
  DIE *VariableSpecDIE = nullptr;
  bool isGlobalVariable = GV.getGlobal() != nullptr;
  if (isGlobalVariable) {
    addToAccelTable = true;
    DIELoc *Loc = new (DIEValueAllocator) DIELoc();
    const MCSymbol *Sym = Asm->getSymbol(GV.getGlobal());
    if (GV.getGlobal()->isThreadLocal()) {
      // FIXME: Make this work with -gsplit-dwarf.
      unsigned PointerSize = Asm->getDataLayout().getPointerSize();
      assert((PointerSize == 4 || PointerSize == 8) &&
             "Add support for other sizes if necessary");
      // Based on GCC's support for TLS:
      if (!DD->useSplitDwarf()) {
        // 1) Start with a constNu of the appropriate pointer size
        addUInt(*Loc, dwarf::DW_FORM_data1,
                PointerSize == 4 ? dwarf::DW_OP_const4u : dwarf::DW_OP_const8u);
        // 2) containing the (relocated) offset of the TLS variable
        //    within the module's TLS block.
        addExpr(*Loc, dwarf::DW_FORM_udata,
                Asm->getObjFileLowering().getDebugThreadLocalSymbol(Sym));
      } else {
        addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_GNU_const_index);
        addUInt(*Loc, dwarf::DW_FORM_udata,
                DD->getAddressPool().getIndex(Sym, /* TLS */ true));
      }
      // 3) followed by a custom OP to make the debugger do a TLS lookup.
      addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_GNU_push_tls_address);
    } else {
      DD->addArangeLabel(SymbolCU(this, Sym));
      addOpAddress(*Loc, Sym);
    }
    // A static member's declaration is already flagged as such.
    if (!SDMDecl.Verify() && !GV.isDefinition())
      addFlag(*VariableDIE, dwarf::DW_AT_declaration);
    // Do not create specification DIE if context is either compile unit
    // or a subprogram.
    if (GVContext && GV.isDefinition() && !GVContext.isCompileUnit() &&
        !GVContext.isFile() && !DD->isSubprogramContext(GVContext)) {
      // Create specification DIE.
      VariableSpecDIE = &createAndAddDIE(dwarf::DW_TAG_variable, UnitDie);
      addDIEEntry(*VariableSpecDIE, dwarf::DW_AT_specification, *VariableDIE);
      addBlock(*VariableSpecDIE, dwarf::DW_AT_location, Loc);
      // A static member's declaration is already flagged as such.
      if (!SDMDecl.Verify())
        addFlag(*VariableDIE, dwarf::DW_AT_declaration);
    } else {
      addBlock(*VariableDIE, dwarf::DW_AT_location, Loc);
    }
    // Add the linkage name.
    StringRef LinkageName = GV.getLinkageName();
    if (!LinkageName.empty())
      // From DWARF4: DIEs to which DW_AT_linkage_name may apply include:
      // TAG_common_block, TAG_constant, TAG_entry_point, TAG_subprogram and
      // TAG_variable.
      addString(IsStaticMember && VariableSpecDIE ? *VariableSpecDIE
                                                  : *VariableDIE,
                DD->getDwarfVersion() >= 4 ? dwarf::DW_AT_linkage_name
                                           : dwarf::DW_AT_MIPS_linkage_name,
                GlobalValue::getRealLinkageName(LinkageName));
  } else if (const ConstantInt *CI =
                 dyn_cast_or_null<ConstantInt>(GV.getConstant())) {
    // AT_const_value was added when the static member was created. To avoid
    // emitting AT_const_value multiple times, we only add AT_const_value when
    // it is not a static member.
    if (!IsStaticMember)
      addConstantValue(*VariableDIE, CI, GTy);
  } else if (const ConstantExpr *CE = getMergedGlobalExpr(GV.getConstant())) {
    addToAccelTable = true;
    // GV is a merged global.
    DIELoc *Loc = new (DIEValueAllocator) DIELoc();
    Value *Ptr = CE->getOperand(0);
    MCSymbol *Sym = Asm->getSymbol(cast<GlobalValue>(Ptr));
    DD->addArangeLabel(SymbolCU(this, Sym));
    addOpAddress(*Loc, Sym);
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    SmallVector<Value *, 3> Idx(CE->op_begin() + 1, CE->op_end());
    addUInt(*Loc, dwarf::DW_FORM_udata,
            Asm->getDataLayout().getIndexedOffset(Ptr->getType(), Idx));
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);
    addBlock(*VariableDIE, dwarf::DW_AT_location, Loc);
  }

  DIE *ResultDIE = VariableSpecDIE ? VariableSpecDIE : VariableDIE;

  if (addToAccelTable) {
    DD->addAccelName(GV.getName(), *ResultDIE);

    // If the linkage name is different than the name, go ahead and output
    // that as well into the name table.
    if (GV.getLinkageName() != "" && GV.getName() != GV.getLinkageName())
      DD->addAccelName(GV.getLinkageName(), *ResultDIE);
  }

  addGlobalName(GV.getName(), *ResultDIE, GV.getContext());
  return ResultDIE;
}

void DwarfCompileUnit::addRange(RangeSpan Range) {
  bool SameAsPrevCU = this == DD->getPrevCU();
  DD->setPrevCU(this);
  // If we have no current ranges just add the range and return, otherwise,
  // check the current section and CU against the previous section and CU we
  // emitted into and the subprogram was contained within. If these are the
  // same then extend our current range, otherwise add this as a new range.
  if (CURanges.empty() || !SameAsPrevCU ||
      (&CURanges.back().getEnd()->getSection() !=
       &Range.getEnd()->getSection())) {
    CURanges.push_back(Range);
    return;
  }

  CURanges.back().setEnd(Range.getEnd());
}

void DwarfCompileUnit::initStmtList(MCSymbol *DwarfLineSectionSym) {
  // Define start line table label for each Compile Unit.
  MCSymbol *LineTableStartSym =
      Asm->OutStreamer.getDwarfLineTableSymbol(getUniqueID());

  stmtListIndex = UnitDie.getValues().size();

  // DW_AT_stmt_list is a offset of line number information for this
  // compile unit in debug_line section. For split dwarf this is
  // left in the skeleton CU and so not included.
  // The line table entries are not always emitted in assembly, so it
  // is not okay to use line_table_start here.
  if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    addSectionLabel(UnitDie, dwarf::DW_AT_stmt_list, LineTableStartSym);
  else
    addSectionDelta(UnitDie, dwarf::DW_AT_stmt_list, LineTableStartSym,
                    DwarfLineSectionSym);
}

void DwarfCompileUnit::applyStmtList(DIE &D) {
  D.addValue(dwarf::DW_AT_stmt_list,
             UnitDie.getAbbrev().getData()[stmtListIndex].getForm(),
             UnitDie.getValues()[stmtListIndex]);
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
DIE &DwarfCompileUnit::updateSubprogramScopeDIE(DISubprogram SP) {
  DIE *SPDie = getOrCreateSubprogramDIE(SP);

  attachLowHighPC(*SPDie, DD->getFunctionBeginSym(), DD->getFunctionEndSym());
  if (!DD->getCurrentFunction()->getTarget().Options.DisableFramePointerElim(
          *DD->getCurrentFunction()))
    addFlag(*SPDie, dwarf::DW_AT_APPLE_omit_frame_ptr);

  // Only include DW_AT_frame_base in full debug info
  if (getCUNode().getEmissionKind() != DIBuilder::LineTablesOnly) {
    const TargetRegisterInfo *RI =
        Asm->TM.getSubtargetImpl()->getRegisterInfo();
    MachineLocation Location(RI->getFrameRegister(*Asm->MF));
    addAddress(*SPDie, dwarf::DW_AT_frame_base, Location);
  }

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_subprogram nodes.
  DD->addSubprogramNames(SP, *SPDie);

  return *SPDie;
}

// Construct a DIE for this scope.
void DwarfCompileUnit::constructScopeDIE(
    LexicalScope *Scope, SmallVectorImpl<std::unique_ptr<DIE>> &FinalChildren) {
  if (!Scope || !Scope->getScopeNode())
    return;

  DIScope DS(Scope->getScopeNode());

  assert((Scope->getInlinedAt() || !DS.isSubprogram()) &&
         "Only handle inlined subprograms here, use "
         "constructSubprogramScopeDIE for non-inlined "
         "subprograms");

  SmallVector<std::unique_ptr<DIE>, 8> Children;

  // We try to create the scope DIE first, then the children DIEs. This will
  // avoid creating un-used children then removing them later when we find out
  // the scope DIE is null.
  std::unique_ptr<DIE> ScopeDIE;
  if (Scope->getParent() && DS.isSubprogram()) {
    ScopeDIE = DD->constructInlinedScopeDIE(*this, Scope);
    if (!ScopeDIE)
      return;
    // We create children when the scope DIE is not null.
    DD->createScopeChildrenDIE(*this, Scope, Children);
  } else {
    // Early exit when we know the scope DIE is going to be null.
    if (DD->isLexicalScopeDIENull(Scope))
      return;

    unsigned ChildScopeCount;

    // We create children here when we know the scope DIE is not going to be
    // null and the children will be added to the scope DIE.
    DD->createScopeChildrenDIE(*this, Scope, Children, &ChildScopeCount);

    // There is no need to emit empty lexical block DIE.
    for (const auto &E : DD->findImportedEntitiesForScope(DS))
      Children.push_back(
          constructImportedEntityDIE(DIImportedEntity(E.second)));
    // If there are only other scopes as children, put them directly in the
    // parent instead, as this scope would serve no purpose.
    if (Children.size() == ChildScopeCount) {
      FinalChildren.insert(FinalChildren.end(),
                           std::make_move_iterator(Children.begin()),
                           std::make_move_iterator(Children.end()));
      return;
    }
    ScopeDIE = DD->constructLexicalScopeDIE(*this, Scope);
    assert(ScopeDIE && "Scope DIE should not be null.");
  }

  // Add children
  for (auto &I : Children)
    ScopeDIE->addChild(std::move(I));

  FinalChildren.push_back(std::move(ScopeDIE));
}

} // end llvm namespace
