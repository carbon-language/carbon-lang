//===-- llvm/CodeGen/DwarfDebug.cpp - Dwarf Debug Framework ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dwarfdebug"
#include "DwarfDebug.h"
#include "DIE.h"
#include "DIEHash.h"
#include "DwarfAccelTable.h"
#include "DwarfUnit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/DIBuilder.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

static cl::opt<bool>
DisableDebugInfoPrinting("disable-debug-info-print", cl::Hidden,
                         cl::desc("Disable debug info printing"));

static cl::opt<bool> UnknownLocations(
    "use-unknown-locations", cl::Hidden,
    cl::desc("Make an absence of debug location information explicit."),
    cl::init(false));

static cl::opt<bool> GenerateCUHash("generate-cu-hash", cl::Hidden,
                                    cl::desc("Add the CU hash as the dwo_id."),
                                    cl::init(false));

static cl::opt<bool>
GenerateGnuPubSections("generate-gnu-dwarf-pub-sections", cl::Hidden,
                       cl::desc("Generate GNU-style pubnames and pubtypes"),
                       cl::init(false));

static cl::opt<bool> GenerateARangeSection("generate-arange-section",
                                           cl::Hidden,
                                           cl::desc("Generate dwarf aranges"),
                                           cl::init(false));

namespace {
enum DefaultOnOff { Default, Enable, Disable };
}

static cl::opt<DefaultOnOff>
DwarfAccelTables("dwarf-accel-tables", cl::Hidden,
                 cl::desc("Output prototype dwarf accelerator tables."),
                 cl::values(clEnumVal(Default, "Default for platform"),
                            clEnumVal(Enable, "Enabled"),
                            clEnumVal(Disable, "Disabled"), clEnumValEnd),
                 cl::init(Default));

static cl::opt<DefaultOnOff>
SplitDwarf("split-dwarf", cl::Hidden,
           cl::desc("Output DWARF5 split debug info."),
           cl::values(clEnumVal(Default, "Default for platform"),
                      clEnumVal(Enable, "Enabled"),
                      clEnumVal(Disable, "Disabled"), clEnumValEnd),
           cl::init(Default));

static cl::opt<DefaultOnOff>
DwarfPubSections("generate-dwarf-pub-sections", cl::Hidden,
                 cl::desc("Generate DWARF pubnames and pubtypes sections"),
                 cl::values(clEnumVal(Default, "Default for platform"),
                            clEnumVal(Enable, "Enabled"),
                            clEnumVal(Disable, "Disabled"), clEnumValEnd),
                 cl::init(Default));

static cl::opt<unsigned>
DwarfVersionNumber("dwarf-version", cl::Hidden,
                   cl::desc("Generate DWARF for dwarf version."), cl::init(0));

static cl::opt<bool>
DwarfCURanges("generate-dwarf-cu-ranges", cl::Hidden,
              cl::desc("Generate DW_AT_ranges for compile units"),
              cl::init(false));

static const char *const DWARFGroupName = "DWARF Emission";
static const char *const DbgTimerName = "DWARF Debug Writer";

//===----------------------------------------------------------------------===//

namespace llvm {

/// resolve - Look in the DwarfDebug map for the MDNode that
/// corresponds to the reference.
template <typename T> T DbgVariable::resolve(DIRef<T> Ref) const {
  return DD->resolve(Ref);
}

DIType DbgVariable::getType() const {
  DIType Ty = Var.getType();
  // FIXME: isBlockByrefVariable should be reformulated in terms of complex
  // addresses instead.
  if (Var.isBlockByrefVariable()) {
    /* Byref variables, in Blocks, are declared by the programmer as
       "SomeType VarName;", but the compiler creates a
       __Block_byref_x_VarName struct, and gives the variable VarName
       either the struct, or a pointer to the struct, as its type.  This
       is necessary for various behind-the-scenes things the compiler
       needs to do with by-reference variables in blocks.

       However, as far as the original *programmer* is concerned, the
       variable should still have type 'SomeType', as originally declared.

       The following function dives into the __Block_byref_x_VarName
       struct to find the original type of the variable.  This will be
       passed back to the code generating the type for the Debug
       Information Entry for the variable 'VarName'.  'VarName' will then
       have the original type 'SomeType' in its debug information.

       The original type 'SomeType' will be the type of the field named
       'VarName' inside the __Block_byref_x_VarName struct.

       NOTE: In order for this to not completely fail on the debugger
       side, the Debug Information Entry for the variable VarName needs to
       have a DW_AT_location that tells the debugger how to unwind through
       the pointers and __Block_byref_x_VarName struct to find the actual
       value of the variable.  The function addBlockByrefType does this.  */
    DIType subType = Ty;
    uint16_t tag = Ty.getTag();

    if (tag == dwarf::DW_TAG_pointer_type)
      subType = resolve(DIDerivedType(Ty).getTypeDerivedFrom());

    DIArray Elements = DICompositeType(subType).getTypeArray();
    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIDerivedType DT(Elements.getElement(i));
      if (getName() == DT.getName())
        return (resolve(DT.getTypeDerivedFrom()));
    }
  }
  return Ty;
}

} // end llvm namespace

/// Return Dwarf Version by checking module flags.
static unsigned getDwarfVersionFromModule(const Module *M) {
  Value *Val = M->getModuleFlag("Dwarf Version");
  if (!Val)
    return dwarf::DWARF_VERSION;
  return cast<ConstantInt>(Val)->getZExtValue();
}

DwarfDebug::DwarfDebug(AsmPrinter *A, Module *M)
    : Asm(A), MMI(Asm->MMI), FirstCU(0), SourceIdMap(DIEValueAllocator),
      PrevLabel(NULL), GlobalRangeCount(0),
      InfoHolder(A, "info_string", DIEValueAllocator), HasCURanges(false),
      UsedNonDefaultText(false),
      SkeletonHolder(A, "skel_string", DIEValueAllocator) {

  DwarfInfoSectionSym = DwarfAbbrevSectionSym = DwarfStrSectionSym = 0;
  DwarfDebugRangeSectionSym = DwarfDebugLocSectionSym = DwarfLineSectionSym = 0;
  DwarfAddrSectionSym = 0;
  DwarfAbbrevDWOSectionSym = DwarfStrDWOSectionSym = 0;
  FunctionBeginSym = FunctionEndSym = 0;
  CurFn = 0;
  CurMI = 0;

  // Turn on accelerator tables for Darwin by default, pubnames by
  // default for non-Darwin, and handle split dwarf.
  bool IsDarwin = Triple(A->getTargetTriple()).isOSDarwin();

  if (DwarfAccelTables == Default)
    HasDwarfAccelTables = IsDarwin;
  else
    HasDwarfAccelTables = DwarfAccelTables == Enable;

  if (SplitDwarf == Default)
    HasSplitDwarf = false;
  else
    HasSplitDwarf = SplitDwarf == Enable;

  if (DwarfPubSections == Default)
    HasDwarfPubSections = !IsDarwin;
  else
    HasDwarfPubSections = DwarfPubSections == Enable;

  DwarfVersion = DwarfVersionNumber
                     ? DwarfVersionNumber
                     : getDwarfVersionFromModule(MMI->getModule());

  {
    NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
    beginModule();
  }
}

// Switch to the specified MCSection and emit an assembler
// temporary label to it if SymbolStem is specified.
static MCSymbol *emitSectionSym(AsmPrinter *Asm, const MCSection *Section,
                                const char *SymbolStem = 0) {
  Asm->OutStreamer.SwitchSection(Section);
  if (!SymbolStem)
    return 0;

  MCSymbol *TmpSym = Asm->GetTempSymbol(SymbolStem);
  Asm->OutStreamer.EmitLabel(TmpSym);
  return TmpSym;
}

DwarfFile::~DwarfFile() {
  for (SmallVectorImpl<DwarfUnit *>::iterator I = CUs.begin(), E = CUs.end();
       I != E; ++I)
    delete *I;
}

MCSymbol *DwarfFile::getStringPoolSym() {
  return Asm->GetTempSymbol(StringPref);
}

MCSymbol *DwarfFile::getStringPoolEntry(StringRef Str) {
  std::pair<MCSymbol *, unsigned> &Entry =
      StringPool.GetOrCreateValue(Str).getValue();
  if (Entry.first)
    return Entry.first;

  Entry.second = NextStringPoolNumber++;
  return Entry.first = Asm->GetTempSymbol(StringPref, Entry.second);
}

unsigned DwarfFile::getStringPoolIndex(StringRef Str) {
  std::pair<MCSymbol *, unsigned> &Entry =
      StringPool.GetOrCreateValue(Str).getValue();
  if (Entry.first)
    return Entry.second;

  Entry.second = NextStringPoolNumber++;
  Entry.first = Asm->GetTempSymbol(StringPref, Entry.second);
  return Entry.second;
}

unsigned DwarfFile::getAddrPoolIndex(const MCSymbol *Sym, bool TLS) {
  std::pair<AddrPool::iterator, bool> P = AddressPool.insert(
      std::make_pair(Sym, AddressPoolEntry(NextAddrPoolNumber, TLS)));
  if (P.second)
    ++NextAddrPoolNumber;
  return P.first->second.Number;
}

// Define a unique number for the abbreviation.
//
void DwarfFile::assignAbbrevNumber(DIEAbbrev &Abbrev) {
  // Check the set for priors.
  DIEAbbrev *InSet = AbbreviationsSet.GetOrInsertNode(&Abbrev);

  // If it's newly added.
  if (InSet == &Abbrev) {
    // Add to abbreviation list.
    Abbreviations.push_back(&Abbrev);

    // Assign the vector position + 1 as its number.
    Abbrev.setNumber(Abbreviations.size());
  } else {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  }
}

static bool isObjCClass(StringRef Name) {
  return Name.startswith("+") || Name.startswith("-");
}

static bool hasObjCCategory(StringRef Name) {
  if (!isObjCClass(Name))
    return false;

  return Name.find(") ") != StringRef::npos;
}

static void getObjCClassCategory(StringRef In, StringRef &Class,
                                 StringRef &Category) {
  if (!hasObjCCategory(In)) {
    Class = In.slice(In.find('[') + 1, In.find(' '));
    Category = "";
    return;
  }

  Class = In.slice(In.find('[') + 1, In.find('('));
  Category = In.slice(In.find('[') + 1, In.find(' '));
  return;
}

static StringRef getObjCMethodName(StringRef In) {
  return In.slice(In.find(' ') + 1, In.find(']'));
}

// Helper for sorting sections into a stable output order.
static bool SectionSort(const MCSection *A, const MCSection *B) {
  std::string LA = (A ? A->getLabelBeginName() : "");
  std::string LB = (B ? B->getLabelBeginName() : "");
  return LA < LB;
}

// Add the various names to the Dwarf accelerator table names.
// TODO: Determine whether or not we should add names for programs
// that do not have a DW_AT_name or DW_AT_linkage_name field - this
// is only slightly different than the lookup of non-standard ObjC names.
static void addSubprogramNames(DwarfUnit *TheU, DISubprogram SP, DIE *Die) {
  if (!SP.isDefinition())
    return;
  TheU->addAccelName(SP.getName(), Die);

  // If the linkage name is different than the name, go ahead and output
  // that as well into the name table.
  if (SP.getLinkageName() != "" && SP.getName() != SP.getLinkageName())
    TheU->addAccelName(SP.getLinkageName(), Die);

  // If this is an Objective-C selector name add it to the ObjC accelerator
  // too.
  if (isObjCClass(SP.getName())) {
    StringRef Class, Category;
    getObjCClassCategory(SP.getName(), Class, Category);
    TheU->addAccelObjC(Class, Die);
    if (Category != "")
      TheU->addAccelObjC(Category, Die);
    // Also add the base method name to the name table.
    TheU->addAccelName(getObjCMethodName(SP.getName()), Die);
  }
}

/// isSubprogramContext - Return true if Context is either a subprogram
/// or another context nested inside a subprogram.
bool DwarfDebug::isSubprogramContext(const MDNode *Context) {
  if (!Context)
    return false;
  DIDescriptor D(Context);
  if (D.isSubprogram())
    return true;
  if (D.isType())
    return isSubprogramContext(resolve(DIType(Context).getContext()));
  return false;
}

// Find DIE for the given subprogram and attach appropriate DW_AT_low_pc
// and DW_AT_high_pc attributes. If there are global variables in this
// scope then create and insert DIEs for these variables.
DIE *DwarfDebug::updateSubprogramScopeDIE(DwarfCompileUnit *SPCU,
                                          DISubprogram SP) {
  DIE *SPDie = SPCU->getDIE(SP);

  assert(SPDie && "Unable to find subprogram DIE!");

  // If we're updating an abstract DIE, then we will be adding the children and
  // object pointer later on. But what we don't want to do is process the
  // concrete DIE twice.
  if (DIE *AbsSPDIE = AbstractSPDies.lookup(SP)) {
    // Pick up abstract subprogram DIE.
    SPDie =
        SPCU->createAndAddDIE(dwarf::DW_TAG_subprogram, *SPCU->getUnitDie());
    SPCU->addDIEEntry(SPDie, dwarf::DW_AT_abstract_origin, AbsSPDIE);
  } else {
    DISubprogram SPDecl = SP.getFunctionDeclaration();
    if (!SPDecl.isSubprogram()) {
      // There is not any need to generate specification DIE for a function
      // defined at compile unit level. If a function is defined inside another
      // function then gdb prefers the definition at top level and but does not
      // expect specification DIE in parent function. So avoid creating
      // specification DIE for a function defined inside a function.
      DIScope SPContext = resolve(SP.getContext());
      if (SP.isDefinition() && !SPContext.isCompileUnit() &&
          !SPContext.isFile() && !isSubprogramContext(SPContext)) {
        SPCU->addFlag(SPDie, dwarf::DW_AT_declaration);

        // Add arguments.
        DICompositeType SPTy = SP.getType();
        DIArray Args = SPTy.getTypeArray();
        uint16_t SPTag = SPTy.getTag();
        if (SPTag == dwarf::DW_TAG_subroutine_type)
          SPCU->constructSubprogramArguments(*SPDie, Args);
        DIE *SPDeclDie = SPDie;
        SPDie = SPCU->createAndAddDIE(dwarf::DW_TAG_subprogram,
                                      *SPCU->getUnitDie());
        SPCU->addDIEEntry(SPDie, dwarf::DW_AT_specification, SPDeclDie);
      }
    }
  }

  SPCU->addLabelAddress(SPDie, dwarf::DW_AT_low_pc, FunctionBeginSym);
  SPCU->addLabelAddress(SPDie, dwarf::DW_AT_high_pc, FunctionEndSym);

  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  MachineLocation Location(RI->getFrameRegister(*Asm->MF));
  SPCU->addAddress(SPDie, dwarf::DW_AT_frame_base, Location);

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_subprogram nodes.
  addSubprogramNames(SPCU, SP, SPDie);

  return SPDie;
}

/// Check whether we should create a DIE for the given Scope, return true
/// if we don't create a DIE (the corresponding DIE is null).
bool DwarfDebug::isLexicalScopeDIENull(LexicalScope *Scope) {
  if (Scope->isAbstractScope())
    return false;

  // We don't create a DIE if there is no Range.
  const SmallVectorImpl<InsnRange> &Ranges = Scope->getRanges();
  if (Ranges.empty())
    return true;

  if (Ranges.size() > 1)
    return false;

  // We don't create a DIE if we have a single Range and the end label
  // is null.
  SmallVectorImpl<InsnRange>::const_iterator RI = Ranges.begin();
  MCSymbol *End = getLabelAfterInsn(RI->second);
  return !End;
}

static void addSectionLabel(AsmPrinter *Asm, DwarfUnit *U, DIE *D,
                            dwarf::Attribute A, const MCSymbol *L,
                            const MCSymbol *Sec) {
  if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    U->addSectionLabel(D, A, L);
  else
    U->addSectionDelta(D, A, L, Sec);
}

void DwarfDebug::addScopeRangeList(DwarfCompileUnit *TheCU, DIE *ScopeDIE,
                                   const SmallVectorImpl<InsnRange> &Range) {
  // Emit offset in .debug_range as a relocatable label. emitDIE will handle
  // emitting it appropriately.
  MCSymbol *RangeSym = Asm->GetTempSymbol("debug_ranges", GlobalRangeCount++);
  addSectionLabel(Asm, TheCU, ScopeDIE, dwarf::DW_AT_ranges, RangeSym,
                  DwarfDebugRangeSectionSym);

  RangeSpanList List(RangeSym);
  for (SmallVectorImpl<InsnRange>::const_iterator RI = Range.begin(),
                                                  RE = Range.end();
       RI != RE; ++RI) {
    RangeSpan Span(getLabelBeforeInsn(RI->first),
                   getLabelAfterInsn(RI->second));
    List.addRange(std::move(Span));
  }

  // Add the range list to the set of ranges to be emitted.
  TheCU->addRangeList(std::move(List));
}

// Construct new DW_TAG_lexical_block for this scope and attach
// DW_AT_low_pc/DW_AT_high_pc labels.
DIE *DwarfDebug::constructLexicalScopeDIE(DwarfCompileUnit *TheCU,
                                          LexicalScope *Scope) {
  if (isLexicalScopeDIENull(Scope))
    return 0;

  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_lexical_block);
  if (Scope->isAbstractScope())
    return ScopeDIE;

  const SmallVectorImpl<InsnRange> &ScopeRanges = Scope->getRanges();

  // If we have multiple ranges, emit them into the range section.
  if (ScopeRanges.size() > 1) {
    addScopeRangeList(TheCU, ScopeDIE, ScopeRanges);
    return ScopeDIE;
  }

  // Construct the address range for this DIE.
  SmallVectorImpl<InsnRange>::const_iterator RI = ScopeRanges.begin();
  MCSymbol *Start = getLabelBeforeInsn(RI->first);
  MCSymbol *End = getLabelAfterInsn(RI->second);
  assert(End && "End label should not be null!");

  assert(Start->isDefined() && "Invalid starting label for an inlined scope!");
  assert(End->isDefined() && "Invalid end label for an inlined scope!");

  TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_low_pc, Start);
  TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_high_pc, End);

  return ScopeDIE;
}

// This scope represents inlined body of a function. Construct DIE to
// represent this concrete inlined copy of the function.
DIE *DwarfDebug::constructInlinedScopeDIE(DwarfCompileUnit *TheCU,
                                          LexicalScope *Scope) {
  const SmallVectorImpl<InsnRange> &ScopeRanges = Scope->getRanges();
  assert(!ScopeRanges.empty() &&
         "LexicalScope does not have instruction markers!");

  if (!Scope->getScopeNode())
    return NULL;
  DIScope DS(Scope->getScopeNode());
  DISubprogram InlinedSP = getDISubprogram(DS);
  DIE *OriginDIE = TheCU->getDIE(InlinedSP);
  if (!OriginDIE) {
    DEBUG(dbgs() << "Unable to find original DIE for an inlined subprogram.");
    return NULL;
  }

  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_inlined_subroutine);
  TheCU->addDIEEntry(ScopeDIE, dwarf::DW_AT_abstract_origin, OriginDIE);

  // If we have multiple ranges, emit them into the range section.
  if (ScopeRanges.size() > 1)
    addScopeRangeList(TheCU, ScopeDIE, ScopeRanges);
  else {
    SmallVectorImpl<InsnRange>::const_iterator RI = ScopeRanges.begin();
    MCSymbol *StartLabel = getLabelBeforeInsn(RI->first);
    MCSymbol *EndLabel = getLabelAfterInsn(RI->second);

    if (StartLabel == 0 || EndLabel == 0)
      llvm_unreachable("Unexpected Start and End labels for an inlined scope!");

    assert(StartLabel->isDefined() &&
           "Invalid starting label for an inlined scope!");
    assert(EndLabel->isDefined() && "Invalid end label for an inlined scope!");

    TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_low_pc, StartLabel);
    TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_high_pc, EndLabel);
  }

  InlinedSubprogramDIEs.insert(OriginDIE);

  // Add the call site information to the DIE.
  DILocation DL(Scope->getInlinedAt());
  TheCU->addUInt(ScopeDIE, dwarf::DW_AT_call_file, None,
                 getOrCreateSourceID(DL.getFilename(), DL.getDirectory(),
                                     TheCU->getUniqueID()));
  TheCU->addUInt(ScopeDIE, dwarf::DW_AT_call_line, None, DL.getLineNumber());

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_inlined_subprogram nodes.
  addSubprogramNames(TheCU, InlinedSP, ScopeDIE);

  return ScopeDIE;
}

DIE *DwarfDebug::createScopeChildrenDIE(DwarfCompileUnit *TheCU,
                                        LexicalScope *Scope,
                                        SmallVectorImpl<DIE *> &Children) {
  DIE *ObjectPointer = NULL;

  // Collect arguments for current function.
  if (LScopes.isCurrentFunctionScope(Scope)) {
    for (unsigned i = 0, N = CurrentFnArguments.size(); i < N; ++i)
      if (DbgVariable *ArgDV = CurrentFnArguments[i])
        if (DIE *Arg =
                TheCU->constructVariableDIE(*ArgDV, Scope->isAbstractScope())) {
          Children.push_back(Arg);
          if (ArgDV->isObjectPointer())
            ObjectPointer = Arg;
        }

    // If this is a variadic function, add an unspecified parameter.
    DISubprogram SP(Scope->getScopeNode());
    DIArray FnArgs = SP.getType().getTypeArray();
    if (FnArgs.getElement(FnArgs.getNumElements() - 1)
            .isUnspecifiedParameter()) {
      DIE *Ellipsis = new DIE(dwarf::DW_TAG_unspecified_parameters);
      Children.push_back(Ellipsis);
    }
  }

  // Collect lexical scope children first.
  const SmallVectorImpl<DbgVariable *> &Variables =
      ScopeVariables.lookup(Scope);
  for (unsigned i = 0, N = Variables.size(); i < N; ++i)
    if (DIE *Variable = TheCU->constructVariableDIE(*Variables[i],
                                                    Scope->isAbstractScope())) {
      Children.push_back(Variable);
      if (Variables[i]->isObjectPointer())
        ObjectPointer = Variable;
    }
  const SmallVectorImpl<LexicalScope *> &Scopes = Scope->getChildren();
  for (unsigned j = 0, M = Scopes.size(); j < M; ++j)
    if (DIE *Nested = constructScopeDIE(TheCU, Scopes[j]))
      Children.push_back(Nested);
  return ObjectPointer;
}

// Construct a DIE for this scope.
DIE *DwarfDebug::constructScopeDIE(DwarfCompileUnit *TheCU,
                                   LexicalScope *Scope) {
  if (!Scope || !Scope->getScopeNode())
    return NULL;

  DIScope DS(Scope->getScopeNode());

  SmallVector<DIE *, 8> Children;
  DIE *ObjectPointer = NULL;
  bool ChildrenCreated = false;

  // We try to create the scope DIE first, then the children DIEs. This will
  // avoid creating un-used children then removing them later when we find out
  // the scope DIE is null.
  DIE *ScopeDIE = NULL;
  if (Scope->getInlinedAt())
    ScopeDIE = constructInlinedScopeDIE(TheCU, Scope);
  else if (DS.isSubprogram()) {
    ProcessedSPNodes.insert(DS);
    if (Scope->isAbstractScope()) {
      ScopeDIE = TheCU->getDIE(DS);
      // Note down abstract DIE.
      if (ScopeDIE)
        AbstractSPDies.insert(std::make_pair(DS, ScopeDIE));
    } else
      ScopeDIE = updateSubprogramScopeDIE(TheCU, DISubprogram(DS));
  } else {
    // Early exit when we know the scope DIE is going to be null.
    if (isLexicalScopeDIENull(Scope))
      return NULL;

    // We create children here when we know the scope DIE is not going to be
    // null and the children will be added to the scope DIE.
    ObjectPointer = createScopeChildrenDIE(TheCU, Scope, Children);
    ChildrenCreated = true;

    // There is no need to emit empty lexical block DIE.
    std::pair<ImportedEntityMap::const_iterator,
              ImportedEntityMap::const_iterator> Range =
        std::equal_range(
            ScopesWithImportedEntities.begin(),
            ScopesWithImportedEntities.end(),
            std::pair<const MDNode *, const MDNode *>(DS, (const MDNode *)0),
            less_first());
    if (Children.empty() && Range.first == Range.second)
      return NULL;
    ScopeDIE = constructLexicalScopeDIE(TheCU, Scope);
    assert(ScopeDIE && "Scope DIE should not be null.");
    for (ImportedEntityMap::const_iterator i = Range.first; i != Range.second;
         ++i)
      constructImportedEntityDIE(TheCU, i->second, ScopeDIE);
  }

  if (!ScopeDIE) {
    assert(Children.empty() &&
           "We create children only when the scope DIE is not null.");
    return NULL;
  }
  if (!ChildrenCreated)
    // We create children when the scope DIE is not null.
    ObjectPointer = createScopeChildrenDIE(TheCU, Scope, Children);

  // Add children
  for (SmallVectorImpl<DIE *>::iterator I = Children.begin(),
                                        E = Children.end();
       I != E; ++I)
    ScopeDIE->addChild(*I);

  if (DS.isSubprogram() && ObjectPointer != NULL)
    TheCU->addDIEEntry(ScopeDIE, dwarf::DW_AT_object_pointer, ObjectPointer);

  return ScopeDIE;
}

// Look up the source id with the given directory and source file names.
// If none currently exists, create a new id and insert it in the
// SourceIds map. This can update DirectoryNames and SourceFileNames maps
// as well.
unsigned DwarfDebug::getOrCreateSourceID(StringRef FileName, StringRef DirName,
                                         unsigned CUID) {
  // If we print assembly, we can't separate .file entries according to
  // compile units. Thus all files will belong to the default compile unit.

  // FIXME: add a better feature test than hasRawTextSupport. Even better,
  // extend .file to support this.
  if (Asm->OutStreamer.hasRawTextSupport())
    CUID = 0;

  // If FE did not provide a file name, then assume stdin.
  if (FileName.empty())
    return getOrCreateSourceID("<stdin>", StringRef(), CUID);

  // TODO: this might not belong here. See if we can factor this better.
  if (DirName == CompilationDir)
    DirName = "";

  // FileIDCUMap stores the current ID for the given compile unit.
  unsigned SrcId = FileIDCUMap[CUID] + 1;

  // We look up the CUID/file/dir by concatenating them with a zero byte.
  SmallString<128> NamePair;
  NamePair += utostr(CUID);
  NamePair += '\0';
  NamePair += DirName;
  NamePair += '\0'; // Zero bytes are not allowed in paths.
  NamePair += FileName;

  StringMapEntry<unsigned> &Ent = SourceIdMap.GetOrCreateValue(NamePair, SrcId);
  if (Ent.getValue() != SrcId)
    return Ent.getValue();

  FileIDCUMap[CUID] = SrcId;
  // Print out a .file directive to specify files for .loc directives.
  Asm->OutStreamer.EmitDwarfFileDirective(SrcId, DirName, FileName, CUID);

  return SrcId;
}

void DwarfDebug::addGnuPubAttributes(DwarfUnit *U, DIE *D) const {
  if (!GenerateGnuPubSections)
    return;

  addSectionLabel(Asm, U, D, dwarf::DW_AT_GNU_pubnames,
                  Asm->GetTempSymbol("gnu_pubnames", U->getUniqueID()),
                  DwarfGnuPubNamesSectionSym);

  addSectionLabel(Asm, U, D, dwarf::DW_AT_GNU_pubtypes,
                  Asm->GetTempSymbol("gnu_pubtypes", U->getUniqueID()),
                  DwarfGnuPubTypesSectionSym);
}

// Create new DwarfCompileUnit for the given metadata node with tag
// DW_TAG_compile_unit.
DwarfCompileUnit *DwarfDebug::constructDwarfCompileUnit(DICompileUnit DIUnit) {
  StringRef FN = DIUnit.getFilename();
  CompilationDir = DIUnit.getDirectory();

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  DwarfCompileUnit *NewCU = new DwarfCompileUnit(
      InfoHolder.getUnits().size(), Die, DIUnit, Asm, this, &InfoHolder);
  InfoHolder.addUnit(NewCU);

  FileIDCUMap[NewCU->getUniqueID()] = 0;

  NewCU->addString(Die, dwarf::DW_AT_producer, DIUnit.getProducer());
  NewCU->addUInt(Die, dwarf::DW_AT_language, dwarf::DW_FORM_data2,
                 DIUnit.getLanguage());
  NewCU->addString(Die, dwarf::DW_AT_name, FN);

  if (!useSplitDwarf()) {
    NewCU->initStmtList(DwarfLineSectionSym);

    // If we're using split dwarf the compilation dir is going to be in the
    // skeleton CU and so we don't need to duplicate it here.
    if (!CompilationDir.empty())
      NewCU->addString(Die, dwarf::DW_AT_comp_dir, CompilationDir);

    addGnuPubAttributes(NewCU, Die);
  }

  if (DIUnit.isOptimized())
    NewCU->addFlag(Die, dwarf::DW_AT_APPLE_optimized);

  StringRef Flags = DIUnit.getFlags();
  if (!Flags.empty())
    NewCU->addString(Die, dwarf::DW_AT_APPLE_flags, Flags);

  if (unsigned RVer = DIUnit.getRunTimeVersion())
    NewCU->addUInt(Die, dwarf::DW_AT_APPLE_major_runtime_vers,
                   dwarf::DW_FORM_data1, RVer);

  if (!FirstCU)
    FirstCU = NewCU;

  if (useSplitDwarf()) {
    NewCU->initSection(Asm->getObjFileLowering().getDwarfInfoDWOSection(),
                       DwarfInfoDWOSectionSym);
    NewCU->setSkeleton(constructSkeletonCU(NewCU));
  } else
    NewCU->initSection(Asm->getObjFileLowering().getDwarfInfoSection(),
                       DwarfInfoSectionSym);

  CUMap.insert(std::make_pair(DIUnit, NewCU));
  CUDieMap.insert(std::make_pair(Die, NewCU));
  return NewCU;
}

// Construct subprogram DIE.
void DwarfDebug::constructSubprogramDIE(DwarfCompileUnit *TheCU,
                                        const MDNode *N) {
  // FIXME: We should only call this routine once, however, during LTO if a
  // program is defined in multiple CUs we could end up calling it out of
  // beginModule as we walk the CUs.

  DwarfCompileUnit *&CURef = SPMap[N];
  if (CURef)
    return;
  CURef = TheCU;

  DISubprogram SP(N);
  if (!SP.isDefinition())
    // This is a method declaration which will be handled while constructing
    // class type.
    return;

  DIE *SubprogramDie = TheCU->getOrCreateSubprogramDIE(SP);

  // Expose as a global name.
  TheCU->addGlobalName(SP.getName(), SubprogramDie, resolve(SP.getContext()));
}

void DwarfDebug::constructImportedEntityDIE(DwarfCompileUnit *TheCU,
                                            const MDNode *N) {
  DIImportedEntity Module(N);
  assert(Module.Verify());
  if (DIE *D = TheCU->getOrCreateContextDIE(Module.getContext()))
    constructImportedEntityDIE(TheCU, Module, D);
}

void DwarfDebug::constructImportedEntityDIE(DwarfCompileUnit *TheCU,
                                            const MDNode *N, DIE *Context) {
  DIImportedEntity Module(N);
  assert(Module.Verify());
  return constructImportedEntityDIE(TheCU, Module, Context);
}

void DwarfDebug::constructImportedEntityDIE(DwarfCompileUnit *TheCU,
                                            const DIImportedEntity &Module,
                                            DIE *Context) {
  assert(Module.Verify() &&
         "Use one of the MDNode * overloads to handle invalid metadata");
  assert(Context && "Should always have a context for an imported_module");
  DIE *IMDie = new DIE(Module.getTag());
  TheCU->insertDIE(Module, IMDie);
  DIE *EntityDie;
  DIDescriptor Entity = Module.getEntity();
  if (Entity.isNameSpace())
    EntityDie = TheCU->getOrCreateNameSpace(DINameSpace(Entity));
  else if (Entity.isSubprogram())
    EntityDie = TheCU->getOrCreateSubprogramDIE(DISubprogram(Entity));
  else if (Entity.isType())
    EntityDie = TheCU->getOrCreateTypeDIE(DIType(Entity));
  else
    EntityDie = TheCU->getDIE(Entity);
  unsigned FileID = getOrCreateSourceID(Module.getContext().getFilename(),
                                        Module.getContext().getDirectory(),
                                        TheCU->getUniqueID());
  TheCU->addUInt(IMDie, dwarf::DW_AT_decl_file, None, FileID);
  TheCU->addUInt(IMDie, dwarf::DW_AT_decl_line, None, Module.getLineNumber());
  TheCU->addDIEEntry(IMDie, dwarf::DW_AT_import, EntityDie);
  StringRef Name = Module.getName();
  if (!Name.empty())
    TheCU->addString(IMDie, dwarf::DW_AT_name, Name);
  Context->addChild(IMDie);
}

// Emit all Dwarf sections that should come prior to the content. Create
// global DIEs and emit initial debug info sections. This is invoked by
// the target AsmPrinter.
void DwarfDebug::beginModule() {
  if (DisableDebugInfoPrinting)
    return;

  const Module *M = MMI->getModule();

  // If module has named metadata anchors then use them, otherwise scan the
  // module using debug info finder to collect debug info.
  NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return;
  TypeIdentifierMap = generateDITypeIdentifierMap(CU_Nodes);

  // Emit initial sections so we can reference labels later.
  emitSectionLabels();

  for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
    DICompileUnit CUNode(CU_Nodes->getOperand(i));
    DwarfCompileUnit *CU = constructDwarfCompileUnit(CUNode);
    DIArray ImportedEntities = CUNode.getImportedEntities();
    for (unsigned i = 0, e = ImportedEntities.getNumElements(); i != e; ++i)
      ScopesWithImportedEntities.push_back(std::make_pair(
          DIImportedEntity(ImportedEntities.getElement(i)).getContext(),
          ImportedEntities.getElement(i)));
    std::sort(ScopesWithImportedEntities.begin(),
              ScopesWithImportedEntities.end(), less_first());
    DIArray GVs = CUNode.getGlobalVariables();
    for (unsigned i = 0, e = GVs.getNumElements(); i != e; ++i)
      CU->createGlobalVariableDIE(DIGlobalVariable(GVs.getElement(i)));
    DIArray SPs = CUNode.getSubprograms();
    for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i)
      constructSubprogramDIE(CU, SPs.getElement(i));
    DIArray EnumTypes = CUNode.getEnumTypes();
    for (unsigned i = 0, e = EnumTypes.getNumElements(); i != e; ++i)
      CU->getOrCreateTypeDIE(EnumTypes.getElement(i));
    DIArray RetainedTypes = CUNode.getRetainedTypes();
    for (unsigned i = 0, e = RetainedTypes.getNumElements(); i != e; ++i)
      CU->getOrCreateTypeDIE(RetainedTypes.getElement(i));
    // Emit imported_modules last so that the relevant context is already
    // available.
    for (unsigned i = 0, e = ImportedEntities.getNumElements(); i != e; ++i)
      constructImportedEntityDIE(CU, ImportedEntities.getElement(i));
  }

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);

  // Prime section data.
  SectionMap[Asm->getObjFileLowering().getTextSection()];
}

// Attach DW_AT_inline attribute with inlined subprogram DIEs.
void DwarfDebug::computeInlinedDIEs() {
  // Attach DW_AT_inline attribute with inlined subprogram DIEs.
  for (SmallPtrSet<DIE *, 4>::iterator AI = InlinedSubprogramDIEs.begin(),
                                       AE = InlinedSubprogramDIEs.end();
       AI != AE; ++AI) {
    DIE *ISP = *AI;
    FirstCU->addUInt(ISP, dwarf::DW_AT_inline, None, dwarf::DW_INL_inlined);
  }
  for (DenseMap<const MDNode *, DIE *>::iterator AI = AbstractSPDies.begin(),
                                                 AE = AbstractSPDies.end();
       AI != AE; ++AI) {
    DIE *ISP = AI->second;
    if (InlinedSubprogramDIEs.count(ISP))
      continue;
    FirstCU->addUInt(ISP, dwarf::DW_AT_inline, None, dwarf::DW_INL_inlined);
  }
}

// Collect info for variables that were optimized out.
void DwarfDebug::collectDeadVariables() {
  const Module *M = MMI->getModule();

  if (NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu")) {
    for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
      DICompileUnit TheCU(CU_Nodes->getOperand(i));
      DIArray Subprograms = TheCU.getSubprograms();
      for (unsigned i = 0, e = Subprograms.getNumElements(); i != e; ++i) {
        DISubprogram SP(Subprograms.getElement(i));
        if (ProcessedSPNodes.count(SP) != 0)
          continue;
        if (!SP.isSubprogram())
          continue;
        if (!SP.isDefinition())
          continue;
        DIArray Variables = SP.getVariables();
        if (Variables.getNumElements() == 0)
          continue;

        // Construct subprogram DIE and add variables DIEs.
        DwarfCompileUnit *SPCU =
            static_cast<DwarfCompileUnit *>(CUMap.lookup(TheCU));
        assert(SPCU && "Unable to find Compile Unit!");
        // FIXME: See the comment in constructSubprogramDIE about duplicate
        // subprogram DIEs.
        constructSubprogramDIE(SPCU, SP);
        DIE *SPDIE = SPCU->getDIE(SP);
        for (unsigned vi = 0, ve = Variables.getNumElements(); vi != ve; ++vi) {
          DIVariable DV(Variables.getElement(vi));
          if (!DV.isVariable())
            continue;
          DbgVariable NewVar(DV, NULL, this);
          if (DIE *VariableDIE = SPCU->constructVariableDIE(NewVar, false))
            SPDIE->addChild(VariableDIE);
        }
      }
    }
  }
}

void DwarfDebug::finalizeModuleInfo() {
  // Collect info for variables that were optimized out.
  collectDeadVariables();

  // Attach DW_AT_inline attribute with inlined subprogram DIEs.
  computeInlinedDIEs();

  // Handle anything that needs to be done on a per-unit basis after
  // all other generation.
  for (SmallVectorImpl<DwarfUnit *>::const_iterator I = getUnits().begin(),
                                                    E = getUnits().end();
       I != E; ++I) {
    DwarfUnit *TheU = *I;
    // Emit DW_AT_containing_type attribute to connect types with their
    // vtable holding type.
    TheU->constructContainingTypeDIEs();

    // Add CU specific attributes if we need to add any.
    if (TheU->getUnitDie()->getTag() == dwarf::DW_TAG_compile_unit) {
      // If we're splitting the dwarf out now that we've got the entire
      // CU then add the dwo id to it.
      DwarfCompileUnit *SkCU =
          static_cast<DwarfCompileUnit *>(TheU->getSkeleton());
      if (useSplitDwarf()) {
        // This should be a unique identifier when we want to build .dwp files.
        uint64_t ID = 0;
        if (GenerateCUHash) {
          DIEHash CUHash(Asm);
          ID = CUHash.computeCUSignature(*TheU->getUnitDie());
        }
        TheU->addUInt(TheU->getUnitDie(), dwarf::DW_AT_GNU_dwo_id,
                      dwarf::DW_FORM_data8, ID);
        SkCU->addUInt(SkCU->getUnitDie(), dwarf::DW_AT_GNU_dwo_id,
                      dwarf::DW_FORM_data8, ID);
      }

      // If we have code split among multiple sections or we've requested
      // it then emit a DW_AT_ranges attribute on the unit that will remain
      // in the .o file, otherwise add a DW_AT_low_pc.
      // FIXME: Also add a high pc if we can.
      // FIXME: We should use ranges if we have multiple compile units or
      // allow reordering of code ala .subsections_via_symbols in mach-o.
      DwarfCompileUnit *U = SkCU ? SkCU : static_cast<DwarfCompileUnit *>(TheU);
      if (useCURanges() && TheU->getRanges().size()) {
        addSectionLabel(Asm, U, U->getUnitDie(), dwarf::DW_AT_ranges,
                        Asm->GetTempSymbol("cu_ranges", U->getUniqueID()),
                        DwarfDebugRangeSectionSym);

        // A DW_AT_low_pc attribute may also be specified in combination with
        // DW_AT_ranges to specify the default base address for use in location
        // lists (see Section 2.6.2) and range lists (see Section 2.17.3).
        U->addUInt(U->getUnitDie(), dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
                   0);
      } else
        U->addUInt(U->getUnitDie(), dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
                   0);
    }
  }

  // Compute DIE offsets and sizes.
  InfoHolder.computeSizeAndOffsets();
  if (useSplitDwarf())
    SkeletonHolder.computeSizeAndOffsets();
}

void DwarfDebug::endSections() {
  // Filter labels by section.
  for (size_t n = 0; n < ArangeLabels.size(); n++) {
    const SymbolCU &SCU = ArangeLabels[n];
    if (SCU.Sym->isInSection()) {
      // Make a note of this symbol and it's section.
      const MCSection *Section = &SCU.Sym->getSection();
      if (!Section->getKind().isMetadata())
        SectionMap[Section].push_back(SCU);
    } else {
      // Some symbols (e.g. common/bss on mach-o) can have no section but still
      // appear in the output. This sucks as we rely on sections to build
      // arange spans. We can do it without, but it's icky.
      SectionMap[NULL].push_back(SCU);
    }
  }

  // Build a list of sections used.
  std::vector<const MCSection *> Sections;
  for (SectionMapType::iterator it = SectionMap.begin(); it != SectionMap.end();
       it++) {
    const MCSection *Section = it->first;
    Sections.push_back(Section);
  }

  // Sort the sections into order.
  // This is only done to ensure consistent output order across different runs.
  std::sort(Sections.begin(), Sections.end(), SectionSort);

  // Add terminating symbols for each section.
  for (unsigned ID = 0; ID < Sections.size(); ID++) {
    const MCSection *Section = Sections[ID];
    MCSymbol *Sym = NULL;

    if (Section) {
      // We can't call MCSection::getLabelEndName, as it's only safe to do so
      // if we know the section name up-front. For user-created sections, the
      // resulting label may not be valid to use as a label. (section names can
      // use a greater set of characters on some systems)
      Sym = Asm->GetTempSymbol("debug_end", ID);
      Asm->OutStreamer.SwitchSection(Section);
      Asm->OutStreamer.EmitLabel(Sym);
    }

    // Insert a final terminator.
    SectionMap[Section].push_back(SymbolCU(NULL, Sym));
  }

  // For now only turn on CU ranges if we've explicitly asked for it,
  // we have -ffunction-sections enabled, we've emitted a function
  // into a unique section, or we're using LTO. If we're using LTO then
  // we can't know that any particular function in the module is correlated
  // to a particular CU and so we need to be conservative. At this point all
  // sections should be finalized except for dwarf sections.
  HasCURanges = DwarfCURanges || UsedNonDefaultText || (CUMap.size() > 1) ||
                TargetMachine::getFunctionSections();
}

// Emit all Dwarf sections that should come after the content.
void DwarfDebug::endModule() {
  assert(CurFn == 0);
  assert(CurMI == 0);

  if (!FirstCU)
    return;

  // End any existing sections.
  // TODO: Does this need to happen?
  endSections();

  // Finalize the debug info for the module.
  finalizeModuleInfo();

  emitDebugStr();

  // Emit all the DIEs into a debug info section.
  emitDebugInfo();

  // Corresponding abbreviations into a abbrev section.
  emitAbbreviations();

  // Emit info into a debug loc section.
  emitDebugLoc();

  // Emit info into a debug aranges section.
  if (GenerateARangeSection)
    emitDebugARanges();

  // Emit info into a debug ranges section.
  emitDebugRanges();

  if (useSplitDwarf()) {
    emitDebugStrDWO();
    emitDebugInfoDWO();
    emitDebugAbbrevDWO();
    // Emit DWO addresses.
    InfoHolder.emitAddresses(Asm->getObjFileLowering().getDwarfAddrSection());
  }

  // Emit info into the dwarf accelerator table sections.
  if (useDwarfAccelTables()) {
    emitAccelNames();
    emitAccelObjC();
    emitAccelNamespaces();
    emitAccelTypes();
  }

  // Emit the pubnames and pubtypes sections if requested.
  if (HasDwarfPubSections) {
    emitDebugPubNames(GenerateGnuPubSections);
    emitDebugPubTypes(GenerateGnuPubSections);
  }

  // clean up.
  SPMap.clear();

  // Reset these for the next Module if we have one.
  FirstCU = NULL;
}

// Find abstract variable, if any, associated with Var.
DbgVariable *DwarfDebug::findAbstractVariable(DIVariable &DV,
                                              DebugLoc ScopeLoc) {
  LLVMContext &Ctx = DV->getContext();
  // More then one inlined variable corresponds to one abstract variable.
  DIVariable Var = cleanseInlinedVariable(DV, Ctx);
  DbgVariable *AbsDbgVariable = AbstractVariables.lookup(Var);
  if (AbsDbgVariable)
    return AbsDbgVariable;

  LexicalScope *Scope = LScopes.findAbstractScope(ScopeLoc.getScope(Ctx));
  if (!Scope)
    return NULL;

  AbsDbgVariable = new DbgVariable(Var, NULL, this);
  addScopeVariable(Scope, AbsDbgVariable);
  AbstractVariables[Var] = AbsDbgVariable;
  return AbsDbgVariable;
}

// If Var is a current function argument then add it to CurrentFnArguments list.
bool DwarfDebug::addCurrentFnArgument(DbgVariable *Var, LexicalScope *Scope) {
  if (!LScopes.isCurrentFunctionScope(Scope))
    return false;
  DIVariable DV = Var->getVariable();
  if (DV.getTag() != dwarf::DW_TAG_arg_variable)
    return false;
  unsigned ArgNo = DV.getArgNumber();
  if (ArgNo == 0)
    return false;

  size_t Size = CurrentFnArguments.size();
  if (Size == 0)
    CurrentFnArguments.resize(CurFn->getFunction()->arg_size());
  // llvm::Function argument size is not good indicator of how many
  // arguments does the function have at source level.
  if (ArgNo > Size)
    CurrentFnArguments.resize(ArgNo * 2);
  CurrentFnArguments[ArgNo - 1] = Var;
  return true;
}

// Collect variable information from side table maintained by MMI.
void DwarfDebug::collectVariableInfoFromMMITable(
    SmallPtrSet<const MDNode *, 16> &Processed) {
  MachineModuleInfo::VariableDbgInfoMapTy &VMap = MMI->getVariableDbgInfo();
  for (MachineModuleInfo::VariableDbgInfoMapTy::iterator VI = VMap.begin(),
                                                         VE = VMap.end();
       VI != VE; ++VI) {
    const MDNode *Var = VI->first;
    if (!Var)
      continue;
    Processed.insert(Var);
    DIVariable DV(Var);
    const std::pair<unsigned, DebugLoc> &VP = VI->second;

    LexicalScope *Scope = LScopes.findLexicalScope(VP.second);

    // If variable scope is not found then skip this variable.
    if (Scope == 0)
      continue;

    DbgVariable *AbsDbgVariable = findAbstractVariable(DV, VP.second);
    DbgVariable *RegVar = new DbgVariable(DV, AbsDbgVariable, this);
    RegVar->setFrameIndex(VP.first);
    if (!addCurrentFnArgument(RegVar, Scope))
      addScopeVariable(Scope, RegVar);
    if (AbsDbgVariable)
      AbsDbgVariable->setFrameIndex(VP.first);
  }
}

// Return true if debug value, encoded by DBG_VALUE instruction, is in a
// defined reg.
static bool isDbgValueInDefinedReg(const MachineInstr *MI) {
  assert(MI->isDebugValue() && "Invalid DBG_VALUE machine instruction!");
  return MI->getNumOperands() == 3 && MI->getOperand(0).isReg() &&
         MI->getOperand(0).getReg() &&
         (MI->getOperand(1).isImm() ||
          (MI->getOperand(1).isReg() && MI->getOperand(1).getReg() == 0U));
}

// Get .debug_loc entry for the instruction range starting at MI.
static DotDebugLocEntry getDebugLocEntry(AsmPrinter *Asm,
                                         const MCSymbol *FLabel,
                                         const MCSymbol *SLabel,
                                         const MachineInstr *MI) {
  const MDNode *Var = MI->getOperand(MI->getNumOperands() - 1).getMetadata();

  assert(MI->getNumOperands() == 3);
  if (MI->getOperand(0).isReg()) {
    MachineLocation MLoc;
    // If the second operand is an immediate, this is a
    // register-indirect address.
    if (!MI->getOperand(1).isImm())
      MLoc.set(MI->getOperand(0).getReg());
    else
      MLoc.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
    return DotDebugLocEntry(FLabel, SLabel, MLoc, Var);
  }
  if (MI->getOperand(0).isImm())
    return DotDebugLocEntry(FLabel, SLabel, MI->getOperand(0).getImm());
  if (MI->getOperand(0).isFPImm())
    return DotDebugLocEntry(FLabel, SLabel, MI->getOperand(0).getFPImm());
  if (MI->getOperand(0).isCImm())
    return DotDebugLocEntry(FLabel, SLabel, MI->getOperand(0).getCImm());

  llvm_unreachable("Unexpected 3 operand DBG_VALUE instruction!");
}

// Find variables for each lexical scope.
void
DwarfDebug::collectVariableInfo(SmallPtrSet<const MDNode *, 16> &Processed) {

  // Grab the variable info that was squirreled away in the MMI side-table.
  collectVariableInfoFromMMITable(Processed);

  for (SmallVectorImpl<const MDNode *>::const_iterator
           UVI = UserVariables.begin(),
           UVE = UserVariables.end();
       UVI != UVE; ++UVI) {
    const MDNode *Var = *UVI;
    if (Processed.count(Var))
      continue;

    // History contains relevant DBG_VALUE instructions for Var and instructions
    // clobbering it.
    SmallVectorImpl<const MachineInstr *> &History = DbgValues[Var];
    if (History.empty())
      continue;
    const MachineInstr *MInsn = History.front();

    DIVariable DV(Var);
    LexicalScope *Scope = NULL;
    if (DV.getTag() == dwarf::DW_TAG_arg_variable &&
        DISubprogram(DV.getContext()).describes(CurFn->getFunction()))
      Scope = LScopes.getCurrentFunctionScope();
    else if (MDNode *IA = DV.getInlinedAt())
      Scope = LScopes.findInlinedScope(DebugLoc::getFromDILocation(IA));
    else
      Scope = LScopes.findLexicalScope(cast<MDNode>(DV->getOperand(1)));
    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    Processed.insert(DV);
    assert(MInsn->isDebugValue() && "History must begin with debug value");
    DbgVariable *AbsVar = findAbstractVariable(DV, MInsn->getDebugLoc());
    DbgVariable *RegVar = new DbgVariable(DV, AbsVar, this);
    if (!addCurrentFnArgument(RegVar, Scope))
      addScopeVariable(Scope, RegVar);
    if (AbsVar)
      AbsVar->setMInsn(MInsn);

    // Simplify ranges that are fully coalesced.
    if (History.size() <= 1 ||
        (History.size() == 2 && MInsn->isIdenticalTo(History.back()))) {
      RegVar->setMInsn(MInsn);
      continue;
    }

    // Handle multiple DBG_VALUE instructions describing one variable.
    RegVar->setDotDebugLocOffset(DotDebugLocEntries.size());

    for (SmallVectorImpl<const MachineInstr *>::const_iterator
             HI = History.begin(),
             HE = History.end();
         HI != HE; ++HI) {
      const MachineInstr *Begin = *HI;
      assert(Begin->isDebugValue() && "Invalid History entry");

      // Check if DBG_VALUE is truncating a range.
      if (Begin->getNumOperands() > 1 && Begin->getOperand(0).isReg() &&
          !Begin->getOperand(0).getReg())
        continue;

      // Compute the range for a register location.
      const MCSymbol *FLabel = getLabelBeforeInsn(Begin);
      const MCSymbol *SLabel = 0;

      if (HI + 1 == HE)
        // If Begin is the last instruction in History then its value is valid
        // until the end of the function.
        SLabel = FunctionEndSym;
      else {
        const MachineInstr *End = HI[1];
        DEBUG(dbgs() << "DotDebugLoc Pair:\n"
                     << "\t" << *Begin << "\t" << *End << "\n");
        if (End->isDebugValue())
          SLabel = getLabelBeforeInsn(End);
        else {
          // End is a normal instruction clobbering the range.
          SLabel = getLabelAfterInsn(End);
          assert(SLabel && "Forgot label after clobber instruction");
          ++HI;
        }
      }

      // The value is valid until the next DBG_VALUE or clobber.
      DotDebugLocEntries.push_back(
          getDebugLocEntry(Asm, FLabel, SLabel, Begin));
    }
    DotDebugLocEntries.push_back(DotDebugLocEntry());
  }

  // Collect info for variables that were optimized out.
  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  DIArray Variables = DISubprogram(FnScope->getScopeNode()).getVariables();
  for (unsigned i = 0, e = Variables.getNumElements(); i != e; ++i) {
    DIVariable DV(Variables.getElement(i));
    if (!DV || !DV.isVariable() || !Processed.insert(DV))
      continue;
    if (LexicalScope *Scope = LScopes.findLexicalScope(DV.getContext()))
      addScopeVariable(Scope, new DbgVariable(DV, NULL, this));
  }
}

// Return Label preceding the instruction.
MCSymbol *DwarfDebug::getLabelBeforeInsn(const MachineInstr *MI) {
  MCSymbol *Label = LabelsBeforeInsn.lookup(MI);
  assert(Label && "Didn't insert label before instruction");
  return Label;
}

// Return Label immediately following the instruction.
MCSymbol *DwarfDebug::getLabelAfterInsn(const MachineInstr *MI) {
  return LabelsAfterInsn.lookup(MI);
}

// Process beginning of an instruction.
void DwarfDebug::beginInstruction(const MachineInstr *MI) {
  assert(CurMI == 0);
  CurMI = MI;
  // Check if source location changes, but ignore DBG_VALUE locations.
  if (!MI->isDebugValue()) {
    DebugLoc DL = MI->getDebugLoc();
    if (DL != PrevInstLoc && (!DL.isUnknown() || UnknownLocations)) {
      unsigned Flags = 0;
      PrevInstLoc = DL;
      if (DL == PrologEndLoc) {
        Flags |= DWARF2_FLAG_PROLOGUE_END;
        PrologEndLoc = DebugLoc();
      }
      if (PrologEndLoc.isUnknown())
        Flags |= DWARF2_FLAG_IS_STMT;

      if (!DL.isUnknown()) {
        const MDNode *Scope = DL.getScope(Asm->MF->getFunction()->getContext());
        recordSourceLine(DL.getLine(), DL.getCol(), Scope, Flags);
      } else
        recordSourceLine(0, 0, 0, 0);
    }
  }

  // Insert labels where requested.
  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
      LabelsBeforeInsn.find(MI);

  // No label needed.
  if (I == LabelsBeforeInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  if (!PrevLabel) {
    PrevLabel = MMI->getContext().CreateTempSymbol();
    Asm->OutStreamer.EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

// Process end of an instruction.
void DwarfDebug::endInstruction() {
  assert(CurMI != 0);
  // Don't create a new label after DBG_VALUE instructions.
  // They don't generate code.
  if (!CurMI->isDebugValue())
    PrevLabel = 0;

  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
      LabelsAfterInsn.find(CurMI);
  CurMI = 0;

  // No label needed.
  if (I == LabelsAfterInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  // We need a label after this instruction.
  if (!PrevLabel) {
    PrevLabel = MMI->getContext().CreateTempSymbol();
    Asm->OutStreamer.EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

// Each LexicalScope has first instruction and last instruction to mark
// beginning and end of a scope respectively. Create an inverse map that list
// scopes starts (and ends) with an instruction. One instruction may start (or
// end) multiple scopes. Ignore scopes that are not reachable.
void DwarfDebug::identifyScopeMarkers() {
  SmallVector<LexicalScope *, 4> WorkList;
  WorkList.push_back(LScopes.getCurrentFunctionScope());
  while (!WorkList.empty()) {
    LexicalScope *S = WorkList.pop_back_val();

    const SmallVectorImpl<LexicalScope *> &Children = S->getChildren();
    if (!Children.empty())
      for (SmallVectorImpl<LexicalScope *>::const_iterator
               SI = Children.begin(),
               SE = Children.end();
           SI != SE; ++SI)
        WorkList.push_back(*SI);

    if (S->isAbstractScope())
      continue;

    const SmallVectorImpl<InsnRange> &Ranges = S->getRanges();
    if (Ranges.empty())
      continue;
    for (SmallVectorImpl<InsnRange>::const_iterator RI = Ranges.begin(),
                                                    RE = Ranges.end();
         RI != RE; ++RI) {
      assert(RI->first && "InsnRange does not have first instruction!");
      assert(RI->second && "InsnRange does not have second instruction!");
      requestLabelBeforeInsn(RI->first);
      requestLabelAfterInsn(RI->second);
    }
  }
}

// Gather pre-function debug information.  Assumes being called immediately
// after the function entry point has been emitted.
void DwarfDebug::beginFunction(const MachineFunction *MF) {
  CurFn = MF;

  // If there's no debug info for the function we're not going to do anything.
  if (!MMI->hasDebugInfo())
    return;

  // Grab the lexical scopes for the function, if we don't have any of those
  // then we're not going to be able to do anything.
  LScopes.initialize(*MF);
  if (LScopes.empty())
    return;

  assert(UserVariables.empty() && DbgValues.empty() && "Maps weren't cleaned");

  // Make sure that each lexical scope will have a begin/end label.
  identifyScopeMarkers();

  // Set DwarfDwarfCompileUnitID in MCContext to the Compile Unit this function
  // belongs to so that we add to the correct per-cu line table in the
  // non-asm case.
  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  DwarfCompileUnit *TheCU = SPMap.lookup(FnScope->getScopeNode());
  assert(TheCU && "Unable to find compile unit!");
  if (Asm->OutStreamer.hasRawTextSupport())
    // Use a single line table if we are generating assembly.
    Asm->OutStreamer.getContext().setDwarfCompileUnitID(0);
  else
    Asm->OutStreamer.getContext().setDwarfCompileUnitID(TheCU->getUniqueID());

  // Check the current section against the standard text section. If different
  // keep track so that we will know when we're emitting functions into multiple
  // sections.
  if (Asm->getObjFileLowering().getTextSection() != Asm->getCurrentSection())
    UsedNonDefaultText = true;

  // Emit a label for the function so that we have a beginning address.
  FunctionBeginSym = Asm->GetTempSymbol("func_begin", Asm->getFunctionNumber());
  // Assumes in correct section after the entry point.
  Asm->OutStreamer.EmitLabel(FunctionBeginSym);

  const TargetRegisterInfo *TRI = Asm->TM.getRegisterInfo();
  // LiveUserVar - Map physreg numbers to the MDNode they contain.
  std::vector<const MDNode *> LiveUserVar(TRI->getNumRegs());

  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end(); I != E;
       ++I) {
    bool AtBlockEntry = true;
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MI = II;

      if (MI->isDebugValue()) {
        assert(MI->getNumOperands() > 1 && "Invalid machine instruction!");

        // Keep track of user variables.
        const MDNode *Var =
            MI->getOperand(MI->getNumOperands() - 1).getMetadata();

        // Variable is in a register, we need to check for clobbers.
        if (isDbgValueInDefinedReg(MI))
          LiveUserVar[MI->getOperand(0).getReg()] = Var;

        // Check the history of this variable.
        SmallVectorImpl<const MachineInstr *> &History = DbgValues[Var];
        if (History.empty()) {
          UserVariables.push_back(Var);
          // The first mention of a function argument gets the FunctionBeginSym
          // label, so arguments are visible when breaking at function entry.
          DIVariable DV(Var);
          if (DV.isVariable() && DV.getTag() == dwarf::DW_TAG_arg_variable &&
              getDISubprogram(DV.getContext()).describes(MF->getFunction()))
            LabelsBeforeInsn[MI] = FunctionBeginSym;
        } else {
          // We have seen this variable before. Try to coalesce DBG_VALUEs.
          const MachineInstr *Prev = History.back();
          if (Prev->isDebugValue()) {
            // Coalesce identical entries at the end of History.
            if (History.size() >= 2 &&
                Prev->isIdenticalTo(History[History.size() - 2])) {
              DEBUG(dbgs() << "Coalescing identical DBG_VALUE entries:\n"
                           << "\t" << *Prev << "\t"
                           << *History[History.size() - 2] << "\n");
              History.pop_back();
            }

            // Terminate old register assignments that don't reach MI;
            MachineFunction::const_iterator PrevMBB = Prev->getParent();
            if (PrevMBB != I && (!AtBlockEntry || std::next(PrevMBB) != I) &&
                isDbgValueInDefinedReg(Prev)) {
              // Previous register assignment needs to terminate at the end of
              // its basic block.
              MachineBasicBlock::const_iterator LastMI =
                  PrevMBB->getLastNonDebugInstr();
              if (LastMI == PrevMBB->end()) {
                // Drop DBG_VALUE for empty range.
                DEBUG(dbgs() << "Dropping DBG_VALUE for empty range:\n"
                             << "\t" << *Prev << "\n");
                History.pop_back();
              } else if (std::next(PrevMBB) != PrevMBB->getParent()->end())
                // Terminate after LastMI.
                History.push_back(LastMI);
            }
          }
        }
        History.push_back(MI);
      } else {
        // Not a DBG_VALUE instruction.
        if (!MI->isLabel())
          AtBlockEntry = false;

        // First known non-DBG_VALUE and non-frame setup location marks
        // the beginning of the function body.
        if (!MI->getFlag(MachineInstr::FrameSetup) &&
            (PrologEndLoc.isUnknown() && !MI->getDebugLoc().isUnknown()))
          PrologEndLoc = MI->getDebugLoc();

        // Check if the instruction clobbers any registers with debug vars.
        for (MachineInstr::const_mop_iterator MOI = MI->operands_begin(),
                                              MOE = MI->operands_end();
             MOI != MOE; ++MOI) {
          if (!MOI->isReg() || !MOI->isDef() || !MOI->getReg())
            continue;
          for (MCRegAliasIterator AI(MOI->getReg(), TRI, true); AI.isValid();
               ++AI) {
            unsigned Reg = *AI;
            const MDNode *Var = LiveUserVar[Reg];
            if (!Var)
              continue;
            // Reg is now clobbered.
            LiveUserVar[Reg] = 0;

            // Was MD last defined by a DBG_VALUE referring to Reg?
            DbgValueHistoryMap::iterator HistI = DbgValues.find(Var);
            if (HistI == DbgValues.end())
              continue;
            SmallVectorImpl<const MachineInstr *> &History = HistI->second;
            if (History.empty())
              continue;
            const MachineInstr *Prev = History.back();
            // Sanity-check: Register assignments are terminated at the end of
            // their block.
            if (!Prev->isDebugValue() || Prev->getParent() != MI->getParent())
              continue;
            // Is the variable still in Reg?
            if (!isDbgValueInDefinedReg(Prev) ||
                Prev->getOperand(0).getReg() != Reg)
              continue;
            // Var is clobbered. Make sure the next instruction gets a label.
            History.push_back(MI);
          }
        }
      }
    }
  }

  for (DbgValueHistoryMap::iterator I = DbgValues.begin(), E = DbgValues.end();
       I != E; ++I) {
    SmallVectorImpl<const MachineInstr *> &History = I->second;
    if (History.empty())
      continue;

    // Make sure the final register assignments are terminated.
    const MachineInstr *Prev = History.back();
    if (Prev->isDebugValue() && isDbgValueInDefinedReg(Prev)) {
      const MachineBasicBlock *PrevMBB = Prev->getParent();
      MachineBasicBlock::const_iterator LastMI =
          PrevMBB->getLastNonDebugInstr();
      if (LastMI == PrevMBB->end())
        // Drop DBG_VALUE for empty range.
        History.pop_back();
      else if (PrevMBB != &PrevMBB->getParent()->back()) {
        // Terminate after LastMI.
        History.push_back(LastMI);
      }
    }
    // Request labels for the full history.
    for (unsigned i = 0, e = History.size(); i != e; ++i) {
      const MachineInstr *MI = History[i];
      if (MI->isDebugValue())
        requestLabelBeforeInsn(MI);
      else
        requestLabelAfterInsn(MI);
    }
  }

  PrevInstLoc = DebugLoc();
  PrevLabel = FunctionBeginSym;

  // Record beginning of function.
  if (!PrologEndLoc.isUnknown()) {
    DebugLoc FnStartDL =
        PrologEndLoc.getFnDebugLoc(MF->getFunction()->getContext());
    recordSourceLine(
        FnStartDL.getLine(), FnStartDL.getCol(),
        FnStartDL.getScope(MF->getFunction()->getContext()),
        // We'd like to list the prologue as "not statements" but GDB behaves
        // poorly if we do that. Revisit this with caution/GDB (7.5+) testing.
        DWARF2_FLAG_IS_STMT);
  }
}

void DwarfDebug::addScopeVariable(LexicalScope *LS, DbgVariable *Var) {
  SmallVectorImpl<DbgVariable *> &Vars = ScopeVariables[LS];
  DIVariable DV = Var->getVariable();
  // Variables with positive arg numbers are parameters.
  if (unsigned ArgNum = DV.getArgNumber()) {
    // Keep all parameters in order at the start of the variable list to ensure
    // function types are correct (no out-of-order parameters)
    //
    // This could be improved by only doing it for optimized builds (unoptimized
    // builds have the right order to begin with), searching from the back (this
    // would catch the unoptimized case quickly), or doing a binary search
    // rather than linear search.
    SmallVectorImpl<DbgVariable *>::iterator I = Vars.begin();
    while (I != Vars.end()) {
      unsigned CurNum = (*I)->getVariable().getArgNumber();
      // A local (non-parameter) variable has been found, insert immediately
      // before it.
      if (CurNum == 0)
        break;
      // A later indexed parameter has been found, insert immediately before it.
      if (CurNum > ArgNum)
        break;
      ++I;
    }
    Vars.insert(I, Var);
    return;
  }

  Vars.push_back(Var);
}

// Gather and emit post-function debug information.
void DwarfDebug::endFunction(const MachineFunction *MF) {
  // Every beginFunction(MF) call should be followed by an endFunction(MF) call,
  // though the beginFunction may not be called at all.
  // We should handle both cases.
  if (CurFn == 0)
    CurFn = MF;
  else
    assert(CurFn == MF);
  assert(CurFn != 0);

  if (!MMI->hasDebugInfo() || LScopes.empty()) {
    CurFn = 0;
    return;
  }

  // Define end label for subprogram.
  FunctionEndSym = Asm->GetTempSymbol("func_end", Asm->getFunctionNumber());
  // Assumes in correct section after the entry point.
  Asm->OutStreamer.EmitLabel(FunctionEndSym);

  // Set DwarfDwarfCompileUnitID in MCContext to default value.
  Asm->OutStreamer.getContext().setDwarfCompileUnitID(0);

  SmallPtrSet<const MDNode *, 16> ProcessedVars;
  collectVariableInfo(ProcessedVars);

  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  DwarfCompileUnit *TheCU = SPMap.lookup(FnScope->getScopeNode());
  assert(TheCU && "Unable to find compile unit!");

  // Construct abstract scopes.
  ArrayRef<LexicalScope *> AList = LScopes.getAbstractScopesList();
  for (unsigned i = 0, e = AList.size(); i != e; ++i) {
    LexicalScope *AScope = AList[i];
    DISubprogram SP(AScope->getScopeNode());
    if (SP.isSubprogram()) {
      // Collect info for variables that were optimized out.
      DIArray Variables = SP.getVariables();
      for (unsigned i = 0, e = Variables.getNumElements(); i != e; ++i) {
        DIVariable DV(Variables.getElement(i));
        if (!DV || !DV.isVariable() || !ProcessedVars.insert(DV))
          continue;
        // Check that DbgVariable for DV wasn't created earlier, when
        // findAbstractVariable() was called for inlined instance of DV.
        LLVMContext &Ctx = DV->getContext();
        DIVariable CleanDV = cleanseInlinedVariable(DV, Ctx);
        if (AbstractVariables.lookup(CleanDV))
          continue;
        if (LexicalScope *Scope = LScopes.findAbstractScope(DV.getContext()))
          addScopeVariable(Scope, new DbgVariable(DV, NULL, this));
      }
    }
    if (ProcessedSPNodes.count(AScope->getScopeNode()) == 0)
      constructScopeDIE(TheCU, AScope);
  }

  DIE *CurFnDIE = constructScopeDIE(TheCU, FnScope);
  if (!CurFn->getTarget().Options.DisableFramePointerElim(*CurFn))
    TheCU->addFlag(CurFnDIE, dwarf::DW_AT_APPLE_omit_frame_ptr);

  // Add the range of this function to the list of ranges for the CU.
  RangeSpan Span(FunctionBeginSym, FunctionEndSym);
  TheCU->addRange(std::move(Span));

  // Clear debug info
  for (ScopeVariablesMap::iterator I = ScopeVariables.begin(),
                                   E = ScopeVariables.end();
       I != E; ++I)
    DeleteContainerPointers(I->second);
  ScopeVariables.clear();
  DeleteContainerPointers(CurrentFnArguments);
  UserVariables.clear();
  DbgValues.clear();
  AbstractVariables.clear();
  LabelsBeforeInsn.clear();
  LabelsAfterInsn.clear();
  PrevLabel = NULL;
  CurFn = 0;
}

// Register a source line with debug info. Returns the  unique label that was
// emitted and which provides correspondence to the source line list.
void DwarfDebug::recordSourceLine(unsigned Line, unsigned Col, const MDNode *S,
                                  unsigned Flags) {
  StringRef Fn;
  StringRef Dir;
  unsigned Src = 1;
  unsigned Discriminator = 0;
  if (S) {
    DIDescriptor Scope(S);

    if (Scope.isCompileUnit()) {
      DICompileUnit CU(S);
      Fn = CU.getFilename();
      Dir = CU.getDirectory();
    } else if (Scope.isFile()) {
      DIFile F(S);
      Fn = F.getFilename();
      Dir = F.getDirectory();
    } else if (Scope.isSubprogram()) {
      DISubprogram SP(S);
      Fn = SP.getFilename();
      Dir = SP.getDirectory();
    } else if (Scope.isLexicalBlockFile()) {
      DILexicalBlockFile DBF(S);
      Fn = DBF.getFilename();
      Dir = DBF.getDirectory();
    } else if (Scope.isLexicalBlock()) {
      DILexicalBlock DB(S);
      Fn = DB.getFilename();
      Dir = DB.getDirectory();
      Discriminator = DB.getDiscriminator();
    } else
      llvm_unreachable("Unexpected scope info");

    Src = getOrCreateSourceID(
        Fn, Dir, Asm->OutStreamer.getContext().getDwarfCompileUnitID());
  }
  Asm->OutStreamer.EmitDwarfLocDirective(Src, Line, Col, Flags, 0,
                                         Discriminator, Fn);
}

//===----------------------------------------------------------------------===//
// Emit Methods
//===----------------------------------------------------------------------===//

// Compute the size and offset of a DIE. The offset is relative to start of the
// CU. It returns the offset after laying out the DIE.
unsigned DwarfFile::computeSizeAndOffset(DIE *Die, unsigned Offset) {
  // Get the children.
  const std::vector<DIE *> &Children = Die->getChildren();

  // Record the abbreviation.
  assignAbbrevNumber(Die->getAbbrev());

  // Get the abbreviation for this DIE.
  const DIEAbbrev &Abbrev = Die->getAbbrev();

  // Set DIE offset
  Die->setOffset(Offset);

  // Start the size with the size of abbreviation code.
  Offset += getULEB128Size(Die->getAbbrevNumber());

  const SmallVectorImpl<DIEValue *> &Values = Die->getValues();
  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();

  // Size the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    // Size attribute value.
    Offset += Values[i]->SizeOf(Asm, AbbrevData[i].getForm());

  // Size the DIE children if any.
  if (!Children.empty()) {
    assert(Abbrev.hasChildren() && "Children flag not set");

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      Offset = computeSizeAndOffset(Children[j], Offset);

    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die->setSize(Offset - Die->getOffset());
  return Offset;
}

// Compute the size and offset for each DIE.
void DwarfFile::computeSizeAndOffsets() {
  // Offset from the first CU in the debug info section is 0 initially.
  unsigned SecOffset = 0;

  // Iterate over each compile unit and set the size and offsets for each
  // DIE within each compile unit. All offsets are CU relative.
  for (SmallVectorImpl<DwarfUnit *>::const_iterator I = CUs.begin(),
                                                    E = CUs.end();
       I != E; ++I) {
    (*I)->setDebugInfoOffset(SecOffset);

    // CU-relative offset is reset to 0 here.
    unsigned Offset = sizeof(int32_t) +      // Length of Unit Info
                      (*I)->getHeaderSize(); // Unit-specific headers

    // EndOffset here is CU-relative, after laying out
    // all of the CU DIE.
    unsigned EndOffset = computeSizeAndOffset((*I)->getUnitDie(), Offset);
    SecOffset += EndOffset;
  }
}

// Emit initial Dwarf sections with a label at the start of each one.
void DwarfDebug::emitSectionLabels() {
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  // Dwarf sections base addresses.
  DwarfInfoSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfInfoSection(), "section_info");
  if (useSplitDwarf())
    DwarfInfoDWOSectionSym =
        emitSectionSym(Asm, TLOF.getDwarfInfoDWOSection(), "section_info_dwo");
  DwarfAbbrevSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfAbbrevSection(), "section_abbrev");
  if (useSplitDwarf())
    DwarfAbbrevDWOSectionSym = emitSectionSym(
        Asm, TLOF.getDwarfAbbrevDWOSection(), "section_abbrev_dwo");
  if (GenerateARangeSection)
    emitSectionSym(Asm, TLOF.getDwarfARangesSection());

  DwarfLineSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfLineSection(), "section_line");
  emitSectionSym(Asm, TLOF.getDwarfLocSection());
  if (GenerateGnuPubSections) {
    DwarfGnuPubNamesSectionSym =
        emitSectionSym(Asm, TLOF.getDwarfGnuPubNamesSection());
    DwarfGnuPubTypesSectionSym =
        emitSectionSym(Asm, TLOF.getDwarfGnuPubTypesSection());
  } else if (HasDwarfPubSections) {
    emitSectionSym(Asm, TLOF.getDwarfPubNamesSection());
    emitSectionSym(Asm, TLOF.getDwarfPubTypesSection());
  }

  DwarfStrSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfStrSection(), "info_string");
  if (useSplitDwarf()) {
    DwarfStrDWOSectionSym =
        emitSectionSym(Asm, TLOF.getDwarfStrDWOSection(), "skel_string");
    DwarfAddrSectionSym =
        emitSectionSym(Asm, TLOF.getDwarfAddrSection(), "addr_sec");
  }
  DwarfDebugRangeSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfRangesSection(), "debug_range");

  DwarfDebugLocSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfLocSection(), "section_debug_loc");
}

// Recursively emits a debug information entry.
void DwarfDebug::emitDIE(DIE *Die) {
  // Get the abbreviation for this DIE.
  const DIEAbbrev &Abbrev = Die->getAbbrev();

  // Emit the code (index) for the abbreviation.
  if (Asm->isVerbose())
    Asm->OutStreamer.AddComment("Abbrev [" + Twine(Abbrev.getNumber()) +
                                "] 0x" + Twine::utohexstr(Die->getOffset()) +
                                ":0x" + Twine::utohexstr(Die->getSize()) + " " +
                                dwarf::TagString(Abbrev.getTag()));
  Asm->EmitULEB128(Abbrev.getNumber());

  const SmallVectorImpl<DIEValue *> &Values = Die->getValues();
  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev.getData();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    dwarf::Attribute Attr = AbbrevData[i].getAttribute();
    dwarf::Form Form = AbbrevData[i].getForm();
    assert(Form && "Too many attributes for DIE (check abbreviation)");

    if (Asm->isVerbose()) {
      Asm->OutStreamer.AddComment(dwarf::AttributeString(Attr));
      if (Attr == dwarf::DW_AT_accessibility)
        Asm->OutStreamer.AddComment(dwarf::AccessibilityString(
            cast<DIEInteger>(Values[i])->getValue()));
    }

    // Emit an attribute using the defined form.
    Values[i]->EmitValue(Asm, Form);
  }

  // Emit the DIE children if any.
  if (Abbrev.hasChildren()) {
    const std::vector<DIE *> &Children = Die->getChildren();

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      emitDIE(Children[j]);

    Asm->OutStreamer.AddComment("End Of Children Mark");
    Asm->EmitInt8(0);
  }
}

// Emit the various dwarf units to the unit section USection with
// the abbreviations going into ASection.
void DwarfFile::emitUnits(DwarfDebug *DD, const MCSection *ASection,
                          const MCSymbol *ASectionSym) {
  for (SmallVectorImpl<DwarfUnit *>::iterator I = CUs.begin(), E = CUs.end();
       I != E; ++I) {
    DwarfUnit *TheU = *I;
    DIE *Die = TheU->getUnitDie();
    const MCSection *USection = TheU->getSection();
    Asm->OutStreamer.SwitchSection(USection);

    // Emit the compile units header.
    Asm->OutStreamer.EmitLabel(TheU->getLabelBegin());

    // Emit size of content not including length itself
    Asm->OutStreamer.AddComment("Length of Unit");
    Asm->EmitInt32(TheU->getHeaderSize() + Die->getSize());

    TheU->emitHeader(ASection, ASectionSym);

    DD->emitDIE(Die);
    Asm->OutStreamer.EmitLabel(TheU->getLabelEnd());
  }
}

// Emit the debug info section.
void DwarfDebug::emitDebugInfo() {
  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;

  Holder.emitUnits(this, Asm->getObjFileLowering().getDwarfAbbrevSection(),
                   DwarfAbbrevSectionSym);
}

// Emit the abbreviation section.
void DwarfDebug::emitAbbreviations() {
  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;

  Holder.emitAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevSection());
}

void DwarfFile::emitAbbrevs(const MCSection *Section) {
  // Check to see if it is worth the effort.
  if (!Abbreviations.empty()) {
    // Start the debug abbrev section.
    Asm->OutStreamer.SwitchSection(Section);

    // For each abbrevation.
    for (unsigned i = 0, N = Abbreviations.size(); i < N; ++i) {
      // Get abbreviation data
      const DIEAbbrev *Abbrev = Abbreviations[i];

      // Emit the abbrevations code (base 1 index.)
      Asm->EmitULEB128(Abbrev->getNumber(), "Abbreviation Code");

      // Emit the abbreviations data.
      Abbrev->Emit(Asm);
    }

    // Mark end of abbreviations.
    Asm->EmitULEB128(0, "EOM(3)");
  }
}

// Emit the last address of the section and the end of the line matrix.
void DwarfDebug::emitEndOfLineMatrix(unsigned SectionEnd) {
  // Define last address of section.
  Asm->OutStreamer.AddComment("Extended Op");
  Asm->EmitInt8(0);

  Asm->OutStreamer.AddComment("Op size");
  Asm->EmitInt8(Asm->getDataLayout().getPointerSize() + 1);
  Asm->OutStreamer.AddComment("DW_LNE_set_address");
  Asm->EmitInt8(dwarf::DW_LNE_set_address);

  Asm->OutStreamer.AddComment("Section end label");

  Asm->OutStreamer.EmitSymbolValue(
      Asm->GetTempSymbol("section_end", SectionEnd),
      Asm->getDataLayout().getPointerSize());

  // Mark end of matrix.
  Asm->OutStreamer.AddComment("DW_LNE_end_sequence");
  Asm->EmitInt8(0);
  Asm->EmitInt8(1);
  Asm->EmitInt8(1);
}

// Emit visible names into a hashed accelerator table section.
void DwarfDebug::emitAccelNames() {
  DwarfAccelTable AT(
      DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4));
  for (SmallVectorImpl<DwarfUnit *>::const_iterator I = getUnits().begin(),
                                                    E = getUnits().end();
       I != E; ++I) {
    DwarfUnit *TheU = *I;
    const StringMap<std::vector<const DIE *> > &Names = TheU->getAccelNames();
    for (StringMap<std::vector<const DIE *> >::const_iterator
             GI = Names.begin(),
             GE = Names.end();
         GI != GE; ++GI) {
      StringRef Name = GI->getKey();
      const std::vector<const DIE *> &Entities = GI->second;
      for (std::vector<const DIE *>::const_iterator DI = Entities.begin(),
                                                    DE = Entities.end();
           DI != DE; ++DI)
        AT.AddName(Name, *DI);
    }
  }

  AT.FinalizeTable(Asm, "Names");
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfAccelNamesSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("names_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Emit objective C classes and categories into a hashed accelerator table
// section.
void DwarfDebug::emitAccelObjC() {
  DwarfAccelTable AT(
      DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4));
  for (SmallVectorImpl<DwarfUnit *>::const_iterator I = getUnits().begin(),
                                                    E = getUnits().end();
       I != E; ++I) {
    DwarfUnit *TheU = *I;
    const StringMap<std::vector<const DIE *> > &Names = TheU->getAccelObjC();
    for (StringMap<std::vector<const DIE *> >::const_iterator
             GI = Names.begin(),
             GE = Names.end();
         GI != GE; ++GI) {
      StringRef Name = GI->getKey();
      const std::vector<const DIE *> &Entities = GI->second;
      for (std::vector<const DIE *>::const_iterator DI = Entities.begin(),
                                                    DE = Entities.end();
           DI != DE; ++DI)
        AT.AddName(Name, *DI);
    }
  }

  AT.FinalizeTable(Asm, "ObjC");
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfAccelObjCSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("objc_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Emit namespace dies into a hashed accelerator table.
void DwarfDebug::emitAccelNamespaces() {
  DwarfAccelTable AT(
      DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4));
  for (SmallVectorImpl<DwarfUnit *>::const_iterator I = getUnits().begin(),
                                                    E = getUnits().end();
       I != E; ++I) {
    DwarfUnit *TheU = *I;
    const StringMap<std::vector<const DIE *> > &Names =
        TheU->getAccelNamespace();
    for (StringMap<std::vector<const DIE *> >::const_iterator
             GI = Names.begin(),
             GE = Names.end();
         GI != GE; ++GI) {
      StringRef Name = GI->getKey();
      const std::vector<const DIE *> &Entities = GI->second;
      for (std::vector<const DIE *>::const_iterator DI = Entities.begin(),
                                                    DE = Entities.end();
           DI != DE; ++DI)
        AT.AddName(Name, *DI);
    }
  }

  AT.FinalizeTable(Asm, "namespac");
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfAccelNamespaceSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("namespac_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Emit type dies into a hashed accelerator table.
void DwarfDebug::emitAccelTypes() {
  std::vector<DwarfAccelTable::Atom> Atoms;
  Atoms.push_back(
      DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4));
  Atoms.push_back(
      DwarfAccelTable::Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2));
  Atoms.push_back(
      DwarfAccelTable::Atom(dwarf::DW_ATOM_type_flags, dwarf::DW_FORM_data1));
  DwarfAccelTable AT(Atoms);
  for (SmallVectorImpl<DwarfUnit *>::const_iterator I = getUnits().begin(),
                                                    E = getUnits().end();
       I != E; ++I) {
    DwarfUnit *TheU = *I;
    const StringMap<std::vector<std::pair<const DIE *, unsigned> > > &Names =
        TheU->getAccelTypes();
    for (StringMap<
             std::vector<std::pair<const DIE *, unsigned> > >::const_iterator
             GI = Names.begin(),
             GE = Names.end();
         GI != GE; ++GI) {
      StringRef Name = GI->getKey();
      const std::vector<std::pair<const DIE *, unsigned> > &Entities =
          GI->second;
      for (std::vector<std::pair<const DIE *, unsigned> >::const_iterator
               DI = Entities.begin(),
               DE = Entities.end();
           DI != DE; ++DI)
        AT.AddName(Name, DI->first, DI->second);
    }
  }

  AT.FinalizeTable(Asm, "types");
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfAccelTypesSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("types_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Public name handling.
// The format for the various pubnames:
//
// dwarf pubnames - offset/name pairs where the offset is the offset into the CU
// for the DIE that is named.
//
// gnu pubnames - offset/index value/name tuples where the offset is the offset
// into the CU and the index value is computed according to the type of value
// for the DIE that is named.
//
// For type units the offset is the offset of the skeleton DIE. For split dwarf
// it's the offset within the debug_info/debug_types dwo section, however, the
// reference in the pubname header doesn't change.

/// computeIndexValue - Compute the gdb index value for the DIE and CU.
static dwarf::PubIndexEntryDescriptor computeIndexValue(DwarfUnit *CU,
                                                        const DIE *Die) {
  dwarf::GDBIndexEntryLinkage Linkage = dwarf::GIEL_STATIC;

  // We could have a specification DIE that has our most of our knowledge,
  // look for that now.
  DIEValue *SpecVal = Die->findAttribute(dwarf::DW_AT_specification);
  if (SpecVal) {
    DIE *SpecDIE = cast<DIEEntry>(SpecVal)->getEntry();
    if (SpecDIE->findAttribute(dwarf::DW_AT_external))
      Linkage = dwarf::GIEL_EXTERNAL;
  } else if (Die->findAttribute(dwarf::DW_AT_external))
    Linkage = dwarf::GIEL_EXTERNAL;

  switch (Die->getTag()) {
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
    return dwarf::PubIndexEntryDescriptor(
        dwarf::GIEK_TYPE, CU->getLanguage() != dwarf::DW_LANG_C_plus_plus
                              ? dwarf::GIEL_STATIC
                              : dwarf::GIEL_EXTERNAL);
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_subrange_type:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_TYPE, dwarf::GIEL_STATIC);
  case dwarf::DW_TAG_namespace:
    return dwarf::GIEK_TYPE;
  case dwarf::DW_TAG_subprogram:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_FUNCTION, Linkage);
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_variable:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_VARIABLE, Linkage);
  case dwarf::DW_TAG_enumerator:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_VARIABLE,
                                          dwarf::GIEL_STATIC);
  default:
    return dwarf::GIEK_NONE;
  }
}

/// emitDebugPubNames - Emit visible names into a debug pubnames section.
///
void DwarfDebug::emitDebugPubNames(bool GnuStyle) {
  const MCSection *PSec =
      GnuStyle ? Asm->getObjFileLowering().getDwarfGnuPubNamesSection()
               : Asm->getObjFileLowering().getDwarfPubNamesSection();

  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
  const SmallVectorImpl<DwarfUnit *> &Units = Holder.getUnits();
  for (unsigned i = 0; i != Units.size(); ++i) {
    DwarfUnit *TheU = Units[i];
    unsigned ID = TheU->getUniqueID();

    // Start the dwarf pubnames section.
    Asm->OutStreamer.SwitchSection(PSec);

    // Emit a label so we can reference the beginning of this pubname section.
    if (GnuStyle)
      Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("gnu_pubnames", ID));

    // Emit the header.
    Asm->OutStreamer.AddComment("Length of Public Names Info");
    MCSymbol *BeginLabel = Asm->GetTempSymbol("pubnames_begin", ID);
    MCSymbol *EndLabel = Asm->GetTempSymbol("pubnames_end", ID);
    Asm->EmitLabelDifference(EndLabel, BeginLabel, 4);

    Asm->OutStreamer.EmitLabel(BeginLabel);

    Asm->OutStreamer.AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DW_PUBNAMES_VERSION);

    Asm->OutStreamer.AddComment("Offset of Compilation Unit Info");
    Asm->EmitSectionOffset(TheU->getLabelBegin(), TheU->getSectionSym());

    Asm->OutStreamer.AddComment("Compilation Unit Length");
    Asm->EmitLabelDifference(TheU->getLabelEnd(), TheU->getLabelBegin(), 4);

    // Emit the pubnames for this compilation unit.
    const StringMap<const DIE *> &Globals = getUnits()[ID]->getGlobalNames();
    for (StringMap<const DIE *>::const_iterator GI = Globals.begin(),
                                                GE = Globals.end();
         GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const DIE *Entity = GI->second;

      Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (GnuStyle) {
        dwarf::PubIndexEntryDescriptor Desc = computeIndexValue(TheU, Entity);
        Asm->OutStreamer.AddComment(
            Twine("Kind: ") + dwarf::GDBIndexEntryKindString(Desc.Kind) + ", " +
            dwarf::GDBIndexEntryLinkageString(Desc.Linkage));
        Asm->EmitInt8(Desc.toBits());
      }

      Asm->OutStreamer.AddComment("External Name");
      Asm->OutStreamer.EmitBytes(StringRef(Name, GI->getKeyLength() + 1));
    }

    Asm->OutStreamer.AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer.EmitLabel(EndLabel);
  }
}

void DwarfDebug::emitDebugPubTypes(bool GnuStyle) {
  const MCSection *PSec =
      GnuStyle ? Asm->getObjFileLowering().getDwarfGnuPubTypesSection()
               : Asm->getObjFileLowering().getDwarfPubTypesSection();

  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
  const SmallVectorImpl<DwarfUnit *> &Units = Holder.getUnits();
  for (unsigned i = 0; i != Units.size(); ++i) {
    DwarfUnit *TheU = Units[i];
    unsigned ID = TheU->getUniqueID();

    // Start the dwarf pubtypes section.
    Asm->OutStreamer.SwitchSection(PSec);

    // Emit a label so we can reference the beginning of this pubtype section.
    if (GnuStyle)
      Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("gnu_pubtypes", ID));

    // Emit the header.
    Asm->OutStreamer.AddComment("Length of Public Types Info");
    MCSymbol *BeginLabel = Asm->GetTempSymbol("pubtypes_begin", ID);
    MCSymbol *EndLabel = Asm->GetTempSymbol("pubtypes_end", ID);
    Asm->EmitLabelDifference(EndLabel, BeginLabel, 4);

    Asm->OutStreamer.EmitLabel(BeginLabel);

    Asm->OutStreamer.AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DW_PUBTYPES_VERSION);

    Asm->OutStreamer.AddComment("Offset of Compilation Unit Info");
    Asm->EmitSectionOffset(TheU->getLabelBegin(), TheU->getSectionSym());

    Asm->OutStreamer.AddComment("Compilation Unit Length");
    Asm->EmitLabelDifference(TheU->getLabelEnd(), TheU->getLabelBegin(), 4);

    // Emit the pubtypes.
    const StringMap<const DIE *> &Globals = getUnits()[ID]->getGlobalTypes();
    for (StringMap<const DIE *>::const_iterator GI = Globals.begin(),
                                                GE = Globals.end();
         GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const DIE *Entity = GI->second;

      Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (GnuStyle) {
        dwarf::PubIndexEntryDescriptor Desc = computeIndexValue(TheU, Entity);
        Asm->OutStreamer.AddComment(
            Twine("Kind: ") + dwarf::GDBIndexEntryKindString(Desc.Kind) + ", " +
            dwarf::GDBIndexEntryLinkageString(Desc.Linkage));
        Asm->EmitInt8(Desc.toBits());
      }

      Asm->OutStreamer.AddComment("External Name");

      // Emit the name with a terminating null byte.
      Asm->OutStreamer.EmitBytes(StringRef(Name, GI->getKeyLength() + 1));
    }

    Asm->OutStreamer.AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer.EmitLabel(EndLabel);
  }
}

// Emit strings into a string section.
void DwarfFile::emitStrings(const MCSection *StrSection,
                            const MCSection *OffsetSection = NULL,
                            const MCSymbol *StrSecSym = NULL) {

  if (StringPool.empty())
    return;

  // Start the dwarf str section.
  Asm->OutStreamer.SwitchSection(StrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<
      std::pair<unsigned, StringMapEntry<std::pair<MCSymbol *, unsigned> > *>,
      64> Entries;

  for (StringMap<std::pair<MCSymbol *, unsigned> >::iterator
           I = StringPool.begin(),
           E = StringPool.end();
       I != E; ++I)
    Entries.push_back(std::make_pair(I->second.second, &*I));

  array_pod_sort(Entries.begin(), Entries.end());

  for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
    // Emit a label for reference from debug information entries.
    Asm->OutStreamer.EmitLabel(Entries[i].second->getValue().first);

    // Emit the string itself with a terminating null byte.
    Asm->OutStreamer.EmitBytes(
        StringRef(Entries[i].second->getKeyData(),
                  Entries[i].second->getKeyLength() + 1));
  }

  // If we've got an offset section go ahead and emit that now as well.
  if (OffsetSection) {
    Asm->OutStreamer.SwitchSection(OffsetSection);
    unsigned offset = 0;
    unsigned size = 4; // FIXME: DWARF64 is 8.
    for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
      Asm->OutStreamer.EmitIntValue(offset, size);
      offset += Entries[i].second->getKeyLength() + 1;
    }
  }
}

// Emit addresses into the section given.
void DwarfFile::emitAddresses(const MCSection *AddrSection) {

  if (AddressPool.empty())
    return;

  // Start the dwarf addr section.
  Asm->OutStreamer.SwitchSection(AddrSection);

  // Order the address pool entries by ID
  SmallVector<const MCExpr *, 64> Entries(AddressPool.size());

  for (AddrPool::iterator I = AddressPool.begin(), E = AddressPool.end();
       I != E; ++I)
    Entries[I->second.Number] =
        I->second.TLS
            ? Asm->getObjFileLowering().getDebugThreadLocalSymbol(I->first)
            : MCSymbolRefExpr::Create(I->first, Asm->OutContext);

  for (unsigned i = 0, e = Entries.size(); i != e; ++i)
    Asm->OutStreamer.EmitValue(Entries[i],
                               Asm->getDataLayout().getPointerSize());
}

// Emit visible names into a debug str section.
void DwarfDebug::emitDebugStr() {
  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
  Holder.emitStrings(Asm->getObjFileLowering().getDwarfStrSection());
}

// Emit locations into the debug loc section.
void DwarfDebug::emitDebugLoc() {
  if (DotDebugLocEntries.empty())
    return;

  for (SmallVectorImpl<DotDebugLocEntry>::iterator
           I = DotDebugLocEntries.begin(),
           E = DotDebugLocEntries.end();
       I != E; ++I) {
    DotDebugLocEntry &Entry = *I;
    if (I + 1 != DotDebugLocEntries.end())
      Entry.Merge(I + 1);
  }

  // Start the dwarf loc section.
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfLocSection());
  unsigned char Size = Asm->getDataLayout().getPointerSize();
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_loc", 0));
  unsigned index = 1;
  for (SmallVectorImpl<DotDebugLocEntry>::iterator
           I = DotDebugLocEntries.begin(),
           E = DotDebugLocEntries.end();
       I != E; ++I, ++index) {
    DotDebugLocEntry &Entry = *I;
    if (Entry.isMerged())
      continue;
    if (Entry.isEmpty()) {
      Asm->OutStreamer.EmitIntValue(0, Size);
      Asm->OutStreamer.EmitIntValue(0, Size);
      Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_loc", index));
    } else {
      Asm->OutStreamer.EmitSymbolValue(Entry.getBeginSym(), Size);
      Asm->OutStreamer.EmitSymbolValue(Entry.getEndSym(), Size);
      DIVariable DV(Entry.getVariable());
      Asm->OutStreamer.AddComment("Loc expr size");
      MCSymbol *begin = Asm->OutStreamer.getContext().CreateTempSymbol();
      MCSymbol *end = Asm->OutStreamer.getContext().CreateTempSymbol();
      Asm->EmitLabelDifference(end, begin, 2);
      Asm->OutStreamer.EmitLabel(begin);
      if (Entry.isInt()) {
        DIBasicType BTy(DV.getType());
        if (BTy.Verify() && (BTy.getEncoding() == dwarf::DW_ATE_signed ||
                             BTy.getEncoding() == dwarf::DW_ATE_signed_char)) {
          Asm->OutStreamer.AddComment("DW_OP_consts");
          Asm->EmitInt8(dwarf::DW_OP_consts);
          Asm->EmitSLEB128(Entry.getInt());
        } else {
          Asm->OutStreamer.AddComment("DW_OP_constu");
          Asm->EmitInt8(dwarf::DW_OP_constu);
          Asm->EmitULEB128(Entry.getInt());
        }
      } else if (Entry.isLocation()) {
        MachineLocation Loc = Entry.getLoc();
        if (!DV.hasComplexAddress())
          // Regular entry.
          Asm->EmitDwarfRegOp(Loc, DV.isIndirect());
        else {
          // Complex address entry.
          unsigned N = DV.getNumAddrElements();
          unsigned i = 0;
          if (N >= 2 && DV.getAddrElement(0) == DIBuilder::OpPlus) {
            if (Loc.getOffset()) {
              i = 2;
              Asm->EmitDwarfRegOp(Loc, DV.isIndirect());
              Asm->OutStreamer.AddComment("DW_OP_deref");
              Asm->EmitInt8(dwarf::DW_OP_deref);
              Asm->OutStreamer.AddComment("DW_OP_plus_uconst");
              Asm->EmitInt8(dwarf::DW_OP_plus_uconst);
              Asm->EmitSLEB128(DV.getAddrElement(1));
            } else {
              // If first address element is OpPlus then emit
              // DW_OP_breg + Offset instead of DW_OP_reg + Offset.
              MachineLocation TLoc(Loc.getReg(), DV.getAddrElement(1));
              Asm->EmitDwarfRegOp(TLoc, DV.isIndirect());
              i = 2;
            }
          } else {
            Asm->EmitDwarfRegOp(Loc, DV.isIndirect());
          }

          // Emit remaining complex address elements.
          for (; i < N; ++i) {
            uint64_t Element = DV.getAddrElement(i);
            if (Element == DIBuilder::OpPlus) {
              Asm->EmitInt8(dwarf::DW_OP_plus_uconst);
              Asm->EmitULEB128(DV.getAddrElement(++i));
            } else if (Element == DIBuilder::OpDeref) {
              if (!Loc.isReg())
                Asm->EmitInt8(dwarf::DW_OP_deref);
            } else
              llvm_unreachable("unknown Opcode found in complex address");
          }
        }
      }
      // else ... ignore constant fp. There is not any good way to
      // to represent them here in dwarf.
      Asm->OutStreamer.EmitLabel(end);
    }
  }
}

struct SymbolCUSorter {
  SymbolCUSorter(const MCStreamer &s) : Streamer(s) {}
  const MCStreamer &Streamer;

  bool operator()(const SymbolCU &A, const SymbolCU &B) {
    unsigned IA = A.Sym ? Streamer.GetSymbolOrder(A.Sym) : 0;
    unsigned IB = B.Sym ? Streamer.GetSymbolOrder(B.Sym) : 0;

    // Symbols with no order assigned should be placed at the end.
    // (e.g. section end labels)
    if (IA == 0)
      IA = (unsigned)(-1);
    if (IB == 0)
      IB = (unsigned)(-1);
    return IA < IB;
  }
};

static bool CUSort(const DwarfUnit *A, const DwarfUnit *B) {
  return (A->getUniqueID() < B->getUniqueID());
}

struct ArangeSpan {
  const MCSymbol *Start, *End;
};

// Emit a debug aranges section, containing a CU lookup for any
// address we can tie back to a CU.
void DwarfDebug::emitDebugARanges() {
  // Start the dwarf aranges section.
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfARangesSection());

  typedef DenseMap<DwarfCompileUnit *, std::vector<ArangeSpan> > SpansType;

  SpansType Spans;

  // Build a list of sections used.
  std::vector<const MCSection *> Sections;
  for (SectionMapType::iterator it = SectionMap.begin(); it != SectionMap.end();
       it++) {
    const MCSection *Section = it->first;
    Sections.push_back(Section);
  }

  // Sort the sections into order.
  // This is only done to ensure consistent output order across different runs.
  std::sort(Sections.begin(), Sections.end(), SectionSort);

  // Build a set of address spans, sorted by CU.
  for (size_t SecIdx = 0; SecIdx < Sections.size(); SecIdx++) {
    const MCSection *Section = Sections[SecIdx];
    SmallVector<SymbolCU, 8> &List = SectionMap[Section];
    if (List.size() < 2)
      continue;

    // Sort the symbols by offset within the section.
    SymbolCUSorter sorter(Asm->OutStreamer);
    std::sort(List.begin(), List.end(), sorter);

    // If we have no section (e.g. common), just write out
    // individual spans for each symbol.
    if (Section == NULL) {
      for (size_t n = 0; n < List.size(); n++) {
        const SymbolCU &Cur = List[n];

        ArangeSpan Span;
        Span.Start = Cur.Sym;
        Span.End = NULL;
        if (Cur.CU)
          Spans[Cur.CU].push_back(Span);
      }
    } else {
      // Build spans between each label.
      const MCSymbol *StartSym = List[0].Sym;
      for (size_t n = 1; n < List.size(); n++) {
        const SymbolCU &Prev = List[n - 1];
        const SymbolCU &Cur = List[n];

        // Try and build the longest span we can within the same CU.
        if (Cur.CU != Prev.CU) {
          ArangeSpan Span;
          Span.Start = StartSym;
          Span.End = Cur.Sym;
          Spans[Prev.CU].push_back(Span);
          StartSym = Cur.Sym;
        }
      }
    }
  }

  unsigned PtrSize = Asm->getDataLayout().getPointerSize();

  // Build a list of CUs used.
  std::vector<DwarfCompileUnit *> CUs;
  for (SpansType::iterator it = Spans.begin(); it != Spans.end(); it++) {
    DwarfCompileUnit *CU = it->first;
    CUs.push_back(CU);
  }

  // Sort the CU list (again, to ensure consistent output order).
  std::sort(CUs.begin(), CUs.end(), CUSort);

  // Emit an arange table for each CU we used.
  for (size_t CUIdx = 0; CUIdx < CUs.size(); CUIdx++) {
    DwarfCompileUnit *CU = CUs[CUIdx];
    std::vector<ArangeSpan> &List = Spans[CU];

    // Emit size of content not including length itself.
    unsigned ContentSize =
        sizeof(int16_t) + // DWARF ARange version number
        sizeof(int32_t) + // Offset of CU in the .debug_info section
        sizeof(int8_t) +  // Pointer Size (in bytes)
        sizeof(int8_t);   // Segment Size (in bytes)

    unsigned TupleSize = PtrSize * 2;

    // 7.20 in the Dwarf specs requires the table to be aligned to a tuple.
    unsigned Padding =
        OffsetToAlignment(sizeof(int32_t) + ContentSize, TupleSize);

    ContentSize += Padding;
    ContentSize += (List.size() + 1) * TupleSize;

    // For each compile unit, write the list of spans it covers.
    Asm->OutStreamer.AddComment("Length of ARange Set");
    Asm->EmitInt32(ContentSize);
    Asm->OutStreamer.AddComment("DWARF Arange version number");
    Asm->EmitInt16(dwarf::DW_ARANGES_VERSION);
    Asm->OutStreamer.AddComment("Offset Into Debug Info Section");
    Asm->EmitSectionOffset(CU->getLocalLabelBegin(), CU->getLocalSectionSym());
    Asm->OutStreamer.AddComment("Address Size (in bytes)");
    Asm->EmitInt8(PtrSize);
    Asm->OutStreamer.AddComment("Segment Size (in bytes)");
    Asm->EmitInt8(0);

    Asm->OutStreamer.EmitFill(Padding, 0xff);

    for (unsigned n = 0; n < List.size(); n++) {
      const ArangeSpan &Span = List[n];
      Asm->EmitLabelReference(Span.Start, PtrSize);

      // Calculate the size as being from the span start to it's end.
      if (Span.End) {
        Asm->EmitLabelDifference(Span.End, Span.Start, PtrSize);
      } else {
        // For symbols without an end marker (e.g. common), we
        // write a single arange entry containing just that one symbol.
        uint64_t Size = SymSize[Span.Start];
        if (Size == 0)
          Size = 1;

        Asm->OutStreamer.EmitIntValue(Size, PtrSize);
      }
    }

    Asm->OutStreamer.AddComment("ARange terminator");
    Asm->OutStreamer.EmitIntValue(0, PtrSize);
    Asm->OutStreamer.EmitIntValue(0, PtrSize);
  }
}

// Emit visible names into a debug ranges section.
void DwarfDebug::emitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfRangesSection());

  // Size for our labels.
  unsigned char Size = Asm->getDataLayout().getPointerSize();

  // Grab the specific ranges for the compile units in the module.
  for (MapVector<const MDNode *, DwarfCompileUnit *>::iterator
           I = CUMap.begin(),
           E = CUMap.end();
       I != E; ++I) {
    DwarfCompileUnit *TheCU = I->second;

    // Emit a symbol so we can find the beginning of our ranges.
    Asm->OutStreamer.EmitLabel(TheCU->getLabelRange());

    // Iterate over the misc ranges for the compile units in the module.
    const SmallVectorImpl<RangeSpanList> &RangeLists = TheCU->getRangeLists();
    for (SmallVectorImpl<RangeSpanList>::const_iterator I = RangeLists.begin(),
                                                        E = RangeLists.end();
         I != E; ++I) {
      const RangeSpanList &List = *I;

      // Emit our symbol so we can find the beginning of the range.
      Asm->OutStreamer.EmitLabel(List.getSym());

      for (SmallVectorImpl<RangeSpan>::const_iterator
               RI = List.getRanges().begin(),
               RE = List.getRanges().end();
           RI != RE; ++RI) {
        const RangeSpan &Range = *RI;
        const MCSymbol *Begin = Range.getStart();
        const MCSymbol *End = Range.getEnd();
        assert(Begin && "Range without a begin symbol?");
        assert(End && "Range without an end symbol?");
        Asm->OutStreamer.EmitSymbolValue(Begin, Size);
        Asm->OutStreamer.EmitSymbolValue(End, Size);
      }

      // And terminate the list with two 0 values.
      Asm->OutStreamer.EmitIntValue(0, Size);
      Asm->OutStreamer.EmitIntValue(0, Size);
    }

    // Now emit a range for the CU itself.
    if (useCURanges() && TheCU->getRanges().size()) {
      Asm->OutStreamer.EmitLabel(
          Asm->GetTempSymbol("cu_ranges", TheCU->getUniqueID()));
      const SmallVectorImpl<RangeSpan> &Ranges = TheCU->getRanges();
      for (uint32_t i = 0, e = Ranges.size(); i != e; ++i) {
        RangeSpan Range = Ranges[i];
        const MCSymbol *Begin = Range.getStart();
        const MCSymbol *End = Range.getEnd();
        assert(Begin && "Range without a begin symbol?");
        assert(End && "Range without an end symbol?");
        Asm->OutStreamer.EmitSymbolValue(Begin, Size);
        Asm->OutStreamer.EmitSymbolValue(End, Size);
      }
      // And terminate the list with two 0 values.
      Asm->OutStreamer.EmitIntValue(0, Size);
      Asm->OutStreamer.EmitIntValue(0, Size);
    }
  }
}

// DWARF5 Experimental Separate Dwarf emitters.

void DwarfDebug::initSkeletonUnit(const DwarfUnit *U, DIE *Die,
                                  DwarfUnit *NewU) {
  NewU->addLocalString(Die, dwarf::DW_AT_GNU_dwo_name,
                       U->getCUNode().getSplitDebugFilename());

  // Relocate to the beginning of the addr_base section, else 0 for the
  // beginning of the one for this compile unit.
  if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    NewU->addSectionLabel(Die, dwarf::DW_AT_GNU_addr_base, DwarfAddrSectionSym);
  else
    NewU->addSectionOffset(Die, dwarf::DW_AT_GNU_addr_base, 0);

  if (!CompilationDir.empty())
    NewU->addLocalString(Die, dwarf::DW_AT_comp_dir, CompilationDir);

  addGnuPubAttributes(NewU, Die);

  SkeletonHolder.addUnit(NewU);
}

// This DIE has the following attributes: DW_AT_comp_dir, DW_AT_stmt_list,
// DW_AT_low_pc, DW_AT_high_pc, DW_AT_ranges, DW_AT_dwo_name, DW_AT_dwo_id,
// DW_AT_ranges_base, DW_AT_addr_base.
// TODO: Implement DW_AT_ranges_base.
DwarfCompileUnit *DwarfDebug::constructSkeletonCU(const DwarfCompileUnit *CU) {

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  DwarfCompileUnit *NewCU = new DwarfCompileUnit(
      CU->getUniqueID(), Die, CU->getCUNode(), Asm, this, &SkeletonHolder);
  NewCU->initSection(Asm->getObjFileLowering().getDwarfInfoSection(),
                     DwarfInfoSectionSym);

  NewCU->initStmtList(DwarfLineSectionSym);

  initSkeletonUnit(CU, Die, NewCU);

  return NewCU;
}

// This DIE has the following attributes: DW_AT_comp_dir, DW_AT_dwo_name,
// DW_AT_addr_base.
DwarfTypeUnit *DwarfDebug::constructSkeletonTU(DwarfTypeUnit *TU) {
  DwarfCompileUnit &CU = static_cast<DwarfCompileUnit &>(
      *SkeletonHolder.getUnits()[TU->getCU().getUniqueID()]);

  DIE *Die = new DIE(dwarf::DW_TAG_type_unit);
  DwarfTypeUnit *NewTU =
      new DwarfTypeUnit(TU->getUniqueID(), Die, CU, Asm, this, &SkeletonHolder);
  NewTU->setTypeSignature(TU->getTypeSignature());
  NewTU->setType(NULL);
  NewTU->initSection(
      Asm->getObjFileLowering().getDwarfTypesSection(TU->getTypeSignature()));
  CU.applyStmtList(*Die);

  initSkeletonUnit(TU, Die, NewTU);
  return NewTU;
}

// Emit the .debug_info.dwo section for separated dwarf. This contains the
// compile units that would normally be in debug_info.
void DwarfDebug::emitDebugInfoDWO() {
  assert(useSplitDwarf() && "No split dwarf debug info?");
  InfoHolder.emitUnits(this,
                       Asm->getObjFileLowering().getDwarfAbbrevDWOSection(),
                       DwarfAbbrevDWOSectionSym);
}

// Emit the .debug_abbrev.dwo section for separated dwarf. This contains the
// abbreviations for the .debug_info.dwo section.
void DwarfDebug::emitDebugAbbrevDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  InfoHolder.emitAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevDWOSection());
}

// Emit the .debug_str.dwo section for separated dwarf. This contains the
// string section and is identical in format to traditional .debug_str
// sections.
void DwarfDebug::emitDebugStrDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  const MCSection *OffSec =
      Asm->getObjFileLowering().getDwarfStrOffDWOSection();
  const MCSymbol *StrSym = DwarfStrSectionSym;
  InfoHolder.emitStrings(Asm->getObjFileLowering().getDwarfStrDWOSection(),
                         OffSec, StrSym);
}

void DwarfDebug::addDwarfTypeUnitType(DwarfCompileUnit &CU,
                                      StringRef Identifier, DIE *RefDie,
                                      DICompositeType CTy) {
  // Flag the type unit reference as a declaration so that if it contains
  // members (implicit special members, static data member definitions, member
  // declarations for definitions in this CU, etc) consumers don't get confused
  // and think this is a full definition.
  CU.addFlag(RefDie, dwarf::DW_AT_declaration);

  const DwarfTypeUnit *&TU = DwarfTypeUnits[CTy];
  if (TU) {
    CU.addDIETypeSignature(RefDie, *TU);
    return;
  }

  DIE *UnitDie = new DIE(dwarf::DW_TAG_type_unit);
  DwarfTypeUnit *NewTU = new DwarfTypeUnit(InfoHolder.getUnits().size(),
                                           UnitDie, CU, Asm, this, &InfoHolder);
  TU = NewTU;
  InfoHolder.addUnit(NewTU);

  NewTU->addUInt(UnitDie, dwarf::DW_AT_language, dwarf::DW_FORM_data2,
                 CU.getLanguage());

  MD5 Hash;
  Hash.update(Identifier);
  // ... take the least significant 8 bytes and return those. Our MD5
  // implementation always returns its results in little endian, swap bytes
  // appropriately.
  MD5::MD5Result Result;
  Hash.final(Result);
  uint64_t Signature = *reinterpret_cast<support::ulittle64_t *>(Result + 8);
  NewTU->setTypeSignature(Signature);
  if (useSplitDwarf())
    NewTU->setSkeleton(constructSkeletonTU(NewTU));
  else
    CU.applyStmtList(*UnitDie);

  NewTU->setType(NewTU->createTypeDIE(CTy));

  NewTU->initSection(
      useSplitDwarf()
          ? Asm->getObjFileLowering().getDwarfTypesDWOSection(Signature)
          : Asm->getObjFileLowering().getDwarfTypesSection(Signature));

  CU.addDIETypeSignature(RefDie, *NewTU);
}
