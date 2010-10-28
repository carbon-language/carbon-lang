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
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Path.h"
using namespace llvm;

static cl::opt<bool> PrintDbgScope("print-dbgscope", cl::Hidden,
     cl::desc("Print DbgScope information for each machine instruction"));

static cl::opt<bool> DisableDebugInfoPrinting("disable-debug-info-print",
                                              cl::Hidden,
     cl::desc("Disable debug info printing"));

static cl::opt<bool> UnknownLocations("use-unknown-locations", cl::Hidden,
     cl::desc("Make an absense of debug location information explicit."),
     cl::init(false));

#ifndef NDEBUG
STATISTIC(BlocksWithoutLineNo, "Number of blocks without any line number");
#endif

namespace {
  const char *DWARFGroupName = "DWARF Emission";
  const char *DbgTimerName = "DWARF Debug Writer";
} // end anonymous namespace

//===----------------------------------------------------------------------===//

/// Configuration values for initial hash set sizes (log2).
///
static const unsigned InitAbbreviationsSetSize = 9; // log2(512)

namespace llvm {

//===----------------------------------------------------------------------===//
/// CompileUnit - This dwarf writer support class manages information associate
/// with a source file.
class CompileUnit {
  /// ID - File identifier for source.
  ///
  unsigned ID;

  /// Die - Compile unit debug information entry.
  ///
  const OwningPtr<DIE> CUDie;

  /// IndexTyDie - An anonymous type for index type.  Owned by CUDie.
  DIE *IndexTyDie;

  /// MDNodeToDieMap - Tracks the mapping of unit level debug informaton
  /// variables to debug information entries.
  DenseMap<const MDNode *, DIE *> MDNodeToDieMap;

  /// MDNodeToDIEEntryMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries using a DIEEntry proxy.
  DenseMap<const MDNode *, DIEEntry *> MDNodeToDIEEntryMap;

  /// Globals - A map of globally visible named entities for this unit.
  ///
  StringMap<DIE*> Globals;

  /// GlobalTypes - A map of globally visible types for this unit.
  ///
  StringMap<DIE*> GlobalTypes;

public:
  CompileUnit(unsigned I, DIE *D)
    : ID(I), CUDie(D), IndexTyDie(0) {}

  // Accessors.
  unsigned getID()                  const { return ID; }
  DIE* getCUDie()                   const { return CUDie.get(); }
  const StringMap<DIE*> &getGlobals()     const { return Globals; }
  const StringMap<DIE*> &getGlobalTypes() const { return GlobalTypes; }

  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const { return !CUDie->getChildren().empty(); }

  /// addGlobal - Add a new global entity to the compile unit.
  ///
  void addGlobal(StringRef Name, DIE *Die) { Globals[Name] = Die; }

  /// addGlobalType - Add a new global type to the compile unit.
  ///
  void addGlobalType(StringRef Name, DIE *Die) {
    GlobalTypes[Name] = Die;
  }

  /// getDIE - Returns the debug information entry map slot for the
  /// specified debug variable.
  DIE *getDIE(const MDNode *N) { return MDNodeToDieMap.lookup(N); }

  /// insertDIE - Insert DIE into the map.
  void insertDIE(const MDNode *N, DIE *D) {
    MDNodeToDieMap.insert(std::make_pair(N, D));
  }

  /// getDIEEntry - Returns the debug information entry for the speciefied
  /// debug variable.
  DIEEntry *getDIEEntry(const MDNode *N) {
    DenseMap<const MDNode *, DIEEntry *>::iterator I =
      MDNodeToDIEEntryMap.find(N);
    if (I == MDNodeToDIEEntryMap.end())
      return NULL;
    return I->second;
  }

  /// insertDIEEntry - Insert debug information entry into the map.
  void insertDIEEntry(const MDNode *N, DIEEntry *E) {
    MDNodeToDIEEntryMap.insert(std::make_pair(N, E));
  }

  /// addDie - Adds or interns the DIE to the compile unit.
  ///
  void addDie(DIE *Buffer) {
    this->CUDie->addChild(Buffer);
  }

  // getIndexTyDie - Get an anonymous type for index type.
  DIE *getIndexTyDie() {
    return IndexTyDie;
  }

  // setIndexTyDie - Set D as anonymous type for index which can be reused
  // later.
  void setIndexTyDie(DIE *D) {
    IndexTyDie = D;
  }

};

//===----------------------------------------------------------------------===//
/// DbgVariable - This class is used to track local variable information.
///
class DbgVariable {
  DIVariable Var;                    // Variable Descriptor.
  DIE *TheDIE;                       // Variable DIE.
  unsigned DotDebugLocOffset;        // Offset in DotDebugLocEntries.
public:
  // AbsVar may be NULL.
  DbgVariable(DIVariable V) : Var(V), TheDIE(0), DotDebugLocOffset(~0U) {}

  // Accessors.
  DIVariable getVariable()           const { return Var; }
  void setDIE(DIE *D)                      { TheDIE = D; }
  DIE *getDIE()                      const { return TheDIE; }
  void setDotDebugLocOffset(unsigned O)    { DotDebugLocOffset = O; }
  unsigned getDotDebugLocOffset()    const { return DotDebugLocOffset; }
  StringRef getName()                const { return Var.getName(); }
  unsigned getTag()                  const { return Var.getTag(); }
  bool variableHasComplexAddress()   const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.hasComplexAddress();
  }
  bool isBlockByrefVariable()        const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.isBlockByrefVariable();
  }
  unsigned getNumAddrElements()      const { 
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.getNumAddrElements();
  }
  uint64_t getAddrElement(unsigned i) const {
    return Var.getAddrElement(i);
  }
  DIType getType()               const {
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
      unsigned tag = Ty.getTag();
      
      if (tag == dwarf::DW_TAG_pointer_type) {
        DIDerivedType DTy = DIDerivedType(Ty);
        subType = DTy.getTypeDerivedFrom();
      }
      
      DICompositeType blockStruct = DICompositeType(subType);
      DIArray Elements = blockStruct.getTypeArray();
      
      for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
        DIDescriptor Element = Elements.getElement(i);
        DIDerivedType DT = DIDerivedType(Element);
        if (getName() == DT.getName())
          return (DT.getTypeDerivedFrom());
      }
      return Ty;
    }
    return Ty;
  }
};

//===----------------------------------------------------------------------===//
/// DbgRange - This is used to track range of instructions with identical
/// debug info scope.
///
typedef std::pair<const MachineInstr *, const MachineInstr *> DbgRange;

//===----------------------------------------------------------------------===//
/// DbgScope - This class is used to track scope information.
///
class DbgScope {
  DbgScope *Parent;                   // Parent to this scope.
  DIDescriptor Desc;                  // Debug info descriptor for scope.
  // Location at which this scope is inlined.
  AssertingVH<const MDNode> InlinedAtLocation;
  bool AbstractScope;                 // Abstract Scope
  const MachineInstr *LastInsn;       // Last instruction of this scope.
  const MachineInstr *FirstInsn;      // First instruction of this scope.
  unsigned DFSIn, DFSOut;
  // Scopes defined in scope.  Contents not owned.
  SmallVector<DbgScope *, 4> Scopes;
  // Variables declared in scope.  Contents owned.
  SmallVector<DbgVariable *, 8> Variables;
  SmallVector<DbgRange, 4> Ranges;
  // Private state for dump()
  mutable unsigned IndentLevel;
public:
  DbgScope(DbgScope *P, DIDescriptor D, const MDNode *I = 0)
    : Parent(P), Desc(D), InlinedAtLocation(I), AbstractScope(false),
      LastInsn(0), FirstInsn(0),
      DFSIn(0), DFSOut(0), IndentLevel(0) {}
  virtual ~DbgScope();

  // Accessors.
  DbgScope *getParent()          const { return Parent; }
  void setParent(DbgScope *P)          { Parent = P; }
  DIDescriptor getDesc()         const { return Desc; }
  const MDNode *getInlinedAt()         const { return InlinedAtLocation; }
  const MDNode *getScopeNode()         const { return Desc; }
  const SmallVector<DbgScope *, 4> &getScopes() { return Scopes; }
  const SmallVector<DbgVariable *, 8> &getDbgVariables() { return Variables; }
  const SmallVector<DbgRange, 4> &getRanges() { return Ranges; }

  /// openInsnRange - This scope covers instruction range starting from MI.
  void openInsnRange(const MachineInstr *MI) {
    if (!FirstInsn)
      FirstInsn = MI;

    if (Parent)
      Parent->openInsnRange(MI);
  }

  /// extendInsnRange - Extend the current instruction range covered by
  /// this scope.
  void extendInsnRange(const MachineInstr *MI) {
    assert (FirstInsn && "MI Range is not open!");
    LastInsn = MI;
    if (Parent)
      Parent->extendInsnRange(MI);
  }

  /// closeInsnRange - Create a range based on FirstInsn and LastInsn collected
  /// until now. This is used when a new scope is encountered while walking
  /// machine instructions.
  void closeInsnRange(DbgScope *NewScope = NULL) {
    assert (LastInsn && "Last insn missing!");
    Ranges.push_back(DbgRange(FirstInsn, LastInsn));
    FirstInsn = NULL;
    LastInsn = NULL;
    // If Parent dominates NewScope then do not close Parent's instruction
    // range.
    if (Parent && (!NewScope || !Parent->dominates(NewScope)))
      Parent->closeInsnRange(NewScope);
  }

  void setAbstractScope() { AbstractScope = true; }
  bool isAbstractScope() const { return AbstractScope; }

  // Depth First Search support to walk and mainpluate DbgScope hierarchy.
  unsigned getDFSOut() const { return DFSOut; }
  void setDFSOut(unsigned O) { DFSOut = O; }
  unsigned getDFSIn() const  { return DFSIn; }
  void setDFSIn(unsigned I)  { DFSIn = I; }
  bool dominates(const DbgScope *S) {
    if (S == this)
      return true;
    if (DFSIn < S->getDFSIn() && DFSOut > S->getDFSOut())
      return true;
    return false;
  }

  /// addScope - Add a scope to the scope.
  ///
  void addScope(DbgScope *S) { Scopes.push_back(S); }

  /// addVariable - Add a variable to the scope.
  ///
  void addVariable(DbgVariable *V) { Variables.push_back(V); }

#ifndef NDEBUG
  void dump() const;
#endif
};

} // end llvm namespace

#ifndef NDEBUG
void DbgScope::dump() const {
  raw_ostream &err = dbgs();
  err.indent(IndentLevel);
  const MDNode *N = Desc;
  N->dump();
  if (AbstractScope)
    err << "Abstract Scope\n";

  IndentLevel += 2;
  if (!Scopes.empty())
    err << "Children ...\n";
  for (unsigned i = 0, e = Scopes.size(); i != e; ++i)
    if (Scopes[i] != this)
      Scopes[i]->dump();

  IndentLevel -= 2;
}
#endif

DbgScope::~DbgScope() {
  for (unsigned j = 0, M = Variables.size(); j < M; ++j)
    delete Variables[j];
}

DwarfDebug::DwarfDebug(AsmPrinter *A, Module *M)
  : Asm(A), MMI(Asm->MMI), FirstCU(0),
    AbbreviationsSet(InitAbbreviationsSetSize),
    CurrentFnDbgScope(0), PrevLabel(NULL) {
  NextStringPoolNumber = 0;

  DwarfFrameSectionSym = DwarfInfoSectionSym = DwarfAbbrevSectionSym = 0;
  DwarfStrSectionSym = TextSectionSym = 0;
  DwarfDebugRangeSectionSym = DwarfDebugLocSectionSym = 0;
  FunctionBeginSym = FunctionEndSym = 0;
  DIEIntegerOne = new (DIEValueAllocator) DIEInteger(1);
  {
    NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
    beginModule(M);
  }
}
DwarfDebug::~DwarfDebug() {
  for (unsigned j = 0, M = DIEBlocks.size(); j < M; ++j)
    DIEBlocks[j]->~DIEBlock();
}

MCSymbol *DwarfDebug::getStringPoolEntry(StringRef Str) {
  std::pair<MCSymbol*, unsigned> &Entry = StringPool[Str];
  if (Entry.first) return Entry.first;

  Entry.second = NextStringPoolNumber++;
  return Entry.first = Asm->GetTempSymbol("string", Entry.second);
}


/// assignAbbrevNumber - Define a unique number for the abbreviation.
///
void DwarfDebug::assignAbbrevNumber(DIEAbbrev &Abbrev) {
  // Profile the node so that we can make it unique.
  FoldingSetNodeID ID;
  Abbrev.Profile(ID);

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

/// createDIEEntry - Creates a new DIEEntry to be a proxy for a debug
/// information entry.
DIEEntry *DwarfDebug::createDIEEntry(DIE *Entry) {
  DIEEntry *Value = new (DIEValueAllocator) DIEEntry(Entry);
  return Value;
}

/// addUInt - Add an unsigned integer attribute data and value.
///
void DwarfDebug::addUInt(DIE *Die, unsigned Attribute,
                         unsigned Form, uint64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(false, Integer);
  DIEValue *Value = Integer == 1 ?
    DIEIntegerOne : new (DIEValueAllocator) DIEInteger(Integer);
  Die->addValue(Attribute, Form, Value);
}

/// addSInt - Add an signed integer attribute data and value.
///
void DwarfDebug::addSInt(DIE *Die, unsigned Attribute,
                         unsigned Form, int64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(true, Integer);
  DIEValue *Value = new (DIEValueAllocator) DIEInteger(Integer);
  Die->addValue(Attribute, Form, Value);
}

/// addString - Add a string attribute data and value. DIEString only
/// keeps string reference.
void DwarfDebug::addString(DIE *Die, unsigned Attribute, unsigned Form,
                           StringRef String) {
  DIEValue *Value = new (DIEValueAllocator) DIEString(String);
  Die->addValue(Attribute, Form, Value);
}

/// addLabel - Add a Dwarf label attribute data and value.
///
void DwarfDebug::addLabel(DIE *Die, unsigned Attribute, unsigned Form,
                          const MCSymbol *Label) {
  DIEValue *Value = new (DIEValueAllocator) DIELabel(Label);
  Die->addValue(Attribute, Form, Value);
}

/// addDelta - Add a label delta attribute data and value.
///
void DwarfDebug::addDelta(DIE *Die, unsigned Attribute, unsigned Form,
                          const MCSymbol *Hi, const MCSymbol *Lo) {
  DIEValue *Value = new (DIEValueAllocator) DIEDelta(Hi, Lo);
  Die->addValue(Attribute, Form, Value);
}

/// addDIEEntry - Add a DIE attribute data and value.
///
void DwarfDebug::addDIEEntry(DIE *Die, unsigned Attribute, unsigned Form,
                             DIE *Entry) {
  Die->addValue(Attribute, Form, createDIEEntry(Entry));
}


/// addBlock - Add block data.
///
void DwarfDebug::addBlock(DIE *Die, unsigned Attribute, unsigned Form,
                          DIEBlock *Block) {
  Block->ComputeSize(Asm);
  DIEBlocks.push_back(Block); // Memoize so we can call the destructor later on.
  Die->addValue(Attribute, Block->BestForm(), Block);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, DIVariable V) {
  // Verify variable.
  if (!V.Verify())
    return;

  unsigned Line = V.getLineNumber();
  if (Line == 0)
    return;
  unsigned FileID = GetOrCreateSourceID(V.getContext().getDirectory(),
                                        V.getContext().getFilename());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, DIGlobalVariable G) {
  // Verify global variable.
  if (!G.Verify())
    return;

  unsigned Line = G.getLineNumber();
  if (Line == 0)
    return;
  unsigned FileID = GetOrCreateSourceID(G.getContext().getDirectory(),
                                        G.getContext().getFilename());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, DISubprogram SP) {
  // Verify subprogram.
  if (!SP.Verify())
    return;
  // If the line number is 0, don't add it.
  if (SP.getLineNumber() == 0)
    return;

  unsigned Line = SP.getLineNumber();
  if (!SP.getContext().Verify())
    return;
  unsigned FileID = GetOrCreateSourceID(SP.getDirectory(),
                                        SP.getFilename());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, DIType Ty) {
  // Verify type.
  if (!Ty.Verify())
    return;

  unsigned Line = Ty.getLineNumber();
  if (Line == 0 || !Ty.getContext().Verify())
    return;
  unsigned FileID = GetOrCreateSourceID(Ty.getDirectory(),
                                        Ty.getFilename());
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, DINameSpace NS) {
  // Verify namespace.
  if (!NS.Verify())
    return;

  unsigned Line = NS.getLineNumber();
  if (Line == 0)
    return;
  StringRef FN = NS.getFilename();
  StringRef Dir = NS.getDirectory();

  unsigned FileID = GetOrCreateSourceID(Dir, FN);
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addVariableAddress - Add DW_AT_location attribute for a DbgVariable based
/// on provided frame index.
void DwarfDebug::addVariableAddress(DbgVariable *&DV, DIE *Die, int64_t FI) {
  MachineLocation Location;
  unsigned FrameReg;
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  int Offset = RI->getFrameIndexReference(*Asm->MF, FI, FrameReg);
  Location.set(FrameReg, Offset);

  if (DV->variableHasComplexAddress())
    addComplexAddress(DV, Die, dwarf::DW_AT_location, Location);
  else if (DV->isBlockByrefVariable())
    addBlockByrefAddress(DV, Die, dwarf::DW_AT_location, Location);
  else
    addAddress(Die, dwarf::DW_AT_location, Location);
}

/// addComplexAddress - Start with the address based on the location provided,
/// and generate the DWARF information necessary to find the actual variable
/// given the extra address information encoded in the DIVariable, starting from
/// the starting location.  Add the DWARF information to the die.
///
void DwarfDebug::addComplexAddress(DbgVariable *&DV, DIE *Die,
                                   unsigned Attribute,
                                   const MachineLocation &Location) {
  DIType Ty = DV->getType();

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();

  if (Location.isReg()) {
    if (Reg < 32) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    } else {
      Reg = Reg - dwarf::DW_OP_reg0;
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32)
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    else {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    addUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  for (unsigned i = 0, N = DV->getNumAddrElements(); i < N; ++i) {
    uint64_t Element = DV->getAddrElement(i);

    if (Element == DIFactory::OpPlus) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
      addUInt(Block, 0, dwarf::DW_FORM_udata, DV->getAddrElement(++i));
    } else if (Element == DIFactory::OpDeref) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    } else llvm_unreachable("unknown DIFactory Opcode");
  }

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, 0, Block);
}

/* Byref variables, in Blocks, are declared by the programmer as "SomeType
   VarName;", but the compiler creates a __Block_byref_x_VarName struct, and
   gives the variable VarName either the struct, or a pointer to the struct, as
   its type.  This is necessary for various behind-the-scenes things the
   compiler needs to do with by-reference variables in Blocks.

   However, as far as the original *programmer* is concerned, the variable
   should still have type 'SomeType', as originally declared.

   The function getBlockByrefType dives into the __Block_byref_x_VarName
   struct to find the original type of the variable, which is then assigned to
   the variable's Debug Information Entry as its real type.  So far, so good.
   However now the debugger will expect the variable VarName to have the type
   SomeType.  So we need the location attribute for the variable to be an
   expression that explains to the debugger how to navigate through the
   pointers and struct to find the actual variable of type SomeType.

   The following function does just that.  We start by getting
   the "normal" location for the variable. This will be the location
   of either the struct __Block_byref_x_VarName or the pointer to the
   struct __Block_byref_x_VarName.

   The struct will look something like:

   struct __Block_byref_x_VarName {
     ... <various fields>
     struct __Block_byref_x_VarName *forwarding;
     ... <various other fields>
     SomeType VarName;
     ... <maybe more fields>
   };

   If we are given the struct directly (as our starting point) we
   need to tell the debugger to:

   1).  Add the offset of the forwarding field.

   2).  Follow that pointer to get the real __Block_byref_x_VarName
   struct to use (the real one may have been copied onto the heap).

   3).  Add the offset for the field VarName, to find the actual variable.

   If we started with a pointer to the struct, then we need to
   dereference that pointer first, before the other steps.
   Translating this into DWARF ops, we will need to append the following
   to the current location description for the variable:

   DW_OP_deref                    -- optional, if we start with a pointer
   DW_OP_plus_uconst <forward_fld_offset>
   DW_OP_deref
   DW_OP_plus_uconst <varName_fld_offset>

   That is what this function does.  */

/// addBlockByrefAddress - Start with the address based on the location
/// provided, and generate the DWARF information necessary to find the
/// actual Block variable (navigating the Block struct) based on the
/// starting location.  Add the DWARF information to the die.  For
/// more information, read large comment just above here.
///
void DwarfDebug::addBlockByrefAddress(DbgVariable *&DV, DIE *Die,
                                      unsigned Attribute,
                                      const MachineLocation &Location) {
  DIType Ty = DV->getType();
  DIType TmpTy = Ty;
  unsigned Tag = Ty.getTag();
  bool isPointer = false;

  StringRef varName = DV->getName();

  if (Tag == dwarf::DW_TAG_pointer_type) {
    DIDerivedType DTy = DIDerivedType(Ty);
    TmpTy = DTy.getTypeDerivedFrom();
    isPointer = true;
  }

  DICompositeType blockStruct = DICompositeType(TmpTy);

  // Find the __forwarding field and the variable field in the __Block_byref
  // struct.
  DIArray Fields = blockStruct.getTypeArray();
  DIDescriptor varField = DIDescriptor();
  DIDescriptor forwardingField = DIDescriptor();

  for (unsigned i = 0, N = Fields.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Fields.getElement(i);
    DIDerivedType DT = DIDerivedType(Element);
    StringRef fieldName = DT.getName();
    if (fieldName == "__forwarding")
      forwardingField = Element;
    else if (fieldName == varName)
      varField = Element;
  }

  // Get the offsets for the forwarding field and the variable field.
  unsigned forwardingFieldOffset =
    DIDerivedType(forwardingField).getOffsetInBits() >> 3;
  unsigned varFieldOffset =
    DIDerivedType(varField).getOffsetInBits() >> 3;

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();

  if (Location.isReg()) {
    if (Reg < 32)
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    else {
      Reg = Reg - dwarf::DW_OP_reg0;
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32)
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    else {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    addUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  // If we started with a pointer to the __Block_byref... struct, then
  // the first thing we need to do is dereference the pointer (DW_OP_deref).
  if (isPointer)
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Next add the offset for the '__forwarding' field:
  // DW_OP_plus_uconst ForwardingFieldOffset.  Note there's no point in
  // adding the offset if it's 0.
  if (forwardingFieldOffset > 0) {
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(Block, 0, dwarf::DW_FORM_udata, forwardingFieldOffset);
  }

  // Now dereference the __forwarding field to get to the real __Block_byref
  // struct:  DW_OP_deref.
  addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Now that we've got the real __Block_byref... struct, add the offset
  // for the variable's field to get to the location of the actual variable:
  // DW_OP_plus_uconst varFieldOffset.  Again, don't add if it's 0.
  if (varFieldOffset > 0) {
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(Block, 0, dwarf::DW_FORM_udata, varFieldOffset);
  }

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, 0, Block);
}

/// addAddress - Add an address attribute to a die based on the location
/// provided.
void DwarfDebug::addAddress(DIE *Die, unsigned Attribute,
                            const MachineLocation &Location) {
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  const TargetRegisterInfo *TRI = Asm->TM.getRegisterInfo();

  if (TRI->getFrameRegister(*Asm->MF) == Location.getReg()
      && Location.getOffset()) {
    // If variable offset is based in frame register then use fbreg.
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_fbreg);
    addSInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
    addBlock(Die, Attribute, 0, Block);
    return;
  }

  if (Location.isReg()) {
    if (Reg < 32) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    } else {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_regx);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    } else {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      addUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    addUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  addBlock(Die, Attribute, 0, Block);
}

/// addRegisterAddress - Add register location entry in variable DIE.
bool DwarfDebug::addRegisterAddress(DIE *Die, const MCSymbol *VS,
                                    const MachineOperand &MO) {
  assert (MO.isReg() && "Invalid machine operand!");
  if (!MO.getReg())
    return false;
  MachineLocation Location;
  Location.set(MO.getReg());
  addAddress(Die, dwarf::DW_AT_location, Location);
  if (VS)
    addLabel(Die, dwarf::DW_AT_start_scope, dwarf::DW_FORM_addr, VS);
  return true;
}

/// addConstantValue - Add constant value entry in variable DIE.
bool DwarfDebug::addConstantValue(DIE *Die, const MCSymbol *VS,
                                  const MachineOperand &MO) {
  assert (MO.isImm() && "Invalid machine operand!");
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  unsigned Imm = MO.getImm();
  addUInt(Block, 0, dwarf::DW_FORM_udata, Imm);
  addBlock(Die, dwarf::DW_AT_const_value, 0, Block);
  if (VS)
    addLabel(Die, dwarf::DW_AT_start_scope, dwarf::DW_FORM_addr, VS);
  return true;
}

/// addConstantFPValue - Add constant value entry in variable DIE.
bool DwarfDebug::addConstantFPValue(DIE *Die, const MCSymbol *VS,
                                    const MachineOperand &MO) {
  assert (MO.isFPImm() && "Invalid machine operand!");
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
  APFloat FPImm = MO.getFPImm()->getValueAPF();

  // Get the raw data form of the floating point.
  const APInt FltVal = FPImm.bitcastToAPInt();
  const char *FltPtr = (const char*)FltVal.getRawData();

  int NumBytes = FltVal.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getTargetData().isLittleEndian();
  int Incr = (LittleEndian ? 1 : -1);
  int Start = (LittleEndian ? 0 : NumBytes - 1);
  int Stop = (LittleEndian ? NumBytes : -1);

  // Output the constant to DWARF one byte at a time.
  for (; Start != Stop; Start += Incr)
    addUInt(Block, 0, dwarf::DW_FORM_data1,
            (unsigned char)0xFF & FltPtr[Start]);

  addBlock(Die, dwarf::DW_AT_const_value, 0, Block);
  if (VS)
    addLabel(Die, dwarf::DW_AT_start_scope, dwarf::DW_FORM_addr, VS);
  return true;
}


/// addToContextOwner - Add Die into the list of its context owner's children.
void DwarfDebug::addToContextOwner(DIE *Die, DIDescriptor Context) {
  if (Context.isType()) {
    DIE *ContextDIE = getOrCreateTypeDIE(DIType(Context));
    ContextDIE->addChild(Die);
  } else if (Context.isNameSpace()) {
    DIE *ContextDIE = getOrCreateNameSpace(DINameSpace(Context));
    ContextDIE->addChild(Die);
  } else if (Context.isSubprogram()) {
    DIE *ContextDIE = createSubprogramDIE(DISubprogram(Context));
    ContextDIE->addChild(Die);
  } else if (DIE *ContextDIE = getCompileUnit(Context)->getDIE(Context))
    ContextDIE->addChild(Die);
  else
    getCompileUnit(Context)->addDie(Die);
}

/// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
/// given DIType.
DIE *DwarfDebug::getOrCreateTypeDIE(DIType Ty) {
  CompileUnit *TypeCU = getCompileUnit(Ty);
  DIE *TyDIE = TypeCU->getDIE(Ty);
  if (TyDIE)
    return TyDIE;

  // Create new type.
  TyDIE = new DIE(dwarf::DW_TAG_base_type);
  TypeCU->insertDIE(Ty, TyDIE);
  if (Ty.isBasicType())
    constructTypeDIE(*TyDIE, DIBasicType(Ty));
  else if (Ty.isCompositeType())
    constructTypeDIE(*TyDIE, DICompositeType(Ty));
  else {
    assert(Ty.isDerivedType() && "Unknown kind of DIType");
    constructTypeDIE(*TyDIE, DIDerivedType(Ty));
  }

  addToContextOwner(TyDIE, Ty.getContext());
  return TyDIE;
}

/// addType - Add a new type attribute to the specified entity.
void DwarfDebug::addType(DIE *Entity, DIType Ty) {
  if (!Ty.Verify())
    return;

  // Check for pre-existence.
  CompileUnit *TypeCU = getCompileUnit(Ty);
  DIEEntry *Entry = TypeCU->getDIEEntry(Ty);
  // If it exists then use the existing value.
  if (Entry) {
    Entity->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Entry);
    return;
  }

  // Construct type.
  DIE *Buffer = getOrCreateTypeDIE(Ty);

  // Set up proxy.
  Entry = createDIEEntry(Buffer);
  TypeCU->insertDIEEntry(Ty, Entry);

  Entity->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Entry);
}

/// constructTypeDIE - Construct basic type die from DIBasicType.
void DwarfDebug::constructTypeDIE(DIE &Buffer, DIBasicType BTy) {
  // Get core information.
  StringRef Name = BTy.getName();
  Buffer.setTag(dwarf::DW_TAG_base_type);
  addUInt(&Buffer, dwarf::DW_AT_encoding,  dwarf::DW_FORM_data1,
          BTy.getEncoding());

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  uint64_t Size = BTy.getSizeInBits() >> 3;
  addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);
}

/// constructTypeDIE - Construct derived type die from DIDerivedType.
void DwarfDebug::constructTypeDIE(DIE &Buffer, DIDerivedType DTy) {
  // Get core information.
  StringRef Name = DTy.getName();
  uint64_t Size = DTy.getSizeInBits() >> 3;
  unsigned Tag = DTy.getTag();

  // FIXME - Workaround for templates.
  if (Tag == dwarf::DW_TAG_inheritance) Tag = dwarf::DW_TAG_reference_type;

  Buffer.setTag(Tag);

  // Map to main type, void will not have a type.
  DIType FromTy = DTy.getTypeDerivedFrom();
  addType(&Buffer, FromTy);

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  // Add size if non-zero (derived types might be zero-sized.)
  if (Size)
    addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);

  // Add source line info if available and TyDesc is not a forward declaration.
  if (!DTy.isForwardDecl())
    addSourceLine(&Buffer, DTy);
}

/// constructTypeDIE - Construct type DIE from DICompositeType.
void DwarfDebug::constructTypeDIE(DIE &Buffer, DICompositeType CTy) {
  // Get core information.
  StringRef Name = CTy.getName();

  uint64_t Size = CTy.getSizeInBits() >> 3;
  unsigned Tag = CTy.getTag();
  Buffer.setTag(Tag);

  switch (Tag) {
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_array_type:
    constructArrayTypeDIE(Buffer, &CTy);
    break;
  case dwarf::DW_TAG_enumeration_type: {
    DIArray Elements = CTy.getTypeArray();

    // Add enumerators to enumeration type.
    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIE *ElemDie = NULL;
      DIDescriptor Enum(Elements.getElement(i));
      if (Enum.isEnumerator()) {
        ElemDie = constructEnumTypeDIE(DIEnumerator(Enum));
        Buffer.addChild(ElemDie);
      }
    }
  }
    break;
  case dwarf::DW_TAG_subroutine_type: {
    // Add return type.
    DIArray Elements = CTy.getTypeArray();
    DIDescriptor RTy = Elements.getElement(0);
    addType(&Buffer, DIType(RTy));

    bool isPrototyped = true;
    // Add arguments.
    for (unsigned i = 1, N = Elements.getNumElements(); i < N; ++i) {
      DIDescriptor Ty = Elements.getElement(i);
      if (Ty.isUnspecifiedParameter()) {
        DIE *Arg = new DIE(dwarf::DW_TAG_unspecified_parameters);
        Buffer.addChild(Arg);
        isPrototyped = false;
      } else {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        addType(Arg, DIType(Ty));
        Buffer.addChild(Arg);
      }
    }
    // Add prototype flag.
    if (isPrototyped)
      addUInt(&Buffer, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);
  }
    break;
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type: {
    // Add elements to structure type.
    DIArray Elements = CTy.getTypeArray();

    // A forward struct declared type may not have elements available.
    unsigned N = Elements.getNumElements();
    if (N == 0)
      break;

    // Add elements to structure type.
    for (unsigned i = 0; i < N; ++i) {
      DIDescriptor Element = Elements.getElement(i);
      DIE *ElemDie = NULL;
      if (Element.isSubprogram()) {
        DISubprogram SP(Element);
        ElemDie = createSubprogramDIE(DISubprogram(Element));
        if (SP.isProtected())
          addUInt(ElemDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
                  dwarf::DW_ACCESS_protected);
        else if (SP.isPrivate())
          addUInt(ElemDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
                  dwarf::DW_ACCESS_private);
        else 
          addUInt(ElemDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_public);
        if (SP.isExplicit())
          addUInt(ElemDie, dwarf::DW_AT_explicit, dwarf::DW_FORM_flag, 1);
      }
      else if (Element.isVariable()) {
        DIVariable DV(Element);
        ElemDie = new DIE(dwarf::DW_TAG_variable);
        addString(ElemDie, dwarf::DW_AT_name, dwarf::DW_FORM_string,
                  DV.getName());
        addType(ElemDie, DV.getType());
        addUInt(ElemDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
        addUInt(ElemDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);
        addSourceLine(ElemDie, DV);
      } else if (Element.isDerivedType())
        ElemDie = createMemberDIE(DIDerivedType(Element));
      else
        continue;
      Buffer.addChild(ElemDie);
    }

    if (CTy.isAppleBlockExtension())
      addUInt(&Buffer, dwarf::DW_AT_APPLE_block, dwarf::DW_FORM_flag, 1);

    unsigned RLang = CTy.getRunTimeLang();
    if (RLang)
      addUInt(&Buffer, dwarf::DW_AT_APPLE_runtime_class,
              dwarf::DW_FORM_data1, RLang);

    DICompositeType ContainingType = CTy.getContainingType();
    if (DIDescriptor(ContainingType).isCompositeType())
      addDIEEntry(&Buffer, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4,
                  getOrCreateTypeDIE(DIType(ContainingType)));
    else {
      DIDescriptor Context = CTy.getContext();
      addToContextOwner(&Buffer, Context);
    }
    break;
  }
  default:
    break;
  }

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  if (Tag == dwarf::DW_TAG_enumeration_type || Tag == dwarf::DW_TAG_class_type
      || Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type)
    {
    // Add size if non-zero (derived types might be zero-sized.)
    if (Size)
      addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);
    else {
      // Add zero size if it is not a forward declaration.
      if (CTy.isForwardDecl())
        addUInt(&Buffer, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
      else
        addUInt(&Buffer, dwarf::DW_AT_byte_size, 0, 0);
    }

    // Add source line info if available.
    if (!CTy.isForwardDecl())
      addSourceLine(&Buffer, CTy);
  }
}

/// constructSubrangeDIE - Construct subrange DIE from DISubrange.
void DwarfDebug::constructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy){
  int64_t L = SR.getLo();
  int64_t H = SR.getHi();
  DIE *DW_Subrange = new DIE(dwarf::DW_TAG_subrange_type);

  addDIEEntry(DW_Subrange, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IndexTy);
  if (L)
    addSInt(DW_Subrange, dwarf::DW_AT_lower_bound, 0, L);
  addSInt(DW_Subrange, dwarf::DW_AT_upper_bound, 0, H);

  Buffer.addChild(DW_Subrange);
}

/// constructArrayTypeDIE - Construct array type DIE from DICompositeType.
void DwarfDebug::constructArrayTypeDIE(DIE &Buffer,
                                       DICompositeType *CTy) {
  Buffer.setTag(dwarf::DW_TAG_array_type);
  if (CTy->getTag() == dwarf::DW_TAG_vector_type)
    addUInt(&Buffer, dwarf::DW_AT_GNU_vector, dwarf::DW_FORM_flag, 1);

  // Emit derived type.
  addType(&Buffer, CTy->getTypeDerivedFrom());
  DIArray Elements = CTy->getTypeArray();

  // Get an anonymous type for index type.
  CompileUnit *TheCU = getCompileUnit(*CTy);
  DIE *IdxTy = TheCU->getIndexTyDie();
  if (!IdxTy) {
    // Construct an anonymous type for index type.
    IdxTy = new DIE(dwarf::DW_TAG_base_type);
    addUInt(IdxTy, dwarf::DW_AT_byte_size, 0, sizeof(int32_t));
    addUInt(IdxTy, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
            dwarf::DW_ATE_signed);
    TheCU->addDie(IdxTy);
    TheCU->setIndexTyDie(IdxTy);
  }

  // Add subranges to array type.
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    if (Element.getTag() == dwarf::DW_TAG_subrange_type)
      constructSubrangeDIE(Buffer, DISubrange(Element), IdxTy);
  }
}

/// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
DIE *DwarfDebug::constructEnumTypeDIE(DIEnumerator ETy) {
  DIE *Enumerator = new DIE(dwarf::DW_TAG_enumerator);
  StringRef Name = ETy.getName();
  addString(Enumerator, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  int64_t Value = ETy.getEnumValue();
  addSInt(Enumerator, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata, Value);
  return Enumerator;
}

/// getRealLinkageName - If special LLVM prefix that is used to inform the asm
/// printer to not emit usual symbol prefix before the symbol name is used then
/// return linkage name after skipping this special LLVM prefix.
static StringRef getRealLinkageName(StringRef LinkageName) {
  char One = '\1';
  if (LinkageName.startswith(StringRef(&One, 1)))
    return LinkageName.substr(1);
  return LinkageName;
}

/// createMemberDIE - Create new member DIE.
DIE *DwarfDebug::createMemberDIE(DIDerivedType DT) {
  DIE *MemberDie = new DIE(DT.getTag());
  StringRef Name = DT.getName();
  if (!Name.empty())
    addString(MemberDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  addType(MemberDie, DT.getTypeDerivedFrom());

  addSourceLine(MemberDie, DT);

  DIEBlock *MemLocationDie = new (DIEValueAllocator) DIEBlock();
  addUInt(MemLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);

  uint64_t Size = DT.getSizeInBits();
  uint64_t FieldSize = DT.getOriginalTypeSize();

  if (Size != FieldSize) {
    // Handle bitfield.
    addUInt(MemberDie, dwarf::DW_AT_byte_size, 0, DT.getOriginalTypeSize()>>3);
    addUInt(MemberDie, dwarf::DW_AT_bit_size, 0, DT.getSizeInBits());

    uint64_t Offset = DT.getOffsetInBits();
    uint64_t AlignMask = ~(DT.getAlignInBits() - 1);
    uint64_t HiMark = (Offset + FieldSize) & AlignMask;
    uint64_t FieldOffset = (HiMark - FieldSize);
    Offset -= FieldOffset;

    // Maybe we need to work from the other end.
    if (Asm->getTargetData().isLittleEndian())
      Offset = FieldSize - (Offset + Size);
    addUInt(MemberDie, dwarf::DW_AT_bit_offset, 0, Offset);

    // Here WD_AT_data_member_location points to the anonymous
    // field that includes this bit field.
    addUInt(MemLocationDie, 0, dwarf::DW_FORM_udata, FieldOffset >> 3);

  } else
    // This is not a bitfield.
    addUInt(MemLocationDie, 0, dwarf::DW_FORM_udata, DT.getOffsetInBits() >> 3);

  if (DT.getTag() == dwarf::DW_TAG_inheritance
      && DT.isVirtual()) {

    // For C++, virtual base classes are not at fixed offset. Use following
    // expression to extract appropriate offset from vtable.
    // BaseAddr = ObAddr + *((*ObAddr) - Offset)

    DIEBlock *VBaseLocationDie = new (DIEValueAllocator) DIEBlock();
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_dup);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_udata, DT.getOffsetInBits());
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_minus);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(VBaseLocationDie, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);

    addBlock(MemberDie, dwarf::DW_AT_data_member_location, 0,
             VBaseLocationDie);
  } else
    addBlock(MemberDie, dwarf::DW_AT_data_member_location, 0, MemLocationDie);

  if (DT.isProtected())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_protected);
  else if (DT.isPrivate())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_private);
  // Otherwise C++ member and base classes are considered public.
  else if (DT.getCompileUnit().getLanguage() == dwarf::DW_LANG_C_plus_plus)
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_public);
  if (DT.isVirtual())
    addUInt(MemberDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_flag,
            dwarf::DW_VIRTUALITY_virtual);
  return MemberDie;
}

/// createSubprogramDIE - Create new DIE using SP.
DIE *DwarfDebug::createSubprogramDIE(DISubprogram SP) {
  CompileUnit *SPCU = getCompileUnit(SP);
  DIE *SPDie = SPCU->getDIE(SP);
  if (SPDie)
    return SPDie;

  SPDie = new DIE(dwarf::DW_TAG_subprogram);
  // Constructors and operators for anonymous aggregates do not have names.
  if (!SP.getName().empty())
    addString(SPDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, SP.getName());

  StringRef LinkageName = SP.getLinkageName();
  if (!LinkageName.empty())
    addString(SPDie, dwarf::DW_AT_MIPS_linkage_name, dwarf::DW_FORM_string,
              getRealLinkageName(LinkageName));

  addSourceLine(SPDie, SP);

  if (SP.isPrototyped()) 
    addUInt(SPDie, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);

  // Add Return Type.
  DICompositeType SPTy = SP.getType();
  DIArray Args = SPTy.getTypeArray();
  unsigned SPTag = SPTy.getTag();

  if (Args.getNumElements() == 0 || SPTag != dwarf::DW_TAG_subroutine_type)
    addType(SPDie, SPTy);
  else
    addType(SPDie, DIType(Args.getElement(0)));

  unsigned VK = SP.getVirtuality();
  if (VK) {
    addUInt(SPDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_flag, VK);
    DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(Block, 0, dwarf::DW_FORM_data1, SP.getVirtualIndex());
    addBlock(SPDie, dwarf::DW_AT_vtable_elem_location, 0, Block);
    ContainingTypeMap.insert(std::make_pair(SPDie,
                                            SP.getContainingType()));
  }

  if (!SP.isDefinition()) {
    addUInt(SPDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);

    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled while processing variables.
    DICompositeType SPTy = SP.getType();
    DIArray Args = SPTy.getTypeArray();
    unsigned SPTag = SPTy.getTag();

    if (SPTag == dwarf::DW_TAG_subroutine_type)
      for (unsigned i = 1, N =  Args.getNumElements(); i < N; ++i) {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        DIType ATy = DIType(DIType(Args.getElement(i)));
        addType(Arg, ATy);
        if (ATy.isArtificial())
          addUInt(Arg, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);
        SPDie->addChild(Arg);
      }
  }

  if (SP.isArtificial())
    addUInt(SPDie, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);

  if (!SP.isLocalToUnit())
    addUInt(SPDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);

  if (SP.isOptimized())
    addUInt(SPDie, dwarf::DW_AT_APPLE_optimized, dwarf::DW_FORM_flag, 1);

  if (unsigned isa = Asm->getISAEncoding()) {
    addUInt(SPDie, dwarf::DW_AT_APPLE_isa, dwarf::DW_FORM_flag, isa);
  }

  // DW_TAG_inlined_subroutine may refer to this DIE.
  SPCU->insertDIE(SP, SPDie);

  // Add to context owner.
  addToContextOwner(SPDie, SP.getContext());

  return SPDie;
}

DbgScope *DwarfDebug::getOrCreateAbstractScope(const MDNode *N) {
  assert(N && "Invalid Scope encoding!");

  DbgScope *AScope = AbstractScopes.lookup(N);
  if (AScope)
    return AScope;

  DbgScope *Parent = NULL;

  DIDescriptor Scope(N);
  if (Scope.isLexicalBlock()) {
    DILexicalBlock DB(N);
    DIDescriptor ParentDesc = DB.getContext();
    Parent = getOrCreateAbstractScope(ParentDesc);
  }

  AScope = new DbgScope(Parent, DIDescriptor(N), NULL);

  if (Parent)
    Parent->addScope(AScope);
  AScope->setAbstractScope();
  AbstractScopes[N] = AScope;
  if (DIDescriptor(N).isSubprogram())
    AbstractScopesList.push_back(AScope);
  return AScope;
}

/// isSubprogramContext - Return true if Context is either a subprogram
/// or another context nested inside a subprogram.
static bool isSubprogramContext(const MDNode *Context) {
  if (!Context)
    return false;
  DIDescriptor D(Context);
  if (D.isSubprogram())
    return true;
  if (D.isType())
    return isSubprogramContext(DIType(Context).getContext());
  return false;
}

/// updateSubprogramScopeDIE - Find DIE for the given subprogram and
/// attach appropriate DW_AT_low_pc and DW_AT_high_pc attributes.
/// If there are global variables in this scope then create and insert
/// DIEs for these variables.
DIE *DwarfDebug::updateSubprogramScopeDIE(const MDNode *SPNode) {
  CompileUnit *SPCU = getCompileUnit(SPNode);
  DIE *SPDie = SPCU->getDIE(SPNode);

  assert(SPDie && "Unable to find subprogram DIE!");
  DISubprogram SP(SPNode);

  // There is not any need to generate specification DIE for a function
  // defined at compile unit level. If a function is defined inside another
  // function then gdb prefers the definition at top level and but does not
  // expect specification DIE in parent function. So avoid creating
  // specification DIE for a function defined inside a function.
  if (SP.isDefinition() && !SP.getContext().isCompileUnit() &&
      !SP.getContext().isFile() &&
      !isSubprogramContext(SP.getContext())) {
    addUInt(SPDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);

    // Add arguments.
    DICompositeType SPTy = SP.getType();
    DIArray Args = SPTy.getTypeArray();
    unsigned SPTag = SPTy.getTag();
    if (SPTag == dwarf::DW_TAG_subroutine_type)
      for (unsigned i = 1, N = Args.getNumElements(); i < N; ++i) {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        DIType ATy = DIType(DIType(Args.getElement(i)));
        addType(Arg, ATy);
        if (ATy.isArtificial())
          addUInt(Arg, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);
        SPDie->addChild(Arg);
      }
    DIE *SPDeclDie = SPDie;
    SPDie = new DIE(dwarf::DW_TAG_subprogram);
    addDIEEntry(SPDie, dwarf::DW_AT_specification, dwarf::DW_FORM_ref4,
                SPDeclDie);
    SPCU->addDie(SPDie);
  }

  // Pick up abstract subprogram DIE.
  if (DIE *AbsSPDIE = AbstractSPDies.lookup(SPNode)) {
    SPDie = new DIE(dwarf::DW_TAG_subprogram);
    addDIEEntry(SPDie, dwarf::DW_AT_abstract_origin,
                dwarf::DW_FORM_ref4, AbsSPDIE);
    SPCU->addDie(SPDie);
  }

  addLabel(SPDie, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
           Asm->GetTempSymbol("func_begin", Asm->getFunctionNumber()));
  addLabel(SPDie, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
           Asm->GetTempSymbol("func_end", Asm->getFunctionNumber()));
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  MachineLocation Location(RI->getFrameRegister(*Asm->MF));
  addAddress(SPDie, dwarf::DW_AT_frame_base, Location);

  return SPDie;
}

/// constructLexicalScope - Construct new DW_TAG_lexical_block
/// for this scope and attach DW_AT_low_pc/DW_AT_high_pc labels.
DIE *DwarfDebug::constructLexicalScopeDIE(DbgScope *Scope) {

  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_lexical_block);
  if (Scope->isAbstractScope())
    return ScopeDIE;

  const SmallVector<DbgRange, 4> &Ranges = Scope->getRanges();
  if (Ranges.empty())
    return 0;

  SmallVector<DbgRange, 4>::const_iterator RI = Ranges.begin();
  if (Ranges.size() > 1) {
    // .debug_range section has not been laid out yet. Emit offset in
    // .debug_range as a uint, size 4, for now. emitDIE will handle
    // DW_AT_ranges appropriately.
    addUInt(ScopeDIE, dwarf::DW_AT_ranges, dwarf::DW_FORM_data4,
            DebugRangeSymbols.size() * Asm->getTargetData().getPointerSize());
    for (SmallVector<DbgRange, 4>::const_iterator RI = Ranges.begin(),
         RE = Ranges.end(); RI != RE; ++RI) {
      DebugRangeSymbols.push_back(getLabelBeforeInsn(RI->first));
      DebugRangeSymbols.push_back(getLabelAfterInsn(RI->second));
    }
    DebugRangeSymbols.push_back(NULL);
    DebugRangeSymbols.push_back(NULL);
    return ScopeDIE;
  }

  const MCSymbol *Start = getLabelBeforeInsn(RI->first);
  const MCSymbol *End = getLabelAfterInsn(RI->second);

  if (End == 0) return 0;

  assert(Start->isDefined() && "Invalid starting label for an inlined scope!");
  assert(End->isDefined() && "Invalid end label for an inlined scope!");

  addLabel(ScopeDIE, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr, Start);
  addLabel(ScopeDIE, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr, End);

  return ScopeDIE;
}

/// constructInlinedScopeDIE - This scope represents inlined body of
/// a function. Construct DIE to represent this concrete inlined copy
/// of the function.
DIE *DwarfDebug::constructInlinedScopeDIE(DbgScope *Scope) {

  const SmallVector<DbgRange, 4> &Ranges = Scope->getRanges();
  assert (Ranges.empty() == false
          && "DbgScope does not have instruction markers!");

  // FIXME : .debug_inlined section specification does not clearly state how
  // to emit inlined scope that is split into multiple instruction ranges.
  // For now, use first instruction range and emit low_pc/high_pc pair and
  // corresponding .debug_inlined section entry for this pair.
  SmallVector<DbgRange, 4>::const_iterator RI = Ranges.begin();
  const MCSymbol *StartLabel = getLabelBeforeInsn(RI->first);
  const MCSymbol *EndLabel = getLabelAfterInsn(RI->second);

  if (StartLabel == 0 || EndLabel == 0) {
    assert (0 && "Unexpected Start and End  labels for a inlined scope!");
    return 0;
  }
  assert(StartLabel->isDefined() &&
         "Invalid starting label for an inlined scope!");
  assert(EndLabel->isDefined() &&
         "Invalid end label for an inlined scope!");

  if (!Scope->getScopeNode())
    return NULL;
  DIScope DS(Scope->getScopeNode());
  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_inlined_subroutine);

  DISubprogram InlinedSP = getDISubprogram(DS);
  CompileUnit *TheCU = getCompileUnit(InlinedSP);
  DIE *OriginDIE = TheCU->getDIE(InlinedSP);
  assert(OriginDIE && "Unable to find Origin DIE!");
  addDIEEntry(ScopeDIE, dwarf::DW_AT_abstract_origin,
              dwarf::DW_FORM_ref4, OriginDIE);

  addLabel(ScopeDIE, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr, StartLabel);
  addLabel(ScopeDIE, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr, EndLabel);

  InlinedSubprogramDIEs.insert(OriginDIE);

  // Track the start label for this inlined function.
  DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> >::iterator
    I = InlineInfo.find(InlinedSP);

  if (I == InlineInfo.end()) {
    InlineInfo[InlinedSP].push_back(std::make_pair(StartLabel,
                                                             ScopeDIE));
    InlinedSPNodes.push_back(InlinedSP);
  } else
    I->second.push_back(std::make_pair(StartLabel, ScopeDIE));

  DILocation DL(Scope->getInlinedAt());
  addUInt(ScopeDIE, dwarf::DW_AT_call_file, 0, TheCU->getID());
  addUInt(ScopeDIE, dwarf::DW_AT_call_line, 0, DL.getLineNumber());

  return ScopeDIE;
}


/// constructVariableDIE - Construct a DIE for the given DbgVariable.
DIE *DwarfDebug::constructVariableDIE(DbgVariable *DV, DbgScope *Scope) {
  StringRef Name = DV->getName();
  if (Name.empty())
    return NULL;

  // Translate tag to proper Dwarf tag.  The result variable is dropped for
  // now.
  unsigned Tag;
  switch (DV->getTag()) {
  case dwarf::DW_TAG_return_variable:
    return NULL;
  case dwarf::DW_TAG_arg_variable:
    Tag = dwarf::DW_TAG_formal_parameter;
    break;
  case dwarf::DW_TAG_auto_variable:    // fall thru
  default:
    Tag = dwarf::DW_TAG_variable;
    break;
  }

  // Define variable debug information entry.
  DIE *VariableDie = new DIE(Tag);

  DIE *AbsDIE = NULL;
  DenseMap<const DbgVariable *, const DbgVariable *>::iterator
    V2AVI = VarToAbstractVarMap.find(DV);
  if (V2AVI != VarToAbstractVarMap.end())
    AbsDIE = V2AVI->second->getDIE();

  if (AbsDIE)
    addDIEEntry(VariableDie, dwarf::DW_AT_abstract_origin,
                dwarf::DW_FORM_ref4, AbsDIE);
  else {
    addString(VariableDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
    addSourceLine(VariableDie, DV->getVariable());

    // Add variable type.
    addType(VariableDie, DV->getType());
  }

  if (Tag == dwarf::DW_TAG_formal_parameter && DV->getType().isArtificial())
    addUInt(VariableDie, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);
  else if (DIVariable(DV->getVariable()).isArtificial())
    addUInt(VariableDie, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1);

  if (Scope->isAbstractScope()) {
    DV->setDIE(VariableDie);
    return VariableDie;
  }

  // Add variable address.

  unsigned Offset = DV->getDotDebugLocOffset();
  if (Offset != ~0U) {
    addLabel(VariableDie, dwarf::DW_AT_location, dwarf::DW_FORM_data4,
             Asm->GetTempSymbol("debug_loc", Offset));
    DV->setDIE(VariableDie);
    UseDotDebugLocEntry.insert(VariableDie);
    return VariableDie;
  }

  // Check if variable is described by a  DBG_VALUE instruction.
  DenseMap<const DbgVariable *, const MachineInstr *>::iterator DVI =
    DbgVariableToDbgInstMap.find(DV);
  if (DVI != DbgVariableToDbgInstMap.end()) {
    const MachineInstr *DVInsn = DVI->second;
    const MCSymbol *DVLabel = findVariableLabel(DV);
    bool updated = false;
    // FIXME : Handle getNumOperands != 3
    if (DVInsn->getNumOperands() == 3) {
      if (DVInsn->getOperand(0).isReg()) {
        const MachineOperand RegOp = DVInsn->getOperand(0);
        const TargetRegisterInfo *TRI = Asm->TM.getRegisterInfo();
        if (DVInsn->getOperand(1).isImm() &&
            TRI->getFrameRegister(*Asm->MF) == RegOp.getReg()) {
          addVariableAddress(DV, VariableDie, DVInsn->getOperand(1).getImm());
          updated = true;
        } else
          updated = addRegisterAddress(VariableDie, DVLabel, RegOp);
      }
      else if (DVInsn->getOperand(0).isImm())
        updated = addConstantValue(VariableDie, DVLabel, DVInsn->getOperand(0));
      else if (DVInsn->getOperand(0).isFPImm())
        updated =
          addConstantFPValue(VariableDie, DVLabel, DVInsn->getOperand(0));
    } else {
      MachineLocation Location = Asm->getDebugValueLocation(DVInsn);
      if (Location.getReg()) {
        addAddress(VariableDie, dwarf::DW_AT_location, Location);
        if (DVLabel)
          addLabel(VariableDie, dwarf::DW_AT_start_scope, dwarf::DW_FORM_addr,
                   DVLabel);
        updated = true;
      }
    }
    if (!updated) {
      // If variableDie is not updated then DBG_VALUE instruction does not
      // have valid variable info.
      delete VariableDie;
      return NULL;
    }
    DV->setDIE(VariableDie);
    return VariableDie;
  }

  // .. else use frame index, if available.
  int FI = 0;
  if (findVariableFrameIndex(DV, &FI))
    addVariableAddress(DV, VariableDie, FI);
  
  DV->setDIE(VariableDie);
  return VariableDie;

}

void DwarfDebug::addPubTypes(DISubprogram SP) {
  DICompositeType SPTy = SP.getType();
  unsigned SPTag = SPTy.getTag();
  if (SPTag != dwarf::DW_TAG_subroutine_type)
    return;

  DIArray Args = SPTy.getTypeArray();
  for (unsigned i = 0, e = Args.getNumElements(); i != e; ++i) {
    DIType ATy(Args.getElement(i));
    if (!ATy.Verify())
      continue;
    DICompositeType CATy = getDICompositeType(ATy);
    if (DIDescriptor(CATy).Verify() && !CATy.getName().empty()
        && !CATy.isForwardDecl()) {
      CompileUnit *TheCU = getCompileUnit(CATy);
      if (DIEEntry *Entry = TheCU->getDIEEntry(CATy))
        TheCU->addGlobalType(CATy.getName(), Entry->getEntry());
    }
  }
}

/// constructScopeDIE - Construct a DIE for this scope.
DIE *DwarfDebug::constructScopeDIE(DbgScope *Scope) {
  if (!Scope || !Scope->getScopeNode())
    return NULL;

  DIScope DS(Scope->getScopeNode());
  DIE *ScopeDIE = NULL;
  if (Scope->getInlinedAt())
    ScopeDIE = constructInlinedScopeDIE(Scope);
  else if (DS.isSubprogram()) {
    ProcessedSPNodes.insert(DS);
    if (Scope->isAbstractScope()) {
      ScopeDIE = getCompileUnit(DS)->getDIE(DS);
      // Note down abstract DIE.
      if (ScopeDIE)
        AbstractSPDies.insert(std::make_pair(DS, ScopeDIE));
    }
    else
      ScopeDIE = updateSubprogramScopeDIE(DS);
  }
  else
    ScopeDIE = constructLexicalScopeDIE(Scope);
  if (!ScopeDIE) return NULL;

  // Add variables to scope.
  const SmallVector<DbgVariable *, 8> &Variables = Scope->getDbgVariables();
  for (unsigned i = 0, N = Variables.size(); i < N; ++i) {
    DIE *VariableDIE = constructVariableDIE(Variables[i], Scope);
    if (VariableDIE)
      ScopeDIE->addChild(VariableDIE);
  }

  // Add nested scopes.
  const SmallVector<DbgScope *, 4> &Scopes = Scope->getScopes();
  for (unsigned j = 0, M = Scopes.size(); j < M; ++j) {
    // Define the Scope debug information entry.
    DIE *NestedDIE = constructScopeDIE(Scopes[j]);
    if (NestedDIE)
      ScopeDIE->addChild(NestedDIE);
  }

  if (DS.isSubprogram())
    addPubTypes(DISubprogram(DS));

 return ScopeDIE;
}

/// GetOrCreateSourceID - Look up the source id with the given directory and
/// source file names. If none currently exists, create a new id and insert it
/// in the SourceIds map. This can update DirectoryNames and SourceFileNames
/// maps as well.
unsigned DwarfDebug::GetOrCreateSourceID(StringRef DirName, StringRef FileName){
  unsigned DId;
  assert (DirName.empty() == false && "Invalid directory name!");

  // If FE did not provide a file name, then assume stdin.
  if (FileName.empty())
    return GetOrCreateSourceID(DirName, "<stdin>");

  StringMap<unsigned>::iterator DI = DirectoryIdMap.find(DirName);
  if (DI != DirectoryIdMap.end()) {
    DId = DI->getValue();
  } else {
    DId = DirectoryNames.size() + 1;
    DirectoryIdMap[DirName] = DId;
    DirectoryNames.push_back(DirName);
  }

  unsigned FId;
  StringMap<unsigned>::iterator FI = SourceFileIdMap.find(FileName);
  if (FI != SourceFileIdMap.end()) {
    FId = FI->getValue();
  } else {
    FId = SourceFileNames.size() + 1;
    SourceFileIdMap[FileName] = FId;
    SourceFileNames.push_back(FileName);
  }

  DenseMap<std::pair<unsigned, unsigned>, unsigned>::iterator SI =
    SourceIdMap.find(std::make_pair(DId, FId));
  if (SI != SourceIdMap.end())
    return SI->second;

  unsigned SrcId = SourceIds.size() + 1;  // DW_AT_decl_file cannot be 0.
  SourceIdMap[std::make_pair(DId, FId)] = SrcId;
  SourceIds.push_back(std::make_pair(DId, FId));

  return SrcId;
}

/// getOrCreateNameSpace - Create a DIE for DINameSpace.
DIE *DwarfDebug::getOrCreateNameSpace(DINameSpace NS) {
  CompileUnit *TheCU = getCompileUnit(NS);
  DIE *NDie = TheCU->getDIE(NS);
  if (NDie)
    return NDie;
  NDie = new DIE(dwarf::DW_TAG_namespace);
  TheCU->insertDIE(NS, NDie);
  if (!NS.getName().empty())
    addString(NDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, NS.getName());
  addSourceLine(NDie, NS);
  addToContextOwner(NDie, NS.getContext());
  return NDie;
}

/// constructCompileUnit - Create new CompileUnit for the given
/// metadata node with tag DW_TAG_compile_unit.
void DwarfDebug::constructCompileUnit(const MDNode *N) {
  DICompileUnit DIUnit(N);
  StringRef FN = DIUnit.getFilename();
  StringRef Dir = DIUnit.getDirectory();
  unsigned ID = GetOrCreateSourceID(Dir, FN);

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  addString(Die, dwarf::DW_AT_producer, dwarf::DW_FORM_string,
            DIUnit.getProducer());
  addUInt(Die, dwarf::DW_AT_language, dwarf::DW_FORM_data1,
          DIUnit.getLanguage());
  addString(Die, dwarf::DW_AT_name, dwarf::DW_FORM_string, FN);
  // Use DW_AT_entry_pc instead of DW_AT_low_pc/DW_AT_high_pc pair. This
  // simplifies debug range entries.
  addUInt(Die, dwarf::DW_AT_entry_pc, dwarf::DW_FORM_addr, 0);
  // DW_AT_stmt_list is a offset of line number information for this
  // compile unit in debug_line section.
  if (Asm->MAI->doesDwarfUsesAbsoluteLabelForStmtList())
    addLabel(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_addr,
             Asm->GetTempSymbol("section_line"));
  else
    addUInt(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4, 0);

  if (!Dir.empty())
    addString(Die, dwarf::DW_AT_comp_dir, dwarf::DW_FORM_string, Dir);
  if (DIUnit.isOptimized())
    addUInt(Die, dwarf::DW_AT_APPLE_optimized, dwarf::DW_FORM_flag, 1);

  StringRef Flags = DIUnit.getFlags();
  if (!Flags.empty())
    addString(Die, dwarf::DW_AT_APPLE_flags, dwarf::DW_FORM_string, Flags);

  unsigned RVer = DIUnit.getRunTimeVersion();
  if (RVer)
    addUInt(Die, dwarf::DW_AT_APPLE_major_runtime_vers,
            dwarf::DW_FORM_data1, RVer);

  CompileUnit *NewCU = new CompileUnit(ID, Die);
  if (!FirstCU)
    FirstCU = NewCU;
  CUMap.insert(std::make_pair(N, NewCU));
}

/// getCompielUnit - Get CompileUnit DIE.
CompileUnit *DwarfDebug::getCompileUnit(const MDNode *N) const {
  assert (N && "Invalid DwarfDebug::getCompileUnit argument!");
  DIDescriptor D(N);
  const MDNode *CUNode = NULL;
  if (D.isCompileUnit())
    CUNode = N;
  else if (D.isSubprogram())
    CUNode = DISubprogram(N).getCompileUnit();
  else if (D.isType())
    CUNode = DIType(N).getCompileUnit();
  else if (D.isGlobalVariable())
    CUNode = DIGlobalVariable(N).getCompileUnit();
  else if (D.isVariable())
    CUNode = DIVariable(N).getCompileUnit();
  else if (D.isNameSpace())
    CUNode = DINameSpace(N).getCompileUnit();
  else if (D.isFile())
    CUNode = DIFile(N).getCompileUnit();
  else
    return FirstCU;

  DenseMap<const MDNode *, CompileUnit *>::const_iterator I
    = CUMap.find(CUNode);
  if (I == CUMap.end())
    return FirstCU;
  return I->second;
}

/// isUnsignedDIType - Return true if type encoding is unsigned.
static bool isUnsignedDIType(DIType Ty) {
  DIDerivedType DTy(Ty);
  if (DTy.Verify())
    return isUnsignedDIType(DTy.getTypeDerivedFrom());

  DIBasicType BTy(Ty);
  if (BTy.Verify()) {
    unsigned Encoding = BTy.getEncoding();
    if (Encoding == dwarf::DW_ATE_unsigned ||
        Encoding == dwarf::DW_ATE_unsigned_char)
      return true;
  }
  return false;
}

/// constructGlobalVariableDIE - Construct global variable DIE.
void DwarfDebug::constructGlobalVariableDIE(const MDNode *N) {
  DIGlobalVariable GV(N);

  // If debug information is malformed then ignore it.
  if (GV.Verify() == false)
    return;

  // Check for pre-existence.
  CompileUnit *TheCU = getCompileUnit(N);
  if (TheCU->getDIE(GV))
    return;

  DIType GTy = GV.getType();
  DIE *VariableDIE = new DIE(GV.getTag());

  bool isGlobalVariable = GV.getGlobal() != NULL;

  // Add name.
  addString(VariableDIE, dwarf::DW_AT_name, dwarf::DW_FORM_string,
            GV.getDisplayName());
  StringRef LinkageName = GV.getLinkageName();
  if (!LinkageName.empty() && isGlobalVariable)
    addString(VariableDIE, dwarf::DW_AT_MIPS_linkage_name, dwarf::DW_FORM_string,
              getRealLinkageName(LinkageName));
  // Add type.
  addType(VariableDIE, GTy);
  if (GTy.isCompositeType() && !GTy.getName().empty()
      && !GTy.isForwardDecl()) {
    DIEEntry *Entry = TheCU->getDIEEntry(GTy);
    assert(Entry && "Missing global type!");
    TheCU->addGlobalType(GTy.getName(), Entry->getEntry());
  }
  // Add scoping info.
  if (!GV.isLocalToUnit()) {
    addUInt(VariableDIE, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);
    // Expose as global. 
    TheCU->addGlobal(GV.getName(), VariableDIE);
  }
  // Add line number info.
  addSourceLine(VariableDIE, GV);
  // Add to map.
  TheCU->insertDIE(N, VariableDIE);
  // Add to context owner.
  DIDescriptor GVContext = GV.getContext();
  addToContextOwner(VariableDIE, GVContext);
  // Add location.
  if (isGlobalVariable) {
    DIEBlock *Block = new (DIEValueAllocator) DIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addLabel(Block, 0, dwarf::DW_FORM_udata,
             Asm->Mang->getSymbol(GV.getGlobal()));
    // Do not create specification DIE if context is either compile unit
    // or a subprogram.
    if (GV.isDefinition() && !GVContext.isCompileUnit() &&
        !GVContext.isFile() && !isSubprogramContext(GVContext)) {
      // Create specification DIE.
      DIE *VariableSpecDIE = new DIE(dwarf::DW_TAG_variable);
      addDIEEntry(VariableSpecDIE, dwarf::DW_AT_specification,
                  dwarf::DW_FORM_ref4, VariableDIE);
      addBlock(VariableSpecDIE, dwarf::DW_AT_location, 0, Block);
      addUInt(VariableDIE, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
      TheCU->addDie(VariableSpecDIE);
    } else {
      addBlock(VariableDIE, dwarf::DW_AT_location, 0, Block);
    } 
  } else if (Constant *C = GV.getConstant()) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
      if (isUnsignedDIType(GTy))
          addUInt(VariableDIE, dwarf::DW_AT_const_value, dwarf::DW_FORM_udata,
                  CI->getZExtValue());
        else
          addSInt(VariableDIE, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata,
                 CI->getSExtValue());
    }
  }
  return;
}

/// construct SubprogramDIE - Construct subprogram DIE.
void DwarfDebug::constructSubprogramDIE(const MDNode *N) {
  DISubprogram SP(N);

  // Check for pre-existence.
  CompileUnit *TheCU = getCompileUnit(N);
  if (TheCU->getDIE(N))
    return;

  if (!SP.isDefinition())
    // This is a method declaration which will be handled while constructing
    // class type.
    return;

  DIE *SubprogramDie = createSubprogramDIE(SP);

  // Add to map.
  TheCU->insertDIE(N, SubprogramDie);

  // Add to context owner.
  addToContextOwner(SubprogramDie, SP.getContext());

  // Expose as global.
  TheCU->addGlobal(SP.getName(), SubprogramDie);

  return;
}

/// beginModule - Emit all Dwarf sections that should come prior to the
/// content. Create global DIEs and emit initial debug info sections.
/// This is inovked by the target AsmPrinter.
void DwarfDebug::beginModule(Module *M) {
  if (DisableDebugInfoPrinting)
    return;

  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(*M);

  bool HasDebugInfo = false;

  // Scan all the compile-units to see if there are any marked as the main unit.
  // if not, we do not generate debug info.
  for (DebugInfoFinder::iterator I = DbgFinder.compile_unit_begin(),
       E = DbgFinder.compile_unit_end(); I != E; ++I) {
    if (DICompileUnit(*I).isMain()) {
      HasDebugInfo = true;
      break;
    }
  }

  if (!HasDebugInfo) return;

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);

  // Emit initial sections.
  EmitSectionLabels();

  // Create all the compile unit DIEs.
  for (DebugInfoFinder::iterator I = DbgFinder.compile_unit_begin(),
         E = DbgFinder.compile_unit_end(); I != E; ++I)
    constructCompileUnit(*I);

  // Create DIEs for each subprogram.
  for (DebugInfoFinder::iterator I = DbgFinder.subprogram_begin(),
         E = DbgFinder.subprogram_end(); I != E; ++I)
    constructSubprogramDIE(*I);

  // Create DIEs for each global variable.
  for (DebugInfoFinder::iterator I = DbgFinder.global_variable_begin(),
         E = DbgFinder.global_variable_end(); I != E; ++I)
    constructGlobalVariableDIE(*I);

  //getOrCreateTypeDIE
  if (NamedMDNode *NMD = M->getNamedMetadata("llvm.dbg.enum"))
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      getOrCreateTypeDIE(DIType(NMD->getOperand(i)));

  if (NamedMDNode *NMD = M->getNamedMetadata("llvm.dbg.ty"))
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      getOrCreateTypeDIE(DIType(NMD->getOperand(i)));

  // Prime section data.
  SectionMap.insert(Asm->getObjFileLowering().getTextSection());

  // Print out .file directives to specify files for .loc directives. These are
  // printed out early so that they precede any .loc directives.
  if (Asm->MAI->hasDotLocAndDotFile()) {
    for (unsigned i = 1, e = getNumSourceIds()+1; i != e; ++i) {
      // Remember source id starts at 1.
      std::pair<unsigned, unsigned> Id = getSourceDirectoryAndFileIds(i);
      // FIXME: don't use sys::path for this!  This should not depend on the
      // host.
      sys::Path FullPath(getSourceDirectoryName(Id.first));
      bool AppendOk =
        FullPath.appendComponent(getSourceFileName(Id.second));
      assert(AppendOk && "Could not append filename to directory!");
      AppendOk = false;
      Asm->OutStreamer.EmitDwarfFileDirective(i, FullPath.str());
    }
  }
}

/// endModule - Emit all Dwarf sections that should come after the content.
///
void DwarfDebug::endModule() {
  if (!FirstCU) return;
  const Module *M = MMI->getModule();
  DenseMap<const MDNode *, DbgScope *> DeadFnScopeMap;
  if (NamedMDNode *AllSPs = M->getNamedMetadata("llvm.dbg.sp")) {
    for (unsigned SI = 0, SE = AllSPs->getNumOperands(); SI != SE; ++SI) {
      if (ProcessedSPNodes.count(AllSPs->getOperand(SI)) != 0) continue;
      DISubprogram SP(AllSPs->getOperand(SI));
      if (!SP.Verify()) continue;

      // Collect info for variables that were optimized out.
      if (!SP.isDefinition()) continue;
      StringRef FName = SP.getLinkageName();
      if (FName.empty())
        FName = SP.getName();
      NamedMDNode *NMD =
        M->getNamedMetadata(Twine("llvm.dbg.lv.", getRealLinkageName(FName)));
      if (!NMD) continue;
      unsigned E = NMD->getNumOperands();
      if (!E) continue;
      DbgScope *Scope = new DbgScope(NULL, DIDescriptor(SP), NULL);
      DeadFnScopeMap[SP] = Scope;
      for (unsigned I = 0; I != E; ++I) {
        DIVariable DV(NMD->getOperand(I));
        if (!DV.Verify()) continue;
        Scope->addVariable(new DbgVariable(DV));
      }

      // Construct subprogram DIE and add variables DIEs.
      constructSubprogramDIE(SP);
      DIE *ScopeDIE = getCompileUnit(SP)->getDIE(SP);
      const SmallVector<DbgVariable *, 8> &Variables = Scope->getDbgVariables();
      for (unsigned i = 0, N = Variables.size(); i < N; ++i) {
        DIE *VariableDIE = constructVariableDIE(Variables[i], Scope);
        if (VariableDIE)
          ScopeDIE->addChild(VariableDIE);
      }
    }
  }

  // Attach DW_AT_inline attribute with inlined subprogram DIEs.
  for (SmallPtrSet<DIE *, 4>::iterator AI = InlinedSubprogramDIEs.begin(),
         AE = InlinedSubprogramDIEs.end(); AI != AE; ++AI) {
    DIE *ISP = *AI;
    addUInt(ISP, dwarf::DW_AT_inline, 0, dwarf::DW_INL_inlined);
  }

  for (DenseMap<DIE *, const MDNode *>::iterator CI = ContainingTypeMap.begin(),
         CE = ContainingTypeMap.end(); CI != CE; ++CI) {
    DIE *SPDie = CI->first;
    const MDNode *N = dyn_cast_or_null<MDNode>(CI->second);
    if (!N) continue;
    DIE *NDie = getCompileUnit(N)->getDIE(N);
    if (!NDie) continue;
    addDIEEntry(SPDie, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4, NDie);
  }

  // Standard sections final addresses.
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getTextSection());
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("text_end"));
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getDataSection());
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("data_end"));

  // End text sections.
  for (unsigned i = 1, N = SectionMap.size(); i <= N; ++i) {
    Asm->OutStreamer.SwitchSection(SectionMap[i]);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("section_end", i));
  }

  // Emit common frame information.
  emitCommonDebugFrame();

  // Emit function debug frame information
  for (std::vector<FunctionDebugFrameInfo>::iterator I = DebugFrames.begin(),
         E = DebugFrames.end(); I != E; ++I)
    emitFunctionDebugFrame(*I);

  // Compute DIE offsets and sizes.
  computeSizeAndOffsets();

  // Emit all the DIEs into a debug info section
  emitDebugInfo();

  // Corresponding abbreviations into a abbrev section.
  emitAbbreviations();

  // Emit source line correspondence into a debug line section.
  emitDebugLines();

  // Emit info into a debug pubnames section.
  emitDebugPubNames();

  // Emit info into a debug pubtypes section.
  emitDebugPubTypes();

  // Emit info into a debug loc section.
  emitDebugLoc();

  // Emit info into a debug aranges section.
  EmitDebugARanges();

  // Emit info into a debug ranges section.
  emitDebugRanges();

  // Emit info into a debug macinfo section.
  emitDebugMacInfo();

  // Emit inline info.
  emitDebugInlineInfo();

  // Emit info into a debug str section.
  emitDebugStr();

  // clean up.
  DeleteContainerSeconds(DeadFnScopeMap);
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I)
    delete I->second;
  FirstCU = NULL;  // Reset for the next Module, if any.
}

/// findAbstractVariable - Find abstract variable, if any, associated with Var.
DbgVariable *DwarfDebug::findAbstractVariable(DIVariable &Var,
                                              DebugLoc ScopeLoc) {

  DbgVariable *AbsDbgVariable = AbstractVariables.lookup(Var);
  if (AbsDbgVariable)
    return AbsDbgVariable;

  LLVMContext &Ctx = Var->getContext();
  DbgScope *Scope = AbstractScopes.lookup(ScopeLoc.getScope(Ctx));
  if (!Scope)
    return NULL;

  AbsDbgVariable = new DbgVariable(Var);
  Scope->addVariable(AbsDbgVariable);
  AbstractVariables[Var] = AbsDbgVariable;
  return AbsDbgVariable;
}

/// collectVariableInfoFromMMITable - Collect variable information from
/// side table maintained by MMI.
void
DwarfDebug::collectVariableInfoFromMMITable(const MachineFunction * MF,
                                   SmallPtrSet<const MDNode *, 16> &Processed) {
  const LLVMContext &Ctx = Asm->MF->getFunction()->getContext();
  MachineModuleInfo::VariableDbgInfoMapTy &VMap = MMI->getVariableDbgInfo();
  for (MachineModuleInfo::VariableDbgInfoMapTy::iterator VI = VMap.begin(),
         VE = VMap.end(); VI != VE; ++VI) {
    const MDNode *Var = VI->first;
    if (!Var) continue;
    Processed.insert(Var);
    DIVariable DV(Var);
    const std::pair<unsigned, DebugLoc> &VP = VI->second;

    DbgScope *Scope = 0;
    if (const MDNode *IA = VP.second.getInlinedAt(Ctx))
      Scope = ConcreteScopes.lookup(IA);
    if (Scope == 0)
      Scope = DbgScopeMap.lookup(VP.second.getScope(Ctx));

    // If variable scope is not found then skip this variable.
    if (Scope == 0)
      continue;

    DbgVariable *AbsDbgVariable = findAbstractVariable(DV, VP.second);
    DbgVariable *RegVar = new DbgVariable(DV);
    recordVariableFrameIndex(RegVar, VP.first);
    Scope->addVariable(RegVar);
    if (AbsDbgVariable) {
      recordVariableFrameIndex(AbsDbgVariable, VP.first);
      VarToAbstractVarMap[RegVar] = AbsDbgVariable;
    }
  }
}

/// isDbgValueInUndefinedReg - Return true if debug value, encoded by
/// DBG_VALUE instruction, is in undefined reg.
static bool isDbgValueInUndefinedReg(const MachineInstr *MI) {
  assert (MI->isDebugValue() && "Invalid DBG_VALUE machine instruction!");
  if (MI->getOperand(0).isReg() && !MI->getOperand(0).getReg())
    return true;
  return false;
}

/// isDbgValueInDefinedReg - Return true if debug value, encoded by
/// DBG_VALUE instruction, is in a defined reg.
static bool isDbgValueInDefinedReg(const MachineInstr *MI) {
  assert (MI->isDebugValue() && "Invalid DBG_VALUE machine instruction!");
  if (MI->getOperand(0).isReg() && MI->getOperand(0).getReg())
    return true;
  return false;
}

/// collectVariableInfo - Populate DbgScope entries with variables' info.
void
DwarfDebug::collectVariableInfo(const MachineFunction *MF,
                                SmallPtrSet<const MDNode *, 16> &Processed) {

  /// collection info from MMI table.
  collectVariableInfoFromMMITable(MF, Processed);

  SmallVector<const MachineInstr *, 8> DbgValues;
  // Collect variable information from DBG_VALUE machine instructions;
  for (MachineFunction::const_iterator I = Asm->MF->begin(), E = Asm->MF->end();
       I != E; ++I)
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;
      if (!MInsn->isDebugValue() || isDbgValueInUndefinedReg(MInsn))
        continue;
      DbgValues.push_back(MInsn);
    }

  // This is a collection of DBV_VALUE instructions describing same variable.
  SmallVector<const MachineInstr *, 4> MultipleValues;
  for(SmallVector<const MachineInstr *, 8>::iterator I = DbgValues.begin(),
        E = DbgValues.end(); I != E; ++I) {
    const MachineInstr *MInsn = *I;
    MultipleValues.clear();
    if (isDbgValueInDefinedReg(MInsn))
      MultipleValues.push_back(MInsn);
    DIVariable DV(MInsn->getOperand(MInsn->getNumOperands() - 1).getMetadata());
    if (Processed.count(DV) != 0)
      continue;

    const MachineInstr *PrevMI = MInsn;
    for (SmallVector<const MachineInstr *, 8>::iterator MI = I+1,
           ME = DbgValues.end(); MI != ME; ++MI) {
      const MDNode *Var =
        (*MI)->getOperand((*MI)->getNumOperands()-1).getMetadata();
      if (Var == DV && isDbgValueInDefinedReg(*MI) &&
          !PrevMI->isIdenticalTo(*MI))
        MultipleValues.push_back(*MI);
      PrevMI = *MI;
    }

    DbgScope *Scope = findDbgScope(MInsn);
    bool CurFnArg = false;
    if (DV.getTag() == dwarf::DW_TAG_arg_variable &&
        DISubprogram(DV.getContext()).describes(MF->getFunction()))
      CurFnArg = true;
    if (!Scope && CurFnArg)
      Scope = CurrentFnDbgScope;
    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    Processed.insert(DV);
    DbgVariable *RegVar = new DbgVariable(DV);
    Scope->addVariable(RegVar);
    if (!CurFnArg)
      DbgVariableLabelsMap[RegVar] = getLabelBeforeInsn(MInsn);
    if (DbgVariable *AbsVar = findAbstractVariable(DV, MInsn->getDebugLoc())) {
      DbgVariableToDbgInstMap[AbsVar] = MInsn;
      VarToAbstractVarMap[RegVar] = AbsVar;
    }
    if (MultipleValues.size() <= 1) {
      DbgVariableToDbgInstMap[RegVar] = MInsn;
      continue;
    }

    // handle multiple DBG_VALUE instructions describing one variable.
    if (DotDebugLocEntries.empty())
      RegVar->setDotDebugLocOffset(0);
    else
      RegVar->setDotDebugLocOffset(DotDebugLocEntries.size());
    const MachineInstr *Begin = NULL;
    const MachineInstr *End = NULL;
    for (SmallVector<const MachineInstr *, 4>::iterator
           MVI = MultipleValues.begin(), MVE = MultipleValues.end();
         MVI != MVE; ++MVI) {
      if (!Begin) {
        Begin = *MVI;
        continue;
      }
      End = *MVI;
      MachineLocation MLoc;
      if (Begin->getNumOperands() == 3) {
        if (Begin->getOperand(0).isReg() && Begin->getOperand(1).isImm())
          MLoc.set(Begin->getOperand(0).getReg(), Begin->getOperand(1).getImm());
      } else
        MLoc = Asm->getDebugValueLocation(Begin);

      const MCSymbol *FLabel = getLabelBeforeInsn(Begin);
      const MCSymbol *SLabel = getLabelBeforeInsn(End);
      if (MLoc.getReg())
        DotDebugLocEntries.push_back(DotDebugLocEntry(FLabel, SLabel, MLoc));

      Begin = End;
      if (MVI + 1 == MVE) {
        // If End is the last instruction then its value is valid
        // until the end of the funtion.
        MachineLocation EMLoc;
        if (End->getNumOperands() == 3) {
          if (End->getOperand(0).isReg() && Begin->getOperand(1).isImm())
          EMLoc.set(Begin->getOperand(0).getReg(), Begin->getOperand(1).getImm());
        } else
          EMLoc = Asm->getDebugValueLocation(End);
        if (EMLoc.getReg()) 
          DotDebugLocEntries.
            push_back(DotDebugLocEntry(SLabel, FunctionEndSym, EMLoc));
      }
    }
    DotDebugLocEntries.push_back(DotDebugLocEntry());
  }

  // Collect info for variables that were optimized out.
  const Function *F = MF->getFunction();
  const Module *M = F->getParent();
  if (NamedMDNode *NMD =
      M->getNamedMetadata(Twine("llvm.dbg.lv.",
                                getRealLinkageName(F->getName())))) {
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
      DIVariable DV(cast<MDNode>(NMD->getOperand(i)));
      if (!DV || !Processed.insert(DV))
        continue;
      DbgScope *Scope = DbgScopeMap.lookup(DV.getContext());
      if (Scope)
        Scope->addVariable(new DbgVariable(DV));
    }
  }
}

/// getLabelBeforeInsn - Return Label preceding the instruction.
const MCSymbol *DwarfDebug::getLabelBeforeInsn(const MachineInstr *MI) {
  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
    LabelsBeforeInsn.find(MI);
  if (I == LabelsBeforeInsn.end())
    // FunctionBeginSym always preceeds all the instruction in current function.
    return FunctionBeginSym;
  return I->second;
}

/// getLabelAfterInsn - Return Label immediately following the instruction.
const MCSymbol *DwarfDebug::getLabelAfterInsn(const MachineInstr *MI) {
  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
    LabelsAfterInsn.find(MI);
  if (I == LabelsAfterInsn.end())
    return NULL;
  return I->second;
}

/// beginInstruction - Process beginning of an instruction.
void DwarfDebug::beginInstruction(const MachineInstr *MI) {
  if (InsnNeedsLabel.count(MI) == 0) {
    LabelsBeforeInsn[MI] = PrevLabel;
    return;
  }

  // Check location.
  DebugLoc DL = MI->getDebugLoc();
  if (!DL.isUnknown()) {
    const MDNode *Scope = DL.getScope(Asm->MF->getFunction()->getContext());
    PrevLabel = recordSourceLine(DL.getLine(), DL.getCol(), Scope);
    PrevInstLoc = DL;
    LabelsBeforeInsn[MI] = PrevLabel;
    return;
  }

  // If location is unknown then use temp label for this DBG_VALUE
  // instruction.
  if (MI->isDebugValue()) {
    PrevLabel = MMI->getContext().CreateTempSymbol();
    Asm->OutStreamer.EmitLabel(PrevLabel);
    LabelsBeforeInsn[MI] = PrevLabel;
    return;
  }

  if (UnknownLocations) {
    PrevLabel = recordSourceLine(0, 0, 0);
    LabelsBeforeInsn[MI] = PrevLabel;
    return;
  }

  assert (0 && "Instruction is not processed!");
}

/// endInstruction - Process end of an instruction.
void DwarfDebug::endInstruction(const MachineInstr *MI) {
  if (InsnsEndScopeSet.count(MI) != 0) {
    // Emit a label if this instruction ends a scope.
    MCSymbol *Label = MMI->getContext().CreateTempSymbol();
    Asm->OutStreamer.EmitLabel(Label);
    LabelsAfterInsn[MI] = Label;
  }
}

/// getOrCreateDbgScope - Create DbgScope for the scope.
DbgScope *DwarfDebug::getOrCreateDbgScope(const MDNode *Scope,
                                          const MDNode *InlinedAt) {
  if (!InlinedAt) {
    DbgScope *WScope = DbgScopeMap.lookup(Scope);
    if (WScope)
      return WScope;
    WScope = new DbgScope(NULL, DIDescriptor(Scope), NULL);
    DbgScopeMap.insert(std::make_pair(Scope, WScope));
    if (DIDescriptor(Scope).isLexicalBlock()) {
      DbgScope *Parent =
        getOrCreateDbgScope(DILexicalBlock(Scope).getContext(), NULL);
      WScope->setParent(Parent);
      Parent->addScope(WScope);
    }

    if (!WScope->getParent()) {
      StringRef SPName = DISubprogram(Scope).getLinkageName();
      // We used to check only for a linkage name, but that fails
      // since we began omitting the linkage name for private
      // functions.  The new way is to check for the name in metadata,
      // but that's not supported in old .ll test cases.  Ergo, we
      // check both.
      if (SPName == Asm->MF->getFunction()->getName() ||
          DISubprogram(Scope).getFunction() == Asm->MF->getFunction())
        CurrentFnDbgScope = WScope;
    }

    return WScope;
  }

  getOrCreateAbstractScope(Scope);
  DbgScope *WScope = DbgScopeMap.lookup(InlinedAt);
  if (WScope)
    return WScope;

  WScope = new DbgScope(NULL, DIDescriptor(Scope), InlinedAt);
  DbgScopeMap.insert(std::make_pair(InlinedAt, WScope));
  DILocation DL(InlinedAt);
  DbgScope *Parent =
    getOrCreateDbgScope(DL.getScope(), DL.getOrigLocation());
  WScope->setParent(Parent);
  Parent->addScope(WScope);

  ConcreteScopes[InlinedAt] = WScope;

  return WScope;
}

/// hasValidLocation - Return true if debug location entry attached with
/// machine instruction encodes valid location info.
static bool hasValidLocation(LLVMContext &Ctx,
                             const MachineInstr *MInsn,
                             const MDNode *&Scope, const MDNode *&InlinedAt) {
  DebugLoc DL = MInsn->getDebugLoc();
  if (DL.isUnknown()) return false;

  const MDNode *S = DL.getScope(Ctx);

  // There is no need to create another DIE for compile unit. For all
  // other scopes, create one DbgScope now. This will be translated
  // into a scope DIE at the end.
  if (DIScope(S).isCompileUnit()) return false;

  Scope = S;
  InlinedAt = DL.getInlinedAt(Ctx);
  return true;
}

/// calculateDominanceGraph - Calculate dominance graph for DbgScope
/// hierarchy.
static void calculateDominanceGraph(DbgScope *Scope) {
  assert (Scope && "Unable to calculate scop edominance graph!");
  SmallVector<DbgScope *, 4> WorkStack;
  WorkStack.push_back(Scope);
  unsigned Counter = 0;
  while (!WorkStack.empty()) {
    DbgScope *WS = WorkStack.back();
    const SmallVector<DbgScope *, 4> &Children = WS->getScopes();
    bool visitedChildren = false;
    for (SmallVector<DbgScope *, 4>::const_iterator SI = Children.begin(),
           SE = Children.end(); SI != SE; ++SI) {
      DbgScope *ChildScope = *SI;
      if (!ChildScope->getDFSOut()) {
        WorkStack.push_back(ChildScope);
        visitedChildren = true;
        ChildScope->setDFSIn(++Counter);
        break;
      }
    }
    if (!visitedChildren) {
      WorkStack.pop_back();
      WS->setDFSOut(++Counter);
    }
  }
}

/// printDbgScopeInfo - Print DbgScope info for each machine instruction.
static
void printDbgScopeInfo(LLVMContext &Ctx, const MachineFunction *MF,
                       DenseMap<const MachineInstr *, DbgScope *> &MI2ScopeMap)
{
#ifndef NDEBUG
  unsigned PrevDFSIn = 0;
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;
      const MDNode *Scope = NULL;
      const MDNode *InlinedAt = NULL;

      // Check if instruction has valid location information.
      if (hasValidLocation(Ctx, MInsn, Scope, InlinedAt)) {
        dbgs() << " [ ";
        if (InlinedAt)
          dbgs() << "*";
        DenseMap<const MachineInstr *, DbgScope *>::iterator DI =
          MI2ScopeMap.find(MInsn);
        if (DI != MI2ScopeMap.end()) {
          DbgScope *S = DI->second;
          dbgs() << S->getDFSIn();
          PrevDFSIn = S->getDFSIn();
        } else
          dbgs() << PrevDFSIn;
      } else
        dbgs() << " [ x" << PrevDFSIn;
      dbgs() << " ]";
      MInsn->dump();
    }
    dbgs() << "\n";
  }
#endif
}
/// extractScopeInformation - Scan machine instructions in this function
/// and collect DbgScopes. Return true, if at least one scope was found.
bool DwarfDebug::extractScopeInformation() {
  // If scope information was extracted using .dbg intrinsics then there is not
  // any need to extract these information by scanning each instruction.
  if (!DbgScopeMap.empty())
    return false;

  // Scan each instruction and create scopes. First build working set of scopes.
  LLVMContext &Ctx = Asm->MF->getFunction()->getContext();
  SmallVector<DbgRange, 4> MIRanges;
  DenseMap<const MachineInstr *, DbgScope *> MI2ScopeMap;
  const MDNode *PrevScope = NULL;
  const MDNode *PrevInlinedAt = NULL;
  const MachineInstr *RangeBeginMI = NULL;
  const MachineInstr *PrevMI = NULL;
  for (MachineFunction::const_iterator I = Asm->MF->begin(), E = Asm->MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;
      const MDNode *Scope = NULL;
      const MDNode *InlinedAt = NULL;

      // Check if instruction has valid location information.
      if (!hasValidLocation(Ctx, MInsn, Scope, InlinedAt)) {
        PrevMI = MInsn;
        continue;
      }

      // If scope has not changed then skip this instruction.
      if (Scope == PrevScope && PrevInlinedAt == InlinedAt) {
        PrevMI = MInsn;
        continue;
      }

      if (RangeBeginMI) {
        // If we have alread seen a beginning of a instruction range and
        // current instruction scope does not match scope of first instruction
        // in this range then create a new instruction range.
        DbgRange R(RangeBeginMI, PrevMI);
        MI2ScopeMap[RangeBeginMI] = getOrCreateDbgScope(PrevScope,
                                                        PrevInlinedAt);
        MIRanges.push_back(R);
      }

      // This is a beginning of a new instruction range.
      RangeBeginMI = MInsn;

      // Reset previous markers.
      PrevMI = MInsn;
      PrevScope = Scope;
      PrevInlinedAt = InlinedAt;
    }
  }

  // Create last instruction range.
  if (RangeBeginMI && PrevMI && PrevScope) {
    DbgRange R(RangeBeginMI, PrevMI);
    MIRanges.push_back(R);
    MI2ScopeMap[RangeBeginMI] = getOrCreateDbgScope(PrevScope, PrevInlinedAt);
  }

  if (!CurrentFnDbgScope)
    return false;

  calculateDominanceGraph(CurrentFnDbgScope);
  if (PrintDbgScope)
    printDbgScopeInfo(Ctx, Asm->MF, MI2ScopeMap);

  // Find ranges of instructions covered by each DbgScope;
  DbgScope *PrevDbgScope = NULL;
  for (SmallVector<DbgRange, 4>::const_iterator RI = MIRanges.begin(),
         RE = MIRanges.end(); RI != RE; ++RI) {
    const DbgRange &R = *RI;
    DbgScope *S = MI2ScopeMap.lookup(R.first);
    assert (S && "Lost DbgScope for a machine instruction!");
    if (PrevDbgScope && !PrevDbgScope->dominates(S))
      PrevDbgScope->closeInsnRange(S);
    S->openInsnRange(R.first);
    S->extendInsnRange(R.second);
    PrevDbgScope = S;
  }

  if (PrevDbgScope)
    PrevDbgScope->closeInsnRange();

  identifyScopeMarkers();

  return !DbgScopeMap.empty();
}

/// identifyScopeMarkers() -
/// Each DbgScope has first instruction and last instruction to mark beginning
/// and end of a scope respectively. Create an inverse map that list scopes
/// starts (and ends) with an instruction. One instruction may start (or end)
/// multiple scopes. Ignore scopes that are not reachable.
void DwarfDebug::identifyScopeMarkers() {
  SmallVector<DbgScope *, 4> WorkList;
  WorkList.push_back(CurrentFnDbgScope);
  while (!WorkList.empty()) {
    DbgScope *S = WorkList.pop_back_val();

    const SmallVector<DbgScope *, 4> &Children = S->getScopes();
    if (!Children.empty())
      for (SmallVector<DbgScope *, 4>::const_iterator SI = Children.begin(),
             SE = Children.end(); SI != SE; ++SI)
        WorkList.push_back(*SI);

    if (S->isAbstractScope())
      continue;

    const SmallVector<DbgRange, 4> &Ranges = S->getRanges();
    if (Ranges.empty())
      continue;
    for (SmallVector<DbgRange, 4>::const_iterator RI = Ranges.begin(),
           RE = Ranges.end(); RI != RE; ++RI) {
      assert(RI->first && "DbgRange does not have first instruction!");
      assert(RI->second && "DbgRange does not have second instruction!");
      InsnsEndScopeSet.insert(RI->second);
    }
  }
}

/// FindFirstDebugLoc - Find the first debug location in the function. This
/// is intended to be an approximation for the source position of the
/// beginning of the function.
static DebugLoc FindFirstDebugLoc(const MachineFunction *MF) {
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I)
    for (MachineBasicBlock::const_iterator MBBI = I->begin(), MBBE = I->end();
         MBBI != MBBE; ++MBBI) {
      DebugLoc DL = MBBI->getDebugLoc();
      if (!DL.isUnknown())
        return DL;
    }
  return DebugLoc();
}

#ifndef NDEBUG
/// CheckLineNumbers - Count basicblocks whose instructions do not have any
/// line number information.
static void CheckLineNumbers(const MachineFunction *MF) {
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    bool FoundLineNo = false;
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MI = II;
      if (!MI->getDebugLoc().isUnknown()) {
        FoundLineNo = true;
        break;
      }
    }
    if (!FoundLineNo && I->size())
      ++BlocksWithoutLineNo;      
  }
}
#endif

/// beginFunction - Gather pre-function debug information.  Assumes being
/// emitted immediately after the function entry point.
void DwarfDebug::beginFunction(const MachineFunction *MF) {
  if (!MMI->hasDebugInfo()) return;
  if (!extractScopeInformation()) return;

#ifndef NDEBUG
  CheckLineNumbers(MF);
#endif

  FunctionBeginSym = Asm->GetTempSymbol("func_begin",
                                        Asm->getFunctionNumber());
  // Assumes in correct section after the entry point.
  Asm->OutStreamer.EmitLabel(FunctionBeginSym);

  // Emit label for the implicitly defined dbg.stoppoint at the start of the
  // function.
  DebugLoc FDL = FindFirstDebugLoc(MF);
  if (FDL.isUnknown()) return;

  const MDNode *Scope = FDL.getScope(MF->getFunction()->getContext());
  const MDNode *TheScope = 0;

  DISubprogram SP = getDISubprogram(Scope);
  unsigned Line, Col;
  if (SP.Verify()) {
    Line = SP.getLineNumber();
    Col = 0;
    TheScope = SP;
  } else {
    Line = FDL.getLine();
    Col = FDL.getCol();
    TheScope = Scope;
  }

  recordSourceLine(Line, Col, TheScope);

  /// ProcessedArgs - Collection of arguments already processed.
  SmallPtrSet<const MDNode *, 8> ProcessedArgs;

  DebugLoc PrevLoc;
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I)
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MI = II;
      DebugLoc DL = MI->getDebugLoc();
      if (MI->isDebugValue()) {
        assert (MI->getNumOperands() > 1 && "Invalid machine instruction!");
        DIVariable DV(MI->getOperand(MI->getNumOperands() - 1).getMetadata());
        if (!DV.Verify()) continue;
        // If DBG_VALUE is for a local variable then it needs a label.
        if (DV.getTag() != dwarf::DW_TAG_arg_variable
            && isDbgValueInUndefinedReg(MI) == false)
          InsnNeedsLabel.insert(MI);
        // DBG_VALUE for inlined functions argument needs a label.
        else if (!DISubprogram(getDISubprogram(DV.getContext())).
                 describes(MF->getFunction()))
          InsnNeedsLabel.insert(MI);
        // DBG_VALUE indicating argument location change needs a label.
        else if (isDbgValueInUndefinedReg(MI) == false
                 && !ProcessedArgs.insert(DV))
          InsnNeedsLabel.insert(MI);
      } else {
        // If location is unknown then instruction needs a location only if
        // UnknownLocations flag is set.
        if (DL.isUnknown()) {
          if (UnknownLocations && !PrevLoc.isUnknown())
            InsnNeedsLabel.insert(MI);
        } else if (DL != PrevLoc)
          // Otherwise, instruction needs a location only if it is new location.
          InsnNeedsLabel.insert(MI);
      }

      if (!DL.isUnknown() || UnknownLocations)
        PrevLoc = DL;
    }

  PrevLabel = FunctionBeginSym;
}

/// endFunction - Gather and emit post-function debug information.
///
void DwarfDebug::endFunction(const MachineFunction *MF) {
  if (!MMI->hasDebugInfo() || DbgScopeMap.empty()) return;

  if (CurrentFnDbgScope) {

    // Define end label for subprogram.
    FunctionEndSym = Asm->GetTempSymbol("func_end",
                                        Asm->getFunctionNumber());
    // Assumes in correct section after the entry point.
    Asm->OutStreamer.EmitLabel(FunctionEndSym);

    SmallPtrSet<const MDNode *, 16> ProcessedVars;
    collectVariableInfo(MF, ProcessedVars);

    // Get function line info.
    if (!Lines.empty()) {
      // Get section line info.
      unsigned ID = SectionMap.insert(Asm->getCurrentSection());
      if (SectionSourceLines.size() < ID) SectionSourceLines.resize(ID);
      std::vector<SrcLineInfo> &SectionLineInfos = SectionSourceLines[ID-1];
      // Append the function info to section info.
      SectionLineInfos.insert(SectionLineInfos.end(),
                              Lines.begin(), Lines.end());
    }

    // Construct abstract scopes.
    for (SmallVector<DbgScope *, 4>::iterator AI = AbstractScopesList.begin(),
           AE = AbstractScopesList.end(); AI != AE; ++AI) {
      DISubprogram SP((*AI)->getScopeNode());
      if (SP.Verify()) {
        // Collect info for variables that were optimized out.
        StringRef FName = SP.getLinkageName();
        if (FName.empty())
          FName = SP.getName();
        const Module *M = MF->getFunction()->getParent();
        if (NamedMDNode *NMD =
            M->getNamedMetadata(Twine("llvm.dbg.lv.",
                                      getRealLinkageName(FName)))) {
          for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
          DIVariable DV(cast<MDNode>(NMD->getOperand(i)));
          if (!DV || !ProcessedVars.insert(DV))
            continue;
          DbgScope *Scope = AbstractScopes.lookup(DV.getContext());
          if (Scope)
            Scope->addVariable(new DbgVariable(DV));
          }
        }
      }
      if (ProcessedSPNodes.count((*AI)->getScopeNode()) == 0)
        constructScopeDIE(*AI);
    }

    DIE *CurFnDIE = constructScopeDIE(CurrentFnDbgScope);

    if (!DisableFramePointerElim(*MF))
      addUInt(CurFnDIE, dwarf::DW_AT_APPLE_omit_frame_ptr,
              dwarf::DW_FORM_flag, 1);


    DebugFrames.push_back(FunctionDebugFrameInfo(Asm->getFunctionNumber(),
                                                 MMI->getFrameMoves()));
  }

  // Clear debug info
  CurrentFnDbgScope = NULL;
  InsnNeedsLabel.clear();
  DbgVariableToFrameIndexMap.clear();
  VarToAbstractVarMap.clear();
  DbgVariableToDbgInstMap.clear();
  DbgVariableLabelsMap.clear();
  DeleteContainerSeconds(DbgScopeMap);
  InsnsEndScopeSet.clear();
  ConcreteScopes.clear();
  DeleteContainerSeconds(AbstractScopes);
  AbstractScopesList.clear();
  AbstractVariables.clear();
  LabelsBeforeInsn.clear();
  LabelsAfterInsn.clear();
  Lines.clear();
  PrevLabel = NULL;
}

/// recordVariableFrameIndex - Record a variable's index.
void DwarfDebug::recordVariableFrameIndex(const DbgVariable *V, int Index) {
  assert (V && "Invalid DbgVariable!");
  DbgVariableToFrameIndexMap[V] = Index;
}

/// findVariableFrameIndex - Return true if frame index for the variable
/// is found. Update FI to hold value of the index.
bool DwarfDebug::findVariableFrameIndex(const DbgVariable *V, int *FI) {
  assert (V && "Invalid DbgVariable!");
  DenseMap<const DbgVariable *, int>::iterator I =
    DbgVariableToFrameIndexMap.find(V);
  if (I == DbgVariableToFrameIndexMap.end())
    return false;
  *FI = I->second;
  return true;
}

/// findVariableLabel - Find MCSymbol for the variable.
const MCSymbol *DwarfDebug::findVariableLabel(const DbgVariable *V) {
  DenseMap<const DbgVariable *, const MCSymbol *>::iterator I
    = DbgVariableLabelsMap.find(V);
  if (I == DbgVariableLabelsMap.end())
    return NULL;
  else return I->second;
}

/// findDbgScope - Find DbgScope for the debug loc attached with an
/// instruction.
DbgScope *DwarfDebug::findDbgScope(const MachineInstr *MInsn) {
  DbgScope *Scope = NULL;
  LLVMContext &Ctx =
    MInsn->getParent()->getParent()->getFunction()->getContext();
  DebugLoc DL = MInsn->getDebugLoc();

  if (DL.isUnknown())
    return Scope;

  if (const MDNode *IA = DL.getInlinedAt(Ctx))
    Scope = ConcreteScopes.lookup(IA);
  if (Scope == 0)
    Scope = DbgScopeMap.lookup(DL.getScope(Ctx));

  return Scope;
}


/// recordSourceLine - Register a source line with debug info. Returns the
/// unique label that was emitted and which provides correspondence to
/// the source line list.
MCSymbol *DwarfDebug::recordSourceLine(unsigned Line, unsigned Col,
                                       const MDNode *S) {
  StringRef Dir;
  StringRef Fn;

  unsigned Src = 1;
  if (S) {
    DIDescriptor Scope(S);

    if (Scope.isCompileUnit()) {
      DICompileUnit CU(S);
      Dir = CU.getDirectory();
      Fn = CU.getFilename();
    } else if (Scope.isFile()) {
      DIFile F(S);
      Dir = F.getDirectory();
      Fn = F.getFilename();
    } else if (Scope.isSubprogram()) {
      DISubprogram SP(S);
      Dir = SP.getDirectory();
      Fn = SP.getFilename();
    } else if (Scope.isLexicalBlock()) {
      DILexicalBlock DB(S);
      Dir = DB.getDirectory();
      Fn = DB.getFilename();
    } else
      assert(0 && "Unexpected scope info");

    Src = GetOrCreateSourceID(Dir, Fn);
  }

  MCSymbol *Label = MMI->getContext().CreateTempSymbol();
  Lines.push_back(SrcLineInfo(Line, Col, Src, Label));

  Asm->OutStreamer.EmitLabel(Label);
  return Label;
}

//===----------------------------------------------------------------------===//
// Emit Methods
//===----------------------------------------------------------------------===//

/// computeSizeAndOffset - Compute the size and offset of a DIE.
///
unsigned
DwarfDebug::computeSizeAndOffset(DIE *Die, unsigned Offset, bool Last) {
  // Get the children.
  const std::vector<DIE *> &Children = Die->getChildren();

  // If not last sibling and has children then add sibling offset attribute.
  if (!Last && !Children.empty())
    Die->addSiblingOffset(DIEValueAllocator);

  // Record the abbreviation.
  assignAbbrevNumber(Die->getAbbrev());

  // Get the abbreviation for this DIE.
  unsigned AbbrevNumber = Die->getAbbrevNumber();
  const DIEAbbrev *Abbrev = Abbreviations[AbbrevNumber - 1];

  // Set DIE offset
  Die->setOffset(Offset);

  // Start the size with the size of abbreviation code.
  Offset += MCAsmInfo::getULEB128Size(AbbrevNumber);

  const SmallVector<DIEValue*, 32> &Values = Die->getValues();
  const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev->getData();

  // Size the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    // Size attribute value.
    Offset += Values[i]->SizeOf(Asm, AbbrevData[i].getForm());

  // Size the DIE children if any.
  if (!Children.empty()) {
    assert(Abbrev->getChildrenFlag() == dwarf::DW_CHILDREN_yes &&
           "Children flag not set");

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      Offset = computeSizeAndOffset(Children[j], Offset, (j + 1) == M);

    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die->setSize(Offset - Die->getOffset());
  return Offset;
}

/// computeSizeAndOffsets - Compute the size and offset of all the DIEs.
///
void DwarfDebug::computeSizeAndOffsets() {
  unsigned PrevOffset = 0;
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    // Compute size of compile unit header.
    static unsigned Offset = PrevOffset +
      sizeof(int32_t) + // Length of Compilation Unit Info
      sizeof(int16_t) + // DWARF version number
      sizeof(int32_t) + // Offset Into Abbrev. Section
      sizeof(int8_t);   // Pointer Size (in bytes)
    computeSizeAndOffset(I->second->getCUDie(), Offset, true);
    PrevOffset = Offset;
  }
}

/// EmitSectionSym - Switch to the specified MCSection and emit an assembler
/// temporary label to it if SymbolStem is specified.
static MCSymbol *EmitSectionSym(AsmPrinter *Asm, const MCSection *Section,
                                const char *SymbolStem = 0) {
  Asm->OutStreamer.SwitchSection(Section);
  if (!SymbolStem) return 0;

  MCSymbol *TmpSym = Asm->GetTempSymbol(SymbolStem);
  Asm->OutStreamer.EmitLabel(TmpSym);
  return TmpSym;
}

/// EmitSectionLabels - Emit initial Dwarf sections with a label at
/// the start of each one.
void DwarfDebug::EmitSectionLabels() {
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  // Dwarf sections base addresses.
  if (Asm->MAI->doesDwarfRequireFrameSection()) {
    DwarfFrameSectionSym =
      EmitSectionSym(Asm, TLOF.getDwarfFrameSection(), "section_debug_frame");
   }

  DwarfInfoSectionSym =
    EmitSectionSym(Asm, TLOF.getDwarfInfoSection(), "section_info");
  DwarfAbbrevSectionSym =
    EmitSectionSym(Asm, TLOF.getDwarfAbbrevSection(), "section_abbrev");
  EmitSectionSym(Asm, TLOF.getDwarfARangesSection());

  if (const MCSection *MacroInfo = TLOF.getDwarfMacroInfoSection())
    EmitSectionSym(Asm, MacroInfo);

  EmitSectionSym(Asm, TLOF.getDwarfLineSection(), "section_line");
  EmitSectionSym(Asm, TLOF.getDwarfLocSection());
  EmitSectionSym(Asm, TLOF.getDwarfPubNamesSection());
  EmitSectionSym(Asm, TLOF.getDwarfPubTypesSection());
  DwarfStrSectionSym =
    EmitSectionSym(Asm, TLOF.getDwarfStrSection(), "section_str");
  DwarfDebugRangeSectionSym = EmitSectionSym(Asm, TLOF.getDwarfRangesSection(),
                                             "debug_range");

  DwarfDebugLocSectionSym = EmitSectionSym(Asm, TLOF.getDwarfLocSection(),
                                           "section_debug_loc");

  TextSectionSym = EmitSectionSym(Asm, TLOF.getTextSection(), "text_begin");
  EmitSectionSym(Asm, TLOF.getDataSection());
}

/// emitDIE - Recusively Emits a debug information entry.
///
void DwarfDebug::emitDIE(DIE *Die) {
  // Get the abbreviation for this DIE.
  unsigned AbbrevNumber = Die->getAbbrevNumber();
  const DIEAbbrev *Abbrev = Abbreviations[AbbrevNumber - 1];

  // Emit the code (index) for the abbreviation.
  if (Asm->isVerbose())
    Asm->OutStreamer.AddComment("Abbrev [" + Twine(AbbrevNumber) + "] 0x" +
                                Twine::utohexstr(Die->getOffset()) + ":0x" +
                                Twine::utohexstr(Die->getSize()) + " " +
                                dwarf::TagString(Abbrev->getTag()));
  Asm->EmitULEB128(AbbrevNumber);

  const SmallVector<DIEValue*, 32> &Values = Die->getValues();
  const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev->getData();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    unsigned Attr = AbbrevData[i].getAttribute();
    unsigned Form = AbbrevData[i].getForm();
    assert(Form && "Too many attributes for DIE (check abbreviation)");

    if (Asm->isVerbose())
      Asm->OutStreamer.AddComment(dwarf::AttributeString(Attr));

    switch (Attr) {
    case dwarf::DW_AT_sibling:
      Asm->EmitInt32(Die->getSiblingOffset());
      break;
    case dwarf::DW_AT_abstract_origin: {
      DIEEntry *E = cast<DIEEntry>(Values[i]);
      DIE *Origin = E->getEntry();
      unsigned Addr = Origin->getOffset();
      Asm->EmitInt32(Addr);
      break;
    }
    case dwarf::DW_AT_ranges: {
      // DW_AT_range Value encodes offset in debug_range section.
      DIEInteger *V = cast<DIEInteger>(Values[i]);

      if (Asm->MAI->doesDwarfUsesLabelOffsetForRanges()) {
        Asm->EmitLabelPlusOffset(DwarfDebugRangeSectionSym,
                                 V->getValue(),
                                 4);
      } else {
        Asm->EmitLabelOffsetDifference(DwarfDebugRangeSectionSym,
                                       V->getValue(),
                                       DwarfDebugRangeSectionSym,
                                       4);
      }
      break;
    }
    case dwarf::DW_AT_location: {
      if (UseDotDebugLocEntry.count(Die) != 0) {
        DIELabel *L = cast<DIELabel>(Values[i]);
        Asm->EmitLabelDifference(L->getValue(), DwarfDebugLocSectionSym, 4);
      } else
        Values[i]->EmitValue(Asm, Form);
      break;
    }
    case dwarf::DW_AT_accessibility: {
      if (Asm->isVerbose()) {
        DIEInteger *V = cast<DIEInteger>(Values[i]);
        Asm->OutStreamer.AddComment(dwarf::AccessibilityString(V->getValue()));
      }
      Values[i]->EmitValue(Asm, Form);
      break;
    }
    default:
      // Emit an attribute using the defined form.
      Values[i]->EmitValue(Asm, Form);
      break;
    }
  }

  // Emit the DIE children if any.
  if (Abbrev->getChildrenFlag() == dwarf::DW_CHILDREN_yes) {
    const std::vector<DIE *> &Children = Die->getChildren();

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      emitDIE(Children[j]);

    if (Asm->isVerbose())
      Asm->OutStreamer.AddComment("End Of Children Mark");
    Asm->EmitInt8(0);
  }
}

/// emitDebugInfo - Emit the debug info section.
///
void DwarfDebug::emitDebugInfo() {
  // Start debug info section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfInfoSection());
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    DIE *Die = TheCU->getCUDie();

    // Emit the compile units header.
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("info_begin",
                                                  TheCU->getID()));

    // Emit size of content not including length itself
    unsigned ContentSize = Die->getSize() +
      sizeof(int16_t) + // DWARF version number
      sizeof(int32_t) + // Offset Into Abbrev. Section
      sizeof(int8_t) +  // Pointer Size (in bytes)
      sizeof(int32_t);  // FIXME - extra pad for gdb bug.

    Asm->OutStreamer.AddComment("Length of Compilation Unit Info");
    Asm->EmitInt32(ContentSize);
    Asm->OutStreamer.AddComment("DWARF version number");
    Asm->EmitInt16(dwarf::DWARF_VERSION);
    Asm->OutStreamer.AddComment("Offset Into Abbrev. Section");
    Asm->EmitSectionOffset(Asm->GetTempSymbol("abbrev_begin"),
                           DwarfAbbrevSectionSym);
    Asm->OutStreamer.AddComment("Address Size (in bytes)");
    Asm->EmitInt8(Asm->getTargetData().getPointerSize());

    emitDIE(Die);
    // FIXME - extra padding for gdb bug.
    Asm->OutStreamer.AddComment("4 extra padding bytes for GDB");
    Asm->EmitInt8(0);
    Asm->EmitInt8(0);
    Asm->EmitInt8(0);
    Asm->EmitInt8(0);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("info_end", TheCU->getID()));
  }
}

/// emitAbbreviations - Emit the abbreviation section.
///
void DwarfDebug::emitAbbreviations() const {
  // Check to see if it is worth the effort.
  if (!Abbreviations.empty()) {
    // Start the debug abbrev section.
    Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfAbbrevSection());

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("abbrev_begin"));

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

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("abbrev_end"));
  }
}

/// emitEndOfLineMatrix - Emit the last address of the section and the end of
/// the line matrix.
///
void DwarfDebug::emitEndOfLineMatrix(unsigned SectionEnd) {
  // Define last address of section.
  Asm->OutStreamer.AddComment("Extended Op");
  Asm->EmitInt8(0);

  Asm->OutStreamer.AddComment("Op size");
  Asm->EmitInt8(Asm->getTargetData().getPointerSize() + 1);
  Asm->OutStreamer.AddComment("DW_LNE_set_address");
  Asm->EmitInt8(dwarf::DW_LNE_set_address);

  Asm->OutStreamer.AddComment("Section end label");

  Asm->OutStreamer.EmitSymbolValue(Asm->GetTempSymbol("section_end",SectionEnd),
                                   Asm->getTargetData().getPointerSize(),
                                   0/*AddrSpace*/);

  // Mark end of matrix.
  Asm->OutStreamer.AddComment("DW_LNE_end_sequence");
  Asm->EmitInt8(0);
  Asm->EmitInt8(1);
  Asm->EmitInt8(1);
}

/// emitDebugLines - Emit source line information.
///
void DwarfDebug::emitDebugLines() {
  // If the target is using .loc/.file, the assembler will be emitting the
  // .debug_line table automatically.
  if (Asm->MAI->hasDotLocAndDotFile())
    return;

  // Minimum line delta, thus ranging from -10..(255-10).
  const int MinLineDelta = -(dwarf::DW_LNS_fixed_advance_pc + 1);
  // Maximum line delta, thus ranging from -10..(255-10).
  const int MaxLineDelta = 255 + MinLineDelta;

  // Start the dwarf line section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfLineSection());

  // Construct the section header.
  Asm->OutStreamer.AddComment("Length of Source Line Info");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("line_end"),
                           Asm->GetTempSymbol("line_begin"), 4);
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("line_begin"));

  Asm->OutStreamer.AddComment("DWARF version number");
  Asm->EmitInt16(dwarf::DWARF_VERSION);

  Asm->OutStreamer.AddComment("Prolog Length");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("line_prolog_end"),
                           Asm->GetTempSymbol("line_prolog_begin"), 4);
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("line_prolog_begin"));

  Asm->OutStreamer.AddComment("Minimum Instruction Length");
  Asm->EmitInt8(1);
  Asm->OutStreamer.AddComment("Default is_stmt_start flag");
  Asm->EmitInt8(1);
  Asm->OutStreamer.AddComment("Line Base Value (Special Opcodes)");
  Asm->EmitInt8(MinLineDelta);
  Asm->OutStreamer.AddComment("Line Range Value (Special Opcodes)");
  Asm->EmitInt8(MaxLineDelta);
  Asm->OutStreamer.AddComment("Special Opcode Base");
  Asm->EmitInt8(-MinLineDelta);

  // Line number standard opcode encodings argument count
  Asm->OutStreamer.AddComment("DW_LNS_copy arg count");
  Asm->EmitInt8(0);
  Asm->OutStreamer.AddComment("DW_LNS_advance_pc arg count");
  Asm->EmitInt8(1);
  Asm->OutStreamer.AddComment("DW_LNS_advance_line arg count");
  Asm->EmitInt8(1);
  Asm->OutStreamer.AddComment("DW_LNS_set_file arg count");
  Asm->EmitInt8(1);
  Asm->OutStreamer.AddComment("DW_LNS_set_column arg count");
  Asm->EmitInt8(1);
  Asm->OutStreamer.AddComment("DW_LNS_negate_stmt arg count");
  Asm->EmitInt8(0);
  Asm->OutStreamer.AddComment("DW_LNS_set_basic_block arg count");
  Asm->EmitInt8(0);
  Asm->OutStreamer.AddComment("DW_LNS_const_add_pc arg count");
  Asm->EmitInt8(0);
  Asm->OutStreamer.AddComment("DW_LNS_fixed_advance_pc arg count");
  Asm->EmitInt8(1);

  // Emit directories.
  for (unsigned DI = 1, DE = getNumSourceDirectories()+1; DI != DE; ++DI) {
    const std::string &Dir = getSourceDirectoryName(DI);
    if (Asm->isVerbose()) Asm->OutStreamer.AddComment("Directory");
    Asm->OutStreamer.EmitBytes(StringRef(Dir.c_str(), Dir.size()+1), 0);
  }

  Asm->OutStreamer.AddComment("End of directories");
  Asm->EmitInt8(0);

  // Emit files.
  for (unsigned SI = 1, SE = getNumSourceIds()+1; SI != SE; ++SI) {
    // Remember source id starts at 1.
    std::pair<unsigned, unsigned> Id = getSourceDirectoryAndFileIds(SI);
    const std::string &FN = getSourceFileName(Id.second);
    if (Asm->isVerbose()) Asm->OutStreamer.AddComment("Source");
    Asm->OutStreamer.EmitBytes(StringRef(FN.c_str(), FN.size()+1), 0);

    Asm->EmitULEB128(Id.first, "Directory #");
    Asm->EmitULEB128(0, "Mod date");
    Asm->EmitULEB128(0, "File size");
  }

  Asm->OutStreamer.AddComment("End of files");
  Asm->EmitInt8(0);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("line_prolog_end"));

  // A sequence for each text section.
  unsigned SecSrcLinesSize = SectionSourceLines.size();

  for (unsigned j = 0; j < SecSrcLinesSize; ++j) {
    // Isolate current sections line info.
    const std::vector<SrcLineInfo> &LineInfos = SectionSourceLines[j];

    // Dwarf assumes we start with first line of first source file.
    unsigned Source = 1;
    unsigned Line = 1;

    // Construct rows of the address, source, line, column matrix.
    for (unsigned i = 0, N = LineInfos.size(); i < N; ++i) {
      const SrcLineInfo &LineInfo = LineInfos[i];
      MCSymbol *Label = LineInfo.getLabel();
      if (!Label->isDefined()) continue; // Not emitted, in dead code.

      if (Asm->isVerbose()) {
        std::pair<unsigned, unsigned> SrcID =
          getSourceDirectoryAndFileIds(LineInfo.getSourceID());
        Asm->OutStreamer.AddComment(Twine(getSourceDirectoryName(SrcID.first)) +
                                    "/" +
                                    Twine(getSourceFileName(SrcID.second)) +
                                    ":" + Twine(LineInfo.getLine()));
      }

      // Define the line address.
      Asm->OutStreamer.AddComment("Extended Op");
      Asm->EmitInt8(0);
      Asm->OutStreamer.AddComment("Op size");
      Asm->EmitInt8(Asm->getTargetData().getPointerSize() + 1);

      Asm->OutStreamer.AddComment("DW_LNE_set_address");
      Asm->EmitInt8(dwarf::DW_LNE_set_address);

      Asm->OutStreamer.AddComment("Location label");
      Asm->OutStreamer.EmitSymbolValue(Label,
                                       Asm->getTargetData().getPointerSize(),
                                       0/*AddrSpace*/);

      // If change of source, then switch to the new source.
      if (Source != LineInfo.getSourceID()) {
        Source = LineInfo.getSourceID();
        Asm->OutStreamer.AddComment("DW_LNS_set_file");
        Asm->EmitInt8(dwarf::DW_LNS_set_file);
        Asm->EmitULEB128(Source, "New Source");
      }

      // If change of line.
      if (Line != LineInfo.getLine()) {
        // Determine offset.
        int Offset = LineInfo.getLine() - Line;
        int Delta = Offset - MinLineDelta;

        // Update line.
        Line = LineInfo.getLine();

        // If delta is small enough and in range...
        if (Delta >= 0 && Delta < (MaxLineDelta - 1)) {
          // ... then use fast opcode.
          Asm->OutStreamer.AddComment("Line Delta");
          Asm->EmitInt8(Delta - MinLineDelta);
        } else {
          // ... otherwise use long hand.
          Asm->OutStreamer.AddComment("DW_LNS_advance_line");
          Asm->EmitInt8(dwarf::DW_LNS_advance_line);
          Asm->EmitSLEB128(Offset, "Line Offset");
          Asm->OutStreamer.AddComment("DW_LNS_copy");
          Asm->EmitInt8(dwarf::DW_LNS_copy);
        }
      } else {
        // Copy the previous row (different address or source)
        Asm->OutStreamer.AddComment("DW_LNS_copy");
        Asm->EmitInt8(dwarf::DW_LNS_copy);
      }
    }

    emitEndOfLineMatrix(j + 1);
  }

  if (SecSrcLinesSize == 0)
    // Because we're emitting a debug_line section, we still need a line
    // table. The linker and friends expect it to exist. If there's nothing to
    // put into it, emit an empty table.
    emitEndOfLineMatrix(1);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("line_end"));
}

/// emitCommonDebugFrame - Emit common frame info into a debug frame section.
///
void DwarfDebug::emitCommonDebugFrame() {
  if (!Asm->MAI->doesDwarfRequireFrameSection())
    return;

  int stackGrowth = Asm->getTargetData().getPointerSize();
  if (Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
      TargetFrameInfo::StackGrowsDown)
    stackGrowth *= -1;

  // Start the dwarf frame section.
  Asm->OutStreamer.SwitchSection(
                              Asm->getObjFileLowering().getDwarfFrameSection());

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_frame_common"));
  Asm->OutStreamer.AddComment("Length of Common Information Entry");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("debug_frame_common_end"),
                           Asm->GetTempSymbol("debug_frame_common_begin"), 4);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_frame_common_begin"));
  Asm->OutStreamer.AddComment("CIE Identifier Tag");
  Asm->EmitInt32((int)dwarf::DW_CIE_ID);
  Asm->OutStreamer.AddComment("CIE Version");
  Asm->EmitInt8(dwarf::DW_CIE_VERSION);
  Asm->OutStreamer.AddComment("CIE Augmentation");
  Asm->OutStreamer.EmitIntValue(0, 1, /*addrspace*/0); // nul terminator.
  Asm->EmitULEB128(1, "CIE Code Alignment Factor");
  Asm->EmitSLEB128(stackGrowth, "CIE Data Alignment Factor");
  Asm->OutStreamer.AddComment("CIE RA Column");
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), false));

  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);

  Asm->EmitFrameMoves(Moves, 0, false);

  Asm->EmitAlignment(2);
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_frame_common_end"));
}

/// emitFunctionDebugFrame - Emit per function frame info into a debug frame
/// section.
void DwarfDebug::
emitFunctionDebugFrame(const FunctionDebugFrameInfo &DebugFrameInfo) {
  if (!Asm->MAI->doesDwarfRequireFrameSection())
    return;

  // Start the dwarf frame section.
  Asm->OutStreamer.SwitchSection(
                              Asm->getObjFileLowering().getDwarfFrameSection());

  Asm->OutStreamer.AddComment("Length of Frame Information Entry");
  MCSymbol *DebugFrameBegin =
    Asm->GetTempSymbol("debug_frame_begin", DebugFrameInfo.Number);
  MCSymbol *DebugFrameEnd =
    Asm->GetTempSymbol("debug_frame_end", DebugFrameInfo.Number);
  Asm->EmitLabelDifference(DebugFrameEnd, DebugFrameBegin, 4);

  Asm->OutStreamer.EmitLabel(DebugFrameBegin);

  Asm->OutStreamer.AddComment("FDE CIE offset");
  Asm->EmitSectionOffset(Asm->GetTempSymbol("debug_frame_common"),
                         DwarfFrameSectionSym);

  Asm->OutStreamer.AddComment("FDE initial location");
  MCSymbol *FuncBeginSym =
    Asm->GetTempSymbol("func_begin", DebugFrameInfo.Number);
  Asm->OutStreamer.EmitSymbolValue(FuncBeginSym,
                                   Asm->getTargetData().getPointerSize(),
                                   0/*AddrSpace*/);


  Asm->OutStreamer.AddComment("FDE address range");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("func_end",DebugFrameInfo.Number),
                           FuncBeginSym, Asm->getTargetData().getPointerSize());

  Asm->EmitFrameMoves(DebugFrameInfo.Moves, FuncBeginSym, false);

  Asm->EmitAlignment(2);
  Asm->OutStreamer.EmitLabel(DebugFrameEnd);
}

/// emitDebugPubNames - Emit visible names into a debug pubnames section.
///
void DwarfDebug::emitDebugPubNames() {
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    // Start the dwarf pubnames section.
    Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfPubNamesSection());

    Asm->OutStreamer.AddComment("Length of Public Names Info");
    Asm->EmitLabelDifference(
      Asm->GetTempSymbol("pubnames_end", TheCU->getID()),
      Asm->GetTempSymbol("pubnames_begin", TheCU->getID()), 4);

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubnames_begin",
                                                  TheCU->getID()));

    Asm->OutStreamer.AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DWARF_VERSION);

    Asm->OutStreamer.AddComment("Offset of Compilation Unit Info");
    Asm->EmitSectionOffset(Asm->GetTempSymbol("info_begin", TheCU->getID()),
                           DwarfInfoSectionSym);

    Asm->OutStreamer.AddComment("Compilation Unit Length");
    Asm->EmitLabelDifference(Asm->GetTempSymbol("info_end", TheCU->getID()),
                             Asm->GetTempSymbol("info_begin", TheCU->getID()),
                             4);

    const StringMap<DIE*> &Globals = TheCU->getGlobals();
    for (StringMap<DIE*>::const_iterator
           GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      DIE *Entity = GI->second;

      Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (Asm->isVerbose())
        Asm->OutStreamer.AddComment("External Name");
      Asm->OutStreamer.EmitBytes(StringRef(Name, strlen(Name)+1), 0);
    }

    Asm->OutStreamer.AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubnames_end",
                                                TheCU->getID()));
  }
}

void DwarfDebug::emitDebugPubTypes() {
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    // Start the dwarf pubnames section.
    Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfPubTypesSection());
    Asm->OutStreamer.AddComment("Length of Public Types Info");
    Asm->EmitLabelDifference(
      Asm->GetTempSymbol("pubtypes_end", TheCU->getID()),
      Asm->GetTempSymbol("pubtypes_begin", TheCU->getID()), 4);

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubtypes_begin",
                                                  TheCU->getID()));

    if (Asm->isVerbose()) Asm->OutStreamer.AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DWARF_VERSION);

    Asm->OutStreamer.AddComment("Offset of Compilation Unit Info");
    Asm->EmitSectionOffset(Asm->GetTempSymbol("info_begin", TheCU->getID()),
                           DwarfInfoSectionSym);

    Asm->OutStreamer.AddComment("Compilation Unit Length");
    Asm->EmitLabelDifference(Asm->GetTempSymbol("info_end", TheCU->getID()),
                             Asm->GetTempSymbol("info_begin", TheCU->getID()),
                             4);

    const StringMap<DIE*> &Globals = TheCU->getGlobalTypes();
    for (StringMap<DIE*>::const_iterator
           GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      DIE * Entity = GI->second;

      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("External Name");
      Asm->OutStreamer.EmitBytes(StringRef(Name, GI->getKeyLength()+1), 0);
    }

    Asm->OutStreamer.AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubtypes_end",
                                                  TheCU->getID()));
  }
}

/// emitDebugStr - Emit visible names into a debug str section.
///
void DwarfDebug::emitDebugStr() {
  // Check to see if it is worth the effort.
  if (StringPool.empty()) return;

  // Start the dwarf str section.
  Asm->OutStreamer.SwitchSection(
                                Asm->getObjFileLowering().getDwarfStrSection());

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<std::pair<unsigned,
      StringMapEntry<std::pair<MCSymbol*, unsigned> >*>, 64> Entries;

  for (StringMap<std::pair<MCSymbol*, unsigned> >::iterator
       I = StringPool.begin(), E = StringPool.end(); I != E; ++I)
    Entries.push_back(std::make_pair(I->second.second, &*I));

  array_pod_sort(Entries.begin(), Entries.end());

  for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
    // Emit a label for reference from debug information entries.
    Asm->OutStreamer.EmitLabel(Entries[i].second->getValue().first);

    // Emit the string itself.
    Asm->OutStreamer.EmitBytes(Entries[i].second->getKey(), 0/*addrspace*/);
  }
}

/// emitDebugLoc - Emit visible names into a debug loc section.
///
void DwarfDebug::emitDebugLoc() {
  if (DotDebugLocEntries.empty())
    return;

  // Start the dwarf loc section.
  Asm->OutStreamer.SwitchSection(
    Asm->getObjFileLowering().getDwarfLocSection());
  unsigned char Size = Asm->getTargetData().getPointerSize();
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_loc", 0));
  unsigned index = 1;
  for (SmallVector<DotDebugLocEntry, 4>::iterator
         I = DotDebugLocEntries.begin(), E = DotDebugLocEntries.end();
       I != E; ++I, ++index) {
    DotDebugLocEntry Entry = *I;
    if (Entry.isEmpty()) {
      Asm->OutStreamer.EmitIntValue(0, Size, /*addrspace*/0);
      Asm->OutStreamer.EmitIntValue(0, Size, /*addrspace*/0);
      Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_loc", index));
    } else {
      Asm->OutStreamer.EmitSymbolValue(Entry.Begin, Size, 0);
      Asm->OutStreamer.EmitSymbolValue(Entry.End, Size, 0);
      const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
      unsigned Reg = RI->getDwarfRegNum(Entry.Loc.getReg(), false);
      if (int Offset =  Entry.Loc.getOffset()) {
        // If the value is at a certain offset from frame register then
        // use DW_OP_fbreg.
        unsigned OffsetSize = Offset ? MCAsmInfo::getSLEB128Size(Offset) : 1;
        Asm->OutStreamer.AddComment("Loc expr size");
        Asm->EmitInt16(1 + OffsetSize);
        Asm->OutStreamer.AddComment(
          dwarf::OperationEncodingString(dwarf::DW_OP_fbreg));
        Asm->EmitInt8(dwarf::DW_OP_fbreg);
        Asm->OutStreamer.AddComment("Offset");
        Asm->EmitSLEB128(Offset);
      } else {
        if (Reg < 32) {
          Asm->OutStreamer.AddComment("Loc expr size");
          Asm->EmitInt16(1);
          Asm->OutStreamer.AddComment(
            dwarf::OperationEncodingString(dwarf::DW_OP_reg0 + Reg));
          Asm->EmitInt8(dwarf::DW_OP_reg0 + Reg);
        } else {
          Asm->OutStreamer.AddComment("Loc expr size");
          Asm->EmitInt16(1 + MCAsmInfo::getULEB128Size(Reg));
          Asm->EmitInt8(dwarf::DW_OP_regx);
          Asm->EmitULEB128(Reg);
        }
      }
    }
  }
}

/// EmitDebugARanges - Emit visible names into a debug aranges section.
///
void DwarfDebug::EmitDebugARanges() {
  // Start the dwarf aranges section.
  Asm->OutStreamer.SwitchSection(
                          Asm->getObjFileLowering().getDwarfARangesSection());
}

/// emitDebugRanges - Emit visible names into a debug ranges section.
///
void DwarfDebug::emitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->OutStreamer.SwitchSection(
    Asm->getObjFileLowering().getDwarfRangesSection());
  unsigned char Size = Asm->getTargetData().getPointerSize();
  for (SmallVector<const MCSymbol *, 8>::iterator
         I = DebugRangeSymbols.begin(), E = DebugRangeSymbols.end();
       I != E; ++I) {
    if (*I)
      Asm->OutStreamer.EmitSymbolValue(const_cast<MCSymbol*>(*I), Size, 0);
    else
      Asm->OutStreamer.EmitIntValue(0, Size, /*addrspace*/0);
  }
}

/// emitDebugMacInfo - Emit visible names into a debug macinfo section.
///
void DwarfDebug::emitDebugMacInfo() {
  if (const MCSection *LineInfo =
      Asm->getObjFileLowering().getDwarfMacroInfoSection()) {
    // Start the dwarf macinfo section.
    Asm->OutStreamer.SwitchSection(LineInfo);
  }
}

/// emitDebugInlineInfo - Emit inline info using following format.
/// Section Header:
/// 1. length of section
/// 2. Dwarf version number
/// 3. address size.
///
/// Entries (one "entry" for each function that was inlined):
///
/// 1. offset into __debug_str section for MIPS linkage name, if exists;
///   otherwise offset into __debug_str for regular function name.
/// 2. offset into __debug_str section for regular function name.
/// 3. an unsigned LEB128 number indicating the number of distinct inlining
/// instances for the function.
///
/// The rest of the entry consists of a {die_offset, low_pc} pair for each
/// inlined instance; the die_offset points to the inlined_subroutine die in the
/// __debug_info section, and the low_pc is the starting address for the
/// inlining instance.
void DwarfDebug::emitDebugInlineInfo() {
  if (!Asm->MAI->doesDwarfUsesInlineInfoSection())
    return;

  if (!FirstCU)
    return;

  Asm->OutStreamer.SwitchSection(
                        Asm->getObjFileLowering().getDwarfDebugInlineSection());

  Asm->OutStreamer.AddComment("Length of Debug Inlined Information Entry");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("debug_inlined_end", 1),
                           Asm->GetTempSymbol("debug_inlined_begin", 1), 4);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_inlined_begin", 1));

  Asm->OutStreamer.AddComment("Dwarf Version");
  Asm->EmitInt16(dwarf::DWARF_VERSION);
  Asm->OutStreamer.AddComment("Address Size (in bytes)");
  Asm->EmitInt8(Asm->getTargetData().getPointerSize());

  for (SmallVector<const MDNode *, 4>::iterator I = InlinedSPNodes.begin(),
         E = InlinedSPNodes.end(); I != E; ++I) {

    const MDNode *Node = *I;
    DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> >::iterator II
      = InlineInfo.find(Node);
    SmallVector<InlineInfoLabels, 4> &Labels = II->second;
    DISubprogram SP(Node);
    StringRef LName = SP.getLinkageName();
    StringRef Name = SP.getName();

    Asm->OutStreamer.AddComment("MIPS linkage name");
    if (LName.empty()) {
      Asm->OutStreamer.EmitBytes(Name, 0);
      Asm->OutStreamer.EmitIntValue(0, 1, 0); // nul terminator.
    } else
      Asm->EmitSectionOffset(getStringPoolEntry(getRealLinkageName(LName)),
                             DwarfStrSectionSym);

    Asm->OutStreamer.AddComment("Function name");
    Asm->EmitSectionOffset(getStringPoolEntry(Name), DwarfStrSectionSym);
    Asm->EmitULEB128(Labels.size(), "Inline count");

    for (SmallVector<InlineInfoLabels, 4>::iterator LI = Labels.begin(),
           LE = Labels.end(); LI != LE; ++LI) {
      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(LI->second->getOffset());

      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("low_pc");
      Asm->OutStreamer.EmitSymbolValue(LI->first,
                                       Asm->getTargetData().getPointerSize(),0);
    }
  }

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_inlined_end", 1));
}
