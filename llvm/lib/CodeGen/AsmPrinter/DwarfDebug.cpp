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
#include "llvm/Module.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Path.h"
using namespace llvm;

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
  DIE *CUDie;

  /// IndexTyDie - An anonymous type for index type.
  DIE *IndexTyDie;

  /// GVToDieMap - Tracks the mapping of unit level debug informaton
  /// variables to debug information entries.
  /// FIXME : Rename GVToDieMap -> NodeToDieMap
  DenseMap<MDNode *, DIE *> GVToDieMap;

  /// GVToDIEEntryMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries using a DIEEntry proxy.
  /// FIXME : Rename
  DenseMap<MDNode *, DIEEntry *> GVToDIEEntryMap;

  /// Globals - A map of globally visible named entities for this unit.
  ///
  StringMap<DIE*> Globals;

  /// GlobalTypes - A map of globally visible types for this unit.
  ///
  StringMap<DIE*> GlobalTypes;

public:
  CompileUnit(unsigned I, DIE *D)
    : ID(I), CUDie(D), IndexTyDie(0) {}
  ~CompileUnit() { delete CUDie; delete IndexTyDie; }

  // Accessors.
  unsigned getID()                  const { return ID; }
  DIE* getCUDie()                   const { return CUDie; }
  const StringMap<DIE*> &getGlobals()     const { return Globals; }
  const StringMap<DIE*> &getGlobalTypes() const { return GlobalTypes; }

  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const { return !CUDie->getChildren().empty(); }

  /// addGlobal - Add a new global entity to the compile unit.
  ///
  void addGlobal(const std::string &Name, DIE *Die) { Globals[Name] = Die; }

  /// addGlobalType - Add a new global type to the compile unit.
  ///
  void addGlobalType(const std::string &Name, DIE *Die) { 
    GlobalTypes[Name] = Die; 
  }

  /// getDIE - Returns the debug information entry map slot for the
  /// specified debug variable.
  DIE *getDIE(MDNode *N) { return GVToDieMap.lookup(N); }

  /// insertDIE - Insert DIE into the map.
  void insertDIE(MDNode *N, DIE *D) {
    GVToDieMap.insert(std::make_pair(N, D));
  }

  /// getDIEEntry - Returns the debug information entry for the speciefied
  /// debug variable.
  DIEEntry *getDIEEntry(MDNode *N) { 
    DenseMap<MDNode *, DIEEntry *>::iterator I = GVToDIEEntryMap.find(N);
    if (I == GVToDIEEntryMap.end())
      return NULL;
    return I->second;
  }

  /// insertDIEEntry - Insert debug information entry into the map.
  void insertDIEEntry(MDNode *N, DIEEntry *E) {
    GVToDIEEntryMap.insert(std::make_pair(N, E));
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
  unsigned FrameIndex;               // Variable frame index.
  DbgVariable *AbstractVar;          // Abstract variable for this variable.
  DIE *TheDIE;
public:
  DbgVariable(DIVariable V, unsigned I)
    : Var(V), FrameIndex(I), AbstractVar(0), TheDIE(0)  {}

  // Accessors.
  DIVariable getVariable()           const { return Var; }
  unsigned getFrameIndex()           const { return FrameIndex; }
  void setAbstractVariable(DbgVariable *V) { AbstractVar = V; }
  DbgVariable *getAbstractVariable() const { return AbstractVar; }
  void setDIE(DIE *D)                      { TheDIE = D; }
  DIE *getDIE()                      const { return TheDIE; }
};

//===----------------------------------------------------------------------===//
/// DbgScope - This class is used to track scope information.
///
class DbgScope {
  DbgScope *Parent;                   // Parent to this scope.
  DIDescriptor Desc;                  // Debug info descriptor for scope.
  MDNode * InlinedAtLocation;           // Location at which scope is inlined.
  bool AbstractScope;                 // Abstract Scope
  unsigned StartLabelID;              // Label ID of the beginning of scope.
  unsigned EndLabelID;                // Label ID of the end of scope.
  const MachineInstr *LastInsn;       // Last instruction of this scope.
  const MachineInstr *FirstInsn;      // First instruction of this scope.
  SmallVector<DbgScope *, 4> Scopes;  // Scopes defined in scope.
  SmallVector<DbgVariable *, 8> Variables;// Variables declared in scope.

  // Private state for dump()
  mutable unsigned IndentLevel;
public:
  DbgScope(DbgScope *P, DIDescriptor D, MDNode *I = 0)
    : Parent(P), Desc(D), InlinedAtLocation(I), AbstractScope(false),
      StartLabelID(0), EndLabelID(0),
      LastInsn(0), FirstInsn(0), IndentLevel(0) {}
  virtual ~DbgScope();

  // Accessors.
  DbgScope *getParent()          const { return Parent; }
  void setParent(DbgScope *P)          { Parent = P; }
  DIDescriptor getDesc()         const { return Desc; }
  MDNode *getInlinedAt()         const {
    return dyn_cast_or_null<MDNode>(InlinedAtLocation);
  }
  MDNode *getScopeNode()         const { return Desc.getNode(); }
  unsigned getStartLabelID()     const { return StartLabelID; }
  unsigned getEndLabelID()       const { return EndLabelID; }
  SmallVector<DbgScope *, 4> &getScopes() { return Scopes; }
  SmallVector<DbgVariable *, 8> &getVariables() { return Variables; }
  void setStartLabelID(unsigned S) { StartLabelID = S; }
  void setEndLabelID(unsigned E)   { EndLabelID = E; }
  void setLastInsn(const MachineInstr *MI) { LastInsn = MI; }
  const MachineInstr *getLastInsn()      { return LastInsn; }
  void setFirstInsn(const MachineInstr *MI) { FirstInsn = MI; }
  void setAbstractScope() { AbstractScope = true; }
  bool isAbstractScope() const { return AbstractScope; }
  const MachineInstr *getFirstInsn()      { return FirstInsn; }

  /// addScope - Add a scope to the scope.
  ///
  void addScope(DbgScope *S) { Scopes.push_back(S); }

  /// addVariable - Add a variable to the scope.
  ///
  void addVariable(DbgVariable *V) { Variables.push_back(V); }

  void fixInstructionMarkers(DenseMap<const MachineInstr *, 
                             unsigned> &MIIndexMap) {
    assert (getFirstInsn() && "First instruction is missing!");
    
    // Use the end of last child scope as end of this scope.
    SmallVector<DbgScope *, 4> &Scopes = getScopes();
    const MachineInstr *LastInsn = getFirstInsn();
    unsigned LIndex = 0;
    if (Scopes.empty()) {
      assert (getLastInsn() && "Inner most scope does not have last insn!");
      return;
    }
    for (SmallVector<DbgScope *, 4>::iterator SI = Scopes.begin(),
           SE = Scopes.end(); SI != SE; ++SI) {
      DbgScope *DS = *SI;
      DS->fixInstructionMarkers(MIIndexMap);
      const MachineInstr *DSLastInsn = DS->getLastInsn();
      unsigned DSI = MIIndexMap[DSLastInsn];
      if (DSI > LIndex) {
        LastInsn = DSLastInsn;
        LIndex = DSI;
      }
    }
    setLastInsn(LastInsn);
  }

#ifndef NDEBUG
  void dump() const;
#endif
};

#ifndef NDEBUG
void DbgScope::dump() const {
  raw_ostream &err = dbgs();
  err.indent(IndentLevel);
  MDNode *N = Desc.getNode();
  N->dump();
  err << " [" << StartLabelID << ", " << EndLabelID << "]\n";
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
  for (unsigned i = 0, N = Scopes.size(); i < N; ++i)
    delete Scopes[i];
  for (unsigned j = 0, M = Variables.size(); j < M; ++j)
    delete Variables[j];
}

} // end llvm namespace

DwarfDebug::DwarfDebug(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T)
  : Dwarf(OS, A, T, "dbg"), ModuleCU(0),
    AbbreviationsSet(InitAbbreviationsSetSize), Abbreviations(),
    DIEValues(), StringPool(),
    SectionSourceLines(), didInitial(false), shouldEmit(false),
    CurrentFnDbgScope(0), DebugTimer(0) {
  if (TimePassesIsEnabled)
    DebugTimer = new Timer("Dwarf Debug Writer");
}
DwarfDebug::~DwarfDebug() {
  for (unsigned j = 0, M = DIEValues.size(); j < M; ++j)
    delete DIEValues[j];

  delete DebugTimer;
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
  DIEEntry *Value = new DIEEntry(Entry);
  DIEValues.push_back(Value);
  return Value;
}

/// addUInt - Add an unsigned integer attribute data and value.
///
void DwarfDebug::addUInt(DIE *Die, unsigned Attribute,
                         unsigned Form, uint64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(false, Integer);
  DIEValue *Value = new DIEInteger(Integer);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addSInt - Add an signed integer attribute data and value.
///
void DwarfDebug::addSInt(DIE *Die, unsigned Attribute,
                         unsigned Form, int64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(true, Integer);
  DIEValue *Value = new DIEInteger(Integer);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addString - Add a string attribute data and value. DIEString only
/// keeps string reference. 
void DwarfDebug::addString(DIE *Die, unsigned Attribute, unsigned Form,
                           const StringRef String) {
  DIEValue *Value = new DIEString(String);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addLabel - Add a Dwarf label attribute data and value.
///
void DwarfDebug::addLabel(DIE *Die, unsigned Attribute, unsigned Form,
                          const DWLabel &Label) {
  DIEValue *Value = new DIEDwarfLabel(Label);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addObjectLabel - Add an non-Dwarf label attribute data and value.
///
void DwarfDebug::addObjectLabel(DIE *Die, unsigned Attribute, unsigned Form,
                                const MCSymbol *Sym) {
  DIEValue *Value = new DIEObjectLabel(Sym);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addSectionOffset - Add a section offset label attribute data and value.
///
void DwarfDebug::addSectionOffset(DIE *Die, unsigned Attribute, unsigned Form,
                                  const DWLabel &Label, const DWLabel &Section,
                                  bool isEH, bool useSet) {
  DIEValue *Value = new DIESectionOffset(Label, Section, isEH, useSet);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addDelta - Add a label delta attribute data and value.
///
void DwarfDebug::addDelta(DIE *Die, unsigned Attribute, unsigned Form,
                          const DWLabel &Hi, const DWLabel &Lo) {
  DIEValue *Value = new DIEDelta(Hi, Lo);
  DIEValues.push_back(Value);
  Die->addValue(Attribute, Form, Value);
}

/// addBlock - Add block data.
///
void DwarfDebug::addBlock(DIE *Die, unsigned Attribute, unsigned Form,
                          DIEBlock *Block) {
  Block->ComputeSize(TD);
  DIEValues.push_back(Block);
  Die->addValue(Attribute, Block->BestForm(), Block);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, const DIVariable *V) {
  // If there is no compile unit specified, don't add a line #.
  if (V->getCompileUnit().isNull())
    return;

  unsigned Line = V->getLineNumber();
  unsigned FileID = findCompileUnit(V->getCompileUnit())->getID();
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, const DIGlobal *G) {
  // If there is no compile unit specified, don't add a line #.
  if (G->getCompileUnit().isNull())
    return;

  unsigned Line = G->getLineNumber();
  unsigned FileID = findCompileUnit(G->getCompileUnit())->getID();
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, const DISubprogram *SP) {
  // If there is no compile unit specified, don't add a line #.
  if (SP->getCompileUnit().isNull())
    return;
  // If the line number is 0, don't add it.
  if (SP->getLineNumber() == 0)
    return;


  unsigned Line = SP->getLineNumber();
  unsigned FileID = findCompileUnit(SP->getCompileUnit())->getID();
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, const DIType *Ty) {
  // If there is no compile unit specified, don't add a line #.
  DICompileUnit CU = Ty->getCompileUnit();
  if (CU.isNull())
    return;

  unsigned Line = Ty->getLineNumber();
  unsigned FileID = findCompileUnit(CU)->getID();
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// addSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::addSourceLine(DIE *Die, const DINameSpace *NS) {
  // If there is no compile unit specified, don't add a line #.
  if (NS->getCompileUnit().isNull())
    return;

  unsigned Line = NS->getLineNumber();
  StringRef FN = NS->getFilename();
  StringRef Dir = NS->getDirectory();

  unsigned FileID = GetOrCreateSourceID(Dir, FN);
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

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

/// Find the type the programmer originally declared the variable to be
/// and return that type.
///
DIType DwarfDebug::getBlockByrefType(DIType Ty, std::string Name) {

  DIType subType = Ty;
  unsigned tag = Ty.getTag();

  if (tag == dwarf::DW_TAG_pointer_type) {
    DIDerivedType DTy = DIDerivedType(Ty.getNode());
    subType = DTy.getTypeDerivedFrom();
  }

  DICompositeType blockStruct = DICompositeType(subType.getNode());

  DIArray Elements = blockStruct.getTypeArray();

  if (Elements.isNull())
    return Ty;

  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    DIDerivedType DT = DIDerivedType(Element.getNode());
    if (Name == DT.getName())
      return (DT.getTypeDerivedFrom());
  }

  return Ty;
}

/// addComplexAddress - Start with the address based on the location provided,
/// and generate the DWARF information necessary to find the actual variable
/// given the extra address information encoded in the DIVariable, starting from
/// the starting location.  Add the DWARF information to the die.
///
void DwarfDebug::addComplexAddress(DbgVariable *&DV, DIE *Die,
                                   unsigned Attribute,
                                   const MachineLocation &Location) {
  const DIVariable &VD = DV->getVariable();
  DIType Ty = VD.getType();

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new DIEBlock();

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

  for (unsigned i = 0, N = VD.getNumAddrElements(); i < N; ++i) {
    uint64_t Element = VD.getAddrElement(i);

    if (Element == DIFactory::OpPlus) {
      addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
      addUInt(Block, 0, dwarf::DW_FORM_udata, VD.getAddrElement(++i));
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

   2).  Follow that pointer to get the the real __Block_byref_x_VarName
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
  const DIVariable &VD = DV->getVariable();
  DIType Ty = VD.getType();
  DIType TmpTy = Ty;
  unsigned Tag = Ty.getTag();
  bool isPointer = false;

  StringRef varName = VD.getName();

  if (Tag == dwarf::DW_TAG_pointer_type) {
    DIDerivedType DTy = DIDerivedType(Ty.getNode());
    TmpTy = DTy.getTypeDerivedFrom();
    isPointer = true;
  }

  DICompositeType blockStruct = DICompositeType(TmpTy.getNode());

  // Find the __forwarding field and the variable field in the __Block_byref
  // struct.
  DIArray Fields = blockStruct.getTypeArray();
  DIDescriptor varField = DIDescriptor();
  DIDescriptor forwardingField = DIDescriptor();


  for (unsigned i = 0, N = Fields.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Fields.getElement(i);
    DIDerivedType DT = DIDerivedType(Element.getNode());
    StringRef fieldName = DT.getName();
    if (fieldName == "__forwarding")
      forwardingField = Element;
    else if (fieldName == varName)
      varField = Element;
  }

  assert(!varField.isNull() && "Can't find byref variable in Block struct");
  assert(!forwardingField.isNull()
         && "Can't find forwarding field in Block struct");

  // Get the offsets for the forwarding field and the variable field.
  unsigned int forwardingFieldOffset =
    DIDerivedType(forwardingField.getNode()).getOffsetInBits() >> 3;
  unsigned int varFieldOffset =
    DIDerivedType(varField.getNode()).getOffsetInBits() >> 3;

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new DIEBlock();

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
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new DIEBlock();

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

/// addToContextOwner - Add Die into the list of its context owner's children.
void DwarfDebug::addToContextOwner(DIE *Die, DIDescriptor Context) {
  if (Context.isNull())
    ModuleCU->addDie(Die);
  else if (Context.isType()) {
    DIE *ContextDIE = getOrCreateTypeDIE(DIType(Context.getNode()));
    ContextDIE->addChild(Die);
  } else if (Context.isNameSpace()) {
    DIE *ContextDIE = getOrCreateNameSpace(DINameSpace(Context.getNode()));
    ContextDIE->addChild(Die);
  } else if (DIE *ContextDIE = ModuleCU->getDIE(Context.getNode()))
    ContextDIE->addChild(Die);
  else 
    ModuleCU->addDie(Die);
}

/// getOrCreateTypeDIE - Find existing DIE or create new DIE for the
/// given DIType.
DIE *DwarfDebug::getOrCreateTypeDIE(DIType Ty) {
  DIE *TyDIE = ModuleCU->getDIE(Ty.getNode());
  if (TyDIE)
    return TyDIE;

  // Create new type.
  TyDIE = new DIE(dwarf::DW_TAG_base_type);
  ModuleCU->insertDIE(Ty.getNode(), TyDIE);
  if (Ty.isBasicType())
    constructTypeDIE(*TyDIE, DIBasicType(Ty.getNode()));
  else if (Ty.isCompositeType())
    constructTypeDIE(*TyDIE, DICompositeType(Ty.getNode()));
  else {
    assert(Ty.isDerivedType() && "Unknown kind of DIType");
    constructTypeDIE(*TyDIE, DIDerivedType(Ty.getNode()));
  }

  addToContextOwner(TyDIE, Ty.getContext());
  return TyDIE;
}

/// addType - Add a new type attribute to the specified entity.
void DwarfDebug::addType(DIE *Entity, DIType Ty) {
  if (Ty.isNull())
    return;

  // Check for pre-existence.
  DIEEntry *Entry = ModuleCU->getDIEEntry(Ty.getNode());
  // If it exists then use the existing value.
  if (Entry) {
    Entity->addValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Entry);
    return;
  }

  // Set up proxy.
  Entry = createDIEEntry();
  ModuleCU->insertDIEEntry(Ty.getNode(), Entry);

  // Construct type.
  DIE *Buffer = getOrCreateTypeDIE(Ty);

  Entry->setEntry(Buffer);
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
    addSourceLine(&Buffer, &DTy);
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
      DIEnumerator Enum(Elements.getElement(i).getNode());
      if (!Enum.isNull()) {
        ElemDie = constructEnumTypeDIE(&Enum);
        Buffer.addChild(ElemDie);
      }
    }
  }
    break;
  case dwarf::DW_TAG_subroutine_type: {
    // Add return type.
    DIArray Elements = CTy.getTypeArray();
    DIDescriptor RTy = Elements.getElement(0);
    addType(&Buffer, DIType(RTy.getNode()));

    // Add prototype flag.
    addUInt(&Buffer, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);

    // Add arguments.
    for (unsigned i = 1, N = Elements.getNumElements(); i < N; ++i) {
      DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
      DIDescriptor Ty = Elements.getElement(i);
      addType(Arg, DIType(Ty.getNode()));
      Buffer.addChild(Arg);
    }
  }
    break;
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type: {
    // Add elements to structure type.
    DIArray Elements = CTy.getTypeArray();

    // A forward struct declared type may not have elements available.
    if (Elements.isNull())
      break;

    // Add elements to structure type.
    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIDescriptor Element = Elements.getElement(i);
      if (Element.isNull())
        continue;
      DIE *ElemDie = NULL;
      if (Element.getTag() == dwarf::DW_TAG_subprogram)
        ElemDie = createSubprogramDIE(DISubprogram(Element.getNode()));
      else
        ElemDie = createMemberDIE(DIDerivedType(Element.getNode()));
      Buffer.addChild(ElemDie);
    }

    if (CTy.isAppleBlockExtension())
      addUInt(&Buffer, dwarf::DW_AT_APPLE_block, dwarf::DW_FORM_flag, 1);

    unsigned RLang = CTy.getRunTimeLang();
    if (RLang)
      addUInt(&Buffer, dwarf::DW_AT_APPLE_runtime_class,
              dwarf::DW_FORM_data1, RLang);
    break;
  }
  default:
    break;
  }

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  if (Tag == dwarf::DW_TAG_enumeration_type ||
      Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type) {
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
      addSourceLine(&Buffer, &CTy);
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
  DIE *IdxTy = ModuleCU->getIndexTyDie();
  if (!IdxTy) {
    // Construct an anonymous type for index type.
    IdxTy = new DIE(dwarf::DW_TAG_base_type);
    addUInt(IdxTy, dwarf::DW_AT_byte_size, 0, sizeof(int32_t));
    addUInt(IdxTy, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
            dwarf::DW_ATE_signed);
    ModuleCU->addDie(IdxTy);
    ModuleCU->setIndexTyDie(IdxTy);
  }

  // Add subranges to array type.
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    if (Element.getTag() == dwarf::DW_TAG_subrange_type)
      constructSubrangeDIE(Buffer, DISubrange(Element.getNode()), IdxTy);
  }
}

/// constructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
DIE *DwarfDebug::constructEnumTypeDIE(DIEnumerator *ETy) {
  DIE *Enumerator = new DIE(dwarf::DW_TAG_enumerator);
  StringRef Name = ETy->getName();
  addString(Enumerator, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  int64_t Value = ETy->getEnumValue();
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

/// createGlobalVariableDIE - Create new DIE using GV.
DIE *DwarfDebug::createGlobalVariableDIE(const DIGlobalVariable &GV) {
  // If the global variable was optmized out then no need to create debug info
  // entry.
  if (!GV.getGlobal()) return NULL;
  if (GV.getDisplayName().empty()) return NULL;

  DIE *GVDie = new DIE(dwarf::DW_TAG_variable);
  addString(GVDie, dwarf::DW_AT_name, dwarf::DW_FORM_string,
            GV.getDisplayName());

  StringRef LinkageName = GV.getLinkageName();
  if (!LinkageName.empty())
    addString(GVDie, dwarf::DW_AT_MIPS_linkage_name, dwarf::DW_FORM_string,
              getRealLinkageName(LinkageName));

  addType(GVDie, GV.getType());
  if (!GV.isLocalToUnit())
    addUInt(GVDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);
  addSourceLine(GVDie, &GV);

  return GVDie;
}

/// createMemberDIE - Create new member DIE.
DIE *DwarfDebug::createMemberDIE(const DIDerivedType &DT) {
  DIE *MemberDie = new DIE(DT.getTag());
  StringRef Name = DT.getName();
  if (!Name.empty())
    addString(MemberDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  
  addType(MemberDie, DT.getTypeDerivedFrom());

  addSourceLine(MemberDie, &DT);

  DIEBlock *MemLocationDie = new DIEBlock();
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
    if (TD->isLittleEndian()) Offset = FieldSize - (Offset + Size);
    addUInt(MemberDie, dwarf::DW_AT_bit_offset, 0, Offset);

    // Here WD_AT_data_member_location points to the anonymous
    // field that includes this bit field.
    addUInt(MemLocationDie, 0, dwarf::DW_FORM_udata, FieldOffset >> 3);

  } else
    // This is not a bitfield.
    addUInt(MemLocationDie, 0, dwarf::DW_FORM_udata, DT.getOffsetInBits() >> 3);

  addBlock(MemberDie, dwarf::DW_AT_data_member_location, 0, MemLocationDie);

  if (DT.isProtected())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_protected);
  else if (DT.isPrivate())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_private);
  else if (DT.getTag() == dwarf::DW_TAG_inheritance)
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_flag,
            dwarf::DW_ACCESS_public);
  if (DT.isVirtual())
    addUInt(MemberDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_flag,
            dwarf::DW_VIRTUALITY_virtual);
  return MemberDie;
}

/// createSubprogramDIE - Create new DIE using SP.
DIE *DwarfDebug::createSubprogramDIE(const DISubprogram &SP, bool MakeDecl) {
  DIE *SPDie = ModuleCU->getDIE(SP.getNode());
  if (SPDie)
    return SPDie;

  SPDie = new DIE(dwarf::DW_TAG_subprogram);
  addString(SPDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, SP.getName());

  StringRef LinkageName = SP.getLinkageName();
  if (!LinkageName.empty())
    addString(SPDie, dwarf::DW_AT_MIPS_linkage_name, dwarf::DW_FORM_string,
              getRealLinkageName(LinkageName));

  addSourceLine(SPDie, &SP);

  // Add prototyped tag, if C or ObjC.
  unsigned Lang = SP.getCompileUnit().getLanguage();
  if (Lang == dwarf::DW_LANG_C99 || Lang == dwarf::DW_LANG_C89 ||
      Lang == dwarf::DW_LANG_ObjC)
    addUInt(SPDie, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);

  // Add Return Type.
  DICompositeType SPTy = SP.getType();
  DIArray Args = SPTy.getTypeArray();
  unsigned SPTag = SPTy.getTag();

  if (Args.isNull() || SPTag != dwarf::DW_TAG_subroutine_type)
    addType(SPDie, SPTy);
  else
    addType(SPDie, DIType(Args.getElement(0).getNode()));

  unsigned VK = SP.getVirtuality();
  if (VK) {
    addUInt(SPDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_flag, VK);
    DIEBlock *Block = new DIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(Block, 0, dwarf::DW_FORM_data1, SP.getVirtualIndex());
    addBlock(SPDie, dwarf::DW_AT_vtable_elem_location, 0, Block);
    ContainingTypeMap.insert(std::make_pair(SPDie, 
                                            SP.getContainingType().getNode()));
  }

  if (MakeDecl || !SP.isDefinition()) {
    addUInt(SPDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);

    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled while processing variables.
    DICompositeType SPTy = SP.getType();
    DIArray Args = SPTy.getTypeArray();
    unsigned SPTag = SPTy.getTag();

    if (SPTag == dwarf::DW_TAG_subroutine_type)
      for (unsigned i = 1, N =  Args.getNumElements(); i < N; ++i) {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        addType(Arg, DIType(Args.getElement(i).getNode()));
        addUInt(Arg, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1); // ??
        SPDie->addChild(Arg);
      }
  }

  // DW_TAG_inlined_subroutine may refer to this DIE.
  ModuleCU->insertDIE(SP.getNode(), SPDie);
  return SPDie;
}

/// findCompileUnit - Get the compile unit for the given descriptor.
///
CompileUnit *DwarfDebug::findCompileUnit(DICompileUnit Unit) {
  DenseMap<Value *, CompileUnit *>::const_iterator I =
    CompileUnitMap.find(Unit.getNode());
  if (I == CompileUnitMap.end())
    return constructCompileUnit(Unit.getNode());
  return I->second;
}

/// getUpdatedDbgScope - Find or create DbgScope assicated with the instruction.
/// Initialize scope and update scope hierarchy.
DbgScope *DwarfDebug::getUpdatedDbgScope(MDNode *N, const MachineInstr *MI,
  MDNode *InlinedAt) {
  assert (N && "Invalid Scope encoding!");
  assert (MI && "Missing machine instruction!");
  bool GetConcreteScope = (MI && InlinedAt);

  DbgScope *NScope = NULL;

  if (InlinedAt)
    NScope = DbgScopeMap.lookup(InlinedAt);
  else
    NScope = DbgScopeMap.lookup(N);
  assert (NScope && "Unable to find working scope!");

  if (NScope->getFirstInsn())
    return NScope;

  DbgScope *Parent = NULL;
  if (GetConcreteScope) {
    DILocation IL(InlinedAt);
    Parent = getUpdatedDbgScope(IL.getScope().getNode(), MI,
                         IL.getOrigLocation().getNode());
    assert (Parent && "Unable to find Parent scope!");
    NScope->setParent(Parent);
    Parent->addScope(NScope);
  } else if (DIDescriptor(N).isLexicalBlock()) {
    DILexicalBlock DB(N);
    if (!DB.getContext().isNull()) {
      Parent = getUpdatedDbgScope(DB.getContext().getNode(), MI, InlinedAt);
      NScope->setParent(Parent);
      Parent->addScope(NScope);
    }
  }

  NScope->setFirstInsn(MI);

  if (!Parent && !InlinedAt) {
    StringRef SPName = DISubprogram(N).getLinkageName();
    if (SPName == MF->getFunction()->getName())
      CurrentFnDbgScope = NScope;
  }

  if (GetConcreteScope) {
    ConcreteScopes[InlinedAt] = NScope;
    getOrCreateAbstractScope(N);
  }

  return NScope;
}

DbgScope *DwarfDebug::getOrCreateAbstractScope(MDNode *N) {
  assert (N && "Invalid Scope encoding!");

  DbgScope *AScope = AbstractScopes.lookup(N);
  if (AScope)
    return AScope;

  DbgScope *Parent = NULL;

  DIDescriptor Scope(N);
  if (Scope.isLexicalBlock()) {
    DILexicalBlock DB(N);
    DIDescriptor ParentDesc = DB.getContext();
    if (!ParentDesc.isNull())
      Parent = getOrCreateAbstractScope(ParentDesc.getNode());
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

/// updateSubprogramScopeDIE - Find DIE for the given subprogram and
/// attach appropriate DW_AT_low_pc and DW_AT_high_pc attributes.
/// If there are global variables in this scope then create and insert
/// DIEs for these variables.
DIE *DwarfDebug::updateSubprogramScopeDIE(MDNode *SPNode) {

 DIE *SPDie = ModuleCU->getDIE(SPNode);
 assert (SPDie && "Unable to find subprogram DIE!");
 DISubprogram SP(SPNode);
 if (SP.isDefinition() && !SP.getContext().isCompileUnit()) {
   addUInt(SPDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
  // Add arguments. 
   DICompositeType SPTy = SP.getType();
   DIArray Args = SPTy.getTypeArray();
   unsigned SPTag = SPTy.getTag();
   if (SPTag == dwarf::DW_TAG_subroutine_type)
     for (unsigned i = 1, N =  Args.getNumElements(); i < N; ++i) {
       DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
       addType(Arg, DIType(Args.getElement(i).getNode()));
       addUInt(Arg, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1); // ??
       SPDie->addChild(Arg);
     }
   DIE *SPDeclDie = SPDie;
   SPDie = new DIE(dwarf::DW_TAG_subprogram);
   addDIEEntry(SPDie, dwarf::DW_AT_specification, dwarf::DW_FORM_ref4, 
               SPDeclDie);
   ModuleCU->addDie(SPDie);
 }
   
 addLabel(SPDie, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
          DWLabel("func_begin", SubprogramCount));
 addLabel(SPDie, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
          DWLabel("func_end", SubprogramCount));
 MachineLocation Location(RI->getFrameRegister(*MF));
 addAddress(SPDie, dwarf::DW_AT_frame_base, Location);

 if (!DISubprogram(SPNode).isLocalToUnit())
   addUInt(SPDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);

 return SPDie;
}

/// constructLexicalScope - Construct new DW_TAG_lexical_block
/// for this scope and attach DW_AT_low_pc/DW_AT_high_pc labels.
DIE *DwarfDebug::constructLexicalScopeDIE(DbgScope *Scope) {
  unsigned StartID = MMI->MappedLabel(Scope->getStartLabelID());
  unsigned EndID = MMI->MappedLabel(Scope->getEndLabelID());

  // Ignore empty scopes.
  if (StartID == EndID && StartID != 0)
    return NULL;

  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_lexical_block);
  if (Scope->isAbstractScope())
    return ScopeDIE;

  addLabel(ScopeDIE, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
           StartID ?
             DWLabel("label", StartID)
           : DWLabel("func_begin", SubprogramCount));
  addLabel(ScopeDIE, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
           EndID ?
             DWLabel("label", EndID)
           : DWLabel("func_end", SubprogramCount));



  return ScopeDIE;
}

/// constructInlinedScopeDIE - This scope represents inlined body of
/// a function. Construct DIE to represent this concrete inlined copy
/// of the function.
DIE *DwarfDebug::constructInlinedScopeDIE(DbgScope *Scope) {
  unsigned StartID = MMI->MappedLabel(Scope->getStartLabelID());
  unsigned EndID = MMI->MappedLabel(Scope->getEndLabelID());
  assert (StartID && "Invalid starting label for an inlined scope!");
  assert (EndID && "Invalid end label for an inlined scope!");
  // Ignore empty scopes.
  if (StartID == EndID && StartID != 0)
    return NULL;

  DIScope DS(Scope->getScopeNode());
  if (DS.isNull())
    return NULL;
  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_inlined_subroutine);

  DISubprogram InlinedSP = getDISubprogram(DS.getNode());
  DIE *OriginDIE = ModuleCU->getDIE(InlinedSP.getNode());
  assert (OriginDIE && "Unable to find Origin DIE!");
  addDIEEntry(ScopeDIE, dwarf::DW_AT_abstract_origin,
              dwarf::DW_FORM_ref4, OriginDIE);

  addLabel(ScopeDIE, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
           DWLabel("label", StartID));
  addLabel(ScopeDIE, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
           DWLabel("label", EndID));

  InlinedSubprogramDIEs.insert(OriginDIE);

  // Track the start label for this inlined function.
  DenseMap<MDNode *, SmallVector<InlineInfoLabels, 4> >::iterator
    I = InlineInfo.find(InlinedSP.getNode());

  if (I == InlineInfo.end()) {
    InlineInfo[InlinedSP.getNode()].push_back(std::make_pair(StartID,
                                                             ScopeDIE));
    InlinedSPNodes.push_back(InlinedSP.getNode());
  } else
    I->second.push_back(std::make_pair(StartID, ScopeDIE));

  StringPool.insert(InlinedSP.getName());
  StringPool.insert(getRealLinkageName(InlinedSP.getLinkageName()));

  DILocation DL(Scope->getInlinedAt());
  addUInt(ScopeDIE, dwarf::DW_AT_call_file, 0, ModuleCU->getID());
  addUInt(ScopeDIE, dwarf::DW_AT_call_line, 0, DL.getLineNumber());

  return ScopeDIE;
}


/// constructVariableDIE - Construct a DIE for the given DbgVariable.
DIE *DwarfDebug::constructVariableDIE(DbgVariable *DV, DbgScope *Scope) {
  // Get the descriptor.
  const DIVariable &VD = DV->getVariable();
  StringRef Name = VD.getName();
  if (Name.empty())
    return NULL;

  // Translate tag to proper Dwarf tag.  The result variable is dropped for
  // now.
  unsigned Tag;
  switch (VD.getTag()) {
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
  if (DbgVariable *AV = DV->getAbstractVariable())
    AbsDIE = AV->getDIE();

  if (AbsDIE) {
    DIScope DS(Scope->getScopeNode());
    DISubprogram InlinedSP = getDISubprogram(DS.getNode());
    DIE *OriginSPDIE = ModuleCU->getDIE(InlinedSP.getNode());
    (void) OriginSPDIE;
    assert (OriginSPDIE && "Unable to find Origin DIE for the SP!");
    DIE *AbsDIE = DV->getAbstractVariable()->getDIE();
    assert (AbsDIE && "Unable to find Origin DIE for the Variable!");
    addDIEEntry(VariableDie, dwarf::DW_AT_abstract_origin,
                dwarf::DW_FORM_ref4, AbsDIE);
  }
  else {
    addString(VariableDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
    addSourceLine(VariableDie, &VD);

    // Add variable type.
    // FIXME: isBlockByrefVariable should be reformulated in terms of complex
    // addresses instead.
    if (VD.isBlockByrefVariable())
      addType(VariableDie, getBlockByrefType(VD.getType(), Name));
    else
      addType(VariableDie, VD.getType());
  }

  // Add variable address.
  if (!Scope->isAbstractScope()) {
    MachineLocation Location;
    unsigned FrameReg;
    int Offset = RI->getFrameIndexReference(*MF, DV->getFrameIndex(), FrameReg);
    Location.set(FrameReg, Offset);

    if (VD.hasComplexAddress())
      addComplexAddress(DV, VariableDie, dwarf::DW_AT_location, Location);
    else if (VD.isBlockByrefVariable())
      addBlockByrefAddress(DV, VariableDie, dwarf::DW_AT_location, Location);
    else
      addAddress(VariableDie, dwarf::DW_AT_location, Location);
  }
  DV->setDIE(VariableDie);
  return VariableDie;

}

void DwarfDebug::addPubTypes(DISubprogram SP) {
  DICompositeType SPTy = SP.getType();
  unsigned SPTag = SPTy.getTag();
  if (SPTag != dwarf::DW_TAG_subroutine_type) 
    return;

  DIArray Args = SPTy.getTypeArray();
  if (Args.isNull()) 
    return;

  for (unsigned i = 0, e = Args.getNumElements(); i != e; ++i) {
    DIType ATy(Args.getElement(i).getNode());
    if (ATy.isNull())
      continue;
    DICompositeType CATy = getDICompositeType(ATy);
    if (!CATy.isNull() && !CATy.getName().empty()) {
      if (DIEEntry *Entry = ModuleCU->getDIEEntry(CATy.getNode()))
        ModuleCU->addGlobalType(CATy.getName(), Entry->getEntry());
    }
  }
}

/// constructScopeDIE - Construct a DIE for this scope.
DIE *DwarfDebug::constructScopeDIE(DbgScope *Scope) {
 if (!Scope)
  return NULL;
 DIScope DS(Scope->getScopeNode());
 if (DS.isNull())
   return NULL;

 DIE *ScopeDIE = NULL;
 if (Scope->getInlinedAt())
   ScopeDIE = constructInlinedScopeDIE(Scope);
 else if (DS.isSubprogram()) {
   if (Scope->isAbstractScope())
     ScopeDIE = ModuleCU->getDIE(DS.getNode());
   else
     ScopeDIE = updateSubprogramScopeDIE(DS.getNode());
 }
 else {
   ScopeDIE = constructLexicalScopeDIE(Scope);
   if (!ScopeDIE) return NULL;
 }

  // Add variables to scope.
  SmallVector<DbgVariable *, 8> &Variables = Scope->getVariables();
  for (unsigned i = 0, N = Variables.size(); i < N; ++i) {
    DIE *VariableDIE = constructVariableDIE(Variables[i], Scope);
    if (VariableDIE)
      ScopeDIE->addChild(VariableDIE);
  }

  // Add nested scopes.
  SmallVector<DbgScope *, 4> &Scopes = Scope->getScopes();
  for (unsigned j = 0, M = Scopes.size(); j < M; ++j) {
    // Define the Scope debug information entry.
    DIE *NestedDIE = constructScopeDIE(Scopes[j]);
    if (NestedDIE)
      ScopeDIE->addChild(NestedDIE);
  }

  if (DS.isSubprogram()) 
    addPubTypes(DISubprogram(DS.getNode()));
 
 return ScopeDIE;
}

/// GetOrCreateSourceID - Look up the source id with the given directory and
/// source file names. If none currently exists, create a new id and insert it
/// in the SourceIds map. This can update DirectoryNames and SourceFileNames
/// maps as well.
unsigned DwarfDebug::GetOrCreateSourceID(StringRef DirName, StringRef FileName) {
  unsigned DId;
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
  DIE *NDie = ModuleCU->getDIE(NS.getNode());
  if (NDie)
    return NDie;
  NDie = new DIE(dwarf::DW_TAG_namespace);
  ModuleCU->insertDIE(NS.getNode(), NDie);
  if (!NS.getName().empty())
    addString(NDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, NS.getName());
  addSourceLine(NDie, &NS);
  addToContextOwner(NDie, NS.getContext());
  return NDie;
}

CompileUnit *DwarfDebug::constructCompileUnit(MDNode *N) {
  DICompileUnit DIUnit(N);
  StringRef FN = DIUnit.getFilename();
  StringRef Dir = DIUnit.getDirectory();
  unsigned ID = GetOrCreateSourceID(Dir, FN);

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  addSectionOffset(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4,
                   DWLabel("section_line", 0), DWLabel("section_line", 0),
                   false);
  addString(Die, dwarf::DW_AT_producer, dwarf::DW_FORM_string,
            DIUnit.getProducer());
  addUInt(Die, dwarf::DW_AT_language, dwarf::DW_FORM_data1,
          DIUnit.getLanguage());
  addString(Die, dwarf::DW_AT_name, dwarf::DW_FORM_string, FN);

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

  CompileUnit *Unit = new CompileUnit(ID, Die);
  if (!ModuleCU && DIUnit.isMain()) {
    // Use first compile unit marked as isMain as the compile unit
    // for this module.
    ModuleCU = Unit;
  }

  CompileUnitMap[DIUnit.getNode()] = Unit;
  CompileUnits.push_back(Unit);
  return Unit;
}

void DwarfDebug::constructGlobalVariableDIE(MDNode *N) {
  DIGlobalVariable DI_GV(N);

  // If debug information is malformed then ignore it.
  if (DI_GV.Verify() == false)
    return;

  // Check for pre-existence.
  if (ModuleCU->getDIE(DI_GV.getNode()))
    return;

  DIE *VariableDie = createGlobalVariableDIE(DI_GV);
  if (!VariableDie)
    return;

  // Add to map.
  ModuleCU->insertDIE(N, VariableDie);

  // Add to context owner.
  DIDescriptor GVContext = DI_GV.getContext();
  // Do not create specification DIE if context is either compile unit
  // or a subprogram.
  if (DI_GV.isDefinition() && !GVContext.isCompileUnit()
      && !GVContext.isSubprogram()) {
    // Create specification DIE.
    DIE *VariableSpecDIE = new DIE(dwarf::DW_TAG_variable);
    addDIEEntry(VariableSpecDIE, dwarf::DW_AT_specification,
                dwarf::DW_FORM_ref4, VariableDie);
    DIEBlock *Block = new DIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addObjectLabel(Block, 0, dwarf::DW_FORM_udata,
                   Asm->GetGlobalValueSymbol(DI_GV.getGlobal()));
    addBlock(VariableSpecDIE, dwarf::DW_AT_location, 0, Block);
    ModuleCU->addDie(VariableSpecDIE);
  } else {
    DIEBlock *Block = new DIEBlock();
    addUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addObjectLabel(Block, 0, dwarf::DW_FORM_udata,
                   Asm->GetGlobalValueSymbol(DI_GV.getGlobal()));
    addBlock(VariableDie, dwarf::DW_AT_location, 0, Block);
  }
  addToContextOwner(VariableDie, GVContext);
  
  // Expose as global. FIXME - need to check external flag.
  ModuleCU->addGlobal(DI_GV.getName(), VariableDie);

  DIType GTy = DI_GV.getType();
  if (GTy.isCompositeType() && !GTy.getName().empty()) {
    DIEEntry *Entry = ModuleCU->getDIEEntry(GTy.getNode());
    assert (Entry && "Missing global type!");
    ModuleCU->addGlobalType(GTy.getName(), Entry->getEntry());
  }
  return;
}

void DwarfDebug::constructSubprogramDIE(MDNode *N) {
  DISubprogram SP(N);

  // Check for pre-existence.
  if (ModuleCU->getDIE(N))
    return;

  if (!SP.isDefinition())
    // This is a method declaration which will be handled while constructing
    // class type.
    return;

  DIE *SubprogramDie = createSubprogramDIE(SP);

  // Add to map.
  ModuleCU->insertDIE(N, SubprogramDie);

  // Add to context owner.
  addToContextOwner(SubprogramDie, SP.getContext());

  // Expose as global.
  ModuleCU->addGlobal(SP.getName(), SubprogramDie);

  return;
}

/// beginModule - Emit all Dwarf sections that should come prior to the
/// content. Create global DIEs and emit initial debug info sections.
/// This is inovked by the target AsmPrinter.
void DwarfDebug::beginModule(Module *M, MachineModuleInfo *mmi) {
  this->M = M;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  if (!MAI->doesSupportDebugInformation())
    return;

  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(*M);

  // Create all the compile unit DIEs.
  for (DebugInfoFinder::iterator I = DbgFinder.compile_unit_begin(),
         E = DbgFinder.compile_unit_end(); I != E; ++I)
    constructCompileUnit(*I);

  if (CompileUnits.empty()) {
    if (TimePassesIsEnabled)
      DebugTimer->stopTimer();

    return;
  }

  // If main compile unit for this module is not seen than randomly
  // select first compile unit.
  if (!ModuleCU)
    ModuleCU = CompileUnits[0];

  // Create DIEs for each subprogram.
  for (DebugInfoFinder::iterator I = DbgFinder.subprogram_begin(),
         E = DbgFinder.subprogram_end(); I != E; ++I)
    constructSubprogramDIE(*I);

  // Create DIEs for each global variable.
  for (DebugInfoFinder::iterator I = DbgFinder.global_variable_begin(),
         E = DbgFinder.global_variable_end(); I != E; ++I)
    constructGlobalVariableDIE(*I);

  MMI = mmi;
  shouldEmit = true;
  MMI->setDebugInfoAvailability(true);

  // Prime section data.
  SectionMap.insert(Asm->getObjFileLowering().getTextSection());

  // Print out .file directives to specify files for .loc directives. These are
  // printed out early so that they precede any .loc directives.
  if (MAI->hasDotLocAndDotFile()) {
    for (unsigned i = 1, e = getNumSourceIds()+1; i != e; ++i) {
      // Remember source id starts at 1.
      std::pair<unsigned, unsigned> Id = getSourceDirectoryAndFileIds(i);
      sys::Path FullPath(getSourceDirectoryName(Id.first));
      bool AppendOk =
        FullPath.appendComponent(getSourceFileName(Id.second));
      assert(AppendOk && "Could not append filename to directory!");
      AppendOk = false;
      Asm->EmitFile(i, FullPath.str());
      Asm->EOL();
    }
  }

  // Emit initial sections
  emitInitial();

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// endModule - Emit all Dwarf sections that should come after the content.
///
void DwarfDebug::endModule() {
  if (!ModuleCU)
    return;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  // Attach DW_AT_inline attribute with inlined subprogram DIEs.
  for (SmallPtrSet<DIE *, 4>::iterator AI = InlinedSubprogramDIEs.begin(),
         AE = InlinedSubprogramDIEs.end(); AI != AE; ++AI) {
    DIE *ISP = *AI;
    addUInt(ISP, dwarf::DW_AT_inline, 0, dwarf::DW_INL_inlined);
  }

  // Insert top level DIEs.
  for (SmallVector<DIE *, 4>::iterator TI = TopLevelDIEsVector.begin(),
         TE = TopLevelDIEsVector.end(); TI != TE; ++TI)
    ModuleCU->getCUDie()->addChild(*TI);

  for (DenseMap<DIE *, MDNode *>::iterator CI = ContainingTypeMap.begin(),
         CE = ContainingTypeMap.end(); CI != CE; ++CI) {
    DIE *SPDie = CI->first;
    MDNode *N = dyn_cast_or_null<MDNode>(CI->second);
    if (!N) continue;
    DIE *NDie = ModuleCU->getDIE(N);
    if (!NDie) continue;
    addDIEEntry(SPDie, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4, NDie);
    // FIXME - This is not the correct approach.
    // addDIEEntry(NDie, dwarf::DW_AT_containing_type, dwarf::DW_FORM_ref4, NDie);
  }

  // Standard sections final addresses.
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getTextSection());
  EmitLabel("text_end", 0);
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getDataSection());
  EmitLabel("data_end", 0);

  // End text sections.
  for (unsigned i = 1, N = SectionMap.size(); i <= N; ++i) {
    Asm->OutStreamer.SwitchSection(SectionMap[i]);
    EmitLabel("section_end", i);
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

  // Emit info into a debug str section.
  emitDebugStr();

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

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// findAbstractVariable - Find abstract variable, if any, associated with Var.
DbgVariable *DwarfDebug::findAbstractVariable(DIVariable &Var,
                                              unsigned FrameIdx,
                                              DILocation &ScopeLoc) {

  DbgVariable *AbsDbgVariable = AbstractVariables.lookup(Var.getNode());
  if (AbsDbgVariable)
    return AbsDbgVariable;

  DbgScope *Scope = AbstractScopes.lookup(ScopeLoc.getScope().getNode());
  if (!Scope)
    return NULL;

  AbsDbgVariable = new DbgVariable(Var, FrameIdx);
  Scope->addVariable(AbsDbgVariable);
  AbstractVariables[Var.getNode()] = AbsDbgVariable;
  return AbsDbgVariable;
}

/// collectVariableInfo - Populate DbgScope entries with variables' info.
void DwarfDebug::collectVariableInfo() {
  if (!MMI) return;

  MachineModuleInfo::VariableDbgInfoMapTy &VMap = MMI->getVariableDbgInfo();
  for (MachineModuleInfo::VariableDbgInfoMapTy::iterator VI = VMap.begin(),
         VE = VMap.end(); VI != VE; ++VI) {
    MetadataBase *MB = VI->first;
    MDNode *Var = dyn_cast_or_null<MDNode>(MB);
    if (!Var) continue;
    DIVariable DV (Var);
    std::pair< unsigned, MDNode *> VP = VI->second;
    DILocation ScopeLoc(VP.second);

    DbgScope *Scope =
      ConcreteScopes.lookup(ScopeLoc.getOrigLocation().getNode());
    if (!Scope)
      Scope = DbgScopeMap.lookup(ScopeLoc.getScope().getNode());
    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    DbgVariable *RegVar = new DbgVariable(DV, VP.first);
    Scope->addVariable(RegVar);
    if (DbgVariable *AbsDbgVariable = findAbstractVariable(DV, VP.first,
                                                           ScopeLoc))
      RegVar->setAbstractVariable(AbsDbgVariable);
  }
}

/// beginScope - Process beginning of a scope starting at Label.
void DwarfDebug::beginScope(const MachineInstr *MI, unsigned Label) {
  InsnToDbgScopeMapTy::iterator I = DbgScopeBeginMap.find(MI);
  if (I == DbgScopeBeginMap.end())
    return;
  ScopeVector &SD = I->second;
  for (ScopeVector::iterator SDI = SD.begin(), SDE = SD.end();
       SDI != SDE; ++SDI)
    (*SDI)->setStartLabelID(Label);
}

/// endScope - Process end of a scope.
void DwarfDebug::endScope(const MachineInstr *MI) {
  InsnToDbgScopeMapTy::iterator I = DbgScopeEndMap.find(MI);
  if (I == DbgScopeEndMap.end())
    return;

  unsigned Label = MMI->NextLabelID();
  Asm->printLabel(Label);
  O << '\n';

  SmallVector<DbgScope *, 2> &SD = I->second;
  for (SmallVector<DbgScope *, 2>::iterator SDI = SD.begin(), SDE = SD.end();
       SDI != SDE; ++SDI)
    (*SDI)->setEndLabelID(Label);
  return;
}

/// createDbgScope - Create DbgScope for the scope.
void DwarfDebug::createDbgScope(MDNode *Scope, MDNode *InlinedAt) {

  if (!InlinedAt) {
    DbgScope *WScope = DbgScopeMap.lookup(Scope);
    if (WScope)
      return;
    WScope = new DbgScope(NULL, DIDescriptor(Scope), NULL);
    DbgScopeMap.insert(std::make_pair(Scope, WScope));
    if (DIDescriptor(Scope).isLexicalBlock())
      createDbgScope(DILexicalBlock(Scope).getContext().getNode(), NULL);
    return;
  }

  DbgScope *WScope = DbgScopeMap.lookup(InlinedAt);
  if (WScope)
    return;

  WScope = new DbgScope(NULL, DIDescriptor(Scope), InlinedAt);
  DbgScopeMap.insert(std::make_pair(InlinedAt, WScope));
  DILocation DL(InlinedAt);
  createDbgScope(DL.getScope().getNode(), DL.getOrigLocation().getNode());
}

/// extractScopeInformation - Scan machine instructions in this function
/// and collect DbgScopes. Return true, if atleast one scope was found.
bool DwarfDebug::extractScopeInformation(MachineFunction *MF) {
  // If scope information was extracted using .dbg intrinsics then there is not
  // any need to extract these information by scanning each instruction.
  if (!DbgScopeMap.empty())
    return false;

  DenseMap<const MachineInstr *, unsigned> MIIndexMap;
  unsigned MIIndex = 0;
  // Scan each instruction and create scopes. First build working set of scopes.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;
      MIIndexMap[MInsn] = MIIndex++;
      DebugLoc DL = MInsn->getDebugLoc();
      if (DL.isUnknown()) continue;
      DILocation DLT = MF->getDILocation(DL);
      DIScope DLTScope = DLT.getScope();
      if (DLTScope.isNull()) continue;
      // There is no need to create another DIE for compile unit. For all
      // other scopes, create one DbgScope now. This will be translated
      // into a scope DIE at the end.
      if (DLTScope.isCompileUnit()) continue;
      createDbgScope(DLTScope.getNode(), DLT.getOrigLocation().getNode());
    }
  }


  // Build scope hierarchy using working set of scopes.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;
      DebugLoc DL = MInsn->getDebugLoc();
      if (DL.isUnknown())  continue;
      DILocation DLT = MF->getDILocation(DL);
      DIScope DLTScope = DLT.getScope();
      if (DLTScope.isNull())  continue;
      // There is no need to create another DIE for compile unit. For all
      // other scopes, create one DbgScope now. This will be translated
      // into a scope DIE at the end.
      if (DLTScope.isCompileUnit()) continue;
      DbgScope *Scope = getUpdatedDbgScope(DLTScope.getNode(), MInsn, 
                                           DLT.getOrigLocation().getNode());
      Scope->setLastInsn(MInsn);
    }
  }

  if (!CurrentFnDbgScope)
    return false;

  CurrentFnDbgScope->fixInstructionMarkers(MIIndexMap);

  // Each scope has first instruction and last instruction to mark beginning
  // and end of a scope respectively. Create an inverse map that list scopes
  // starts (and ends) with an instruction. One instruction may start (or end)
  // multiple scopes. Ignore scopes that are not reachable.
  SmallVector<DbgScope *, 4> WorkList;
  WorkList.push_back(CurrentFnDbgScope);
  while (!WorkList.empty()) {
    DbgScope *S = WorkList.back(); WorkList.pop_back();

    SmallVector<DbgScope *, 4> &Children = S->getScopes();
    if (!Children.empty()) 
      for (SmallVector<DbgScope *, 4>::iterator SI = Children.begin(),
             SE = Children.end(); SI != SE; ++SI)
        WorkList.push_back(*SI);

    if (S->isAbstractScope())
      continue;
    const MachineInstr *MI = S->getFirstInsn();
    assert (MI && "DbgScope does not have first instruction!");

    InsnToDbgScopeMapTy::iterator IDI = DbgScopeBeginMap.find(MI);
    if (IDI != DbgScopeBeginMap.end())
      IDI->second.push_back(S);
    else
      DbgScopeBeginMap[MI].push_back(S);

    MI = S->getLastInsn();
    assert (MI && "DbgScope does not have last instruction!");
    IDI = DbgScopeEndMap.find(MI);
    if (IDI != DbgScopeEndMap.end())
      IDI->second.push_back(S);
    else
      DbgScopeEndMap[MI].push_back(S);
  }

  return !DbgScopeMap.empty();
}

/// beginFunction - Gather pre-function debug information.  Assumes being
/// emitted immediately after the function entry point.
void DwarfDebug::beginFunction(MachineFunction *MF) {
  this->MF = MF;

  if (!ShouldEmitDwarfDebug()) return;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  if (!extractScopeInformation(MF))
    return;

  collectVariableInfo();

  // Begin accumulating function debug information.
  MMI->BeginFunction(MF);

  // Assumes in correct section after the entry point.
  EmitLabel("func_begin", ++SubprogramCount);

  // Emit label for the implicitly defined dbg.stoppoint at the start of the
  // function.
  DebugLoc FDL = MF->getDefaultDebugLoc();
  if (!FDL.isUnknown()) {
    DILocation DLT = MF->getDILocation(FDL);
    unsigned LabelID = 0;
    DISubprogram SP = getDISubprogram(DLT.getScope().getNode());
    if (!SP.isNull())
      LabelID = recordSourceLine(SP.getLineNumber(), 0, 
                                 DLT.getScope().getNode());
    else
      LabelID = recordSourceLine(DLT.getLineNumber(), 
                                 DLT.getColumnNumber(), 
                                 DLT.getScope().getNode());
    Asm->printLabel(LabelID);
    O << '\n';
  }
  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// endFunction - Gather and emit post-function debug information.
///
void DwarfDebug::endFunction(MachineFunction *MF) {
  if (!ShouldEmitDwarfDebug()) return;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  if (DbgScopeMap.empty())
    return;

  if (CurrentFnDbgScope) {
    // Define end label for subprogram.
    EmitLabel("func_end", SubprogramCount);
    
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
           AE = AbstractScopesList.end(); AI != AE; ++AI)
      constructScopeDIE(*AI);
    
    constructScopeDIE(CurrentFnDbgScope);
    
    DebugFrames.push_back(FunctionDebugFrameInfo(SubprogramCount,
                                                 MMI->getFrameMoves()));
  }

  // Clear debug info
  CurrentFnDbgScope = NULL;
  DbgScopeMap.clear();
  DbgScopeBeginMap.clear();
  DbgScopeEndMap.clear();
  ConcreteScopes.clear();
  AbstractScopesList.clear();
  Lines.clear();
  
  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// recordSourceLine - Records location information and associates it with a
/// label. Returns a unique label ID used to generate a label and provide
/// correspondence to the source line list.
unsigned DwarfDebug::recordSourceLine(unsigned Line, unsigned Col,
                                      MDNode *S) {
  if (!MMI)
    return 0;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  StringRef Dir;
  StringRef Fn;

  DIDescriptor Scope(S);
  if (Scope.isCompileUnit()) {
    DICompileUnit CU(S);
    Dir = CU.getDirectory();
    Fn = CU.getFilename();
  } else if (Scope.isSubprogram()) {
    DISubprogram SP(S);
    Dir = SP.getDirectory();
    Fn = SP.getFilename();
  } else if (Scope.isLexicalBlock()) {
    DILexicalBlock DB(S);
    Dir = DB.getDirectory();
    Fn = DB.getFilename();
  } else
    assert (0 && "Unexpected scope info");

  unsigned Src = GetOrCreateSourceID(Dir, Fn);
  unsigned ID = MMI->NextLabelID();
  Lines.push_back(SrcLineInfo(Line, Col, Src, ID));

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();

  return ID;
}

/// getOrCreateSourceID - Public version of GetOrCreateSourceID. This can be
/// timed. Look up the source id with the given directory and source file
/// names. If none currently exists, create a new id and insert it in the
/// SourceIds map. This can update DirectoryNames and SourceFileNames maps as
/// well.
unsigned DwarfDebug::getOrCreateSourceID(const std::string &DirName,
                                         const std::string &FileName) {
  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  unsigned SrcId = GetOrCreateSourceID(DirName.c_str(), FileName.c_str());

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();

  return SrcId;
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
  if (!Last && !Children.empty()) Die->addSiblingOffset();

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
    Offset += Values[i]->SizeOf(TD, AbbrevData[i].getForm());

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
  // Compute size of compile unit header.
  static unsigned Offset =
    sizeof(int32_t) + // Length of Compilation Unit Info
    sizeof(int16_t) + // DWARF version number
    sizeof(int32_t) + // Offset Into Abbrev. Section
    sizeof(int8_t);   // Pointer Size (in bytes)

  computeSizeAndOffset(ModuleCU->getCUDie(), Offset, true);
  CompileUnitOffsets[ModuleCU] = 0;
}

/// emitInitial - Emit initial Dwarf declarations.  This is necessary for cc
/// tools to recognize the object file contains Dwarf information.
void DwarfDebug::emitInitial() {
  // Check to see if we already emitted intial headers.
  if (didInitial) return;
  didInitial = true;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  // Dwarf sections base addresses.
  if (MAI->doesDwarfRequireFrameSection()) {
    Asm->OutStreamer.SwitchSection(TLOF.getDwarfFrameSection());
    EmitLabel("section_debug_frame", 0);
  }

  Asm->OutStreamer.SwitchSection(TLOF.getDwarfInfoSection());
  EmitLabel("section_info", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfAbbrevSection());
  EmitLabel("section_abbrev", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfARangesSection());
  EmitLabel("section_aranges", 0);

  if (const MCSection *LineInfoDirective = TLOF.getDwarfMacroInfoSection()) {
    Asm->OutStreamer.SwitchSection(LineInfoDirective);
    EmitLabel("section_macinfo", 0);
  }

  Asm->OutStreamer.SwitchSection(TLOF.getDwarfLineSection());
  EmitLabel("section_line", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfLocSection());
  EmitLabel("section_loc", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfPubNamesSection());
  EmitLabel("section_pubnames", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfPubTypesSection());
  EmitLabel("section_pubtypes", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfStrSection());
  EmitLabel("section_str", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfRangesSection());
  EmitLabel("section_ranges", 0);

  Asm->OutStreamer.SwitchSection(TLOF.getTextSection());
  EmitLabel("text_begin", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDataSection());
  EmitLabel("data_begin", 0);
}

/// emitDIE - Recusively Emits a debug information entry.
///
void DwarfDebug::emitDIE(DIE *Die) {
  // Get the abbreviation for this DIE.
  unsigned AbbrevNumber = Die->getAbbrevNumber();
  const DIEAbbrev *Abbrev = Abbreviations[AbbrevNumber - 1];

  Asm->EOL();

  // Emit the code (index) for the abbreviation.
  Asm->EmitULEB128Bytes(AbbrevNumber);

  Asm->EOL("Abbrev [" + Twine(AbbrevNumber) + "] 0x" +
           Twine::utohexstr(Die->getOffset()) + ":0x" +
           Twine::utohexstr(Die->getSize()) + " " +
           dwarf::TagString(Abbrev->getTag()));

  SmallVector<DIEValue*, 32> &Values = Die->getValues();
  const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev->getData();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    unsigned Attr = AbbrevData[i].getAttribute();
    unsigned Form = AbbrevData[i].getForm();
    assert(Form && "Too many attributes for DIE (check abbreviation)");

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
    default:
      // Emit an attribute using the defined form.
      Values[i]->EmitValue(this, Form);
      break;
    }

    Asm->EOL(dwarf::AttributeString(Attr));
  }

  // Emit the DIE children if any.
  if (Abbrev->getChildrenFlag() == dwarf::DW_CHILDREN_yes) {
    const std::vector<DIE *> &Children = Die->getChildren();

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      emitDIE(Children[j]);

    Asm->EmitInt8(0); Asm->EOL("End Of Children Mark");
  }
}

/// emitDebugInfo - Emit the debug info section.
///
void DwarfDebug::emitDebugInfo() {
  // Start debug info section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfInfoSection());
  DIE *Die = ModuleCU->getCUDie();

  // Emit the compile units header.
  EmitLabel("info_begin", ModuleCU->getID());

  // Emit size of content not including length itself
  unsigned ContentSize = Die->getSize() +
    sizeof(int16_t) + // DWARF version number
    sizeof(int32_t) + // Offset Into Abbrev. Section
    sizeof(int8_t) +  // Pointer Size (in bytes)
    sizeof(int32_t);  // FIXME - extra pad for gdb bug.

  Asm->EmitInt32(ContentSize);  Asm->EOL("Length of Compilation Unit Info");
  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("DWARF version number");
  EmitSectionOffset("abbrev_begin", "section_abbrev", 0, 0, true, false);
  Asm->EOL("Offset Into Abbrev. Section");
  Asm->EmitInt8(TD->getPointerSize()); Asm->EOL("Address Size (in bytes)");

  emitDIE(Die);
  // FIXME - extra padding for gdb bug.
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  EmitLabel("info_end", ModuleCU->getID());

  Asm->EOL();
}

/// emitAbbreviations - Emit the abbreviation section.
///
void DwarfDebug::emitAbbreviations() const {
  // Check to see if it is worth the effort.
  if (!Abbreviations.empty()) {
    // Start the debug abbrev section.
    Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfAbbrevSection());

    EmitLabel("abbrev_begin", 0);

    // For each abbrevation.
    for (unsigned i = 0, N = Abbreviations.size(); i < N; ++i) {
      // Get abbreviation data
      const DIEAbbrev *Abbrev = Abbreviations[i];

      // Emit the abbrevations code (base 1 index.)
      Asm->EmitULEB128Bytes(Abbrev->getNumber());
      Asm->EOL("Abbreviation Code");

      // Emit the abbreviations data.
      Abbrev->Emit(Asm);

      Asm->EOL();
    }

    // Mark end of abbreviations.
    Asm->EmitULEB128Bytes(0); Asm->EOL("EOM(3)");

    EmitLabel("abbrev_end", 0);
    Asm->EOL();
  }
}

/// emitEndOfLineMatrix - Emit the last address of the section and the end of
/// the line matrix.
///
void DwarfDebug::emitEndOfLineMatrix(unsigned SectionEnd) {
  // Define last address of section.
  Asm->EmitInt8(0); Asm->EOL("Extended Op");
  Asm->EmitInt8(TD->getPointerSize() + 1); Asm->EOL("Op size");
  Asm->EmitInt8(dwarf::DW_LNE_set_address); Asm->EOL("DW_LNE_set_address");
  EmitReference("section_end", SectionEnd); Asm->EOL("Section end label");

  // Mark end of matrix.
  Asm->EmitInt8(0); Asm->EOL("DW_LNE_end_sequence");
  Asm->EmitULEB128Bytes(1); Asm->EOL();
  Asm->EmitInt8(1); Asm->EOL();
}

/// emitDebugLines - Emit source line information.
///
void DwarfDebug::emitDebugLines() {
  // If the target is using .loc/.file, the assembler will be emitting the
  // .debug_line table automatically.
  if (MAI->hasDotLocAndDotFile())
    return;

  // Minimum line delta, thus ranging from -10..(255-10).
  const int MinLineDelta = -(dwarf::DW_LNS_fixed_advance_pc + 1);
  // Maximum line delta, thus ranging from -10..(255-10).
  const int MaxLineDelta = 255 + MinLineDelta;

  // Start the dwarf line section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfLineSection());

  // Construct the section header.
  EmitDifference("line_end", 0, "line_begin", 0, true);
  Asm->EOL("Length of Source Line Info");
  EmitLabel("line_begin", 0);

  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("DWARF version number");

  EmitDifference("line_prolog_end", 0, "line_prolog_begin", 0, true);
  Asm->EOL("Prolog Length");
  EmitLabel("line_prolog_begin", 0);

  Asm->EmitInt8(1); Asm->EOL("Minimum Instruction Length");

  Asm->EmitInt8(1); Asm->EOL("Default is_stmt_start flag");

  Asm->EmitInt8(MinLineDelta); Asm->EOL("Line Base Value (Special Opcodes)");

  Asm->EmitInt8(MaxLineDelta); Asm->EOL("Line Range Value (Special Opcodes)");

  Asm->EmitInt8(-MinLineDelta); Asm->EOL("Special Opcode Base");

  // Line number standard opcode encodings argument count
  Asm->EmitInt8(0); Asm->EOL("DW_LNS_copy arg count");
  Asm->EmitInt8(1); Asm->EOL("DW_LNS_advance_pc arg count");
  Asm->EmitInt8(1); Asm->EOL("DW_LNS_advance_line arg count");
  Asm->EmitInt8(1); Asm->EOL("DW_LNS_set_file arg count");
  Asm->EmitInt8(1); Asm->EOL("DW_LNS_set_column arg count");
  Asm->EmitInt8(0); Asm->EOL("DW_LNS_negate_stmt arg count");
  Asm->EmitInt8(0); Asm->EOL("DW_LNS_set_basic_block arg count");
  Asm->EmitInt8(0); Asm->EOL("DW_LNS_const_add_pc arg count");
  Asm->EmitInt8(1); Asm->EOL("DW_LNS_fixed_advance_pc arg count");

  // Emit directories.
  for (unsigned DI = 1, DE = getNumSourceDirectories()+1; DI != DE; ++DI) {
    Asm->EmitString(getSourceDirectoryName(DI));
    Asm->EOL("Directory");
  }

  Asm->EmitInt8(0); Asm->EOL("End of directories");

  // Emit files.
  for (unsigned SI = 1, SE = getNumSourceIds()+1; SI != SE; ++SI) {
    // Remember source id starts at 1.
    std::pair<unsigned, unsigned> Id = getSourceDirectoryAndFileIds(SI);
    Asm->EmitString(getSourceFileName(Id.second));
    Asm->EOL("Source");
    Asm->EmitULEB128Bytes(Id.first);
    Asm->EOL("Directory #");
    Asm->EmitULEB128Bytes(0);
    Asm->EOL("Mod date");
    Asm->EmitULEB128Bytes(0);
    Asm->EOL("File size");
  }

  Asm->EmitInt8(0); Asm->EOL("End of files");

  EmitLabel("line_prolog_end", 0);

  // A sequence for each text section.
  unsigned SecSrcLinesSize = SectionSourceLines.size();

  for (unsigned j = 0; j < SecSrcLinesSize; ++j) {
    // Isolate current sections line info.
    const std::vector<SrcLineInfo> &LineInfos = SectionSourceLines[j];

    /*if (Asm->isVerbose()) {
      const MCSection *S = SectionMap[j + 1];
      O << '\t' << MAI->getCommentString() << " Section"
        << S->getName() << '\n';
    }*/
    Asm->EOL();

    // Dwarf assumes we start with first line of first source file.
    unsigned Source = 1;
    unsigned Line = 1;

    // Construct rows of the address, source, line, column matrix.
    for (unsigned i = 0, N = LineInfos.size(); i < N; ++i) {
      const SrcLineInfo &LineInfo = LineInfos[i];
      unsigned LabelID = MMI->MappedLabel(LineInfo.getLabelID());
      if (!LabelID) continue;

      if (LineInfo.getLine() == 0) continue;

      if (!Asm->isVerbose())
        Asm->EOL();
      else {
        std::pair<unsigned, unsigned> SourceID =
          getSourceDirectoryAndFileIds(LineInfo.getSourceID());
        O << '\t' << MAI->getCommentString() << ' '
          << getSourceDirectoryName(SourceID.first) << '/'
          << getSourceFileName(SourceID.second)
          << ':' << utostr_32(LineInfo.getLine()) << '\n';
      }

      // Define the line address.
      Asm->EmitInt8(0); Asm->EOL("Extended Op");
      Asm->EmitInt8(TD->getPointerSize() + 1); Asm->EOL("Op size");
      Asm->EmitInt8(dwarf::DW_LNE_set_address); Asm->EOL("DW_LNE_set_address");
      EmitReference("label",  LabelID); Asm->EOL("Location label");

      // If change of source, then switch to the new source.
      if (Source != LineInfo.getSourceID()) {
        Source = LineInfo.getSourceID();
        Asm->EmitInt8(dwarf::DW_LNS_set_file); Asm->EOL("DW_LNS_set_file");
        Asm->EmitULEB128Bytes(Source); Asm->EOL("New Source");
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
          Asm->EmitInt8(Delta - MinLineDelta); Asm->EOL("Line Delta");
        } else {
          // ... otherwise use long hand.
          Asm->EmitInt8(dwarf::DW_LNS_advance_line);
          Asm->EOL("DW_LNS_advance_line");
          Asm->EmitSLEB128Bytes(Offset); Asm->EOL("Line Offset");
          Asm->EmitInt8(dwarf::DW_LNS_copy); Asm->EOL("DW_LNS_copy");
        }
      } else {
        // Copy the previous row (different address or source)
        Asm->EmitInt8(dwarf::DW_LNS_copy); Asm->EOL("DW_LNS_copy");
      }
    }

    emitEndOfLineMatrix(j + 1);
  }

  if (SecSrcLinesSize == 0)
    // Because we're emitting a debug_line section, we still need a line
    // table. The linker and friends expect it to exist. If there's nothing to
    // put into it, emit an empty table.
    emitEndOfLineMatrix(1);

  EmitLabel("line_end", 0);
  Asm->EOL();
}

/// emitCommonDebugFrame - Emit common frame info into a debug frame section.
///
void DwarfDebug::emitCommonDebugFrame() {
  if (!MAI->doesDwarfRequireFrameSection())
    return;

  int stackGrowth =
    Asm->TM.getFrameInfo()->getStackGrowthDirection() ==
      TargetFrameInfo::StackGrowsUp ?
    TD->getPointerSize() : -TD->getPointerSize();

  // Start the dwarf frame section.
  Asm->OutStreamer.SwitchSection(
                              Asm->getObjFileLowering().getDwarfFrameSection());

  EmitLabel("debug_frame_common", 0);
  EmitDifference("debug_frame_common_end", 0,
                 "debug_frame_common_begin", 0, true);
  Asm->EOL("Length of Common Information Entry");

  EmitLabel("debug_frame_common_begin", 0);
  Asm->EmitInt32((int)dwarf::DW_CIE_ID);
  Asm->EOL("CIE Identifier Tag");
  Asm->EmitInt8(dwarf::DW_CIE_VERSION);
  Asm->EOL("CIE Version");
  Asm->EmitString("");
  Asm->EOL("CIE Augmentation");
  Asm->EmitULEB128Bytes(1);
  Asm->EOL("CIE Code Alignment Factor");
  Asm->EmitSLEB128Bytes(stackGrowth);
  Asm->EOL("CIE Data Alignment Factor");
  Asm->EmitInt8(RI->getDwarfRegNum(RI->getRARegister(), false));
  Asm->EOL("CIE RA Column");

  std::vector<MachineMove> Moves;
  RI->getInitialFrameState(Moves);

  EmitFrameMoves(NULL, 0, Moves, false);

  Asm->EmitAlignment(2, 0, 0, false);
  EmitLabel("debug_frame_common_end", 0);

  Asm->EOL();
}

/// emitFunctionDebugFrame - Emit per function frame info into a debug frame
/// section.
void
DwarfDebug::emitFunctionDebugFrame(const FunctionDebugFrameInfo&DebugFrameInfo){
  if (!MAI->doesDwarfRequireFrameSection())
    return;

  // Start the dwarf frame section.
  Asm->OutStreamer.SwitchSection(
                              Asm->getObjFileLowering().getDwarfFrameSection());

  EmitDifference("debug_frame_end", DebugFrameInfo.Number,
                 "debug_frame_begin", DebugFrameInfo.Number, true);
  Asm->EOL("Length of Frame Information Entry");

  EmitLabel("debug_frame_begin", DebugFrameInfo.Number);

  EmitSectionOffset("debug_frame_common", "section_debug_frame",
                    0, 0, true, false);
  Asm->EOL("FDE CIE offset");

  EmitReference("func_begin", DebugFrameInfo.Number);
  Asm->EOL("FDE initial location");
  EmitDifference("func_end", DebugFrameInfo.Number,
                 "func_begin", DebugFrameInfo.Number);
  Asm->EOL("FDE address range");

  EmitFrameMoves("func_begin", DebugFrameInfo.Number, DebugFrameInfo.Moves,
                 false);

  Asm->EmitAlignment(2, 0, 0, false);
  EmitLabel("debug_frame_end", DebugFrameInfo.Number);

  Asm->EOL();
}

/// emitDebugPubNames - Emit visible names into a debug pubnames section.
///
void DwarfDebug::emitDebugPubNames() {
  // Start the dwarf pubnames section.
  Asm->OutStreamer.SwitchSection(
                          Asm->getObjFileLowering().getDwarfPubNamesSection());

  EmitDifference("pubnames_end", ModuleCU->getID(),
                 "pubnames_begin", ModuleCU->getID(), true);
  Asm->EOL("Length of Public Names Info");

  EmitLabel("pubnames_begin", ModuleCU->getID());

  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("DWARF Version");

  EmitSectionOffset("info_begin", "section_info",
                    ModuleCU->getID(), 0, true, false);
  Asm->EOL("Offset of Compilation Unit Info");

  EmitDifference("info_end", ModuleCU->getID(), "info_begin", ModuleCU->getID(),
                 true);
  Asm->EOL("Compilation Unit Length");

  const StringMap<DIE*> &Globals = ModuleCU->getGlobals();
  for (StringMap<DIE*>::const_iterator
         GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
    const char *Name = GI->getKeyData();
    DIE * Entity = GI->second;

    Asm->EmitInt32(Entity->getOffset()); Asm->EOL("DIE offset");
    Asm->EmitString(Name, strlen(Name)); Asm->EOL("External Name");
  }

  Asm->EmitInt32(0); Asm->EOL("End Mark");
  EmitLabel("pubnames_end", ModuleCU->getID());

  Asm->EOL();
}

void DwarfDebug::emitDebugPubTypes() {
  // Start the dwarf pubnames section.
  Asm->OutStreamer.SwitchSection(
                          Asm->getObjFileLowering().getDwarfPubTypesSection());
  EmitDifference("pubtypes_end", ModuleCU->getID(),
                 "pubtypes_begin", ModuleCU->getID(), true);
  Asm->EOL("Length of Public Types Info");

  EmitLabel("pubtypes_begin", ModuleCU->getID());

  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("DWARF Version");

  EmitSectionOffset("info_begin", "section_info",
                    ModuleCU->getID(), 0, true, false);
  Asm->EOL("Offset of Compilation ModuleCU Info");

  EmitDifference("info_end", ModuleCU->getID(), "info_begin", ModuleCU->getID(),
                 true);
  Asm->EOL("Compilation ModuleCU Length");

  const StringMap<DIE*> &Globals = ModuleCU->getGlobalTypes();
  for (StringMap<DIE*>::const_iterator
         GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
    const char *Name = GI->getKeyData();
    DIE * Entity = GI->second;

    Asm->EmitInt32(Entity->getOffset()); Asm->EOL("DIE offset");
    Asm->EmitString(Name, strlen(Name)); Asm->EOL("External Name");
  }

  Asm->EmitInt32(0); Asm->EOL("End Mark");
  EmitLabel("pubtypes_end", ModuleCU->getID());

  Asm->EOL();
}

/// emitDebugStr - Emit visible names into a debug str section.
///
void DwarfDebug::emitDebugStr() {
  // Check to see if it is worth the effort.
  if (!StringPool.empty()) {
    // Start the dwarf str section.
    Asm->OutStreamer.SwitchSection(
                                Asm->getObjFileLowering().getDwarfStrSection());

    // For each of strings in the string pool.
    for (unsigned StringID = 1, N = StringPool.size();
         StringID <= N; ++StringID) {
      // Emit a label for reference from debug information entries.
      EmitLabel("string", StringID);

      // Emit the string itself.
      const std::string &String = StringPool[StringID];
      Asm->EmitString(String); Asm->EOL();
    }

    Asm->EOL();
  }
}

/// emitDebugLoc - Emit visible names into a debug loc section.
///
void DwarfDebug::emitDebugLoc() {
  // Start the dwarf loc section.
  Asm->OutStreamer.SwitchSection(
                              Asm->getObjFileLowering().getDwarfLocSection());
  Asm->EOL();
}

/// EmitDebugARanges - Emit visible names into a debug aranges section.
///
void DwarfDebug::EmitDebugARanges() {
  // Start the dwarf aranges section.
  Asm->OutStreamer.SwitchSection(
                          Asm->getObjFileLowering().getDwarfARangesSection());

  // FIXME - Mock up
#if 0
  CompileUnit *Unit = GetBaseCompileUnit();

  // Don't include size of length
  Asm->EmitInt32(0x1c); Asm->EOL("Length of Address Ranges Info");

  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("Dwarf Version");

  EmitReference("info_begin", Unit->getID());
  Asm->EOL("Offset of Compilation Unit Info");

  Asm->EmitInt8(TD->getPointerSize()); Asm->EOL("Size of Address");

  Asm->EmitInt8(0); Asm->EOL("Size of Segment Descriptor");

  Asm->EmitInt16(0);  Asm->EOL("Pad (1)");
  Asm->EmitInt16(0);  Asm->EOL("Pad (2)");

  // Range 1
  EmitReference("text_begin", 0); Asm->EOL("Address");
  EmitDifference("text_end", 0, "text_begin", 0, true); Asm->EOL("Length");

  Asm->EmitInt32(0); Asm->EOL("EOM (1)");
  Asm->EmitInt32(0); Asm->EOL("EOM (2)");
#endif

  Asm->EOL();
}

/// emitDebugRanges - Emit visible names into a debug ranges section.
///
void DwarfDebug::emitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfRangesSection());
  Asm->EOL();
}

/// emitDebugMacInfo - Emit visible names into a debug macinfo section.
///
void DwarfDebug::emitDebugMacInfo() {
  if (const MCSection *LineInfo =
      Asm->getObjFileLowering().getDwarfMacroInfoSection()) {
    // Start the dwarf macinfo section.
    Asm->OutStreamer.SwitchSection(LineInfo);
    Asm->EOL();
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
  if (!MAI->doesDwarfUsesInlineInfoSection())
    return;

  if (!ModuleCU)
    return;

  Asm->OutStreamer.SwitchSection(
                        Asm->getObjFileLowering().getDwarfDebugInlineSection());
  Asm->EOL();
  EmitDifference("debug_inlined_end", 1,
                 "debug_inlined_begin", 1, true);
  Asm->EOL("Length of Debug Inlined Information Entry");

  EmitLabel("debug_inlined_begin", 1);

  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("Dwarf Version");
  Asm->EmitInt8(TD->getPointerSize()); Asm->EOL("Address Size (in bytes)");

  for (SmallVector<MDNode *, 4>::iterator I = InlinedSPNodes.begin(),
         E = InlinedSPNodes.end(); I != E; ++I) {

    MDNode *Node = *I;
    DenseMap<MDNode *, SmallVector<InlineInfoLabels, 4> >::iterator II
      = InlineInfo.find(Node);
    SmallVector<InlineInfoLabels, 4> &Labels = II->second;
    DISubprogram SP(Node);
    StringRef LName = SP.getLinkageName();
    StringRef Name = SP.getName();

    if (LName.empty())
      Asm->EmitString(Name);
    else 
      EmitSectionOffset("string", "section_str",
                        StringPool.idFor(getRealLinkageName(LName)), false, true);

    Asm->EOL("MIPS linkage name");
    EmitSectionOffset("string", "section_str",
                      StringPool.idFor(Name), false, true);
    Asm->EOL("Function name");
    Asm->EmitULEB128Bytes(Labels.size()); Asm->EOL("Inline count");

    for (SmallVector<InlineInfoLabels, 4>::iterator LI = Labels.begin(),
           LE = Labels.end(); LI != LE; ++LI) {
      DIE *SP = LI->second;
      Asm->EmitInt32(SP->getOffset()); Asm->EOL("DIE offset");

      if (TD->getPointerSize() == sizeof(int32_t))
        O << MAI->getData32bitsDirective();
      else
        O << MAI->getData64bitsDirective();

      PrintLabelName("label", LI->first); Asm->EOL("low_pc");
    }
  }

  EmitLabel("debug_inlined_end", 1);
  Asm->EOL();
}
