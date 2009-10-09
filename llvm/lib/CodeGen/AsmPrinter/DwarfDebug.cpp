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
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/Debug.h"
#include "llvm/System/Path.h"
using namespace llvm;

static TimerGroup &getDwarfTimerGroup() {
  static TimerGroup DwarfTimerGroup("Dwarf Debugging");
  return DwarfTimerGroup;
}

//===----------------------------------------------------------------------===//

/// Configuration values for initial hash set sizes (log2).
///
static const unsigned InitDiesSetSize          = 9; // log2(512)
static const unsigned InitAbbreviationsSetSize = 9; // log2(512)
static const unsigned InitValuesSetSize        = 9; // log2(512)

namespace llvm {

//===----------------------------------------------------------------------===//
/// CompileUnit - This dwarf writer support class manages information associate
/// with a source file.
class VISIBILITY_HIDDEN CompileUnit {
  /// ID - File identifier for source.
  ///
  unsigned ID;

  /// Die - Compile unit debug information entry.
  ///
  DIE *Die;

  /// GVToDieMap - Tracks the mapping of unit level debug informaton
  /// variables to debug information entries.
  /// FIXME : Rename GVToDieMap -> NodeToDieMap
  std::map<MDNode *, DIE *> GVToDieMap;

  /// GVToDIEEntryMap - Tracks the mapping of unit level debug informaton
  /// descriptors to debug information entries using a DIEEntry proxy.
  /// FIXME : Rename
  std::map<MDNode *, DIEEntry *> GVToDIEEntryMap;

  /// Globals - A map of globally visible named entities for this unit.
  ///
  StringMap<DIE*> Globals;

  /// DiesSet - Used to uniquely define dies within the compile unit.
  ///
  FoldingSet<DIE> DiesSet;
public:
  CompileUnit(unsigned I, DIE *D)
    : ID(I), Die(D), DiesSet(InitDiesSetSize) {}
  ~CompileUnit() { delete Die; }

  // Accessors.
  unsigned getID() const { return ID; }
  DIE* getDie() const { return Die; }
  StringMap<DIE*> &getGlobals() { return Globals; }

  /// hasContent - Return true if this compile unit has something to write out.
  ///
  bool hasContent() const { return !Die->getChildren().empty(); }

  /// AddGlobal - Add a new global entity to the compile unit.
  ///
  void AddGlobal(const std::string &Name, DIE *Die) { Globals[Name] = Die; }

  /// getDieMapSlotFor - Returns the debug information entry map slot for the
  /// specified debug variable.
  DIE *&getDieMapSlotFor(MDNode *N) { return GVToDieMap[N]; }

  /// getDIEEntrySlotFor - Returns the debug information entry proxy slot for
  /// the specified debug variable.
  DIEEntry *&getDIEEntrySlotFor(MDNode *N) {
    return GVToDIEEntryMap[N];
  }

  /// AddDie - Adds or interns the DIE to the compile unit.
  ///
  DIE *AddDie(DIE &Buffer) {
    FoldingSetNodeID ID;
    Buffer.Profile(ID);
    void *Where;
    DIE *Die = DiesSet.FindNodeOrInsertPos(ID, Where);

    if (!Die) {
      Die = new DIE(Buffer);
      DiesSet.InsertNode(Die, Where);
      this->Die->AddChild(Die);
      Buffer.Detach();
    }

    return Die;
  }
};

//===----------------------------------------------------------------------===//
/// DbgVariable - This class is used to track local variable information.
///
class VISIBILITY_HIDDEN DbgVariable {
  DIVariable Var;                    // Variable Descriptor.
  unsigned FrameIndex;               // Variable frame index.
  bool InlinedFnVar;                 // Variable for an inlined function.
public:
  DbgVariable(DIVariable V, unsigned I, bool IFV)
    : Var(V), FrameIndex(I), InlinedFnVar(IFV)  {}

  // Accessors.
  DIVariable getVariable() const { return Var; }
  unsigned getFrameIndex() const { return FrameIndex; }
  bool isInlinedFnVar() const { return InlinedFnVar; }
};

//===----------------------------------------------------------------------===//
/// DbgScope - This class is used to track scope information.
///
class DbgConcreteScope;
class VISIBILITY_HIDDEN DbgScope {
  DbgScope *Parent;                   // Parent to this scope.
  DIDescriptor Desc;                  // Debug info descriptor for scope.
                                      // Either subprogram or block.
  unsigned StartLabelID;              // Label ID of the beginning of scope.
  unsigned EndLabelID;                // Label ID of the end of scope.
  const MachineInstr *LastInsn;       // Last instruction of this scope.
  const MachineInstr *FirstInsn;      // First instruction of this scope.
  SmallVector<DbgScope *, 4> Scopes;  // Scopes defined in scope.
  SmallVector<DbgVariable *, 8> Variables;// Variables declared in scope.
  SmallVector<DbgConcreteScope *, 8> ConcreteInsts;// Concrete insts of funcs.

  // Private state for dump()
  mutable unsigned IndentLevel;
public:
  DbgScope(DbgScope *P, DIDescriptor D)
    : Parent(P), Desc(D), StartLabelID(0), EndLabelID(0), LastInsn(0),
      FirstInsn(0), IndentLevel(0) {}
  virtual ~DbgScope();

  // Accessors.
  DbgScope *getParent()          const { return Parent; }
  DIDescriptor getDesc()         const { return Desc; }
  unsigned getStartLabelID()     const { return StartLabelID; }
  unsigned getEndLabelID()       const { return EndLabelID; }
  SmallVector<DbgScope *, 4> &getScopes() { return Scopes; }
  SmallVector<DbgVariable *, 8> &getVariables() { return Variables; }
  SmallVector<DbgConcreteScope*,8> &getConcreteInsts() { return ConcreteInsts; }
  void setStartLabelID(unsigned S) { StartLabelID = S; }
  void setEndLabelID(unsigned E)   { EndLabelID = E; }
  void setLastInsn(const MachineInstr *MI) { LastInsn = MI; }
  const MachineInstr *getLastInsn()      { return LastInsn; }
  void setFirstInsn(const MachineInstr *MI) { FirstInsn = MI; }
  const MachineInstr *getFirstInsn()      { return FirstInsn; }
  /// AddScope - Add a scope to the scope.
  ///
  void AddScope(DbgScope *S) { Scopes.push_back(S); }

  /// AddVariable - Add a variable to the scope.
  ///
  void AddVariable(DbgVariable *V) { Variables.push_back(V); }

  /// AddConcreteInst - Add a concrete instance to the scope.
  ///
  void AddConcreteInst(DbgConcreteScope *C) { ConcreteInsts.push_back(C); }

  void FixInstructionMarkers() {
    assert (getFirstInsn() && "First instruction is missing!");
    if (getLastInsn())
      return;
    
    // If a scope does not have an instruction to mark an end then use
    // the end of last child scope.
    SmallVector<DbgScope *, 4> &Scopes = getScopes();
    assert (!Scopes.empty() && "Inner most scope does not have last insn!");
    DbgScope *L = Scopes.back();
    if (!L->getLastInsn())
      L->FixInstructionMarkers();
    setLastInsn(L->getLastInsn());
  }

#ifndef NDEBUG
  void dump() const;
#endif
};

#ifndef NDEBUG
void DbgScope::dump() const {
  raw_ostream &err = errs();
  err.indent(IndentLevel);
  Desc.dump();
  err << " [" << StartLabelID << ", " << EndLabelID << "]\n";

  IndentLevel += 2;

  for (unsigned i = 0, e = Scopes.size(); i != e; ++i)
    if (Scopes[i] != this)
      Scopes[i]->dump();

  IndentLevel -= 2;
}
#endif

//===----------------------------------------------------------------------===//
/// DbgConcreteScope - This class is used to track a scope that holds concrete
/// instance information.
///
class VISIBILITY_HIDDEN DbgConcreteScope : public DbgScope {
  CompileUnit *Unit;
  DIE *Die;                           // Debug info for this concrete scope.
public:
  DbgConcreteScope(DIDescriptor D) : DbgScope(NULL, D) {}

  // Accessors.
  DIE *getDie() const { return Die; }
  void setDie(DIE *D) { Die = D; }
};

DbgScope::~DbgScope() {
  for (unsigned i = 0, N = Scopes.size(); i < N; ++i)
    delete Scopes[i];
  for (unsigned j = 0, M = Variables.size(); j < M; ++j)
    delete Variables[j];
  for (unsigned k = 0, O = ConcreteInsts.size(); k < O; ++k)
    delete ConcreteInsts[k];
}

} // end llvm namespace

DwarfDebug::DwarfDebug(raw_ostream &OS, AsmPrinter *A, const MCAsmInfo *T)
  : Dwarf(OS, A, T, "dbg"), ModuleCU(0),
    AbbreviationsSet(InitAbbreviationsSetSize), Abbreviations(),
    ValuesSet(InitValuesSetSize), Values(), StringPool(),
    SectionSourceLines(), didInitial(false), shouldEmit(false),
    FunctionDbgScope(0), DebugTimer(0) {
  if (TimePassesIsEnabled)
    DebugTimer = new Timer("Dwarf Debug Writer",
                           getDwarfTimerGroup());
}
DwarfDebug::~DwarfDebug() {
  for (unsigned j = 0, M = Values.size(); j < M; ++j)
    delete Values[j];

  for (DenseMap<const MDNode *, DbgScope *>::iterator
         I = AbstractInstanceRootMap.begin(),
         E = AbstractInstanceRootMap.end(); I != E;++I)
    delete I->second;

  delete DebugTimer;
}

/// AssignAbbrevNumber - Define a unique number for the abbreviation.
///
void DwarfDebug::AssignAbbrevNumber(DIEAbbrev &Abbrev) {
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

/// CreateDIEEntry - Creates a new DIEEntry to be a proxy for a debug
/// information entry.
DIEEntry *DwarfDebug::CreateDIEEntry(DIE *Entry) {
  DIEEntry *Value;

  if (Entry) {
    FoldingSetNodeID ID;
    DIEEntry::Profile(ID, Entry);
    void *Where;
    Value = static_cast<DIEEntry *>(ValuesSet.FindNodeOrInsertPos(ID, Where));

    if (Value) return Value;

    Value = new DIEEntry(Entry);
    ValuesSet.InsertNode(Value, Where);
  } else {
    Value = new DIEEntry(Entry);
  }

  Values.push_back(Value);
  return Value;
}

/// SetDIEEntry - Set a DIEEntry once the debug information entry is defined.
///
void DwarfDebug::SetDIEEntry(DIEEntry *Value, DIE *Entry) {
  Value->setEntry(Entry);

  // Add to values set if not already there.  If it is, we merely have a
  // duplicate in the values list (no harm.)
  ValuesSet.GetOrInsertNode(Value);
}

/// AddUInt - Add an unsigned integer attribute data and value.
///
void DwarfDebug::AddUInt(DIE *Die, unsigned Attribute,
                         unsigned Form, uint64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(false, Integer);

  FoldingSetNodeID ID;
  DIEInteger::Profile(ID, Integer);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIEInteger(Integer);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddSInt - Add an signed integer attribute data and value.
///
void DwarfDebug::AddSInt(DIE *Die, unsigned Attribute,
                         unsigned Form, int64_t Integer) {
  if (!Form) Form = DIEInteger::BestForm(true, Integer);

  FoldingSetNodeID ID;
  DIEInteger::Profile(ID, (uint64_t)Integer);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIEInteger(Integer);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddString - Add a string attribute data and value.
///
void DwarfDebug::AddString(DIE *Die, unsigned Attribute, unsigned Form,
                           const std::string &String) {
  FoldingSetNodeID ID;
  DIEString::Profile(ID, String);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIEString(String);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddLabel - Add a Dwarf label attribute data and value.
///
void DwarfDebug::AddLabel(DIE *Die, unsigned Attribute, unsigned Form,
                          const DWLabel &Label) {
  FoldingSetNodeID ID;
  DIEDwarfLabel::Profile(ID, Label);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIEDwarfLabel(Label);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddObjectLabel - Add an non-Dwarf label attribute data and value.
///
void DwarfDebug::AddObjectLabel(DIE *Die, unsigned Attribute, unsigned Form,
                                const std::string &Label) {
  FoldingSetNodeID ID;
  DIEObjectLabel::Profile(ID, Label);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIEObjectLabel(Label);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddSectionOffset - Add a section offset label attribute data and value.
///
void DwarfDebug::AddSectionOffset(DIE *Die, unsigned Attribute, unsigned Form,
                                  const DWLabel &Label, const DWLabel &Section,
                                  bool isEH, bool useSet) {
  FoldingSetNodeID ID;
  DIESectionOffset::Profile(ID, Label, Section);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIESectionOffset(Label, Section, isEH, useSet);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddDelta - Add a label delta attribute data and value.
///
void DwarfDebug::AddDelta(DIE *Die, unsigned Attribute, unsigned Form,
                          const DWLabel &Hi, const DWLabel &Lo) {
  FoldingSetNodeID ID;
  DIEDelta::Profile(ID, Hi, Lo);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = new DIEDelta(Hi, Lo);
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  }

  Die->AddValue(Attribute, Form, Value);
}

/// AddBlock - Add block data.
///
void DwarfDebug::AddBlock(DIE *Die, unsigned Attribute, unsigned Form,
                          DIEBlock *Block) {
  Block->ComputeSize(TD);
  FoldingSetNodeID ID;
  Block->Profile(ID);
  void *Where;
  DIEValue *Value = ValuesSet.FindNodeOrInsertPos(ID, Where);

  if (!Value) {
    Value = Block;
    ValuesSet.InsertNode(Value, Where);
    Values.push_back(Value);
  } else {
    // Already exists, reuse the previous one.
    delete Block;
    Block = cast<DIEBlock>(Value);
  }

  Die->AddValue(Attribute, Block->BestForm(), Value);
}

/// AddSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::AddSourceLine(DIE *Die, const DIVariable *V) {
  // If there is no compile unit specified, don't add a line #.
  if (V->getCompileUnit().isNull())
    return;

  unsigned Line = V->getLineNumber();
  unsigned FileID = FindCompileUnit(V->getCompileUnit()).getID();
  assert(FileID && "Invalid file id");
  AddUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  AddUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// AddSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::AddSourceLine(DIE *Die, const DIGlobal *G) {
  // If there is no compile unit specified, don't add a line #.
  if (G->getCompileUnit().isNull())
    return;

  unsigned Line = G->getLineNumber();
  unsigned FileID = FindCompileUnit(G->getCompileUnit()).getID();
  assert(FileID && "Invalid file id");
  AddUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  AddUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// AddSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::AddSourceLine(DIE *Die, const DISubprogram *SP) {
  // If there is no compile unit specified, don't add a line #.
  if (SP->getCompileUnit().isNull())
    return;
  // If the line number is 0, don't add it.
  if (SP->getLineNumber() == 0)
    return;


  unsigned Line = SP->getLineNumber();
  unsigned FileID = FindCompileUnit(SP->getCompileUnit()).getID();
  assert(FileID && "Invalid file id");
  AddUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  AddUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
}

/// AddSourceLine - Add location information to specified debug information
/// entry.
void DwarfDebug::AddSourceLine(DIE *Die, const DIType *Ty) {
  // If there is no compile unit specified, don't add a line #.
  DICompileUnit CU = Ty->getCompileUnit();
  if (CU.isNull())
    return;

  unsigned Line = Ty->getLineNumber();
  unsigned FileID = FindCompileUnit(CU).getID();
  assert(FileID && "Invalid file id");
  AddUInt(Die, dwarf::DW_AT_decl_file, 0, FileID);
  AddUInt(Die, dwarf::DW_AT_decl_line, 0, Line);
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
   value of the variable.  The function AddBlockByrefType does this.  */

/// Find the type the programmer originally declared the variable to be
/// and return that type.
///
DIType DwarfDebug::GetBlockByrefType(DIType Ty, std::string Name) {

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
    if (strcmp(Name.c_str(), DT.getName()) == 0)
      return (DT.getTypeDerivedFrom());
  }

  return Ty;
}

/// AddComplexAddress - Start with the address based on the location provided,
/// and generate the DWARF information necessary to find the actual variable
/// given the extra address information encoded in the DIVariable, starting from
/// the starting location.  Add the DWARF information to the die.
///
void DwarfDebug::AddComplexAddress(DbgVariable *&DV, DIE *Die,
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
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    } else {
      Reg = Reg - dwarf::DW_OP_reg0;
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32)
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    else {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    AddUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  for (unsigned i = 0, N = VD.getNumAddrElements(); i < N; ++i) {
    uint64_t Element = VD.getAddrElement(i);

    if (Element == DIFactory::OpPlus) {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, VD.getAddrElement(++i));
    } else if (Element == DIFactory::OpDeref) {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    } else llvm_unreachable("unknown DIFactory Opcode");
  }

  // Now attach the location information to the DIE.
  AddBlock(Die, Attribute, 0, Block);
}

/* Byref variables, in Blocks, are declared by the programmer as "SomeType
   VarName;", but the compiler creates a __Block_byref_x_VarName struct, and
   gives the variable VarName either the struct, or a pointer to the struct, as
   its type.  This is necessary for various behind-the-scenes things the
   compiler needs to do with by-reference variables in Blocks.

   However, as far as the original *programmer* is concerned, the variable
   should still have type 'SomeType', as originally declared.

   The function GetBlockByrefType dives into the __Block_byref_x_VarName
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

/// AddBlockByrefAddress - Start with the address based on the location
/// provided, and generate the DWARF information necessary to find the
/// actual Block variable (navigating the Block struct) based on the
/// starting location.  Add the DWARF information to the die.  For
/// more information, read large comment just above here.
///
void DwarfDebug::AddBlockByrefAddress(DbgVariable *&DV, DIE *Die,
                                      unsigned Attribute,
                                      const MachineLocation &Location) {
  const DIVariable &VD = DV->getVariable();
  DIType Ty = VD.getType();
  DIType TmpTy = Ty;
  unsigned Tag = Ty.getTag();
  bool isPointer = false;

  const char *varName = VD.getName();

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
    const char *fieldName = DT.getName();
    if (strcmp(fieldName, "__forwarding") == 0)
      forwardingField = Element;
    else if (strcmp(fieldName, varName) == 0)
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
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    else {
      Reg = Reg - dwarf::DW_OP_reg0;
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32)
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    else {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    AddUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  // If we started with a pointer to the __Block_byref... struct, then
  // the first thing we need to do is dereference the pointer (DW_OP_deref).
  if (isPointer)
    AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Next add the offset for the '__forwarding' field:
  // DW_OP_plus_uconst ForwardingFieldOffset.  Note there's no point in
  // adding the offset if it's 0.
  if (forwardingFieldOffset > 0) {
    AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    AddUInt(Block, 0, dwarf::DW_FORM_udata, forwardingFieldOffset);
  }

  // Now dereference the __forwarding field to get to the real __Block_byref
  // struct:  DW_OP_deref.
  AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Now that we've got the real __Block_byref... struct, add the offset
  // for the variable's field to get to the location of the actual variable:
  // DW_OP_plus_uconst varFieldOffset.  Again, don't add if it's 0.
  if (varFieldOffset > 0) {
    AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    AddUInt(Block, 0, dwarf::DW_FORM_udata, varFieldOffset);
  }

  // Now attach the location information to the DIE.
  AddBlock(Die, Attribute, 0, Block);
}

/// AddAddress - Add an address attribute to a die based on the location
/// provided.
void DwarfDebug::AddAddress(DIE *Die, unsigned Attribute,
                            const MachineLocation &Location) {
  unsigned Reg = RI->getDwarfRegNum(Location.getReg(), false);
  DIEBlock *Block = new DIEBlock();

  if (Location.isReg()) {
    if (Reg < 32) {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_reg0 + Reg);
    } else {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_regx);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }
  } else {
    if (Reg < 32) {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_breg0 + Reg);
    } else {
      AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_bregx);
      AddUInt(Block, 0, dwarf::DW_FORM_udata, Reg);
    }

    AddUInt(Block, 0, dwarf::DW_FORM_sdata, Location.getOffset());
  }

  AddBlock(Die, Attribute, 0, Block);
}

/// AddType - Add a new type attribute to the specified entity.
void DwarfDebug::AddType(CompileUnit *DW_Unit, DIE *Entity, DIType Ty) {
  if (Ty.isNull())
    return;

  // Check for pre-existence.
  DIEEntry *&Slot = DW_Unit->getDIEEntrySlotFor(Ty.getNode());

  // If it exists then use the existing value.
  if (Slot) {
    Entity->AddValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Slot);
    return;
  }

  // Set up proxy.
  Slot = CreateDIEEntry();

  // Construct type.
  DIE Buffer(dwarf::DW_TAG_base_type);
  if (Ty.isBasicType())
    ConstructTypeDIE(DW_Unit, Buffer, DIBasicType(Ty.getNode()));
  else if (Ty.isCompositeType())
    ConstructTypeDIE(DW_Unit, Buffer, DICompositeType(Ty.getNode()));
  else {
    assert(Ty.isDerivedType() && "Unknown kind of DIType");
    ConstructTypeDIE(DW_Unit, Buffer, DIDerivedType(Ty.getNode()));
  }

  // Add debug information entry to entity and appropriate context.
  DIE *Die = NULL;
  DIDescriptor Context = Ty.getContext();
  if (!Context.isNull())
    Die = DW_Unit->getDieMapSlotFor(Context.getNode());

  if (Die) {
    DIE *Child = new DIE(Buffer);
    Die->AddChild(Child);
    Buffer.Detach();
    SetDIEEntry(Slot, Child);
  } else {
    Die = DW_Unit->AddDie(Buffer);
    SetDIEEntry(Slot, Die);
  }

  Entity->AddValue(dwarf::DW_AT_type, dwarf::DW_FORM_ref4, Slot);
}

/// ConstructTypeDIE - Construct basic type die from DIBasicType.
void DwarfDebug::ConstructTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                                  DIBasicType BTy) {
  // Get core information.
  const char *Name = BTy.getName();
  Buffer.setTag(dwarf::DW_TAG_base_type);
  AddUInt(&Buffer, dwarf::DW_AT_encoding,  dwarf::DW_FORM_data1,
          BTy.getEncoding());

  // Add name if not anonymous or intermediate type.
  if (Name)
    AddString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  uint64_t Size = BTy.getSizeInBits() >> 3;
  AddUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);
}

/// ConstructTypeDIE - Construct derived type die from DIDerivedType.
void DwarfDebug::ConstructTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                                  DIDerivedType DTy) {
  // Get core information.
  const char *Name = DTy.getName();
  uint64_t Size = DTy.getSizeInBits() >> 3;
  unsigned Tag = DTy.getTag();

  // FIXME - Workaround for templates.
  if (Tag == dwarf::DW_TAG_inheritance) Tag = dwarf::DW_TAG_reference_type;

  Buffer.setTag(Tag);

  // Map to main type, void will not have a type.
  DIType FromTy = DTy.getTypeDerivedFrom();
  AddType(DW_Unit, &Buffer, FromTy);

  // Add name if not anonymous or intermediate type.
  if (Name)
    AddString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  // Add size if non-zero (derived types might be zero-sized.)
  if (Size)
    AddUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);

  // Add source line info if available and TyDesc is not a forward declaration.
  if (!DTy.isForwardDecl())
    AddSourceLine(&Buffer, &DTy);
}

/// ConstructTypeDIE - Construct type DIE from DICompositeType.
void DwarfDebug::ConstructTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                                  DICompositeType CTy) {
  // Get core information.
  const char *Name = CTy.getName();

  uint64_t Size = CTy.getSizeInBits() >> 3;
  unsigned Tag = CTy.getTag();
  Buffer.setTag(Tag);

  switch (Tag) {
  case dwarf::DW_TAG_vector_type:
  case dwarf::DW_TAG_array_type:
    ConstructArrayTypeDIE(DW_Unit, Buffer, &CTy);
    break;
  case dwarf::DW_TAG_enumeration_type: {
    DIArray Elements = CTy.getTypeArray();

    // Add enumerators to enumeration type.
    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIE *ElemDie = NULL;
      DIEnumerator Enum(Elements.getElement(i).getNode());
      if (!Enum.isNull()) {
        ElemDie = ConstructEnumTypeDIE(DW_Unit, &Enum);
        Buffer.AddChild(ElemDie);
      }
    }
  }
    break;
  case dwarf::DW_TAG_subroutine_type: {
    // Add return type.
    DIArray Elements = CTy.getTypeArray();
    DIDescriptor RTy = Elements.getElement(0);
    AddType(DW_Unit, &Buffer, DIType(RTy.getNode()));

    // Add prototype flag.
    AddUInt(&Buffer, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);

    // Add arguments.
    for (unsigned i = 1, N = Elements.getNumElements(); i < N; ++i) {
      DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
      DIDescriptor Ty = Elements.getElement(i);
      AddType(DW_Unit, Arg, DIType(Ty.getNode()));
      Buffer.AddChild(Arg);
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
        ElemDie = CreateSubprogramDIE(DW_Unit,
                                      DISubprogram(Element.getNode()));
      else
        ElemDie = CreateMemberDIE(DW_Unit,
                                  DIDerivedType(Element.getNode()));
      Buffer.AddChild(ElemDie);
    }

    if (CTy.isAppleBlockExtension())
      AddUInt(&Buffer, dwarf::DW_AT_APPLE_block, dwarf::DW_FORM_flag, 1);

    unsigned RLang = CTy.getRunTimeLang();
    if (RLang)
      AddUInt(&Buffer, dwarf::DW_AT_APPLE_runtime_class,
              dwarf::DW_FORM_data1, RLang);
    break;
  }
  default:
    break;
  }

  // Add name if not anonymous or intermediate type.
  if (Name)
    AddString(&Buffer, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  if (Tag == dwarf::DW_TAG_enumeration_type ||
      Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type) {
    // Add size if non-zero (derived types might be zero-sized.)
    if (Size)
      AddUInt(&Buffer, dwarf::DW_AT_byte_size, 0, Size);
    else {
      // Add zero size if it is not a forward declaration.
      if (CTy.isForwardDecl())
        AddUInt(&Buffer, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);
      else
        AddUInt(&Buffer, dwarf::DW_AT_byte_size, 0, 0);
    }

    // Add source line info if available.
    if (!CTy.isForwardDecl())
      AddSourceLine(&Buffer, &CTy);
  }
}

/// ConstructSubrangeDIE - Construct subrange DIE from DISubrange.
void DwarfDebug::ConstructSubrangeDIE(DIE &Buffer, DISubrange SR, DIE *IndexTy){
  int64_t L = SR.getLo();
  int64_t H = SR.getHi();
  DIE *DW_Subrange = new DIE(dwarf::DW_TAG_subrange_type);

  AddDIEEntry(DW_Subrange, dwarf::DW_AT_type, dwarf::DW_FORM_ref4, IndexTy);
  if (L)
    AddSInt(DW_Subrange, dwarf::DW_AT_lower_bound, 0, L);
  if (H)
    AddSInt(DW_Subrange, dwarf::DW_AT_upper_bound, 0, H);

  Buffer.AddChild(DW_Subrange);
}

/// ConstructArrayTypeDIE - Construct array type DIE from DICompositeType.
void DwarfDebug::ConstructArrayTypeDIE(CompileUnit *DW_Unit, DIE &Buffer,
                                       DICompositeType *CTy) {
  Buffer.setTag(dwarf::DW_TAG_array_type);
  if (CTy->getTag() == dwarf::DW_TAG_vector_type)
    AddUInt(&Buffer, dwarf::DW_AT_GNU_vector, dwarf::DW_FORM_flag, 1);

  // Emit derived type.
  AddType(DW_Unit, &Buffer, CTy->getTypeDerivedFrom());
  DIArray Elements = CTy->getTypeArray();

  // Construct an anonymous type for index type.
  DIE IdxBuffer(dwarf::DW_TAG_base_type);
  AddUInt(&IdxBuffer, dwarf::DW_AT_byte_size, 0, sizeof(int32_t));
  AddUInt(&IdxBuffer, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
          dwarf::DW_ATE_signed);
  DIE *IndexTy = DW_Unit->AddDie(IdxBuffer);

  // Add subranges to array type.
  for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
    DIDescriptor Element = Elements.getElement(i);
    if (Element.getTag() == dwarf::DW_TAG_subrange_type)
      ConstructSubrangeDIE(Buffer, DISubrange(Element.getNode()), IndexTy);
  }
}

/// ConstructEnumTypeDIE - Construct enum type DIE from DIEnumerator.
DIE *DwarfDebug::ConstructEnumTypeDIE(CompileUnit *DW_Unit, DIEnumerator *ETy) {
  DIE *Enumerator = new DIE(dwarf::DW_TAG_enumerator);
  const char *Name = ETy->getName();
  AddString(Enumerator, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);
  int64_t Value = ETy->getEnumValue();
  AddSInt(Enumerator, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata, Value);
  return Enumerator;
}

/// CreateGlobalVariableDIE - Create new DIE using GV.
DIE *DwarfDebug::CreateGlobalVariableDIE(CompileUnit *DW_Unit,
                                         const DIGlobalVariable &GV) {
  DIE *GVDie = new DIE(dwarf::DW_TAG_variable);
  AddString(GVDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, 
            GV.getDisplayName());

  const char *LinkageName = GV.getLinkageName();
  if (LinkageName) {
    // Skip special LLVM prefix that is used to inform the asm printer to not
    // emit usual symbol prefix before the symbol name. This happens for
    // Objective-C symbol names and symbol whose name is replaced using GCC's
    // __asm__ attribute.
    if (LinkageName[0] == 1)
      LinkageName = &LinkageName[1];
    AddString(GVDie, dwarf::DW_AT_MIPS_linkage_name, dwarf::DW_FORM_string,
              LinkageName);
  }
  AddType(DW_Unit, GVDie, GV.getType());
  if (!GV.isLocalToUnit())
    AddUInt(GVDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);
  AddSourceLine(GVDie, &GV);

  // Add address.
  DIEBlock *Block = new DIEBlock();
  AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
  AddObjectLabel(Block, 0, dwarf::DW_FORM_udata,
                 Asm->Mang->getMangledName(GV.getGlobal()));
  AddBlock(GVDie, dwarf::DW_AT_location, 0, Block);

  return GVDie;
}

/// CreateMemberDIE - Create new member DIE.
DIE *DwarfDebug::CreateMemberDIE(CompileUnit *DW_Unit, const DIDerivedType &DT){
  DIE *MemberDie = new DIE(DT.getTag());
  if (const char *Name = DT.getName())
    AddString(MemberDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  AddType(DW_Unit, MemberDie, DT.getTypeDerivedFrom());

  AddSourceLine(MemberDie, &DT);

  uint64_t Size = DT.getSizeInBits();
  uint64_t FieldSize = DT.getOriginalTypeSize();

  if (Size != FieldSize) {
    // Handle bitfield.
    AddUInt(MemberDie, dwarf::DW_AT_byte_size, 0, DT.getOriginalTypeSize()>>3);
    AddUInt(MemberDie, dwarf::DW_AT_bit_size, 0, DT.getSizeInBits());

    uint64_t Offset = DT.getOffsetInBits();
    uint64_t FieldOffset = Offset;
    uint64_t AlignMask = ~(DT.getAlignInBits() - 1);
    uint64_t HiMark = (Offset + FieldSize) & AlignMask;
    FieldOffset = (HiMark - FieldSize);
    Offset -= FieldOffset;

    // Maybe we need to work from the other end.
    if (TD->isLittleEndian()) Offset = FieldSize - (Offset + Size);
    AddUInt(MemberDie, dwarf::DW_AT_bit_offset, 0, Offset);
  }

  DIEBlock *Block = new DIEBlock();
  AddUInt(Block, 0, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
  AddUInt(Block, 0, dwarf::DW_FORM_udata, DT.getOffsetInBits() >> 3);
  AddBlock(MemberDie, dwarf::DW_AT_data_member_location, 0, Block);

  if (DT.isProtected())
    AddUInt(MemberDie, dwarf::DW_AT_accessibility, 0,
            dwarf::DW_ACCESS_protected);
  else if (DT.isPrivate())
    AddUInt(MemberDie, dwarf::DW_AT_accessibility, 0,
            dwarf::DW_ACCESS_private);

  return MemberDie;
}

/// CreateSubprogramDIE - Create new DIE using SP.
DIE *DwarfDebug::CreateSubprogramDIE(CompileUnit *DW_Unit,
                                     const DISubprogram &SP,
                                     bool IsConstructor,
                                     bool IsInlined) {
  DIE *SPDie = new DIE(dwarf::DW_TAG_subprogram);

  const char * Name = SP.getName();
  AddString(SPDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  const char *LinkageName = SP.getLinkageName();
  if (LinkageName) {
    // Skip special LLVM prefix that is used to inform the asm printer to not emit
    // usual symbol prefix before the symbol name. This happens for Objective-C
    // symbol names and symbol whose name is replaced using GCC's __asm__ attribute.
    if (LinkageName[0] == 1)
      LinkageName = &LinkageName[1];
    AddString(SPDie, dwarf::DW_AT_MIPS_linkage_name, dwarf::DW_FORM_string,
              LinkageName);
  }
  AddSourceLine(SPDie, &SP);

  DICompositeType SPTy = SP.getType();
  DIArray Args = SPTy.getTypeArray();

  // Add prototyped tag, if C or ObjC.
  unsigned Lang = SP.getCompileUnit().getLanguage();
  if (Lang == dwarf::DW_LANG_C99 || Lang == dwarf::DW_LANG_C89 ||
      Lang == dwarf::DW_LANG_ObjC)
    AddUInt(SPDie, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag, 1);

  // Add Return Type.
  unsigned SPTag = SPTy.getTag();
  if (!IsConstructor) {
    if (Args.isNull() || SPTag != dwarf::DW_TAG_subroutine_type)
      AddType(DW_Unit, SPDie, SPTy);
    else
      AddType(DW_Unit, SPDie, DIType(Args.getElement(0).getNode()));
  }

  if (!SP.isDefinition()) {
    AddUInt(SPDie, dwarf::DW_AT_declaration, dwarf::DW_FORM_flag, 1);

    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled through RecordVariable.
    if (SPTag == dwarf::DW_TAG_subroutine_type)
      for (unsigned i = 1, N =  Args.getNumElements(); i < N; ++i) {
        DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
        AddType(DW_Unit, Arg, DIType(Args.getElement(i).getNode()));
        AddUInt(Arg, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag, 1); // ??
        SPDie->AddChild(Arg);
      }
  }

  if (!SP.isLocalToUnit() && !IsInlined)
    AddUInt(SPDie, dwarf::DW_AT_external, dwarf::DW_FORM_flag, 1);

  // DW_TAG_inlined_subroutine may refer to this DIE.
  DIE *&Slot = DW_Unit->getDieMapSlotFor(SP.getNode());
  Slot = SPDie;
  return SPDie;
}

/// FindCompileUnit - Get the compile unit for the given descriptor.
///
CompileUnit &DwarfDebug::FindCompileUnit(DICompileUnit Unit) const {
  DenseMap<Value *, CompileUnit *>::const_iterator I =
    CompileUnitMap.find(Unit.getNode());
  assert(I != CompileUnitMap.end() && "Missing compile unit.");
  return *I->second;
}

/// CreateDbgScopeVariable - Create a new scope variable.
///
DIE *DwarfDebug::CreateDbgScopeVariable(DbgVariable *DV, CompileUnit *Unit) {
  // Get the descriptor.
  const DIVariable &VD = DV->getVariable();

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
  const char *Name = VD.getName();
  AddString(VariableDie, dwarf::DW_AT_name, dwarf::DW_FORM_string, Name);

  // Add source line info if available.
  AddSourceLine(VariableDie, &VD);

  // Add variable type.
  // FIXME: isBlockByrefVariable should be reformulated in terms of complex addresses instead.
  if (VD.isBlockByrefVariable())
    AddType(Unit, VariableDie, GetBlockByrefType(VD.getType(), Name));
  else
    AddType(Unit, VariableDie, VD.getType());

  // Add variable address.
  if (!DV->isInlinedFnVar()) {
    // Variables for abstract instances of inlined functions don't get a
    // location.
    MachineLocation Location;
    Location.set(RI->getFrameRegister(*MF),
                 RI->getFrameIndexOffset(*MF, DV->getFrameIndex()));


    if (VD.hasComplexAddress())
      AddComplexAddress(DV, VariableDie, dwarf::DW_AT_location, Location);
    else if (VD.isBlockByrefVariable())
      AddBlockByrefAddress(DV, VariableDie, dwarf::DW_AT_location, Location);
    else
      AddAddress(VariableDie, dwarf::DW_AT_location, Location);
  }

  return VariableDie;
}

/// getOrCreateScope - Returns the scope associated with the given descriptor.
///
DbgScope *DwarfDebug::getDbgScope(MDNode *N, const MachineInstr *MI) {
  DbgScope *&Slot = DbgScopeMap[N];
  if (Slot) return Slot;

  DbgScope *Parent = NULL;

  DIDescriptor Scope(N);
  if (Scope.isCompileUnit()) {
    return NULL;
  } else if (Scope.isSubprogram()) {
    DISubprogram SP(N);
    DIDescriptor ParentDesc = SP.getContext();
    if (!ParentDesc.isNull() && !ParentDesc.isCompileUnit())
      Parent = getDbgScope(ParentDesc.getNode(), MI);
  } else if (Scope.isLexicalBlock()) {
    DILexicalBlock DB(N);
    DIDescriptor ParentDesc = DB.getContext();
    if (!ParentDesc.isNull())
      Parent = getDbgScope(ParentDesc.getNode(), MI);
  } else
    assert (0 && "Unexpected scope info");

  Slot = new DbgScope(Parent, DIDescriptor(N));
  Slot->setFirstInsn(MI);

  if (Parent)
    Parent->AddScope(Slot);
  else
    // First function is top level function.
    if (!FunctionDbgScope)
      FunctionDbgScope = Slot;

  return Slot;
}


/// getOrCreateScope - Returns the scope associated with the given descriptor.
/// FIXME - Remove this method.
DbgScope *DwarfDebug::getOrCreateScope(MDNode *N) {
  DbgScope *&Slot = DbgScopeMap[N];
  if (Slot) return Slot;

  DbgScope *Parent = NULL;
  DILexicalBlock Block(N);

  // Don't create a new scope if we already created one for an inlined function.
  DenseMap<const MDNode *, DbgScope *>::iterator
    II = AbstractInstanceRootMap.find(N);
  if (II != AbstractInstanceRootMap.end())
    return LexicalScopeStack.back();

  if (!Block.isNull()) {
    DIDescriptor ParentDesc = Block.getContext();
    Parent =
      ParentDesc.isNull() ?  NULL : getOrCreateScope(ParentDesc.getNode());
  }

  Slot = new DbgScope(Parent, DIDescriptor(N));

  if (Parent)
    Parent->AddScope(Slot);
  else
    // First function is top level function.
    FunctionDbgScope = Slot;

  return Slot;
}

/// ConstructDbgScope - Construct the components of a scope.
///
void DwarfDebug::ConstructDbgScope(DbgScope *ParentScope,
                                   unsigned ParentStartID,
                                   unsigned ParentEndID,
                                   DIE *ParentDie, CompileUnit *Unit) {
  // Add variables to scope.
  SmallVector<DbgVariable *, 8> &Variables = ParentScope->getVariables();
  for (unsigned i = 0, N = Variables.size(); i < N; ++i) {
    DIE *VariableDie = CreateDbgScopeVariable(Variables[i], Unit);
    if (VariableDie) ParentDie->AddChild(VariableDie);
  }

  // Add concrete instances to scope.
  SmallVector<DbgConcreteScope *, 8> &ConcreteInsts =
    ParentScope->getConcreteInsts();
  for (unsigned i = 0, N = ConcreteInsts.size(); i < N; ++i) {
    DbgConcreteScope *ConcreteInst = ConcreteInsts[i];
    DIE *Die = ConcreteInst->getDie();

    unsigned StartID = ConcreteInst->getStartLabelID();
    unsigned EndID = ConcreteInst->getEndLabelID();

    // Add the scope bounds.
    if (StartID)
      AddLabel(Die, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
               DWLabel("label", StartID));
    else
      AddLabel(Die, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
               DWLabel("func_begin", SubprogramCount));

    if (EndID)
      AddLabel(Die, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
               DWLabel("label", EndID));
    else
      AddLabel(Die, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
               DWLabel("func_end", SubprogramCount));

    ParentDie->AddChild(Die);
  }

  // Add nested scopes.
  SmallVector<DbgScope *, 4> &Scopes = ParentScope->getScopes();
  for (unsigned j = 0, M = Scopes.size(); j < M; ++j) {
    // Define the Scope debug information entry.
    DbgScope *Scope = Scopes[j];

    unsigned StartID = MMI->MappedLabel(Scope->getStartLabelID());
    unsigned EndID = MMI->MappedLabel(Scope->getEndLabelID());

    // Ignore empty scopes.
    if (StartID == EndID && StartID != 0) continue;

    // Do not ignore inlined scopes even if they don't have any variables or
    // scopes.
    if (Scope->getScopes().empty() && Scope->getVariables().empty() &&
        Scope->getConcreteInsts().empty())
      continue;

    if (StartID == ParentStartID && EndID == ParentEndID) {
      // Just add stuff to the parent scope.
      ConstructDbgScope(Scope, ParentStartID, ParentEndID, ParentDie, Unit);
    } else {
      DIE *ScopeDie = new DIE(dwarf::DW_TAG_lexical_block);

      // Add the scope bounds.
      if (StartID)
        AddLabel(ScopeDie, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
                 DWLabel("label", StartID));
      else
        AddLabel(ScopeDie, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
                 DWLabel("func_begin", SubprogramCount));

      if (EndID)
        AddLabel(ScopeDie, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
                 DWLabel("label", EndID));
      else
        AddLabel(ScopeDie, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
                 DWLabel("func_end", SubprogramCount));

      // Add the scope's contents.
      ConstructDbgScope(Scope, StartID, EndID, ScopeDie, Unit);
      ParentDie->AddChild(ScopeDie);
    }
  }
}

/// ConstructFunctionDbgScope - Construct the scope for the subprogram.
///
void DwarfDebug::ConstructFunctionDbgScope(DbgScope *RootScope,
                                           bool AbstractScope) {
  // Exit if there is no root scope.
  if (!RootScope) return;
  DIDescriptor Desc = RootScope->getDesc();
  if (Desc.isNull())
    return;

  // Get the subprogram debug information entry.
  DISubprogram SPD(Desc.getNode());

  // Get the subprogram die.
  DIE *SPDie = ModuleCU->getDieMapSlotFor(SPD.getNode());
  if (!SPDie) {
    ConstructSubprogram(SPD.getNode());
    SPDie = ModuleCU->getDieMapSlotFor(SPD.getNode());
  }
  assert(SPDie && "Missing subprogram descriptor");

  if (!AbstractScope) {
    // Add the function bounds.
    AddLabel(SPDie, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
             DWLabel("func_begin", SubprogramCount));
    AddLabel(SPDie, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
             DWLabel("func_end", SubprogramCount));
    MachineLocation Location(RI->getFrameRegister(*MF));
    AddAddress(SPDie, dwarf::DW_AT_frame_base, Location);
  }

  ConstructDbgScope(RootScope, 0, 0, SPDie, ModuleCU);
  // If there are global variables at this scope then add their dies.
  for (SmallVector<WeakVH, 4>::iterator SGI = ScopedGVs.begin(), 
       SGE = ScopedGVs.end(); SGI != SGE; ++SGI) {
    MDNode *N = dyn_cast_or_null<MDNode>(*SGI);
    if (!N) continue;
    DIGlobalVariable GV(N);
    if (GV.getContext().getNode() == RootScope->getDesc().getNode()) {
      DIE *ScopedGVDie = CreateGlobalVariableDIE(ModuleCU, GV);
      SPDie->AddChild(ScopedGVDie);
    }
  }
}

/// ConstructDefaultDbgScope - Construct a default scope for the subprogram.
///
void DwarfDebug::ConstructDefaultDbgScope(MachineFunction *MF) {
  StringMap<DIE*> &Globals = ModuleCU->getGlobals();
  StringMap<DIE*>::iterator GI = Globals.find(MF->getFunction()->getName());
  if (GI != Globals.end()) {
    DIE *SPDie = GI->second;

    // Add the function bounds.
    AddLabel(SPDie, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr,
             DWLabel("func_begin", SubprogramCount));
    AddLabel(SPDie, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr,
             DWLabel("func_end", SubprogramCount));

    MachineLocation Location(RI->getFrameRegister(*MF));
    AddAddress(SPDie, dwarf::DW_AT_frame_base, Location);
  }
}

/// GetOrCreateSourceID - Look up the source id with the given directory and
/// source file names. If none currently exists, create a new id and insert it
/// in the SourceIds map. This can update DirectoryNames and SourceFileNames
/// maps as well.
unsigned DwarfDebug::GetOrCreateSourceID(const char *DirName,
                                         const char *FileName) {
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

void DwarfDebug::ConstructCompileUnit(MDNode *N) {
  DICompileUnit DIUnit(N);
  const char *FN = DIUnit.getFilename();
  const char *Dir = DIUnit.getDirectory();
  unsigned ID = GetOrCreateSourceID(Dir, FN);

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  AddSectionOffset(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4,
                   DWLabel("section_line", 0), DWLabel("section_line", 0),
                   false);
  AddString(Die, dwarf::DW_AT_producer, dwarf::DW_FORM_string,
            DIUnit.getProducer());
  AddUInt(Die, dwarf::DW_AT_language, dwarf::DW_FORM_data1,
          DIUnit.getLanguage());
  AddString(Die, dwarf::DW_AT_name, dwarf::DW_FORM_string, FN);

  if (Dir)
    AddString(Die, dwarf::DW_AT_comp_dir, dwarf::DW_FORM_string, Dir);
  if (DIUnit.isOptimized())
    AddUInt(Die, dwarf::DW_AT_APPLE_optimized, dwarf::DW_FORM_flag, 1);

  if (const char *Flags = DIUnit.getFlags())
    AddString(Die, dwarf::DW_AT_APPLE_flags, dwarf::DW_FORM_string, Flags);

  unsigned RVer = DIUnit.getRunTimeVersion();
  if (RVer)
    AddUInt(Die, dwarf::DW_AT_APPLE_major_runtime_vers,
            dwarf::DW_FORM_data1, RVer);

  CompileUnit *Unit = new CompileUnit(ID, Die);
  if (!ModuleCU && DIUnit.isMain()) {
    // Use first compile unit marked as isMain as the compile unit
    // for this module.
    ModuleCU = Unit;
  }

  CompileUnitMap[DIUnit.getNode()] = Unit;
  CompileUnits.push_back(Unit);
}

void DwarfDebug::ConstructGlobalVariableDIE(MDNode *N) {
  DIGlobalVariable DI_GV(N);

  // If debug information is malformed then ignore it.
  if (DI_GV.Verify() == false)
    return;

  // Check for pre-existence.
  DIE *&Slot = ModuleCU->getDieMapSlotFor(DI_GV.getNode());
  if (Slot)
    return;

  DIE *VariableDie = CreateGlobalVariableDIE(ModuleCU, DI_GV);

  // Add to map.
  Slot = VariableDie;

  // Add to context owner.
  ModuleCU->getDie()->AddChild(VariableDie);

  // Expose as global. FIXME - need to check external flag.
  ModuleCU->AddGlobal(DI_GV.getName(), VariableDie);
  return;
}

void DwarfDebug::ConstructSubprogram(MDNode *N) {
  DISubprogram SP(N);

  // Check for pre-existence.
  DIE *&Slot = ModuleCU->getDieMapSlotFor(N);
  if (Slot)
    return;

  if (!SP.isDefinition())
    // This is a method declaration which will be handled while constructing
    // class type.
    return;

  DIE *SubprogramDie = CreateSubprogramDIE(ModuleCU, SP);

  // Add to map.
  Slot = SubprogramDie;

  // Add to context owner.
  ModuleCU->getDie()->AddChild(SubprogramDie);

  // Expose as global.
  ModuleCU->AddGlobal(SP.getName(), SubprogramDie);
  return;
}

/// BeginModule - Emit all Dwarf sections that should come prior to the
/// content. Create global DIEs and emit initial debug info sections.
/// This is inovked by the target AsmPrinter.
void DwarfDebug::BeginModule(Module *M, MachineModuleInfo *mmi) {
  this->M = M;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  DebugInfoFinder DbgFinder;
  DbgFinder.processModule(*M);

  // Create all the compile unit DIEs.
  for (DebugInfoFinder::iterator I = DbgFinder.compile_unit_begin(),
         E = DbgFinder.compile_unit_end(); I != E; ++I)
    ConstructCompileUnit(*I);

  if (CompileUnits.empty()) {
    if (TimePassesIsEnabled)
      DebugTimer->stopTimer();

    return;
  }

  // If main compile unit for this module is not seen than randomly
  // select first compile unit.
  if (!ModuleCU)
    ModuleCU = CompileUnits[0];

  // Create DIEs for each of the externally visible global variables.
  for (DebugInfoFinder::iterator I = DbgFinder.global_variable_begin(),
         E = DbgFinder.global_variable_end(); I != E; ++I) {
    DIGlobalVariable GV(*I);
    if (GV.getContext().getNode() != GV.getCompileUnit().getNode())
      ScopedGVs.push_back(*I);
    else
      ConstructGlobalVariableDIE(*I);
  }

  // Create DIEs for each of the externally visible subprograms.
  for (DebugInfoFinder::iterator I = DbgFinder.subprogram_begin(),
         E = DbgFinder.subprogram_end(); I != E; ++I)
    ConstructSubprogram(*I);

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
  EmitInitial();

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// EndModule - Emit all Dwarf sections that should come after the content.
///
void DwarfDebug::EndModule() {
  if (!ModuleCU)
    return;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

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
  EmitCommonDebugFrame();

  // Emit function debug frame information
  for (std::vector<FunctionDebugFrameInfo>::iterator I = DebugFrames.begin(),
         E = DebugFrames.end(); I != E; ++I)
    EmitFunctionDebugFrame(*I);

  // Compute DIE offsets and sizes.
  SizeAndOffsets();

  // Emit all the DIEs into a debug info section
  EmitDebugInfo();

  // Corresponding abbreviations into a abbrev section.
  EmitAbbreviations();

  // Emit source line correspondence into a debug line section.
  EmitDebugLines();

  // Emit info into a debug pubnames section.
  EmitDebugPubNames();

  // Emit info into a debug str section.
  EmitDebugStr();

  // Emit info into a debug loc section.
  EmitDebugLoc();

  // Emit info into a debug aranges section.
  EmitDebugARanges();

  // Emit info into a debug ranges section.
  EmitDebugRanges();

  // Emit info into a debug macinfo section.
  EmitDebugMacInfo();

  // Emit inline info.
  EmitDebugInlineInfo();

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// CollectVariableInfo - Populate DbgScope entries with variables' info.
void DwarfDebug::CollectVariableInfo() {
  if (!MMI) return;
  MachineModuleInfo::VariableDbgInfoMapTy &VMap = MMI->getVariableDbgInfo();
  for (MachineModuleInfo::VariableDbgInfoMapTy::iterator VI = VMap.begin(),
         VE = VMap.end(); VI != VE; ++VI) {
    MetadataBase *MB = VI->first;
    MDNode *Var = dyn_cast_or_null<MDNode>(MB);
    DIVariable DV (Var);
    if (DV.isNull()) continue;
    unsigned VSlot = VI->second;
    DbgScope *Scope = getDbgScope(DV.getContext().getNode(),  NULL);
    Scope->AddVariable(new DbgVariable(DV, VSlot, false));
  }
}

/// SetDbgScopeBeginLabels - Update DbgScope begin labels for the scopes that
/// start with this machine instruction.
void DwarfDebug::SetDbgScopeBeginLabels(const MachineInstr *MI, unsigned Label) {
  InsnToDbgScopeMapTy::iterator I = DbgScopeBeginMap.find(MI);
  if (I == DbgScopeBeginMap.end())
    return;
  SmallVector<DbgScope *, 2> &SD = I->second;
  for (SmallVector<DbgScope *, 2>::iterator SDI = SD.begin(), SDE = SD.end();
       SDI != SDE; ++SDI) 
    (*SDI)->setStartLabelID(Label);
}

/// SetDbgScopeEndLabels - Update DbgScope end labels for the scopes that
/// end with this machine instruction.
void DwarfDebug::SetDbgScopeEndLabels(const MachineInstr *MI, unsigned Label) {
  InsnToDbgScopeMapTy::iterator I = DbgScopeEndMap.find(MI);
  if (I == DbgScopeEndMap.end())
    return;
  SmallVector<DbgScope *, 2> &SD = I->second;
  for (SmallVector<DbgScope *, 2>::iterator SDI = SD.begin(), SDE = SD.end();
       SDI != SDE; ++SDI) 
    (*SDI)->setEndLabelID(Label);
}

/// ExtractScopeInformation - Scan machine instructions in this function
/// and collect DbgScopes. Return true, if atleast one scope was found.
bool DwarfDebug::ExtractScopeInformation(MachineFunction *MF) {
  // If scope information was extracted using .dbg intrinsics then there is not
  // any need to extract these information by scanning each instruction.
  if (!DbgScopeMap.empty())
    return false;

  // Scan each instruction and create scopes.
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MInsn = II;
      DebugLoc DL = MInsn->getDebugLoc();
      if (DL.isUnknown())
        continue;
      DebugLocTuple DLT = MF->getDebugLocTuple(DL);
      if (!DLT.CompileUnit)
        continue;
      // There is no need to create another DIE for compile unit. For all
      // other scopes, create one DbgScope now. This will be translated 
      // into a scope DIE at the end.
      DIDescriptor D(DLT.CompileUnit);
      if (!D.isCompileUnit()) {
        DbgScope *Scope = getDbgScope(DLT.CompileUnit, MInsn);
        Scope->setLastInsn(MInsn);
      }
    }
  }

  // If a scope's last instruction is not set then use its child scope's
  // last instruction as this scope's last instrunction.
  for (DenseMap<MDNode *, DbgScope *>::iterator DI = DbgScopeMap.begin(),
	 DE = DbgScopeMap.end(); DI != DE; ++DI) {
    assert (DI->second->getFirstInsn() && "Invalid first instruction!");
    DI->second->FixInstructionMarkers();
    assert (DI->second->getLastInsn() && "Invalid last instruction!");
  }

  // Each scope has first instruction and last instruction to mark beginning
  // and end of a scope respectively. Create an inverse map that list scopes
  // starts (and ends) with an instruction. One instruction may start (or end)
  // multiple scopes.
  for (DenseMap<MDNode *, DbgScope *>::iterator DI = DbgScopeMap.begin(),
	 DE = DbgScopeMap.end(); DI != DE; ++DI) {
    DbgScope *S = DI->second;
    assert (S && "DbgScope is missing!");
    const MachineInstr *MI = S->getFirstInsn();
    assert (MI && "DbgScope does not have first instruction!");

    InsnToDbgScopeMapTy::iterator IDI = DbgScopeBeginMap.find(MI);
    if (IDI != DbgScopeBeginMap.end())
      IDI->second.push_back(S);
    else
      DbgScopeBeginMap.insert(std::make_pair(MI, 
                                             SmallVector<DbgScope *, 2>(2, S)));

    MI = S->getLastInsn();
    assert (MI && "DbgScope does not have last instruction!");
    IDI = DbgScopeEndMap.find(MI);
    if (IDI != DbgScopeEndMap.end())
      IDI->second.push_back(S);
    else
      DbgScopeEndMap.insert(std::make_pair(MI,
                                             SmallVector<DbgScope *, 2>(2, S)));
  }

  return !DbgScopeMap.empty();
}

/// BeginFunction - Gather pre-function debug information.  Assumes being
/// emitted immediately after the function entry point.
void DwarfDebug::BeginFunction(MachineFunction *MF) {
  this->MF = MF;

  if (!ShouldEmitDwarfDebug()) return;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

#ifdef ATTACH_DEBUG_INFO_TO_AN_INSN
  if (!ExtractScopeInformation(MF))
    return;
  CollectVariableInfo();
#endif

  // Begin accumulating function debug information.
  MMI->BeginFunction(MF);

  // Assumes in correct section after the entry point.
  EmitLabel("func_begin", ++SubprogramCount);

  // Emit label for the implicitly defined dbg.stoppoint at the start of the
  // function.
#ifdef ATTACH_DEBUG_INFO_TO_AN_INSN
  DebugLoc FDL = MF->getDefaultDebugLoc();
  if (!FDL.isUnknown()) {
    DebugLocTuple DLT = MF->getDebugLocTuple(FDL);
    unsigned LabelID = 0;
    DISubprogram SP(DLT.CompileUnit);
    if (!SP.isNull())
      LabelID = RecordSourceLine(SP.getLineNumber(), 0, DLT.CompileUnit);
    else
      LabelID = RecordSourceLine(DLT.Line, DLT.Col, DLT.CompileUnit);
    Asm->printLabel(LabelID);
    O << '\n';
  }
#else
  DebugLoc FDL = MF->getDefaultDebugLoc();
  if (!FDL.isUnknown()) {
    DebugLocTuple DLT = MF->getDebugLocTuple(FDL);
    unsigned LabelID = RecordSourceLine(DLT.Line, DLT.Col, DLT.CompileUnit);
    Asm->printLabel(LabelID);
    O << '\n';
  }
#endif
  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// EndFunction - Gather and emit post-function debug information.
///
void DwarfDebug::EndFunction(MachineFunction *MF) {
  if (!ShouldEmitDwarfDebug()) return;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

#ifdef ATTACH_DEBUG_INFO_TO_AN_INSN
  if (DbgScopeMap.empty())
    return;
#endif
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

  // Construct the DbgScope for abstract instances.
  for (SmallVector<DbgScope *, 32>::iterator
         I = AbstractInstanceRootList.begin(),
         E = AbstractInstanceRootList.end(); I != E; ++I)
    ConstructFunctionDbgScope(*I);

  // Construct scopes for subprogram.
  if (FunctionDbgScope)
    ConstructFunctionDbgScope(FunctionDbgScope);
  else
    // FIXME: This is wrong. We are essentially getting past a problem with
    // debug information not being able to handle unreachable blocks that have
    // debug information in them. In particular, those unreachable blocks that
    // have "region end" info in them. That situation results in the "root
    // scope" not being created. If that's the case, then emit a "default"
    // scope, i.e., one that encompasses the whole function. This isn't
    // desirable. And a better way of handling this (and all of the debugging
    // information) needs to be explored.
    ConstructDefaultDbgScope(MF);

  DebugFrames.push_back(FunctionDebugFrameInfo(SubprogramCount,
                                               MMI->getFrameMoves()));

  // Clear debug info
  if (FunctionDbgScope) {
    delete FunctionDbgScope;
    DbgScopeMap.clear();
    DbgScopeBeginMap.clear();
    DbgScopeEndMap.clear();
    DbgAbstractScopeMap.clear();
    DbgConcreteScopeMap.clear();
    FunctionDbgScope = NULL;
    LexicalScopeStack.clear();
    AbstractInstanceRootList.clear();
    AbstractInstanceRootMap.clear();
  }

  Lines.clear();

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

/// RecordSourceLine - Records location information and associates it with a
/// label. Returns a unique label ID used to generate a label and provide
/// correspondence to the source line list.
unsigned DwarfDebug::RecordSourceLine(unsigned Line, unsigned Col, 
                                      MDNode *S) {
  if (!MMI)
    return 0;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  const char *Dir = NULL;
  const char *Fn = NULL;

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

/// RecordRegionStart - Indicate the start of a region.
unsigned DwarfDebug::RecordRegionStart(MDNode *N) {
  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  DbgScope *Scope = getOrCreateScope(N);
  unsigned ID = MMI->NextLabelID();
  if (!Scope->getStartLabelID()) Scope->setStartLabelID(ID);
  LexicalScopeStack.push_back(Scope);

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();

  return ID;
}

/// RecordRegionEnd - Indicate the end of a region.
unsigned DwarfDebug::RecordRegionEnd(MDNode *N) {
  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  DbgScope *Scope = getOrCreateScope(N);
  unsigned ID = MMI->NextLabelID();
  Scope->setEndLabelID(ID);
  // FIXME : region.end() may not be in the last basic block.
  // For now, do not pop last lexical scope because next basic
  // block may start new inlined function's body.
  unsigned LSSize = LexicalScopeStack.size();
  if (LSSize != 0 && LSSize != 1)
    LexicalScopeStack.pop_back();

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();

  return ID;
}

/// RecordVariable - Indicate the declaration of a local variable.
void DwarfDebug::RecordVariable(MDNode *N, unsigned FrameIndex) {
  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  DIDescriptor Desc(N);
  DbgScope *Scope = NULL;
  bool InlinedFnVar = false;

  if (Desc.getTag() == dwarf::DW_TAG_variable)
    Scope = getOrCreateScope(DIGlobalVariable(N).getContext().getNode());
  else {
    bool InlinedVar = false;
    MDNode *Context = DIVariable(N).getContext().getNode();
    DISubprogram SP(Context);
    if (!SP.isNull()) {
      // SP is inserted into DbgAbstractScopeMap when inlined function
      // start was recorded by RecordInlineFnStart.
      DenseMap<MDNode *, DbgScope *>::iterator
        I = DbgAbstractScopeMap.find(SP.getNode());
      if (I != DbgAbstractScopeMap.end()) {
        InlinedVar = true;
        Scope = I->second;
      }
    }
    if (!InlinedVar)
      Scope = getOrCreateScope(Context);
  }

  assert(Scope && "Unable to find the variable's scope");
  DbgVariable *DV = new DbgVariable(DIVariable(N), FrameIndex, InlinedFnVar);
  Scope->AddVariable(DV);

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();
}

//// RecordInlinedFnStart - Indicate the start of inlined subroutine.
unsigned DwarfDebug::RecordInlinedFnStart(DISubprogram &SP, DICompileUnit CU,
                                          unsigned Line, unsigned Col) {
  unsigned LabelID = MMI->NextLabelID();

  if (!MAI->doesDwarfUsesInlineInfoSection())
    return LabelID;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  MDNode *Node = SP.getNode();
  DenseMap<const MDNode *, DbgScope *>::iterator
    II = AbstractInstanceRootMap.find(Node);

  if (II == AbstractInstanceRootMap.end()) {
    // Create an abstract instance entry for this inlined function if it doesn't
    // already exist.
    DbgScope *Scope = new DbgScope(NULL, DIDescriptor(Node));

    // Get the compile unit context.
    DIE *SPDie = ModuleCU->getDieMapSlotFor(Node);
    if (!SPDie)
      SPDie = CreateSubprogramDIE(ModuleCU, SP, false, true);

    // Mark as being inlined. This makes this subprogram entry an abstract
    // instance root.
    // FIXME: Our debugger doesn't care about the value of DW_AT_inline, only
    // that it's defined. That probably won't change in the future. However,
    // this could be more elegant.
    AddUInt(SPDie, dwarf::DW_AT_inline, 0, dwarf::DW_INL_declared_not_inlined);

    // Keep track of the abstract scope for this function.
    DbgAbstractScopeMap[Node] = Scope;

    AbstractInstanceRootMap[Node] = Scope;
    AbstractInstanceRootList.push_back(Scope);
  }

  // Create a concrete inlined instance for this inlined function.
  DbgConcreteScope *ConcreteScope = new DbgConcreteScope(DIDescriptor(Node));
  DIE *ScopeDie = new DIE(dwarf::DW_TAG_inlined_subroutine);
  ScopeDie->setAbstractCompileUnit(ModuleCU);

  DIE *Origin = ModuleCU->getDieMapSlotFor(Node);
  AddDIEEntry(ScopeDie, dwarf::DW_AT_abstract_origin,
              dwarf::DW_FORM_ref4, Origin);
  AddUInt(ScopeDie, dwarf::DW_AT_call_file, 0, ModuleCU->getID());
  AddUInt(ScopeDie, dwarf::DW_AT_call_line, 0, Line);
  AddUInt(ScopeDie, dwarf::DW_AT_call_column, 0, Col);

  ConcreteScope->setDie(ScopeDie);
  ConcreteScope->setStartLabelID(LabelID);
  MMI->RecordUsedDbgLabel(LabelID);

  LexicalScopeStack.back()->AddConcreteInst(ConcreteScope);

  // Keep track of the concrete scope that's inlined into this function.
  DenseMap<MDNode *, SmallVector<DbgScope *, 8> >::iterator
    SI = DbgConcreteScopeMap.find(Node);

  if (SI == DbgConcreteScopeMap.end())
    DbgConcreteScopeMap[Node].push_back(ConcreteScope);
  else
    SI->second.push_back(ConcreteScope);

  // Track the start label for this inlined function.
  DenseMap<MDNode *, SmallVector<unsigned, 4> >::iterator
    I = InlineInfo.find(Node);

  if (I == InlineInfo.end())
    InlineInfo[Node].push_back(LabelID);
  else
    I->second.push_back(LabelID);

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();

  return LabelID;
}

/// RecordInlinedFnEnd - Indicate the end of inlined subroutine.
unsigned DwarfDebug::RecordInlinedFnEnd(DISubprogram &SP) {
  if (!MAI->doesDwarfUsesInlineInfoSection())
    return 0;

  if (TimePassesIsEnabled)
    DebugTimer->startTimer();

  MDNode *Node = SP.getNode();
  DenseMap<MDNode *, SmallVector<DbgScope *, 8> >::iterator
    I = DbgConcreteScopeMap.find(Node);

  if (I == DbgConcreteScopeMap.end()) {
    // FIXME: Can this situation actually happen? And if so, should it?
    if (TimePassesIsEnabled)
      DebugTimer->stopTimer();

    return 0;
  }

  SmallVector<DbgScope *, 8> &Scopes = I->second;
  if (Scopes.empty()) {
    // Returned ID is 0 if this is unbalanced "end of inlined
    // scope". This could happen if optimizer eats dbg intrinsics
    // or "beginning of inlined scope" is not recoginized due to
    // missing location info. In such cases, ignore this region.end.
    return 0;
  }

  DbgScope *Scope = Scopes.back(); Scopes.pop_back();
  unsigned ID = MMI->NextLabelID();
  MMI->RecordUsedDbgLabel(ID);
  Scope->setEndLabelID(ID);

  if (TimePassesIsEnabled)
    DebugTimer->stopTimer();

  return ID;
}

//===----------------------------------------------------------------------===//
// Emit Methods
//===----------------------------------------------------------------------===//

/// SizeAndOffsetDie - Compute the size and offset of a DIE.
///
unsigned DwarfDebug::SizeAndOffsetDie(DIE *Die, unsigned Offset, bool Last) {
  // Get the children.
  const std::vector<DIE *> &Children = Die->getChildren();

  // If not last sibling and has children then add sibling offset attribute.
  if (!Last && !Children.empty()) Die->AddSiblingOffset();

  // Record the abbreviation.
  AssignAbbrevNumber(Die->getAbbrev());

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
      Offset = SizeAndOffsetDie(Children[j], Offset, (j + 1) == M);

    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die->setSize(Offset - Die->getOffset());
  return Offset;
}

/// SizeAndOffsets - Compute the size and offset of all the DIEs.
///
void DwarfDebug::SizeAndOffsets() {
  // Compute size of compile unit header.
  static unsigned Offset =
    sizeof(int32_t) + // Length of Compilation Unit Info
    sizeof(int16_t) + // DWARF version number
    sizeof(int32_t) + // Offset Into Abbrev. Section
    sizeof(int8_t);   // Pointer Size (in bytes)

  SizeAndOffsetDie(ModuleCU->getDie(), Offset, true);
  CompileUnitOffsets[ModuleCU] = 0;
}

/// EmitInitial - Emit initial Dwarf declarations.  This is necessary for cc
/// tools to recognize the object file contains Dwarf information.
void DwarfDebug::EmitInitial() {
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
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfStrSection());
  EmitLabel("section_str", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDwarfRangesSection());
  EmitLabel("section_ranges", 0);

  Asm->OutStreamer.SwitchSection(TLOF.getTextSection());
  EmitLabel("text_begin", 0);
  Asm->OutStreamer.SwitchSection(TLOF.getDataSection());
  EmitLabel("data_begin", 0);
}

/// EmitDIE - Recusively Emits a debug information entry.
///
void DwarfDebug::EmitDIE(DIE *Die) {
  // Get the abbreviation for this DIE.
  unsigned AbbrevNumber = Die->getAbbrevNumber();
  const DIEAbbrev *Abbrev = Abbreviations[AbbrevNumber - 1];

  Asm->EOL();

  // Emit the code (index) for the abbreviation.
  Asm->EmitULEB128Bytes(AbbrevNumber);

  if (Asm->isVerbose())
    Asm->EOL(std::string("Abbrev [" +
                         utostr(AbbrevNumber) +
                         "] 0x" + utohexstr(Die->getOffset()) +
                         ":0x" + utohexstr(Die->getSize()) + " " +
                         dwarf::TagString(Abbrev->getTag())));
  else
    Asm->EOL();

  SmallVector<DIEValue*, 32> &Values = Die->getValues();
  const SmallVector<DIEAbbrevData, 8> &AbbrevData = Abbrev->getData();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    unsigned Attr = AbbrevData[i].getAttribute();
    unsigned Form = AbbrevData[i].getForm();
    assert(Form && "Too many attributes for DIE (check abbreviation)");

    switch (Attr) {
    case dwarf::DW_AT_sibling:
      Asm->EmitInt32(Die->SiblingOffset());
      break;
    case dwarf::DW_AT_abstract_origin: {
      DIEEntry *E = cast<DIEEntry>(Values[i]);
      DIE *Origin = E->getEntry();
      unsigned Addr =
        CompileUnitOffsets[Die->getAbstractCompileUnit()] +
        Origin->getOffset();

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
      EmitDIE(Children[j]);

    Asm->EmitInt8(0); Asm->EOL("End Of Children Mark");
  }
}

/// EmitDebugInfo / EmitDebugInfoPerCU - Emit the debug info section.
///
void DwarfDebug::EmitDebugInfoPerCU(CompileUnit *Unit) {
  DIE *Die = Unit->getDie();

  // Emit the compile units header.
  EmitLabel("info_begin", Unit->getID());

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

  EmitDIE(Die);
  // FIXME - extra padding for gdb bug.
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  Asm->EmitInt8(0); Asm->EOL("Extra Pad For GDB");
  EmitLabel("info_end", Unit->getID());

  Asm->EOL();
}

void DwarfDebug::EmitDebugInfo() {
  // Start debug info section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfInfoSection());

  EmitDebugInfoPerCU(ModuleCU);
}

/// EmitAbbreviations - Emit the abbreviation section.
///
void DwarfDebug::EmitAbbreviations() const {
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

/// EmitEndOfLineMatrix - Emit the last address of the section and the end of
/// the line matrix.
///
void DwarfDebug::EmitEndOfLineMatrix(unsigned SectionEnd) {
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

/// EmitDebugLines - Emit source line information.
///
void DwarfDebug::EmitDebugLines() {
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
          << getSourceDirectoryName(SourceID.first) << ' '
          << getSourceFileName(SourceID.second)
          <<" :" << utostr_32(LineInfo.getLine()) << '\n';
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

    EmitEndOfLineMatrix(j + 1);
  }

  if (SecSrcLinesSize == 0)
    // Because we're emitting a debug_line section, we still need a line
    // table. The linker and friends expect it to exist. If there's nothing to
    // put into it, emit an empty table.
    EmitEndOfLineMatrix(1);

  EmitLabel("line_end", 0);
  Asm->EOL();
}

/// EmitCommonDebugFrame - Emit common frame info into a debug frame section.
///
void DwarfDebug::EmitCommonDebugFrame() {
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

/// EmitFunctionDebugFrame - Emit per function frame info into a debug frame
/// section.
void
DwarfDebug::EmitFunctionDebugFrame(const FunctionDebugFrameInfo&DebugFrameInfo){
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

void DwarfDebug::EmitDebugPubNamesPerCU(CompileUnit *Unit) {
  EmitDifference("pubnames_end", Unit->getID(),
                 "pubnames_begin", Unit->getID(), true);
  Asm->EOL("Length of Public Names Info");

  EmitLabel("pubnames_begin", Unit->getID());

  Asm->EmitInt16(dwarf::DWARF_VERSION); Asm->EOL("DWARF Version");

  EmitSectionOffset("info_begin", "section_info",
                    Unit->getID(), 0, true, false);
  Asm->EOL("Offset of Compilation Unit Info");

  EmitDifference("info_end", Unit->getID(), "info_begin", Unit->getID(),
                 true);
  Asm->EOL("Compilation Unit Length");

  StringMap<DIE*> &Globals = Unit->getGlobals();
  for (StringMap<DIE*>::const_iterator
         GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
    const char *Name = GI->getKeyData();
    DIE * Entity = GI->second;

    Asm->EmitInt32(Entity->getOffset()); Asm->EOL("DIE offset");
    Asm->EmitString(Name, strlen(Name)); Asm->EOL("External Name");
  }

  Asm->EmitInt32(0); Asm->EOL("End Mark");
  EmitLabel("pubnames_end", Unit->getID());

  Asm->EOL();
}

/// EmitDebugPubNames - Emit visible names into a debug pubnames section.
///
void DwarfDebug::EmitDebugPubNames() {
  // Start the dwarf pubnames section.
  Asm->OutStreamer.SwitchSection(
                          Asm->getObjFileLowering().getDwarfPubNamesSection());

  EmitDebugPubNamesPerCU(ModuleCU);
}

/// EmitDebugStr - Emit visible names into a debug str section.
///
void DwarfDebug::EmitDebugStr() {
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

/// EmitDebugLoc - Emit visible names into a debug loc section.
///
void DwarfDebug::EmitDebugLoc() {
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

/// EmitDebugRanges - Emit visible names into a debug ranges section.
///
void DwarfDebug::EmitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->OutStreamer.SwitchSection(
                            Asm->getObjFileLowering().getDwarfRangesSection());
  Asm->EOL();
}

/// EmitDebugMacInfo - Emit visible names into a debug macinfo section.
///
void DwarfDebug::EmitDebugMacInfo() {
  if (const MCSection *LineInfo =
      Asm->getObjFileLowering().getDwarfMacroInfoSection()) {
    // Start the dwarf macinfo section.
    Asm->OutStreamer.SwitchSection(LineInfo);
    Asm->EOL();
  }
}

/// EmitDebugInlineInfo - Emit inline info using following format.
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
void DwarfDebug::EmitDebugInlineInfo() {
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

  for (DenseMap<MDNode *, SmallVector<unsigned, 4> >::iterator
         I = InlineInfo.begin(), E = InlineInfo.end(); I != E; ++I) {
    MDNode *Node = I->first;
    SmallVector<unsigned, 4> &Labels = I->second;
    DISubprogram SP(Node);
    const char *LName = SP.getLinkageName();
    const char *Name = SP.getName();

    if (!LName)
      Asm->EmitString(Name);
    else {
      // Skip special LLVM prefix that is used to inform the asm printer to not
      // emit usual symbol prefix before the symbol name. This happens for
      // Objective-C symbol names and symbol whose name is replaced using GCC's
      // __asm__ attribute.
      if (LName[0] == 1)
        LName = &LName[1];
      Asm->EmitString(LName);
    }
    Asm->EOL("MIPS linkage name");

    Asm->EmitString(Name); Asm->EOL("Function name");

    Asm->EmitULEB128Bytes(Labels.size()); Asm->EOL("Inline count");

    for (SmallVector<unsigned, 4>::iterator LI = Labels.begin(),
           LE = Labels.end(); LI != LE; ++LI) {
      DIE *SP = ModuleCU->getDieMapSlotFor(Node);
      Asm->EmitInt32(SP->getOffset()); Asm->EOL("DIE offset");

      if (TD->getPointerSize() == sizeof(int32_t))
        O << MAI->getData32bitsDirective();
      else
        O << MAI->getData64bitsDirective();

      PrintLabelName("label", *LI); Asm->EOL("low_pc");
    }
  }

  EmitLabel("debug_inlined_end", 1);
  Asm->EOL();
}
