//===-- llvm/CodeGen/MachineModuleInfo.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect meta information for a module.  This information should be in a
// neutral form that can be used by different debugging and exception handling
// schemes.
//
// The organization of information is primarily clustered around the source
// compile units.  The main exception is source line correspondence where
// inlining may interleave code from various compile units.
//
// The following information can be retrieved from the MachineModuleInfo.
//
//  -- Source directories - Directories are uniqued based on their canonical
//     string and assigned a sequential numeric ID (base 1.)
//  -- Source files - Files are also uniqued based on their name and directory
//     ID.  A file ID is sequential number (base 1.)
//  -- Source line correspondence - A vector of file ID, line#, column# triples.
//     A DEBUG_LOCATION instruction is generated  by the DAG Legalizer
//     corresponding to each entry in the source line list.  This allows a debug
//     emitter to generate labels referenced by debug information tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFO_H
#define LLVM_CODEGEN_MACHINEMODULEINFO_H

#include "llvm/GlobalValue.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
class AnchoredDesc;
class CompileUnitDesc;
class Constant;
class DebugInfoDesc;
class GlobalVariable;
class MachineBasicBlock;
class MachineFunction;
class MachineMove;
class Module;
class PointerType;
class StructType;
class VariableDesc;

//===----------------------------------------------------------------------===//
/// DIVisitor - Subclasses of this class apply steps to each of the fields in
/// the supplied DebugInfoDesc.
class DIVisitor {
public:
  DIVisitor() {}
  virtual ~DIVisitor() {}

  /// ApplyToFields - Target the visitor to each field of the debug information
  /// descriptor.
  void ApplyToFields(DebugInfoDesc *DD);
  
  /// Apply - Subclasses override each of these methods to perform the
  /// appropriate action for the type of field.
  virtual void Apply(int &Field) = 0;
  virtual void Apply(unsigned &Field) = 0;
  virtual void Apply(int64_t &Field) = 0;
  virtual void Apply(uint64_t &Field) = 0;
  virtual void Apply(bool &Field) = 0;
  virtual void Apply(std::string &Field) = 0;
  virtual void Apply(DebugInfoDesc *&Field) = 0;
  virtual void Apply(GlobalVariable *&Field) = 0;
  virtual void Apply(std::vector<DebugInfoDesc *> &Field) = 0;
};

//===----------------------------------------------------------------------===//
/// DIDeserializer - This class is responsible for casting GlobalVariables
/// into DebugInfoDesc objects.
class DIDeserializer {
  // Previously defined gloabls.
  DenseMap<GlobalVariable*, DebugInfoDesc*> GlobalDescs;
public:
  const DenseMap<GlobalVariable *, DebugInfoDesc *> &getGlobalDescs() const {
    return GlobalDescs;
  }

  /// Deserialize - Reconstitute a GlobalVariable into it's component
  /// DebugInfoDesc objects.
  DebugInfoDesc *Deserialize(Value *V);
  DebugInfoDesc *Deserialize(GlobalVariable *GV);
};

//===----------------------------------------------------------------------===//
/// DISerializer - This class is responsible for casting DebugInfoDesc objects
/// into GlobalVariables.
class DISerializer {
  Module *M;                            // Definition space module.
  PointerType *StrPtrTy;                // A "i8*" type.  Created lazily.
  PointerType *EmptyStructPtrTy;        // A "{ }*" type.  Created lazily.

  // Types per Tag. Created lazily.
  std::map<unsigned, StructType *> TagTypes;

  // Previously defined descriptors.
  DenseMap<DebugInfoDesc *, GlobalVariable *> DescGlobals;

  // Previously defined strings.
  DenseMap<const char *, Constant*> StringCache;
public:
  DISerializer()
    : M(NULL), StrPtrTy(NULL), EmptyStructPtrTy(NULL), TagTypes(),
      DescGlobals(), StringCache()
  {}
  
  // Accessors
  Module *getModule()        const { return M; };
  void setModule(Module *module)  { M = module; }

  /// getStrPtrType - Return a "i8*" type.
  ///
  const PointerType *getStrPtrType();
  
  /// getEmptyStructPtrType - Return a "{ }*" type.
  ///
  const PointerType *getEmptyStructPtrType();
  
  /// getTagType - Return the type describing the specified descriptor (via
  /// tag.)
  const StructType *getTagType(DebugInfoDesc *DD);
  
  /// getString - Construct the string as constant string global.
  ///
  Constant *getString(const std::string &String);
  
  /// Serialize - Recursively cast the specified descriptor into a
  /// GlobalVariable so that it can be serialized to a .bc or .ll file.
  GlobalVariable *Serialize(DebugInfoDesc *DD);

  /// addDescriptor - Directly connect DD with existing GV.
  void addDescriptor(DebugInfoDesc *DD, GlobalVariable *GV);
};

//===----------------------------------------------------------------------===//
/// DIVerifier - This class is responsible for verifying the given network of
/// GlobalVariables are valid as DebugInfoDesc objects.
class DIVerifier {
  enum {
    Unknown = 0,
    Invalid,
    Valid
  };
  DenseMap<GlobalVariable *, unsigned> Validity; // Tracks prior results.
  std::map<unsigned, unsigned> Counts; // Count of fields per Tag type.
public:
  DIVerifier()
    : Validity(), Counts()
  {}
  
  /// Verify - Return true if the GlobalVariable appears to be a valid
  /// serialization of a DebugInfoDesc.
  bool Verify(Value *V);
  bool Verify(GlobalVariable *GV);

  /// isVerified - Return true if the specified GV has already been
  /// verified as a debug information descriptor.
  bool isVerified(GlobalVariable *GV);
};

//===----------------------------------------------------------------------===//
/// SourceLineInfo - This class is used to record source line correspondence.
///
class SourceLineInfo {
  unsigned Line;                        // Source line number.
  unsigned Column;                      // Source column.
  unsigned SourceID;                    // Source ID number.
  unsigned LabelID;                     // Label in code ID number.
public:
  SourceLineInfo(unsigned L, unsigned C, unsigned S, unsigned I)
  : Line(L), Column(C), SourceID(S), LabelID(I) {}
  
  // Accessors
  unsigned getLine()     const { return Line; }
  unsigned getColumn()   const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  unsigned getLabelID()  const { return LabelID; }
};

//===----------------------------------------------------------------------===//
/// SourceFileInfo - This class is used to track source information.
///
class SourceFileInfo {
  unsigned DirectoryID;                 // Directory ID number.
  std::string Name;                     // File name (not including directory.)
public:
  SourceFileInfo(unsigned D, const std::string &N) : DirectoryID(D), Name(N) {}
            
  // Accessors
  unsigned getDirectoryID()    const { return DirectoryID; }
  const std::string &getName() const { return Name; }

  /// operator== - Used by UniqueVector to locate entry.
  ///
  bool operator==(const SourceFileInfo &SI) const {
    return getDirectoryID() == SI.getDirectoryID() && getName() == SI.getName();
  }

  /// operator< - Used by UniqueVector to locate entry.
  ///
  bool operator<(const SourceFileInfo &SI) const {
    return getDirectoryID() < SI.getDirectoryID() ||
          (getDirectoryID() == SI.getDirectoryID() && getName() < SI.getName());
  }
};

//===----------------------------------------------------------------------===//
/// DebugVariable - This class is used to track local variable information.
///
class DebugVariable {
private:
  VariableDesc *Desc;                   // Variable Descriptor.
  unsigned FrameIndex;                  // Variable frame index.

public:
  DebugVariable(VariableDesc *D, unsigned I)
  : Desc(D)
  , FrameIndex(I)
  {}
  
  // Accessors.
  VariableDesc *getDesc()  const { return Desc; }
  unsigned getFrameIndex() const { return FrameIndex; }
};

//===----------------------------------------------------------------------===//
/// DebugScope - This class is used to track scope information.
///
class DebugScope {
private:
  DebugScope *Parent;                   // Parent to this scope.
  DebugInfoDesc *Desc;                  // Debug info descriptor for scope.
                                        // Either subprogram or block.
  unsigned StartLabelID;                // Label ID of the beginning of scope.
  unsigned EndLabelID;                  // Label ID of the end of scope.
  std::vector<DebugScope *> Scopes;     // Scopes defined in scope.
  std::vector<DebugVariable *> Variables;// Variables declared in scope.
  
public:
  DebugScope(DebugScope *P, DebugInfoDesc *D)
  : Parent(P)
  , Desc(D)
  , StartLabelID(0)
  , EndLabelID(0)
  , Scopes()
  , Variables()
  {}
  ~DebugScope();
  
  // Accessors.
  DebugScope *getParent()        const { return Parent; }
  DebugInfoDesc *getDesc()       const { return Desc; }
  unsigned getStartLabelID()     const { return StartLabelID; }
  unsigned getEndLabelID()       const { return EndLabelID; }
  std::vector<DebugScope *> &getScopes() { return Scopes; }
  std::vector<DebugVariable *> &getVariables() { return Variables; }
  void setStartLabelID(unsigned S) { StartLabelID = S; }
  void setEndLabelID(unsigned E)   { EndLabelID = E; }
  
  /// AddScope - Add a scope to the scope.
  ///
  void AddScope(DebugScope *S) { Scopes.push_back(S); }
  
  /// AddVariable - Add a variable to the scope.
  ///
  void AddVariable(DebugVariable *V) { Variables.push_back(V); }
};

//===----------------------------------------------------------------------===//
/// LandingPadInfo - This structure is used to retain landing pad info for
/// the current function.
///
struct LandingPadInfo {
  MachineBasicBlock *LandingPadBlock;   // Landing pad block.
  SmallVector<unsigned, 1> BeginLabels; // Labels prior to invoke.
  SmallVector<unsigned, 1> EndLabels;   // Labels after invoke.
  unsigned LandingPadLabel;             // Label at beginning of landing pad.
  Function *Personality;                // Personality function.
  std::vector<int> TypeIds;             // List of type ids (filters negative)

  explicit LandingPadInfo(MachineBasicBlock *MBB)
  : LandingPadBlock(MBB)
  , LandingPadLabel(0)
  , Personality(NULL)  
  {}
};

//===----------------------------------------------------------------------===//
/// MachineModuleInfo - This class contains meta information specific to a
/// module.  Queries can be made by different debugging and exception handling 
/// schemes and reformated for specific use.
///
class MachineModuleInfo : public ImmutablePass {
private:
  // Use the same deserializer/verifier for the module.
  DIDeserializer DR;
  DIVerifier VR;

  // CompileUnits - Uniquing vector for compile units.
  UniqueVector<CompileUnitDesc *> CompileUnits;
  
  // Directories - Uniquing vector for directories.
  UniqueVector<std::string> Directories;
                                         
  // SourceFiles - Uniquing vector for source files.
  UniqueVector<SourceFileInfo> SourceFiles;

  // Lines - List of of source line correspondence.
  std::vector<SourceLineInfo> Lines;
  
  // LabelIDList - One entry per assigned label.  Normally the entry is equal to
  // the list index(+1).  If the entry is zero then the label has been deleted.
  // Any other value indicates the label has been deleted by is mapped to
  // another label.
  std::vector<unsigned> LabelIDList;
  
  // ScopeMap - Tracks the scopes in the current function.
  std::map<DebugInfoDesc *, DebugScope *> ScopeMap;
  
  // RootScope - Top level scope for the current function.
  //
  DebugScope *RootScope;
  
  // FrameMoves - List of moves done by a function's prolog.  Used to construct
  // frame maps by debug and exception handling consumers.
  std::vector<MachineMove> FrameMoves;
  
  // LandingPads - List of LandingPadInfo describing the landing pad information
  // in the current function.
  std::vector<LandingPadInfo> LandingPads;
  
  // TypeInfos - List of C++ TypeInfo used in the current function.
  //
  std::vector<GlobalVariable *> TypeInfos;

  // FilterIds - List of typeids encoding filters used in the current function.
  //
  std::vector<unsigned> FilterIds;

  // FilterEnds - List of the indices in FilterIds corresponding to filter
  // terminators.
  //
  std::vector<unsigned> FilterEnds;

  // Personalities - Vector of all personality functions ever seen. Used to emit
  // common EH frames.
  std::vector<Function *> Personalities;

  // UsedFunctions - the functions in the llvm.used list in a more easily
  // searchable format.
  SmallPtrSet<const Function *, 32> UsedFunctions;

  bool CallsEHReturn;
  bool CallsUnwindInit;
public:
  static char ID; // Pass identification, replacement for typeid

  MachineModuleInfo();
  ~MachineModuleInfo();
  
  /// doInitialization - Initialize the state for a new module.
  ///
  bool doInitialization();
  
  /// doFinalization - Tear down the state after completion of a module.
  ///
  bool doFinalization();
  
  /// BeginFunction - Begin gathering function meta information.
  ///
  void BeginFunction(MachineFunction *MF);
  
  /// EndFunction - Discard function meta information.
  ///
  void EndFunction();

  /// getDescFor - Convert a Value to a debug information descriptor.
  ///
  // FIXME - use new Value type when available.
  DebugInfoDesc *getDescFor(Value *V);
  
  /// Verify - Verify that a Value is debug information descriptor.
  ///
  bool Verify(Value *V) { return VR.Verify(V); }

  /// isVerified - Return true if the specified GV has already been
  /// verified as a debug information descriptor.
  bool isVerified(GlobalVariable *GV) { return VR.isVerified(GV); }
  
  /// AnalyzeModule - Scan the module for global debug information.
  ///
  void AnalyzeModule(Module &M);
  
  /// hasDebugInfo - Returns true if valid debug info is present.
  ///
  bool hasDebugInfo() const { return !CompileUnits.empty(); }
  
  bool callsEHReturn() const { return CallsEHReturn; }
  void setCallsEHReturn(bool b) { CallsEHReturn = b; }

  bool callsUnwindInit() const { return CallsUnwindInit; }
  void setCallsUnwindInit(bool b) { CallsUnwindInit = b; }
  
  /// NextLabelID - Return the next unique label id.
  ///
  unsigned NextLabelID() {
    unsigned ID = (unsigned)LabelIDList.size() + 1;
    LabelIDList.push_back(ID);
    return ID;
  }
  
  /// RecordSourceLine - Records location information and associates it with a
  /// label.  Returns a unique label ID used to generate a label and 
  /// provide correspondence to the source line list.
  unsigned RecordSourceLine(unsigned Line, unsigned Column, unsigned Source);
  
  /// InvalidateLabel - Inhibit use of the specified label # from
  /// MachineModuleInfo, for example because the code was deleted.
  void InvalidateLabel(unsigned LabelID) {
    // Remap to zero to indicate deletion.
    RemapLabel(LabelID, 0);
  }

  /// RemapLabel - Indicate that a label has been merged into another.
  ///
  void RemapLabel(unsigned OldLabelID, unsigned NewLabelID) {
    assert(0 < OldLabelID && OldLabelID <= LabelIDList.size() &&
          "Old label ID out of range.");
    assert(NewLabelID <= LabelIDList.size() &&
          "New label ID out of range.");
    LabelIDList[OldLabelID - 1] = NewLabelID;
  }
  
  /// MappedLabel - Find out the label's final ID.  Zero indicates deletion.
  /// ID != Mapped ID indicates that the label was folded into another label.
  unsigned MappedLabel(unsigned LabelID) const {
    assert(LabelID <= LabelIDList.size() && "Debug label ID out of range.");
    return LabelID ? LabelIDList[LabelID - 1] : 0;
  }

  /// RecordSource - Register a source file with debug info. Returns an source
  /// ID.
  unsigned RecordSource(const std::string &Directory,
                        const std::string &Source);
  unsigned RecordSource(const CompileUnitDesc *CompileUnit);
  
  /// getDirectories - Return the UniqueVector of std::string representing
  /// directories.
  const UniqueVector<std::string> &getDirectories() const {
    return Directories;
  }
  
  /// getSourceFiles - Return the UniqueVector of source files. 
  ///
  const UniqueVector<SourceFileInfo> &getSourceFiles() const {
    return SourceFiles;
  }
  
  /// getSourceLines - Return a vector of source lines.
  ///
  const std::vector<SourceLineInfo> &getSourceLines() const {
    return Lines;
  }
  
  /// SetupCompileUnits - Set up the unique vector of compile units.
  ///
  void SetupCompileUnits(Module &M);

  /// getCompileUnits - Return a vector of debug compile units.
  ///
  const UniqueVector<CompileUnitDesc *> getCompileUnits() const;
  
  /// getGlobalVariablesUsing - Return all of the GlobalVariables that use the
  /// named GlobalVariable.
  void getGlobalVariablesUsing(Module &M, const std::string &RootName,
                               std::vector<GlobalVariable*> &Result);

  /// getAnchoredDescriptors - Return a vector of anchored debug descriptors.
  ///
  void getAnchoredDescriptors(Module &M, const AnchoredDesc *Desc,
                              std::vector<void*> &AnchoredDescs);
  
  /// RecordRegionStart - Indicate the start of a region.
  ///
  unsigned RecordRegionStart(Value *V);

  /// RecordRegionEnd - Indicate the end of a region.
  ///
  unsigned RecordRegionEnd(Value *V);

  /// RecordVariable - Indicate the declaration of  a local variable.
  ///
  void RecordVariable(GlobalValue *GV, unsigned FrameIndex);
  
  /// getRootScope - Return current functions root scope.
  ///
  DebugScope *getRootScope() { return RootScope; }
  
  /// getOrCreateScope - Returns the scope associated with the given descriptor.
  ///
  DebugScope *getOrCreateScope(DebugInfoDesc *ScopeDesc);
  
  /// getFrameMoves - Returns a reference to a list of moves done in the current
  /// function's prologue.  Used to construct frame maps for debug and exception
  /// handling comsumers.
  std::vector<MachineMove> &getFrameMoves() { return FrameMoves; }
  
  //===-EH-----------------------------------------------------------------===//

  /// getOrCreateLandingPadInfo - Find or create an LandingPadInfo for the
  /// specified MachineBasicBlock.
  LandingPadInfo &getOrCreateLandingPadInfo(MachineBasicBlock *LandingPad);

  /// addInvoke - Provide the begin and end labels of an invoke style call and
  /// associate it with a try landing pad block.
  void addInvoke(MachineBasicBlock *LandingPad, unsigned BeginLabel,
                                                unsigned EndLabel);
  
  /// addLandingPad - Add a new panding pad.  Returns the label ID for the 
  /// landing pad entry.
  unsigned addLandingPad(MachineBasicBlock *LandingPad);
  
  /// addPersonality - Provide the personality function for the exception
  /// information.
  void addPersonality(MachineBasicBlock *LandingPad, Function *Personality);

  /// getPersonalityIndex - Get index of the current personality function inside
  /// Personalitites array
  unsigned getPersonalityIndex() const;

  /// getPersonalities - Return array of personality functions ever seen.
  const std::vector<Function *>& getPersonalities() const {
    return Personalities;
  }

  // UsedFunctions - Return set of the functions in the llvm.used list.
  const SmallPtrSet<const Function *, 32>& getUsedFunctions() const {
    return UsedFunctions;
  }

  /// addCatchTypeInfo - Provide the catch typeinfo for a landing pad.
  ///
  void addCatchTypeInfo(MachineBasicBlock *LandingPad,
                        std::vector<GlobalVariable *> &TyInfo);

  /// addFilterTypeInfo - Provide the filter typeinfo for a landing pad.
  ///
  void addFilterTypeInfo(MachineBasicBlock *LandingPad,
                         std::vector<GlobalVariable *> &TyInfo);

  /// addCleanup - Add a cleanup action for a landing pad.
  ///
  void addCleanup(MachineBasicBlock *LandingPad);

  /// getTypeIDFor - Return the type id for the specified typeinfo.  This is 
  /// function wide.
  unsigned getTypeIDFor(GlobalVariable *TI);

  /// getFilterIDFor - Return the id of the filter encoded by TyIds.  This is
  /// function wide.
  int getFilterIDFor(std::vector<unsigned> &TyIds);

  /// TidyLandingPads - Remap landing pad labels and remove any deleted landing
  /// pads.
  void TidyLandingPads();
                        
  /// getLandingPads - Return a reference to the landing pad info for the
  /// current function.
  const std::vector<LandingPadInfo> &getLandingPads() const {
    return LandingPads;
  }
  
  /// getTypeInfos - Return a reference to the C++ typeinfo for the current
  /// function.
  const std::vector<GlobalVariable *> &getTypeInfos() const {
    return TypeInfos;
  }

  /// getFilterIds - Return a reference to the typeids encoding filters used in
  /// the current function.
  const std::vector<unsigned> &getFilterIds() const {
    return FilterIds;
  }

  /// getPersonality - Return a personality function if available.  The presence
  /// of one is required to emit exception handling info.
  Function *getPersonality() const;

  DIDeserializer *getDIDeserializer() { return &DR; }
}; // End class MachineModuleInfo

} // End llvm namespace

#endif
