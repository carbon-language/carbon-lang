//===--- llvm/Analysis/DebugInfo.h - Debug Information Helpers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a bunch of datatypes that are useful for creating and
// walking debug info in LLVM IR form. They essentially provide wrappers around
// the information in the global variables that's needed when constructing the
// DWARF information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DEBUGINFO_H
#define LLVM_ANALYSIS_DEBUGINFO_H

#include "llvm/Metadata.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {
  class BasicBlock;
  class Constant;
  class Function;
  class GlobalVariable;
  class Module;
  class Type;
  class Value;
  struct DbgStopPointInst;
  struct DbgDeclareInst;
  struct DbgFuncStartInst;
  struct DbgRegionStartInst;
  struct DbgRegionEndInst;
  class DebugLoc;
  struct DebugLocTracker;
  class Instruction;
  class LLVMContext;

  class DIDescriptor {
  protected:    
    MDNode *DbgNode;

    /// DIDescriptor constructor.  If the specified node is non-null, check
    /// to make sure that the tag in the descriptor matches 'RequiredTag'.  If
    /// not, the debug info is corrupt and we ignore it.
    DIDescriptor(MDNode *N, unsigned RequiredTag);

    const std::string &getStringField(unsigned Elt, std::string &Result) const;
    unsigned getUnsignedField(unsigned Elt) const {
      return (unsigned)getUInt64Field(Elt);
    }
    uint64_t getUInt64Field(unsigned Elt) const;
    DIDescriptor getDescriptorField(unsigned Elt) const;

    template <typename DescTy>
    DescTy getFieldAs(unsigned Elt) const {
      return DescTy(getDescriptorField(Elt).getNode());
    }

    GlobalVariable *getGlobalVariableField(unsigned Elt) const;

  public:
    explicit DIDescriptor() : DbgNode(0) {}
    explicit DIDescriptor(MDNode *N) : DbgNode(N) {}

    bool isNull() const { return DbgNode == 0; }

    MDNode *getNode() const { return DbgNode; }

    unsigned getVersion() const {
      return getUnsignedField(0) & LLVMDebugVersionMask;
    }

    unsigned getTag() const {
      return getUnsignedField(0) & ~LLVMDebugVersionMask;
    }

    /// ValidDebugInfo - Return true if N represents valid debug info value.
    static bool ValidDebugInfo(MDNode *N, CodeGenOpt::Level OptLevel);

    /// dump - print descriptor.
    void dump() const;
  };

  /// DISubrange - This is used to represent ranges, for array bounds.
  class DISubrange : public DIDescriptor {
  public:
    explicit DISubrange(MDNode *N = 0)
      : DIDescriptor(N, dwarf::DW_TAG_subrange_type) {}

    int64_t getLo() const { return (int64_t)getUInt64Field(1); }
    int64_t getHi() const { return (int64_t)getUInt64Field(2); }
  };

  /// DIArray - This descriptor holds an array of descriptors.
  class DIArray : public DIDescriptor {
  public:
    explicit DIArray(MDNode *N = 0) 
      : DIDescriptor(N) {}

    unsigned getNumElements() const;
    DIDescriptor getElement(unsigned Idx) const {
      return getDescriptorField(Idx);
    }
  };

  /// DICompileUnit - A wrapper for a compile unit.
  class DICompileUnit : public DIDescriptor {
  public:
    explicit DICompileUnit(MDNode *N = 0)
      : DIDescriptor(N, dwarf::DW_TAG_compile_unit) {}

    unsigned getLanguage() const     { return getUnsignedField(2); }
    const std::string &getFilename(std::string &F) const {
      return getStringField(3, F);
    }
    const std::string &getDirectory(std::string &F) const {
      return getStringField(4, F);
    }
    const std::string &getProducer(std::string &F) const {
      return getStringField(5, F);
    }
    
    /// isMain - Each input file is encoded as a separate compile unit in LLVM
    /// debugging information output. However, many target specific tool chains
    /// prefer to encode only one compile unit in an object file. In this 
    /// situation, the LLVM code generator will include  debugging information
    /// entities in the compile unit that is marked as main compile unit. The 
    /// code generator accepts maximum one main compile unit per module. If a
    /// module does not contain any main compile unit then the code generator 
    /// will emit multiple compile units in the output object file.

    bool isMain() const                { return getUnsignedField(6); }
    bool isOptimized() const           { return getUnsignedField(7); }
    const std::string &getFlags(std::string &F) const {
      return getStringField(8, F);
    }
    unsigned getRunTimeVersion() const { return getUnsignedField(9); }

    /// Verify - Verify that a compile unit is well formed.
    bool Verify() const;

    /// dump - print compile unit.
    void dump() const;
  };

  /// DIEnumerator - A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
  /// FIXME: it seems strange that this doesn't have either a reference to the
  /// type/precision or a file/line pair for location info.
  class DIEnumerator : public DIDescriptor {
  public:
    explicit DIEnumerator(MDNode *N = 0)
      : DIDescriptor(N, dwarf::DW_TAG_enumerator) {}

    const std::string &getName(std::string &F) const {
      return getStringField(1, F);
    }
    uint64_t getEnumValue() const { return getUInt64Field(2); }
  };

  /// DIType - This is a wrapper for a type.
  /// FIXME: Types should be factored much better so that CV qualifiers and
  /// others do not require a huge and empty descriptor full of zeros.
  class DIType : public DIDescriptor {
  public:
    enum {
      FlagPrivate    = 1 << 0,
      FlagProtected  = 1 << 1,
      FlagFwdDecl    = 1 << 2,
      FlagAppleBlock = 1 << 3
    };

  protected:
    DIType(MDNode *N, unsigned Tag) 
      : DIDescriptor(N, Tag) {}
    // This ctor is used when the Tag has already been validated by a derived
    // ctor.
    DIType(MDNode *N, bool, bool) : DIDescriptor(N) {}

  public:
    /// isDerivedType - Return true if the specified tag is legal for
    /// DIDerivedType.
    static bool isDerivedType(unsigned TAG);

    /// isCompositeType - Return true if the specified tag is legal for
    /// DICompositeType.
    static bool isCompositeType(unsigned TAG);

    /// isBasicType - Return true if the specified tag is legal for
    /// DIBasicType.
    static bool isBasicType(unsigned TAG) {
      return TAG == dwarf::DW_TAG_base_type;
    }

    /// Verify - Verify that a type descriptor is well formed.
    bool Verify() const;
  public:
    explicit DIType(MDNode *N);
    explicit DIType() {}
    virtual ~DIType() {}

    DIDescriptor getContext() const     { return getDescriptorField(1); }
    const std::string &getName(std::string &F) const {
      return getStringField(2, F);
    }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(3); }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    uint64_t getSizeInBits() const      { return getUInt64Field(5); }
    uint64_t getAlignInBits() const     { return getUInt64Field(6); }
    // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
    // carry this is just plain insane.
    uint64_t getOffsetInBits() const    { return getUInt64Field(7); }
    unsigned getFlags() const           { return getUnsignedField(8); }
    bool isPrivate() const {
      return (getFlags() & FlagPrivate) != 0; 
    }
    bool isProtected() const {
      return (getFlags() & FlagProtected) != 0; 
    }
    bool isForwardDecl() const {
      return (getFlags() & FlagFwdDecl) != 0; 
    }
    // isAppleBlock - Return true if this is the Apple Blocks extension.
    bool isAppleBlockExtension() const {
      return (getFlags() & FlagAppleBlock) != 0; 
    }

    /// dump - print type.
    void dump() const;
  };

  /// DIBasicType - A basic type, like 'int' or 'float'.
  class DIBasicType : public DIType {
  public:
    explicit DIBasicType(MDNode *N = 0)
      : DIType(N, dwarf::DW_TAG_base_type) {}

    unsigned getEncoding() const { return getUnsignedField(9); }

    /// dump - print basic type.
    void dump() const;
  };

  /// DIDerivedType - A simple derived type, like a const qualified type,
  /// a typedef, a pointer or reference, etc.
  class DIDerivedType : public DIType {
  protected:
    explicit DIDerivedType(MDNode *N, bool, bool)
      : DIType(N, true, true) {}
  public:
    explicit DIDerivedType(MDNode *N = 0)
      : DIType(N, true, true) {
      if (DbgNode && !isDerivedType(getTag()))
        DbgNode = 0;
    }

    DIType getTypeDerivedFrom() const { return getFieldAs<DIType>(9); }

    /// getOriginalTypeSize - If this type is derived from a base type then
    /// return base type size.
    uint64_t getOriginalTypeSize() const;
    /// dump - print derived type.
    void dump() const;

    /// replaceAllUsesWith - Replace all uses of debug info referenced by
    /// this descriptor. After this completes, the current debug info value
    /// is erased.
    void replaceAllUsesWith(DIDescriptor &D);
  };

  /// DICompositeType - This descriptor holds a type that can refer to multiple
  /// other types, like a function or struct.
  /// FIXME: Why is this a DIDerivedType??
  class DICompositeType : public DIDerivedType {
  public:
    explicit DICompositeType(MDNode *N = 0)
      : DIDerivedType(N, true, true) {
      if (N && !isCompositeType(getTag()))
        DbgNode = 0;
    }

    DIArray getTypeArray() const { return getFieldAs<DIArray>(10); }
    unsigned getRunTimeLang() const { return getUnsignedField(11); }

    /// Verify - Verify that a composite type descriptor is well formed.
    bool Verify() const;

    /// dump - print composite type.
    void dump() const;
  };

  /// DIGlobal - This is a common class for global variables and subprograms.
  class DIGlobal : public DIDescriptor {
  protected:
    explicit DIGlobal(MDNode *N, unsigned RequiredTag)
      : DIDescriptor(N, RequiredTag) {}

    /// isSubprogram - Return true if the specified tag is legal for
    /// DISubprogram.
    static bool isSubprogram(unsigned TAG) {
      return TAG == dwarf::DW_TAG_subprogram;
    }

    /// isGlobalVariable - Return true if the specified tag is legal for
    /// DIGlobalVariable.
    static bool isGlobalVariable(unsigned TAG) {
      return TAG == dwarf::DW_TAG_variable;
    }

  public:
    virtual ~DIGlobal() {}

    DIDescriptor getContext() const     { return getDescriptorField(2); }
    const std::string &getName(std::string &F) const {
      return getStringField(3, F);
    }
    const std::string &getDisplayName(std::string &F) const {
      return getStringField(4, F);
    }
    const std::string &getLinkageName(std::string &F) const {
      return getStringField(5, F);
    }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(6); }
    unsigned getLineNumber() const      { return getUnsignedField(7); }
    DIType getType() const              { return getFieldAs<DIType>(8); }

    /// isLocalToUnit - Return true if this subprogram is local to the current
    /// compile unit, like 'static' in C.
    unsigned isLocalToUnit() const      { return getUnsignedField(9); }
    unsigned isDefinition() const       { return getUnsignedField(10); }

    /// dump - print global.
    void dump() const;
  };

  /// DISubprogram - This is a wrapper for a subprogram (e.g. a function).
  class DISubprogram : public DIGlobal {
  public:
    explicit DISubprogram(MDNode *N = 0)
      : DIGlobal(N, dwarf::DW_TAG_subprogram) {}

    DICompositeType getType() const { return getFieldAs<DICompositeType>(8); }

    /// getReturnTypeName - Subprogram return types are encoded either as
    /// DIType or as DICompositeType.
    const std::string &getReturnTypeName(std::string &F) const {
      DICompositeType DCT(getFieldAs<DICompositeType>(8));
      if (!DCT.isNull()) {
        DIArray A = DCT.getTypeArray();
        DIType T(A.getElement(0).getNode());
        return T.getName(F);
      }
      DIType T(getFieldAs<DIType>(8));
      return T.getName(F);
    }

    /// Verify - Verify that a subprogram descriptor is well formed.
    bool Verify() const;

    /// dump - print subprogram.
    void dump() const;

    /// describes - Return true if this subprogram provides debugging
    /// information for the function F.
    bool describes(const Function *F);
  };

  /// DIGlobalVariable - This is a wrapper for a global variable.
  class DIGlobalVariable : public DIGlobal {
  public:
    explicit DIGlobalVariable(MDNode *N = 0)
      : DIGlobal(N, dwarf::DW_TAG_variable) {}

    GlobalVariable *getGlobal() const { return getGlobalVariableField(11); }

    /// Verify - Verify that a global variable descriptor is well formed.
    bool Verify() const;

    /// dump - print global variable.
    void dump() const;
  };

  /// DIVariable - This is a wrapper for a variable (e.g. parameter, local,
  /// global etc).
  class DIVariable : public DIDescriptor {
  public:
    explicit DIVariable(MDNode *N = 0)
      : DIDescriptor(N) {
      if (DbgNode && !isVariable(getTag()))
        DbgNode = 0;
    }

    DIDescriptor getContext() const { return getDescriptorField(1); }
    const std::string &getName(std::string &F) const {
      return getStringField(2, F);
    }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(3); }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    DIType getType() const              { return getFieldAs<DIType>(5); }

    /// isVariable - Return true if the specified tag is legal for DIVariable.
    static bool isVariable(unsigned Tag);

    /// Verify - Verify that a variable descriptor is well formed.
    bool Verify() const;

    /// dump - print variable.
    void dump() const;
  };

  /// DIBlock - This is a wrapper for a block (e.g. a function, scope, etc).
  class DIBlock : public DIDescriptor {
  public:
    explicit DIBlock(MDNode *N = 0)
      : DIDescriptor(N, dwarf::DW_TAG_lexical_block) {}

    DIDescriptor getContext() const { return getDescriptorField(1); }
  };

  /// DIFactory - This object assists with the construction of the various
  /// descriptors.
  class DIFactory {
    Module &M;
    LLVMContext& VMContext;
    
    // Cached values for uniquing and faster lookups.
    const Type *EmptyStructPtr; // "{}*".
    Function *StopPointFn;   // llvm.dbg.stoppoint
    Function *FuncStartFn;   // llvm.dbg.func.start
    Function *RegionStartFn; // llvm.dbg.region.start
    Function *RegionEndFn;   // llvm.dbg.region.end
    Function *DeclareFn;     // llvm.dbg.declare
    StringMap<Constant*> StringCache;
    DenseMap<Constant*, DIDescriptor> SimpleConstantCache;

    DIFactory(const DIFactory &);     // DO NOT IMPLEMENT
    void operator=(const DIFactory&); // DO NOT IMPLEMENT
  public:
    explicit DIFactory(Module &m);

    /// GetOrCreateArray - Create an descriptor for an array of descriptors. 
    /// This implicitly uniques the arrays created.
    DIArray GetOrCreateArray(DIDescriptor *Tys, unsigned NumTys);

    /// GetOrCreateSubrange - Create a descriptor for a value range.  This
    /// implicitly uniques the values returned.
    DISubrange GetOrCreateSubrange(int64_t Lo, int64_t Hi);

    /// CreateCompileUnit - Create a new descriptor for the specified compile
    /// unit.
    DICompileUnit CreateCompileUnit(unsigned LangID,
                                    const std::string &Filename,
                                    const std::string &Directory,
                                    const std::string &Producer,
                                    bool isMain = false,
                                    bool isOptimized = false,
                                    const char *Flags = "",
                                    unsigned RunTimeVer = 0);

    /// CreateEnumerator - Create a single enumerator value.
    DIEnumerator CreateEnumerator(const std::string &Name, uint64_t Val);

    /// CreateBasicType - Create a basic type like int, float, etc.
    DIBasicType CreateBasicType(DIDescriptor Context, const std::string &Name,
                                DICompileUnit CompileUnit, unsigned LineNumber,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                unsigned Encoding);

    /// CreateDerivedType - Create a derived type like const qualified type,
    /// pointer, typedef, etc.
    DIDerivedType CreateDerivedType(unsigned Tag, DIDescriptor Context,
                                    const std::string &Name,
                                    DICompileUnit CompileUnit,
                                    unsigned LineNumber,
                                    uint64_t SizeInBits, uint64_t AlignInBits,
                                    uint64_t OffsetInBits, unsigned Flags,
                                    DIType DerivedFrom);

    /// CreateCompositeType - Create a composite type like array, struct, etc.
    DICompositeType CreateCompositeType(unsigned Tag, DIDescriptor Context,
                                        const std::string &Name,
                                        DICompileUnit CompileUnit,
                                        unsigned LineNumber,
                                        uint64_t SizeInBits,
                                        uint64_t AlignInBits,
                                        uint64_t OffsetInBits, unsigned Flags,
                                        DIType DerivedFrom,
                                        DIArray Elements,
                                        unsigned RunTimeLang = 0);

    /// CreateSubprogram - Create a new descriptor for the specified subprogram.
    /// See comments in DISubprogram for descriptions of these fields.
    DISubprogram CreateSubprogram(DIDescriptor Context, const std::string &Name,
                                  const std::string &DisplayName,
                                  const std::string &LinkageName,
                                  DICompileUnit CompileUnit, unsigned LineNo,
                                  DIType Type, bool isLocalToUnit,
                                  bool isDefinition);

    /// CreateGlobalVariable - Create a new descriptor for the specified global.
    DIGlobalVariable
    CreateGlobalVariable(DIDescriptor Context, const std::string &Name,
                         const std::string &DisplayName,
                         const std::string &LinkageName, 
                         DICompileUnit CompileUnit,
                         unsigned LineNo, DIType Type, bool isLocalToUnit,
                         bool isDefinition, llvm::GlobalVariable *GV);

    /// CreateVariable - Create a new descriptor for the specified variable.
    DIVariable CreateVariable(unsigned Tag, DIDescriptor Context,
                              const std::string &Name,
                              DICompileUnit CompileUnit, unsigned LineNo,
                              DIType Type);

    /// CreateBlock - This creates a descriptor for a lexical block with the
    /// specified parent context.
    DIBlock CreateBlock(DIDescriptor Context);

    /// InsertStopPoint - Create a new llvm.dbg.stoppoint intrinsic invocation,
    /// inserting it at the end of the specified basic block.
    void InsertStopPoint(DICompileUnit CU, unsigned LineNo, unsigned ColNo,
                         BasicBlock *BB);

    /// InsertSubprogramStart - Create a new llvm.dbg.func.start intrinsic to
    /// mark the start of the specified subprogram.
    void InsertSubprogramStart(DISubprogram SP, BasicBlock *BB);

    /// InsertRegionStart - Insert a new llvm.dbg.region.start intrinsic call to
    /// mark the start of a region for the specified scoping descriptor.
    void InsertRegionStart(DIDescriptor D, BasicBlock *BB);

    /// InsertRegionEnd - Insert a new llvm.dbg.region.end intrinsic call to
    /// mark the end of a region for the specified scoping descriptor.
    void InsertRegionEnd(DIDescriptor D, BasicBlock *BB);

    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    void InsertDeclare(llvm::Value *Storage, DIVariable D, BasicBlock *BB);

  private:
    Constant *GetTagConstant(unsigned TAG);
  };

  /// Finds the stoppoint coressponding to this instruction, that is the
  /// stoppoint that dominates this instruction 
  const DbgStopPointInst *findStopPoint(const Instruction *Inst);

  /// Finds the stoppoint corresponding to first real (non-debug intrinsic) 
  /// instruction in this Basic Block, and returns the stoppoint for it.
  const DbgStopPointInst *findBBStopPoint(const BasicBlock *BB);

  /// Finds the dbg.declare intrinsic corresponding to this value if any.
  /// It looks through pointer casts too.
  const DbgDeclareInst *findDbgDeclare(const Value *V, bool stripCasts = true);

  /// Find the debug info descriptor corresponding to this global variable.
  Value *findDbgGlobalDeclare(GlobalVariable *V);

  bool getLocationInfo(const Value *V, std::string &DisplayName, 
                       std::string &Type, unsigned &LineNo, std::string &File,
                       std::string &Dir); 

  /// isValidDebugInfoIntrinsic - Return true if SPI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgStopPointInst &SPI, 
                                 CodeGenOpt::Level OptLev);

  /// isValidDebugInfoIntrinsic - Return true if FSI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgFuncStartInst &FSI,
                                 CodeGenOpt::Level OptLev);

  /// isValidDebugInfoIntrinsic - Return true if RSI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgRegionStartInst &RSI,
                                 CodeGenOpt::Level OptLev);

  /// isValidDebugInfoIntrinsic - Return true if REI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgRegionEndInst &REI,
                                 CodeGenOpt::Level OptLev);

  /// isValidDebugInfoIntrinsic - Return true if DI is a valid debug 
  /// info intrinsic.
  bool isValidDebugInfoIntrinsic(DbgDeclareInst &DI,
                                 CodeGenOpt::Level OptLev);

  /// ExtractDebugLocation - Extract debug location information 
  /// from llvm.dbg.stoppoint intrinsic.
  DebugLoc ExtractDebugLocation(DbgStopPointInst &SPI,
                                DebugLocTracker &DebugLocInfo);

  /// ExtractDebugLocation - Extract debug location information 
  /// from llvm.dbg.func_start intrinsic.
  DebugLoc ExtractDebugLocation(DbgFuncStartInst &FSI,
                                DebugLocTracker &DebugLocInfo);

  /// isInlinedFnStart - Return true if FSI is starting an inlined function.
  bool isInlinedFnStart(DbgFuncStartInst &FSI, const Function *CurrentFn);

  /// isInlinedFnEnd - Return true if REI is ending an inlined function.
  bool isInlinedFnEnd(DbgRegionEndInst &REI, const Function *CurrentFn);
  /// DebugInfoFinder - This object collects DebugInfo from a module.
  class DebugInfoFinder {

  public:
    /// processModule - Process entire module and collect debug info
    /// anchors.
    void processModule(Module &M);
    
  private:
    /// processType - Process DIType.
    void processType(DIType DT);

    /// processSubprogram - Enumberate DISubprogram.
    void processSubprogram(DISubprogram SP);

    /// processStopPoint - Process DbgStopPointInst.
    void processStopPoint(DbgStopPointInst *SPI);

    /// processFuncStart - Process DbgFuncStartInst.
    void processFuncStart(DbgFuncStartInst *FSI);

    /// processRegionStart - Process DbgRegionStart.
    void processRegionStart(DbgRegionStartInst *DRS);

    /// processRegionEnd - Process DbgRegionEnd.
    void processRegionEnd(DbgRegionEndInst *DRE);

    /// processDeclare - Process DbgDeclareInst.
    void processDeclare(DbgDeclareInst *DDI);

    /// addCompileUnit - Add compile unit into CUs.
    bool addCompileUnit(DICompileUnit CU);
    
    /// addGlobalVariable - Add global variable into GVs.
    bool addGlobalVariable(DIGlobalVariable DIG);

    // addSubprogram - Add subprgoram into SPs.
    bool addSubprogram(DISubprogram SP);

    /// addType - Add type into Tys.
    bool addType(DIType DT);

  public:
    typedef SmallVector<MDNode *, 8>::iterator iterator;
    iterator compile_unit_begin()    { return CUs.begin(); }
    iterator compile_unit_end()      { return CUs.end(); }
    iterator subprogram_begin()      { return SPs.begin(); }
    iterator subprogram_end()        { return SPs.end(); }
    iterator global_variable_begin() { return GVs.begin(); }
    iterator global_variable_end()   { return GVs.end(); }
    iterator type_begin()            { return TYs.begin(); }
    iterator type_end()              { return TYs.end(); }

    unsigned compile_unit_count()    { return CUs.size(); }
    unsigned global_variable_count() { return GVs.size(); }
    unsigned subprogram_count()      { return SPs.size(); }
    unsigned type_count()            { return TYs.size(); }

  private:
    SmallVector<MDNode *, 8> CUs;  // Compile Units
    SmallVector<MDNode *, 8> SPs;  // Subprograms
    SmallVector<MDNode *, 8> GVs;  // Global Variables;
    SmallVector<MDNode *, 8> TYs;  // Types
    SmallPtrSet<MDNode *, 64> NodesSeen;
  };
} // end namespace llvm

#endif
