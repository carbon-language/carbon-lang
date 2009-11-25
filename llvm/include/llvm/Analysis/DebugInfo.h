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
#include "llvm/Support/ValueHandle.h"

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

  /// DIDescriptor - A thin wraper around MDNode to access encoded debug info. This should not
  /// be stored in a container, because underly MDNode may change in certain situations.
  class DIDescriptor {
  protected:
    MDNode  *DbgNode;

    /// DIDescriptor constructor.  If the specified node is non-null, check
    /// to make sure that the tag in the descriptor matches 'RequiredTag'.  If
    /// not, the debug info is corrupt and we ignore it.
    DIDescriptor(MDNode *N, unsigned RequiredTag);

    StringRef getStringField(unsigned Elt) const;
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

    bool isDerivedType() const;
    bool isCompositeType() const;
    bool isBasicType() const;
    bool isVariable() const;
    bool isSubprogram() const;
    bool isGlobalVariable() const;
    bool isScope() const;
    bool isCompileUnit() const;
    bool isLexicalBlock() const;
    bool isSubrange() const;
    bool isEnumerator() const;
    bool isType() const;
    bool isGlobal() const;
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

  /// DIScope - A base class for various scopes.
  class DIScope : public DIDescriptor {
  public:
    explicit DIScope(MDNode *N = 0) : DIDescriptor (N) {
      if (DbgNode && !isScope())
        DbgNode = 0;
    }
    virtual ~DIScope() {}

    StringRef getFilename() const;
    StringRef getDirectory() const;
  };

  /// DICompileUnit - A wrapper for a compile unit.
  class DICompileUnit : public DIScope {
  public:
    explicit DICompileUnit(MDNode *N = 0) : DIScope(N) {
      if (DbgNode && !isCompileUnit())
        DbgNode = 0;
    }

    unsigned getLanguage() const     { return getUnsignedField(2); }
    StringRef getFilename() const  { return getStringField(3);   }
    StringRef getDirectory() const { return getStringField(4);   }
    StringRef getProducer() const  { return getStringField(5);   }

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
    StringRef getFlags() const       { return getStringField(8);   }
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

    StringRef getName() const        { return getStringField(1); }
    uint64_t getEnumValue() const      { return getUInt64Field(2); }
  };

  /// DIType - This is a wrapper for a type.
  /// FIXME: Types should be factored much better so that CV qualifiers and
  /// others do not require a huge and empty descriptor full of zeros.
  class DIType : public DIDescriptor {
  public:
    enum {
      FlagPrivate          = 1 << 0,
      FlagProtected        = 1 << 1,
      FlagFwdDecl          = 1 << 2,
      FlagAppleBlock       = 1 << 3,
      FlagBlockByrefStruct = 1 << 4
    };

  protected:
    DIType(MDNode *N, unsigned Tag)
      : DIDescriptor(N, Tag) {}
    // This ctor is used when the Tag has already been validated by a derived
    // ctor.
    DIType(MDNode *N, bool, bool) : DIDescriptor(N) {}

  public:

    /// Verify - Verify that a type descriptor is well formed.
    bool Verify() const;
  public:
    explicit DIType(MDNode *N);
    explicit DIType() {}
    virtual ~DIType() {}

    DIDescriptor getContext() const     { return getDescriptorField(1); }
    StringRef getName() const         { return getStringField(2);     }
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
    bool isBlockByrefStruct() const {
      return (getFlags() & FlagBlockByrefStruct) != 0;
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
      if (DbgNode && !isDerivedType())
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
      if (N && !isCompositeType())
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

  public:
    virtual ~DIGlobal() {}

    DIDescriptor getContext() const     { return getDescriptorField(2); }
    StringRef getName() const         { return getStringField(3); }
    StringRef getDisplayName() const  { return getStringField(4); }
    StringRef getLinkageName() const  { return getStringField(5); }
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
  class DISubprogram : public DIScope {
  public:
    explicit DISubprogram(MDNode *N = 0) : DIScope(N) {
      if (DbgNode && !isSubprogram())
        DbgNode = 0;
    }

    DIDescriptor getContext() const     { return getDescriptorField(2); }
    StringRef getName() const         { return getStringField(3); }
    StringRef getDisplayName() const  { return getStringField(4); }
    StringRef getLinkageName() const  { return getStringField(5); }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(6); }
    unsigned getLineNumber() const      { return getUnsignedField(7); }
    DICompositeType getType() const { return getFieldAs<DICompositeType>(8); }

    /// getReturnTypeName - Subprogram return types are encoded either as
    /// DIType or as DICompositeType.
    StringRef getReturnTypeName() const {
      DICompositeType DCT(getFieldAs<DICompositeType>(8));
      if (!DCT.isNull()) {
        DIArray A = DCT.getTypeArray();
        DIType T(A.getElement(0).getNode());
        return T.getName();
      }
      DIType T(getFieldAs<DIType>(8));
      return T.getName();
    }

    /// isLocalToUnit - Return true if this subprogram is local to the current
    /// compile unit, like 'static' in C.
    unsigned isLocalToUnit() const     { return getUnsignedField(9); }
    unsigned isDefinition() const      { return getUnsignedField(10); }
    StringRef getFilename() const    { return getCompileUnit().getFilename();}
    StringRef getDirectory() const   { return getCompileUnit().getDirectory();}

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
      if (DbgNode && !isVariable())
        DbgNode = 0;
    }

    DIDescriptor getContext() const { return getDescriptorField(1); }
    StringRef getName() const     { return getStringField(2);     }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(3); }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    DIType getType() const              { return getFieldAs<DIType>(5); }


    /// Verify - Verify that a variable descriptor is well formed.
    bool Verify() const;

    /// HasComplexAddr - Return true if the variable has a complex address.
    bool hasComplexAddress() const {
      return getNumAddrElements() > 0;
    }

    unsigned getNumAddrElements() const { return DbgNode->getNumElements()-6; }

    uint64_t getAddrElement(unsigned Idx) const {
      return getUInt64Field(Idx+6);
    }

    /// isBlockByrefVariable - Return true if the variable was declared as
    /// a "__block" variable (Apple Blocks).
    bool isBlockByrefVariable() const {
      return getType().isBlockByrefStruct();
    }

    /// dump - print variable.
    void dump() const;
  };

  /// DILexicalBlock - This is a wrapper for a lexical block.
  class DILexicalBlock : public DIScope {
  public:
    explicit DILexicalBlock(MDNode *N = 0) : DIScope(N) {
      if (DbgNode && !isLexicalBlock())
        DbgNode = 0;
    }
    DIScope getContext() const       { return getFieldAs<DIScope>(1); }
    StringRef getDirectory() const { return getContext().getDirectory(); }
    StringRef getFilename() const  { return getContext().getFilename(); }
  };

  /// DILocation - This object holds location information. This object
  /// is not associated with any DWARF tag.
  class DILocation : public DIDescriptor {
  public:
    explicit DILocation(MDNode *N) : DIDescriptor(N) { ; }

    unsigned getLineNumber() const     { return getUnsignedField(0); }
    unsigned getColumnNumber() const   { return getUnsignedField(1); }
    DIScope  getScope() const          { return getFieldAs<DIScope>(2); }
    DILocation getOrigLocation() const { return getFieldAs<DILocation>(3); }
    StringRef getFilename() const    { return getScope().getFilename(); }
    StringRef getDirectory() const   { return getScope().getDirectory(); }
  };

  /// DIFactory - This object assists with the construction of the various
  /// descriptors.
  class DIFactory {
    Module &M;
    LLVMContext& VMContext;

    const Type *EmptyStructPtr; // "{}*".
    Function *DeclareFn;     // llvm.dbg.declare

    DIFactory(const DIFactory &);     // DO NOT IMPLEMENT
    void operator=(const DIFactory&); // DO NOT IMPLEMENT
  public:
    enum ComplexAddrKind { OpPlus=1, OpDeref };

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
                                    StringRef Filename,
                                    StringRef Directory,
                                    StringRef Producer,
                                    bool isMain = false,
                                    bool isOptimized = false,
                                    StringRef Flags = "",
                                    unsigned RunTimeVer = 0);

    /// CreateEnumerator - Create a single enumerator value.
    DIEnumerator CreateEnumerator(StringRef Name, uint64_t Val);

    /// CreateBasicType - Create a basic type like int, float, etc.
    DIBasicType CreateBasicType(DIDescriptor Context, StringRef Name,
                                DICompileUnit CompileUnit, unsigned LineNumber,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                unsigned Encoding);

    /// CreateBasicType - Create a basic type like int, float, etc.
    DIBasicType CreateBasicTypeEx(DIDescriptor Context, StringRef Name,
                                DICompileUnit CompileUnit, unsigned LineNumber,
                                Constant *SizeInBits, Constant *AlignInBits,
                                Constant *OffsetInBits, unsigned Flags,
                                unsigned Encoding);

    /// CreateDerivedType - Create a derived type like const qualified type,
    /// pointer, typedef, etc.
    DIDerivedType CreateDerivedType(unsigned Tag, DIDescriptor Context,
                                    StringRef Name,
                                    DICompileUnit CompileUnit,
                                    unsigned LineNumber,
                                    uint64_t SizeInBits, uint64_t AlignInBits,
                                    uint64_t OffsetInBits, unsigned Flags,
                                    DIType DerivedFrom);

    /// CreateDerivedType - Create a derived type like const qualified type,
    /// pointer, typedef, etc.
    DIDerivedType CreateDerivedTypeEx(unsigned Tag, DIDescriptor Context,
                                        StringRef Name,
                                    DICompileUnit CompileUnit,
                                    unsigned LineNumber,
                                    Constant *SizeInBits, Constant *AlignInBits,
                                    Constant *OffsetInBits, unsigned Flags,
                                    DIType DerivedFrom);

    /// CreateCompositeType - Create a composite type like array, struct, etc.
    DICompositeType CreateCompositeType(unsigned Tag, DIDescriptor Context,
                                        StringRef Name,
                                        DICompileUnit CompileUnit,
                                        unsigned LineNumber,
                                        uint64_t SizeInBits,
                                        uint64_t AlignInBits,
                                        uint64_t OffsetInBits, unsigned Flags,
                                        DIType DerivedFrom,
                                        DIArray Elements,
                                        unsigned RunTimeLang = 0);

    /// CreateCompositeType - Create a composite type like array, struct, etc.
    DICompositeType CreateCompositeTypeEx(unsigned Tag, DIDescriptor Context,
                                        StringRef Name,
                                        DICompileUnit CompileUnit,
                                        unsigned LineNumber,
                                        Constant *SizeInBits,
                                        Constant *AlignInBits,
                                        Constant *OffsetInBits, unsigned Flags,
                                        DIType DerivedFrom,
                                        DIArray Elements,
                                        unsigned RunTimeLang = 0);

    /// CreateSubprogram - Create a new descriptor for the specified subprogram.
    /// See comments in DISubprogram for descriptions of these fields.
    DISubprogram CreateSubprogram(DIDescriptor Context, StringRef Name,
                                  StringRef DisplayName,
                                  StringRef LinkageName,
                                  DICompileUnit CompileUnit, unsigned LineNo,
                                  DIType Type, bool isLocalToUnit,
                                  bool isDefinition);

    /// CreateGlobalVariable - Create a new descriptor for the specified global.
    DIGlobalVariable
    CreateGlobalVariable(DIDescriptor Context, StringRef Name,
                         StringRef DisplayName,
                         StringRef LinkageName,
                         DICompileUnit CompileUnit,
                         unsigned LineNo, DIType Type, bool isLocalToUnit,
                         bool isDefinition, llvm::GlobalVariable *GV);

    /// CreateVariable - Create a new descriptor for the specified variable.
    DIVariable CreateVariable(unsigned Tag, DIDescriptor Context,
                              StringRef Name,
                              DICompileUnit CompileUnit, unsigned LineNo,
                              DIType Type);

    /// CreateComplexVariable - Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    DIVariable CreateComplexVariable(unsigned Tag, DIDescriptor Context,
                                     const std::string &Name,
                                     DICompileUnit CompileUnit, unsigned LineNo,
                                     DIType Type,
                                     SmallVector<Value *, 9> &addr);

    /// CreateLexicalBlock - This creates a descriptor for a lexical block
    /// with the specified parent context.
    DILexicalBlock CreateLexicalBlock(DIDescriptor Context);

    /// CreateLocation - Creates a debug info location.
    DILocation CreateLocation(unsigned LineNo, unsigned ColumnNo,
                              DIScope S, DILocation OrigLoc);

    /// CreateLocation - Creates a debug info location.
    DILocation CreateLocation(unsigned LineNo, unsigned ColumnNo,
                              DIScope S, MDNode *OrigLoc = 0);

    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    Instruction *InsertDeclare(llvm::Value *Storage, DIVariable D,
                               BasicBlock *InsertAtEnd);

    /// InsertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    Instruction *InsertDeclare(llvm::Value *Storage, DIVariable D,
                               Instruction *InsertBefore);

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
  /// from DILocation.
  DebugLoc ExtractDebugLocation(DILocation &Loc,
                                DebugLocTracker &DebugLocInfo);

  /// ExtractDebugLocation - Extract debug location information
  /// from llvm.dbg.func_start intrinsic.
  DebugLoc ExtractDebugLocation(DbgFuncStartInst &FSI,
                                DebugLocTracker &DebugLocInfo);

  /// getDISubprogram - Find subprogram that is enclosing this scope.
  DISubprogram getDISubprogram(MDNode *Scope);

  /// getDICompositeType - Find underlying composite type.
  DICompositeType getDICompositeType(DIType T);

  class DebugInfoFinder {

  public:
    /// processModule - Process entire module and collect debug info
    /// anchors.
    void processModule(Module &M);

  private:
    /// processType - Process DIType.
    void processType(DIType DT);

    /// processLexicalBlock - Process DILexicalBlock.
    void processLexicalBlock(DILexicalBlock LB);

    /// processSubprogram - Process DISubprogram.
    void processSubprogram(DISubprogram SP);

    /// processDeclare - Process DbgDeclareInst.
    void processDeclare(DbgDeclareInst *DDI);

    /// processLocation - Process DILocation.
    void processLocation(DILocation Loc);

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
