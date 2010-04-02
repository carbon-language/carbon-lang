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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {
  class BasicBlock;
  class Constant;
  class Function;
  class GlobalVariable;
  class Module;
  class Type;
  class Value;
  class DbgDeclareInst;
  class Instruction;
  class MDNode;
  class LLVMContext;

  /// DIDescriptor - A thin wraper around MDNode to access encoded debug info.
  /// This should not be stored in a container, because underly MDNode may
  /// change in certain situations.
  class DIDescriptor {
  protected:
    MDNode *DbgNode;

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

    bool Verify() const { return DbgNode != 0; }

    MDNode *getNode() const { return DbgNode; }

    unsigned getVersion() const {
      return getUnsignedField(0) & LLVMDebugVersionMask;
    }

    unsigned getTag() const {
      return getUnsignedField(0) & ~LLVMDebugVersionMask;
    }

    /// ValidDebugInfo - Return true if N represents valid debug info value.
    static bool ValidDebugInfo(MDNode *N, unsigned OptLevel);

    /// dump - print descriptor.
    void dump() const;

    bool isDerivedType() const;
    bool isCompositeType() const;
    bool isBasicType() const;
    bool isVariable() const;
    bool isSubprogram() const;
    bool isGlobalVariable() const;
    bool isScope() const;
    bool isFile() const;
    bool isCompileUnit() const;
    bool isNameSpace() const;
    bool isLexicalBlock() const;
    bool isSubrange() const;
    bool isEnumerator() const;
    bool isType() const;
    bool isGlobal() const;
  };

  /// DISubrange - This is used to represent ranges, for array bounds.
  class DISubrange : public DIDescriptor {
  public:
    explicit DISubrange(MDNode *N = 0) : DIDescriptor(N) {}

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
    explicit DIScope(MDNode *N = 0) : DIDescriptor (N) {}
    virtual ~DIScope() {}

    StringRef getFilename() const;
    StringRef getDirectory() const;
  };

  /// DICompileUnit - A wrapper for a compile unit.
  class DICompileUnit : public DIScope {
  public:
    explicit DICompileUnit(MDNode *N = 0) : DIScope(N) {}

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

  /// DIFile - This is a wrapper for a file.
  class DIFile : public DIScope {
  public:
    explicit DIFile(MDNode *N = 0) : DIScope(N) {
      if (DbgNode && !isFile())
        DbgNode = 0;
    }
    StringRef getFilename() const  { return getStringField(1);   }
    StringRef getDirectory() const { return getStringField(2);   }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(3); }
  };

  /// DIEnumerator - A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
  /// FIXME: it seems strange that this doesn't have either a reference to the
  /// type/precision or a file/line pair for location info.
  class DIEnumerator : public DIDescriptor {
  public:
    explicit DIEnumerator(MDNode *N = 0) : DIDescriptor(N) {}

    StringRef getName() const        { return getStringField(1); }
    uint64_t getEnumValue() const      { return getUInt64Field(2); }
  };

  /// DIType - This is a wrapper for a type.
  /// FIXME: Types should be factored much better so that CV qualifiers and
  /// others do not require a huge and empty descriptor full of zeros.
  class DIType : public DIScope {
  public:
    enum {
      FlagPrivate          = 1 << 0,
      FlagProtected        = 1 << 1,
      FlagFwdDecl          = 1 << 2,
      FlagAppleBlock       = 1 << 3,
      FlagBlockByrefStruct = 1 << 4,
      FlagVirtual          = 1 << 5,
      FlagArtificial       = 1 << 6  // To identify artificial arguments in
                                     // a subroutine type. e.g. "this" in c++.
    };

  protected:
    // This ctor is used when the Tag has already been validated by a derived
    // ctor.
    DIType(MDNode *N, bool, bool) : DIScope(N) {}

  public:

    /// Verify - Verify that a type descriptor is well formed.
    bool Verify() const;
  public:
    explicit DIType(MDNode *N);
    explicit DIType() {}
    virtual ~DIType() {}

    DIScope getContext() const          { return getFieldAs<DIScope>(1); }
    StringRef getName() const           { return getStringField(2);     }
    DICompileUnit getCompileUnit() const{ 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(3);

      DIFile F = getFieldAs<DIFile>(3);
      return F.getCompileUnit();
    }
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
    bool isVirtual() const {
      return (getFlags() & FlagVirtual) != 0;
    }
    bool isArtificial() const {
      return (getFlags() & FlagArtificial) != 0;
    }
    bool isValid() const {
      return DbgNode && (isBasicType() || isDerivedType() || isCompositeType());
    }
    StringRef getFilename() const    { return getCompileUnit().getFilename();}
    StringRef getDirectory() const   { return getCompileUnit().getDirectory();}
    /// dump - print type.
    void dump() const;
  };

  /// DIBasicType - A basic type, like 'int' or 'float'.
  class DIBasicType : public DIType {
  public:
    explicit DIBasicType(MDNode *N = 0) : DIType(N) {}

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
      : DIType(N, true, true) {}

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
    DICompositeType getContainingType() const {
      return getFieldAs<DICompositeType>(12);
    }

    /// Verify - Verify that a composite type descriptor is well formed.
    bool Verify() const;

    /// dump - print composite type.
    void dump() const;
  };

  /// DIGlobal - This is a common class for global variables and subprograms.
  class DIGlobal : public DIDescriptor {
  protected:
    explicit DIGlobal(MDNode *N) : DIDescriptor(N) {}

  public:
    virtual ~DIGlobal() {}

    DIScope getContext() const          { return getFieldAs<DIScope>(2); }
    StringRef getName() const         { return getStringField(3); }
    StringRef getDisplayName() const  { return getStringField(4); }
    StringRef getLinkageName() const  { return getStringField(5); }
    DICompileUnit getCompileUnit() const{ 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(6);

      DIFile F = getFieldAs<DIFile>(6); 
      return F.getCompileUnit();
    }

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
    explicit DISubprogram(MDNode *N = 0) : DIScope(N) {}

    DIScope getContext() const          { return getFieldAs<DIScope>(2); }
    StringRef getName() const         { return getStringField(3); }
    StringRef getDisplayName() const  { return getStringField(4); }
    StringRef getLinkageName() const  { return getStringField(5); }
    DICompileUnit getCompileUnit() const{ 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(6);

      DIFile F = getFieldAs<DIFile>(6); 
      return F.getCompileUnit();
    }
    unsigned getLineNumber() const      { return getUnsignedField(7); }
    DICompositeType getType() const { return getFieldAs<DICompositeType>(8); }

    /// getReturnTypeName - Subprogram return types are encoded either as
    /// DIType or as DICompositeType.
    StringRef getReturnTypeName() const {
      DICompositeType DCT(getFieldAs<DICompositeType>(8));
      if (DCT.Verify()) {
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

    unsigned getVirtuality() const { return getUnsignedField(11); }
    unsigned getVirtualIndex() const { return getUnsignedField(12); }

    DICompositeType getContainingType() const {
      return getFieldAs<DICompositeType>(13);
    }
    unsigned isArtificial() const    { return getUnsignedField(14); }

    StringRef getFilename() const    { 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getCompileUnit().getFilename();

      DIFile F = getFieldAs<DIFile>(6); 
      return F.getFilename();
    }

    StringRef getDirectory() const   { 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getCompileUnit().getFilename();

      DIFile F = getFieldAs<DIFile>(6); 
      return F.getDirectory();
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
    explicit DIGlobalVariable(MDNode *N = 0) : DIGlobal(N) {}

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
      : DIDescriptor(N) {}

    DIScope getContext() const          { return getFieldAs<DIScope>(1); }
    StringRef getName() const           { return getStringField(2);     }
    DICompileUnit getCompileUnit() const{ 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(3);

      DIFile F = getFieldAs<DIFile>(3); 
      return F.getCompileUnit();
    }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    DIType getType() const              { return getFieldAs<DIType>(5); }


    /// Verify - Verify that a variable descriptor is well formed.
    bool Verify() const;

    /// HasComplexAddr - Return true if the variable has a complex address.
    bool hasComplexAddress() const {
      return getNumAddrElements() > 0;
    }

    unsigned getNumAddrElements() const;
    
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
    explicit DILexicalBlock(MDNode *N = 0) : DIScope(N) {}
    DIScope getContext() const       { return getFieldAs<DIScope>(1);      }
    StringRef getDirectory() const   { return getContext().getDirectory(); }
    StringRef getFilename() const    { return getContext().getFilename();  }
    unsigned getLineNumber() const   { return getUnsignedField(2);         }
    unsigned getColumnNumber() const { return getUnsignedField(3);         }
  };

  /// DINameSpace - A wrapper for a C++ style name space.
  class DINameSpace : public DIScope { 
  public:
    explicit DINameSpace(MDNode *N = 0) : DIScope(N) {}
    DIScope getContext() const     { return getFieldAs<DIScope>(1);      }
    StringRef getName() const      { return getStringField(2);           }
    StringRef getDirectory() const { return getContext().getDirectory(); }
    StringRef getFilename() const  { return getContext().getFilename();  }
    DICompileUnit getCompileUnit() const{ 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(3);

      DIFile F = getFieldAs<DIFile>(3); 
      return F.getCompileUnit();
    }
    unsigned getLineNumber() const { return getUnsignedField(4);         }
  };

  /// DILocation - This object holds location information. This object
  /// is not associated with any DWARF tag.
  class DILocation : public DIDescriptor {
  public:
    explicit DILocation(MDNode *N) : DIDescriptor(N) { }

    unsigned getLineNumber() const     { return getUnsignedField(0); }
    unsigned getColumnNumber() const   { return getUnsignedField(1); }
    DIScope  getScope() const          { return getFieldAs<DIScope>(2); }
    DILocation getOrigLocation() const { return getFieldAs<DILocation>(3); }
    StringRef getFilename() const    { return getScope().getFilename(); }
    StringRef getDirectory() const   { return getScope().getDirectory(); }
    bool Verify() const;
  };

  /// DIFactory - This object assists with the construction of the various
  /// descriptors.
  class DIFactory {
    Module &M;
    LLVMContext& VMContext;

    Function *DeclareFn;     // llvm.dbg.declare
    Function *ValueFn;       // llvm.dbg.value

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

    /// CreateFile -  Create a new descriptor for the specified file.
    DIFile CreateFile(StringRef Filename, StringRef Directory, DICompileUnit CU);

    /// CreateEnumerator - Create a single enumerator value.
    DIEnumerator CreateEnumerator(StringRef Name, uint64_t Val);

    /// CreateBasicType - Create a basic type like int, float, etc.
    DIBasicType CreateBasicType(DIDescriptor Context, StringRef Name,
                                DIFile F, unsigned LineNumber,
                                uint64_t SizeInBits, uint64_t AlignInBits,
                                uint64_t OffsetInBits, unsigned Flags,
                                unsigned Encoding);

    /// CreateBasicType - Create a basic type like int, float, etc.
    DIBasicType CreateBasicTypeEx(DIDescriptor Context, StringRef Name,
                                DIFile F, unsigned LineNumber,
                                Constant *SizeInBits, Constant *AlignInBits,
                                Constant *OffsetInBits, unsigned Flags,
                                unsigned Encoding);

    /// CreateDerivedType - Create a derived type like const qualified type,
    /// pointer, typedef, etc.
    DIDerivedType CreateDerivedType(unsigned Tag, DIDescriptor Context,
                                    StringRef Name,
                                    DIFile F,
                                    unsigned LineNumber,
                                    uint64_t SizeInBits, uint64_t AlignInBits,
                                    uint64_t OffsetInBits, unsigned Flags,
                                    DIType DerivedFrom);

    /// CreateDerivedType - Create a derived type like const qualified type,
    /// pointer, typedef, etc.
    DIDerivedType CreateDerivedTypeEx(unsigned Tag, DIDescriptor Context,
                                      StringRef Name,
                                      DIFile F,
                                      unsigned LineNumber,
                                      Constant *SizeInBits, 
                                      Constant *AlignInBits,
                                      Constant *OffsetInBits, unsigned Flags,
                                      DIType DerivedFrom);

    /// CreateCompositeType - Create a composite type like array, struct, etc.
    DICompositeType CreateCompositeType(unsigned Tag, DIDescriptor Context,
                                        StringRef Name,
                                        DIFile F,
                                        unsigned LineNumber,
                                        uint64_t SizeInBits,
                                        uint64_t AlignInBits,
                                        uint64_t OffsetInBits, unsigned Flags,
                                        DIType DerivedFrom,
                                        DIArray Elements,
                                        unsigned RunTimeLang = 0,
                                        MDNode *ContainingType = 0);

    /// CreateArtificialType - Create a new DIType with "artificial" flag set.
    DIType CreateArtificialType(DIType Ty);

    /// CreateCompositeType - Create a composite type like array, struct, etc.
    DICompositeType CreateCompositeTypeEx(unsigned Tag, DIDescriptor Context,
                                          StringRef Name,
                                          DIFile F,
                                          unsigned LineNumber,
                                          Constant *SizeInBits,
                                          Constant *AlignInBits,
                                          Constant *OffsetInBits, 
                                          unsigned Flags,
                                          DIType DerivedFrom,
                                          DIArray Elements,
                                          unsigned RunTimeLang = 0);

    /// CreateSubprogram - Create a new descriptor for the specified subprogram.
    /// See comments in DISubprogram for descriptions of these fields.
    DISubprogram CreateSubprogram(DIDescriptor Context, StringRef Name,
                                  StringRef DisplayName,
                                  StringRef LinkageName,
                                  DIFile F, unsigned LineNo,
                                  DIType Ty, bool isLocalToUnit,
                                  bool isDefinition,
                                  unsigned VK = 0,
                                  unsigned VIndex = 0,
                                  DIType = DIType(),
                                  bool isArtificial = 0);

    /// CreateSubprogramDefinition - Create new subprogram descriptor for the
    /// given declaration. 
    DISubprogram CreateSubprogramDefinition(DISubprogram &SPDeclaration);

    /// CreateGlobalVariable - Create a new descriptor for the specified global.
    DIGlobalVariable
    CreateGlobalVariable(DIDescriptor Context, StringRef Name,
                         StringRef DisplayName,
                         StringRef LinkageName,
                         DIFile F,
                         unsigned LineNo, DIType Ty, bool isLocalToUnit,
                         bool isDefinition, llvm::GlobalVariable *GV);

    /// CreateVariable - Create a new descriptor for the specified variable.
    DIVariable CreateVariable(unsigned Tag, DIDescriptor Context,
                              StringRef Name,
                              DIFile F, unsigned LineNo,
                              DIType Ty);

    /// CreateComplexVariable - Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    DIVariable CreateComplexVariable(unsigned Tag, DIDescriptor Context,
                                     const std::string &Name,
                                     DIFile F, unsigned LineNo,
                                     DIType Ty,
                                     SmallVector<Value *, 9> &addr);

    /// CreateLexicalBlock - This creates a descriptor for a lexical block
    /// with the specified parent context.
    DILexicalBlock CreateLexicalBlock(DIDescriptor Context, unsigned Line = 0,
                                      unsigned Col = 0);

    /// CreateNameSpace - This creates new descriptor for a namespace
    /// with the specified parent context.
    DINameSpace CreateNameSpace(DIDescriptor Context, StringRef Name,
                                DIFile F, unsigned LineNo);

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

    /// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    Instruction *InsertDbgValueIntrinsic(llvm::Value *V, uint64_t Offset,
                                         DIVariable D, BasicBlock *InsertAtEnd);

    /// InsertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    Instruction *InsertDbgValueIntrinsic(llvm::Value *V, uint64_t Offset,
                                       DIVariable D, Instruction *InsertBefore);
  private:
    Constant *GetTagConstant(unsigned TAG);
  };

  bool getLocationInfo(const Value *V, std::string &DisplayName,
                       std::string &Type, unsigned &LineNo, std::string &File,
                       std::string &Dir);

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
