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
// walking debug info in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUGINFO_H
#define LLVM_SUPPORT_DEBUGINFO_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class BasicBlock;
  class Constant;
  class Function;
  class GlobalVariable;
  class Module;
  class Value;
  
  class DIDescriptor {
  public:
    enum {
      Version6    = 6 << 16,     // Current version of debug information.
      Version5    = 5 << 16,     // Constant for version 5.
      Version4    = 4 << 16,     // Constant for version 4.
      VersionMask = 0xffff0000   // Mask for version number.
    };
    
  protected:    
    GlobalVariable *GV;
    
    /// DIDescriptor constructor.  If the specified GV is non-null, this checks
    /// to make sure that the tag in the descriptor matches 'RequiredTag'.  If
    /// not, the debug info is corrupt and we ignore it.
    DIDescriptor(GlobalVariable *GV, unsigned RequiredTag);
    
    unsigned getTag() const {
      return getUnsignedField(0) & ~VersionMask;
    }
    unsigned getVersion() const {
      return getUnsignedField(0) & VersionMask;
    }
    
    std::string getStringField(unsigned Elt) const;
    unsigned getUnsignedField(unsigned Elt) const {
      return (unsigned)getUInt64Field(Elt);
    }
    uint64_t getUInt64Field(unsigned Elt) const;
    DIDescriptor getDescriptorField(unsigned Elt) const;
    
    template <typename DescTy>
    DescTy getFieldAs(unsigned Elt) const {
      return DescTy(getDescriptorField(6).getGV());
    }
  
    GlobalVariable *getGlobalVariableField(unsigned Elt) const;
    
  public:
    explicit DIDescriptor() : GV(0) {}
    explicit DIDescriptor(GlobalVariable *gv) : GV(gv) {}

    bool isNull() const { return GV == 0; }

    GlobalVariable *getGV() const { return GV; }
    
    /// getCastToEmpty - Return this descriptor as a Constant* with type '{}*'.
    Constant *getCastToEmpty() const;
  };
  
  /// DIAnchor - A wrapper for various anchor descriptors.
  class DIAnchor : public DIDescriptor {
  public:
    explicit DIAnchor(GlobalVariable *GV = 0);
    
    unsigned getAnchorTag() const { return getUnsignedField(1); }
  };
  
  /// DIArray - This descriptor holds an array of descriptors.
  class DIArray : public DIDescriptor {
  public:
    explicit DIArray(GlobalVariable *GV = 0) : DIDescriptor(GV) {}
    
    unsigned getNumElements() const;
    DIDescriptor getElement(unsigned Idx) const;
  };
  
  /// DICompileUnit - A wrapper for a compile unit.
  class DICompileUnit : public DIDescriptor {
  public:
    explicit DICompileUnit(GlobalVariable *GV = 0);
    
    unsigned getLanguage() const     { return getUnsignedField(2); }
    std::string getFilename() const  { return getStringField(3); }
    std::string getDirectory() const { return getStringField(4); }
    std::string getProducer() const  { return getStringField(5); }
  };

  /// DIEnumerator - A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
  /// FIXME: it seems strange that this doesn't have either a reference to the
  /// type/precision or a file/line pair for location info.
  class DIEnumerator : public DIDescriptor {
  public:
    explicit DIEnumerator(GlobalVariable *GV = 0);
    
    std::string getName() const  { return getStringField(1); }
    uint64_t getLanguage() const { return getUInt64Field(2); }
  };
  
  /// DISubrange - This is used to represent ranges, for array bounds.
  class DISubrange : public DIDescriptor {
  public:
    explicit DISubrange(GlobalVariable *GV = 0);
    
    int64_t getLo() const { return (int64_t)getUInt64Field(1); }
    int64_t getHi() const { return (int64_t)getUInt64Field(2); }
  };
  
  /// DIType - This is a wrapper for a type.
  /// FIXME: Types should be factored much better so that CV qualifiers and
  /// others do not require a huge and empty descriptor full of zeros.
  class DIType : public DIDescriptor {
  protected:
    DIType(GlobalVariable *GV, unsigned Tag) : DIDescriptor(GV, Tag) {}
    // This ctor is used when the Tag has already been validated by a derived
    // ctor.
    DIType(GlobalVariable *GV, bool, bool) : DIDescriptor(GV) {}
  public:
    explicit DIType(GlobalVariable *GV);
    explicit DIType() {}
    
    DIDescriptor getContext() const     { return getDescriptorField(1); }
    std::string getName() const         { return getStringField(2); }
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(3); }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    uint64_t getSizeInBits() const      { return getUInt64Field(5); }
    uint64_t getAlignInBits() const     { return getUInt64Field(6); }
    // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
    // carry this is just plain insane.
    uint64_t getOffsetInBits() const    { return getUInt64Field(7); }
    unsigned getFlags() const           { return getUnsignedField(8); }
  };
  
  /// DIBasicType - A basic type, like 'int' or 'float'.
  class DIBasicType : public DIType {
  public:
    explicit DIBasicType(GlobalVariable *GV);
    
    unsigned getEncoding() const { return getUnsignedField(9); }
  };
  
  /// DIDerivedType - A simple derived type, like a const qualified type,
  /// a typedef, a pointer or reference, etc.
  class DIDerivedType : public DIType {
  protected:
    explicit DIDerivedType(GlobalVariable *GV, bool, bool)
      : DIType(GV, true, true) {}
  public:
    explicit DIDerivedType(GlobalVariable *GV);
    
    DIType getTypeDerivedFrom() const { return getFieldAs<DIType>(9); }
    
    /// isDerivedType - Return true if the specified tag is legal for
    /// DIDerivedType.
    static bool isDerivedType(unsigned TAG);
  };

  
  /// DICompositeType - This descriptor holds a type that can refer to multiple
  /// other types, like a function or struct.
  /// FIXME: Why is this a DIDerivedType??
  class DICompositeType : public DIDerivedType {
  public:
    explicit DICompositeType(GlobalVariable *GV);
    
    DIArray getTypeArray() const { return getFieldAs<DIArray>(10); }
    
    /// isCompositeType - Return true if the specified tag is legal for
    /// DICompositeType.
    static bool isCompositeType(unsigned TAG);
  };
  
  /// DIGlobal - This is a common class for global variables and subprograms.
  class DIGlobal : public DIDescriptor {
  protected:
    explicit DIGlobal(GlobalVariable *GV, unsigned RequiredTag)
      : DIDescriptor(GV, RequiredTag) {}
  public:
    
    DIDescriptor getContext() const     { return getDescriptorField(2); }
    std::string getName() const         { return getStringField(3); }
    std::string getDisplayName() const  { return getStringField(4); }
    std::string getLinkageName() const  { return getStringField(5); }
    
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(6); }
    unsigned getLineNumber() const      { return getUnsignedField(7); }
    DIType getType() const              { return getFieldAs<DIType>(8); }
    
    /// isLocalToUnit - Return true if this subprogram is local to the current
    /// compile unit, like 'static' in C.
    unsigned isLocalToUnit() const      { return getUnsignedField(9); }
    unsigned isDefinition() const       { return getUnsignedField(10); }
  };
  
  
  /// DISubprogram - This is a wrapper for a subprogram (e.g. a function).
  class DISubprogram : public DIGlobal {
  public:
    explicit DISubprogram(GlobalVariable *GV = 0);

  };
  
  /// DIGlobalVariable - This is a wrapper for a global variable.
  class DIGlobalVariable : public DIGlobal {
  public:
    explicit DIGlobalVariable(GlobalVariable *GV = 0);
    
    GlobalVariable *getGlobal() const { return getGlobalVariableField(11); }
  };
  
  
  /// DIVariable - This is a wrapper for a variable (e.g. parameter, local,
  /// global etc).
  class DIVariable : public DIDescriptor {
  public:
    explicit DIVariable(GlobalVariable *GV = 0);
    
    DIDescriptor getContext() const { return getDescriptorField(1); }
    std::string getName() const { return getStringField(2); }
    
    DICompileUnit getCompileUnit() const{ return getFieldAs<DICompileUnit>(3); }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    DIType getType() const              { return getFieldAs<DIType>(5); }
    
    /// isVariable - Return true if the specified tag is legal for DIVariable.
    static bool isVariable(unsigned Tag);
  };
  
  
  /// DIBlock - This is a wrapper for a block (e.g. a function, scope, etc).
  class DIBlock : public DIDescriptor {
  public:
    explicit DIBlock(GlobalVariable *GV = 0);
    
    DIDescriptor getContext() const { return getDescriptorField(1); }
  };
  
  /// DIFactory - This object assists with the construction of the various
  /// descriptors.
  class DIFactory {
    Module &M;
    // Cached values for uniquing and faster lookups.
    DIAnchor CompileUnitAnchor, SubProgramAnchor, GlobalVariableAnchor;
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
    explicit DIFactory(Module &m) : M(m) {
      StopPointFn = FuncStartFn = RegionStartFn = RegionEndFn = DeclareFn = 0;
    }
    
    /// GetOrCreateCompileUnitAnchor - Return the anchor for compile units,
    /// creating a new one if there isn't already one in the module.
    DIAnchor GetOrCreateCompileUnitAnchor();

    /// GetOrCreateSubprogramAnchor - Return the anchor for subprograms,
    /// creating a new one if there isn't already one in the module.
    DIAnchor GetOrCreateSubprogramAnchor();

    /// GetOrCreateGlobalVariableAnchor - Return the anchor for globals,
    /// creating a new one if there isn't already one in the module.
    DIAnchor GetOrCreateGlobalVariableAnchor();

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
                                    const std::string &Producer);

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
                                        DIArray Elements);
    
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
    Constant *GetStringConstant(const std::string &String);
    DIAnchor GetOrCreateAnchor(unsigned TAG, const char *Name);
  };
  
  
} // end namespace llvm

#endif
