//===-- llvm/CodeGen/MachineDebugInfo.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect debug information for a module.  This information should be in a
// neutral form that can be used by different debugging schemes.
//
// The organization of information is primarily clustered around the source
// compile units.  The main exception is source line correspondence where
// inlining may interleave code from various compile units.
//
// The following information can be retrieved from the MachineDebugInfo.
//
//  -- Source directories - Directories are uniqued based on their canonical
//     string and assigned a sequential numeric ID (base 1.)
//  -- Source files - Files are also uniqued based on their name and directory
//     ID.  A file ID is sequential number (base 1.)
//  -- Source line coorespondence - A vector of file ID, line#, column# triples.
//     A DEBUG_LOCATION instruction is generated  by the DAG Legalizer
//     corresponding to each entry in the source line list.  This allows a debug
//     emitter to generate labels referenced by debug information tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDEBUGINFO_H
#define LLVM_CODEGEN_MACHINEDEBUGINFO_H

#include "llvm/Support/Dwarf.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/GlobalValue.h"
#include "llvm/Pass.h"
#include "llvm/User.h"

#include <string>
#include <set>

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
class Constant;
class DebugInfoDesc;
class GlobalVariable;
class MachineFunction;
class MachineMove;
class Module;
class PointerType;
class StructType;

//===----------------------------------------------------------------------===//
// Debug info constants.

enum {
  LLVMDebugVersion = (4 << 16),         // Current version of debug information.
  LLVMDebugVersionMask = 0xffff0000     // Mask for version number.
};

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
/// DebugInfoDesc - This class is the base class for debug info descriptors.
///
class DebugInfoDesc {
private:
  unsigned Tag;                         // Content indicator.  Dwarf values are
                                        // used but that does not limit use to
                                        // Dwarf writers.
  
protected:
  DebugInfoDesc(unsigned T) : Tag(T | LLVMDebugVersion) {}
  
public:
  virtual ~DebugInfoDesc() {}

  // Accessors
  unsigned getTag()          const { return Tag & ~LLVMDebugVersionMask; }
  unsigned getVersion()      const { return Tag &  LLVMDebugVersionMask; }
  void setTag(unsigned T)          { Tag = T | LLVMDebugVersion; }
  
  /// TagFromGlobal - Returns the tag number from a debug info descriptor
  /// GlobalVariable.   Return DIIValid if operand is not an unsigned int. 
  static unsigned TagFromGlobal(GlobalVariable *GV);

  /// VersionFromGlobal - Returns the version number from a debug info
  /// descriptor GlobalVariable.  Return DIIValid if operand is not an unsigned
  /// int.
  static unsigned VersionFromGlobal(GlobalVariable *GV);

  /// DescFactory - Create an instance of debug info descriptor based on Tag.
  /// Return NULL if not a recognized Tag.
  static DebugInfoDesc *DescFactory(unsigned Tag);
  
  /// getLinkage - get linkage appropriate for this type of descriptor.
  ///
  virtual GlobalValue::LinkageTypes getLinkage() const;
    
  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following static methods.
  
  // Implement isa/cast/dyncast.
  static bool classof(const DebugInfoDesc *) { return true; }
  
  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following virtual methods.
  
  /// ApplyToFields - Target the vistor to the fields of the descriptor.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const = 0;
  
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const = 0;
  
#ifndef NDEBUG
  virtual void dump() = 0;
#endif
};

//===----------------------------------------------------------------------===//
/// AnchorDesc - Descriptors of this class act as markers for identifying
/// descriptors of certain groups.
class AnchoredDesc;
class AnchorDesc : public DebugInfoDesc {
private:
  unsigned AnchorTag;                   // Tag number of descriptors anchored
                                        // by this object.
  
public:
  AnchorDesc();
  AnchorDesc(AnchoredDesc *D);
  
  // Accessors
  unsigned getAnchorTag() const { return AnchorTag; }

  // Implement isa/cast/dyncast.
  static bool classof(const AnchorDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);

  /// getLinkage - get linkage appropriate for this type of descriptor.
  ///
  virtual GlobalValue::LinkageTypes getLinkage() const;

  /// ApplyToFields - Target the visitor to the fields of the AnchorDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// AnchoredDesc - This class manages anchors for a variety of top level
/// descriptors.
class AnchoredDesc : public DebugInfoDesc {
private:  
  DebugInfoDesc *Anchor;                // Anchor for all descriptors of the
                                        // same type.

protected:

  AnchoredDesc(unsigned T);

public:  
  // Accessors.
  AnchorDesc *getAnchor() const { return static_cast<AnchorDesc *>(Anchor); }
  void setAnchor(AnchorDesc *A) { Anchor = static_cast<DebugInfoDesc *>(A); }

  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following virtual methods.
  
  /// getAnchorString - Return a string used to label descriptor's anchor.
  ///
  virtual const char *getAnchorString() const = 0;
    
  /// ApplyToFields - Target the visitor to the fields of the AnchoredDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);
};

//===----------------------------------------------------------------------===//
/// CompileUnitDesc - This class packages debug information associated with a 
/// source/header file.
class CompileUnitDesc : public AnchoredDesc {
private:  
  unsigned Language;                    // Language number (ex. DW_LANG_C89.)
  std::string FileName;                 // Source file name.
  std::string Directory;                // Source file directory.
  std::string Producer;                 // Compiler string.
  
public:
  CompileUnitDesc();
  
  
  // Accessors
  unsigned getLanguage()                  const { return Language; }
  const std::string &getFileName()        const { return FileName; }
  const std::string &getDirectory()       const { return Directory; }
  const std::string &getProducer()        const { return Producer; }
  void setLanguage(unsigned L)                  { Language = L; }
  void setFileName(const std::string &FN)       { FileName = FN; }
  void setDirectory(const std::string &D)       { Directory = D; }
  void setProducer(const std::string &P)        { Producer = P; }
  
  // FIXME - Need translation unit getter/setter.

  // Implement isa/cast/dyncast.
  static bool classof(const CompileUnitDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the CompileUnitDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  static const char *AnchorString;
  virtual const char *getAnchorString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// TypeDesc - This class packages debug information associated with a type.
///
class TypeDesc : public DebugInfoDesc {
private:
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Type name (may be empty.)
  DebugInfoDesc *File;                  // Defined compile unit (may be NULL.)
  unsigned Line;                        // Defined line# (may be zero.)
  uint64_t Size;                        // Type bit size (may be zero.)
  uint64_t Align;                       // Type bit alignment (may be zero.)
  uint64_t Offset;                      // Type bit offset (may be zero.)

public:
  TypeDesc(unsigned T);

  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  const std::string &getName()               const { return Name; }
  CompileUnitDesc *getFile() const {
    return static_cast<CompileUnitDesc *>(File);
  }
  unsigned getLine()                         const { return Line; }
  uint64_t getSize()                         const { return Size; }
  uint64_t getAlign()                        const { return Align; }
  uint64_t getOffset()                       const { return Offset; }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  void setName(const std::string &N)               { Name = N; }
  void setFile(CompileUnitDesc *U) {
    File = static_cast<DebugInfoDesc *>(U);
  }
  void setLine(unsigned L)                         { Line = L; }
  void setSize(uint64_t S)                         { Size = S; }
  void setAlign(uint64_t A)                        { Align = A; }
  void setOffset(uint64_t O)                       { Offset = O; }
  
  /// ApplyToFields - Target the visitor to the fields of the TypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// BasicTypeDesc - This class packages debug information associated with a
/// basic type (eg. int, bool, double.)
class BasicTypeDesc : public TypeDesc {
private:
  unsigned Encoding;                    // Type encoding.
  
public:
  BasicTypeDesc();
  
  // Accessors
  unsigned getEncoding()                     const { return Encoding; }
  void setEncoding(unsigned E)                     { Encoding = E; }

  // Implement isa/cast/dyncast.
  static bool classof(const BasicTypeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the BasicTypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;

#ifndef NDEBUG
  virtual void dump();
#endif
};


//===----------------------------------------------------------------------===//
/// DerivedTypeDesc - This class packages debug information associated with a
/// derived types (eg., typedef, pointer, reference.)
class DerivedTypeDesc : public TypeDesc {
private:
  DebugInfoDesc *FromType;              // Type derived from.

public:
  DerivedTypeDesc(unsigned T);
  
  // Accessors
  TypeDesc *getFromType() const {
    return static_cast<TypeDesc *>(FromType);
  }
  void setFromType(TypeDesc *F) {
    FromType = static_cast<DebugInfoDesc *>(F);
  }

  // Implement isa/cast/dyncast.
  static bool classof(const DerivedTypeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the DerivedTypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;

#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// CompositeTypeDesc - This class packages debug information associated with a
/// array/struct types (eg., arrays, struct, union, enums.)
class CompositeTypeDesc : public DerivedTypeDesc {
private:
  std::vector<DebugInfoDesc *> Elements;// Information used to compose type.

public:
  CompositeTypeDesc(unsigned T);
  
  // Accessors
  std::vector<DebugInfoDesc *> &getElements() { return Elements; }

  // Implement isa/cast/dyncast.
  static bool classof(const CompositeTypeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the CompositeTypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;

#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// SubrangeDesc - This class packages debug information associated with integer
/// value ranges.
class SubrangeDesc : public DebugInfoDesc {
private:
  int64_t Lo;                           // Low value of range.
  int64_t Hi;                           // High value of range.

public:
  SubrangeDesc();
  
  // Accessors
  int64_t getLo()                            const { return Lo; }
  int64_t getHi()                            const { return Hi; }
  void setLo(int64_t L)                            { Lo = L; }
  void setHi(int64_t H)                            { Hi = H; }

  // Implement isa/cast/dyncast.
  static bool classof(const SubrangeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the SubrangeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;

#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// EnumeratorDesc - This class packages debug information associated with
/// named integer constants.
class EnumeratorDesc : public DebugInfoDesc {
private:
  std::string Name;                     // Enumerator name.
  int64_t Value;                        // Enumerator value.

public:
  EnumeratorDesc();
  
  // Accessors
  const std::string &getName()               const { return Name; }
  int64_t getValue()                         const { return Value; }
  void setName(const std::string &N)               { Name = N; }
  void setValue(int64_t V)                         { Value = V; }

  // Implement isa/cast/dyncast.
  static bool classof(const EnumeratorDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the EnumeratorDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;

#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// VariableDesc - This class packages debug information associated with a
/// subprogram variable.
///
class VariableDesc : public DebugInfoDesc {
private:
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Type name (may be empty.)
  DebugInfoDesc *File;                  // Defined compile unit (may be NULL.)
  unsigned Line;                        // Defined line# (may be zero.)
  DebugInfoDesc *TyDesc;                // Type of variable.

public:
  VariableDesc(unsigned T);

  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  const std::string &getName()               const { return Name; }
  CompileUnitDesc *getFile() const {
    return static_cast<CompileUnitDesc *>(File);
  }
  unsigned getLine()                         const { return Line; }
  TypeDesc *getType() const {
    return static_cast<TypeDesc *>(TyDesc);
  }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  void setName(const std::string &N)               { Name = N; }
  void setFile(CompileUnitDesc *U) {
    File = static_cast<DebugInfoDesc *>(U);
  }
  void setLine(unsigned L)                         { Line = L; }
  void setType(TypeDesc *T) {
    TyDesc = static_cast<DebugInfoDesc *>(T);
  }
  
  // Implement isa/cast/dyncast.
  static bool classof(const VariableDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the VariableDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// GlobalDesc - This class is the base descriptor for global functions and
/// variables.
class GlobalDesc : public AnchoredDesc {
private:
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Global name.
  DebugInfoDesc *File;                  // Defined compile unit (may be NULL.)
  unsigned Line;                        // Defined line# (may be zero.)
  DebugInfoDesc *TyDesc;                // Type debug descriptor.
  bool IsStatic;                        // Is the global a static.
  bool IsDefinition;                    // Is the global defined in context.
  
protected:
  GlobalDesc(unsigned T);

public:
  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  const std::string &getName()               const { return Name; }
  CompileUnitDesc *getFile() const {
    return static_cast<CompileUnitDesc *>(File);
  }
  unsigned getLine()                         const { return Line; }
  TypeDesc *getType() const {
    return static_cast<TypeDesc *>(TyDesc);
  }
  bool isStatic()                            const { return IsStatic; }
  bool isDefinition()                        const { return IsDefinition; }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  void setName(const std::string &N)               { Name = N; }
  void setFile(CompileUnitDesc *U) {
    File = static_cast<DebugInfoDesc *>(U);
  }
  void setLine(unsigned L)                         { Line = L; }
  void setType(TypeDesc *T) {
    TyDesc = static_cast<DebugInfoDesc *>(T);
  }
  void setIsStatic(bool IS)                        { IsStatic = IS; }
  void setIsDefinition(bool ID)                    { IsDefinition = ID; }

  /// ApplyToFields - Target the visitor to the fields of the GlobalDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);
};

//===----------------------------------------------------------------------===//
/// GlobalVariableDesc - This class packages debug information associated with a
/// GlobalVariable.
class GlobalVariableDesc : public GlobalDesc {
private:
  GlobalVariable *Global;               // llvm global.
  
public:
  GlobalVariableDesc();

  // Accessors.
  GlobalVariable *getGlobalVariable()        const { return Global; }
  void setGlobalVariable(GlobalVariable *GV)       { Global = GV; }
 
  // Implement isa/cast/dyncast.
  static bool classof(const GlobalVariableDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the
  /// GlobalVariableDesc.
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  static const char *AnchorString;
  virtual const char *getAnchorString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// SubprogramDesc - This class packages debug information associated with a
/// subprogram/function.
class SubprogramDesc : public GlobalDesc {
private:
  
public:
  SubprogramDesc();
  
  // Accessors
  
  // Implement isa/cast/dyncast.
  static bool classof(const SubprogramDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the SubprogramDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  static const char *AnchorString;
  virtual const char *getAnchorString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// BlockDesc - This descriptor groups variables and blocks nested in a block.
///
class BlockDesc : public DebugInfoDesc {
private:
  DebugInfoDesc *Context;               // Context debug descriptor.

public:
  BlockDesc();
  
  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  
  // Implement isa/cast/dyncast.
  static bool classof(const BlockDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
  
  /// ApplyToFields - Target the visitor to the fields of the BlockDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const;
    
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// DIDeserializer - This class is responsible for casting GlobalVariables
/// into DebugInfoDesc objects.
class DIDeserializer {
private:
  std::map<GlobalVariable *, DebugInfoDesc *> GlobalDescs;
                                        // Previously defined gloabls.
  
public:
  DIDeserializer() {}
  ~DIDeserializer() {}
  
  /// Deserialize - Reconstitute a GlobalVariable into it's component
  /// DebugInfoDesc objects.
  DebugInfoDesc *Deserialize(Value *V);
  DebugInfoDesc *Deserialize(GlobalVariable *GV);
};

//===----------------------------------------------------------------------===//
/// DISerializer - This class is responsible for casting DebugInfoDesc objects
/// into GlobalVariables.
class DISerializer {
private:
  Module *M;                            // Definition space module.
  PointerType *StrPtrTy;                // A "sbyte *" type.  Created lazily.
  PointerType *EmptyStructPtrTy;        // A "{ }*" type.  Created lazily.
  std::map<unsigned, StructType *> TagTypes;
                                        // Types per Tag.  Created lazily.
  std::map<DebugInfoDesc *, GlobalVariable *> DescGlobals;
                                        // Previously defined descriptors.
  std::map<const std::string, Constant *> StringCache;
                                        // Previously defined strings.
                                          
public:
  DISerializer()
  : M(NULL)
  , StrPtrTy(NULL)
  , EmptyStructPtrTy(NULL)
  , TagTypes()
  , DescGlobals()
  , StringCache()
  {}
  ~DISerializer() {}
  
  // Accessors
  Module *getModule()        const { return M; };
  void setModule(Module *module)  { M = module; }

  /// getStrPtrType - Return a "sbyte *" type.
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
};

//===----------------------------------------------------------------------===//
/// DIVerifier - This class is responsible for verifying the given network of
/// GlobalVariables are valid as DebugInfoDesc objects.
class DIVerifier {
private:
  enum {
    Unknown = 0,
    Invalid,
    Valid
  };
  std::map<GlobalVariable *, unsigned> Validity;// Tracks prior results.
  std::map<unsigned, unsigned> Counts;  // Count of fields per Tag type.
  
public:
  DIVerifier()
  : Validity()
  , Counts()
  {}
  ~DIVerifier() {}
  
  /// Verify - Return true if the GlobalVariable appears to be a valid
  /// serialization of a DebugInfoDesc.
  bool Verify(Value *V);
  bool Verify(GlobalVariable *GV);
};

//===----------------------------------------------------------------------===//
/// SourceLineInfo - This class is used to record source line correspondence.
///
class SourceLineInfo {
private:
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
private:
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
/// MachineDebugInfo - This class contains debug information specific to a
/// module.  Queries can be made by different debugging schemes and reformated
/// for specific use.
///
class MachineDebugInfo : public ImmutablePass {
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
  std::vector<SourceLineInfo *> Lines;
  
  // LabelID - Current number assigned to unique label numbers.
  unsigned LabelID;
  
  // ScopeMap - Tracks the scopes in the current function.
  std::map<DebugInfoDesc *, DebugScope *> ScopeMap;
  
  // RootScope - Top level scope for the current function.
  //
  DebugScope *RootScope;
  
  // FrameMoves - List of moves done by a function's prolog.  Used to construct
  // frame maps by debug consumers.
  std::vector<MachineMove *> FrameMoves;

public:
  MachineDebugInfo();
  ~MachineDebugInfo();
  
  /// doInitialization - Initialize the debug state for a new module.
  ///
  bool doInitialization();
  
  /// doFinalization - Tear down the debug state after completion of a module.
  ///
  bool doFinalization();
  
  /// BeginFunction - Begin gathering function debug information.
  ///
  void BeginFunction(MachineFunction *MF);
  
  /// EndFunction - Discard function debug information.
  ///
  void EndFunction();

  /// getDescFor - Convert a Value to a debug information descriptor.
  ///
  // FIXME - use new Value type when available.
  DebugInfoDesc *getDescFor(Value *V);
  
  /// Verify - Verify that a Value is debug information descriptor.
  ///
  bool Verify(Value *V);
  
  /// AnalyzeModule - Scan the module for global debug information.
  ///
  void AnalyzeModule(Module &M);
  
  /// hasInfo - Returns true if valid debug info is present.
  ///
  bool hasInfo() const { return !CompileUnits.empty(); }
  
  /// NextLabelID - Return the next unique label id.
  ///
  unsigned NextLabelID() { return ++LabelID; }
  
  /// RecordLabel - Records location information and associates it with a
  /// debug label.  Returns a unique label ID used to generate a label and 
  /// provide correspondence to the source line list.
  unsigned RecordLabel(unsigned Line, unsigned Column, unsigned Source);
  
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
  std::vector<SourceLineInfo *> &getSourceLines() {
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
  std::vector<GlobalVariable*>
  getGlobalVariablesUsing(Module &M, const std::string &RootName);

  /// getAnchoredDescriptors - Return a vector of anchored debug descriptors.
  ///
  template <class T>std::vector<T *> getAnchoredDescriptors(Module &M) {
    T Desc;
    std::vector<GlobalVariable *> Globals =
                             getGlobalVariablesUsing(M, Desc.getAnchorString());
    std::vector<T *> AnchoredDescs;
    for (unsigned i = 0, N = Globals.size(); i < N; ++i) {
      GlobalVariable *GV = Globals[i];
      
      // FIXME - In the short term, changes are too drastic to continue.
      if (DebugInfoDesc::TagFromGlobal(GV) == Desc.getTag() &&
          DebugInfoDesc::VersionFromGlobal(GV) == LLVMDebugVersion) {
        AnchoredDescs.push_back(cast<T>(DR.Deserialize(GV)));
      }
    }

    return AnchoredDescs;
  }
  
  /// RecordRegionStart - Indicate the start of a region.
  ///
  unsigned RecordRegionStart(Value *V);

  /// RecordRegionEnd - Indicate the end of a region.
  ///
  unsigned RecordRegionEnd(Value *V);

  /// RecordVariable - Indicate the declaration of  a local variable.
  ///
  void RecordVariable(Value *V, unsigned FrameIndex);
  
  /// getRootScope - Return current functions root scope.
  ///
  DebugScope *getRootScope() { return RootScope; }
  
  /// getOrCreateScope - Returns the scope associated with the given descriptor.
  ///
  DebugScope *getOrCreateScope(DebugInfoDesc *ScopeDesc);
  
  /// getFrameMoves - Returns a reference to a list of moves done in the current
  /// function's prologue.  Used to construct frame maps for debug comsumers.
  std::vector<MachineMove *> &getFrameMoves() { return FrameMoves; }

}; // End class MachineDebugInfo

} // End llvm namespace

#endif
