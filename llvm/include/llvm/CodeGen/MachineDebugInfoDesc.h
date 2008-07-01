//===-- llvm/CodeGen/MachineDebugInfoDesc.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Debug descriptor information for a module.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDEBUGINFODESC_H
#define LLVM_CODEGEN_MACHINEDEBUGINFODESC_H

#include "llvm/Support/DataTypes.h"
#include <string>
#include <vector>

namespace llvm {

//===----------------------------------------------------------------------===//
// Forward declarations.
class DIVisitor;
class GlobalVariable;

//===----------------------------------------------------------------------===//
// Debug info description constants.

enum {
  LLVMDebugVersion     = (6 << 16),     // Current version of debug information.
  LLVMDebugVersion5    = (5 << 16),     // Constant for version 5.
  LLVMDebugVersion4    = (4 << 16),     // Constant for version 4.
  LLVMDebugVersionMask = 0xffff0000     // Mask for version number.
};

//===----------------------------------------------------------------------===//
/// DebugInfoDesc - This class is the base class for debug info descriptors.

class DebugInfoDesc {
  // Content indicator. Dwarf values are used but that does not limit use to
  // Dwarf writers.
  unsigned Tag;
protected:
  explicit DebugInfoDesc(unsigned T) : Tag(T | LLVMDebugVersion) {}
public:
  virtual ~DebugInfoDesc();

  // Accessors
  unsigned getTag()          const { return Tag & ~LLVMDebugVersionMask; }
  unsigned getVersion()      const { return Tag &  LLVMDebugVersionMask; }
  void setTag(unsigned T)          { Tag = T | LLVMDebugVersion; }
  
  /// TagFromGlobal - Returns the tag number from a debug info descriptor
  /// GlobalVariable.  Return DIIValid if operand is not an unsigned int. 
  static unsigned TagFromGlobal(GlobalVariable *GV);

  /// VersionFromGlobal - Returns the version number from a debug info
  /// descriptor GlobalVariable. Return DIIValid if operand is not an unsigned
  /// int.
  static unsigned VersionFromGlobal(GlobalVariable *GV);

  /// DescFactory - Create an instance of debug info descriptor based on Tag.
  /// Return NULL if not a recognized Tag.
  static DebugInfoDesc *DescFactory(unsigned Tag);
  
  /// getLinkage - get linkage appropriate for this type of descriptor.
  ///
  virtual unsigned getLinkage() const;
    
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

  //===--------------------------------------------------------------------===//
  // Subclasses should supply the following static methods.
  
  // Implement isa/cast/dyncast.
  static bool classof(const DebugInfoDesc *) { return true; }  
};

//===----------------------------------------------------------------------===//
/// AnchorDesc - Descriptors of this class act as markers for identifying
/// descriptors of certain groups.
class AnchoredDesc;
class AnchorDesc : public DebugInfoDesc {
  // Tag number of descriptors anchored by this object.
  unsigned AnchorTag;
public:
  AnchorDesc();
  explicit AnchorDesc(AnchoredDesc *D);
  
  // Accessors
  unsigned getAnchorTag() const { return AnchorTag; }

  /// getLinkage - get linkage appropriate for this type of descriptor.
  ///
  virtual unsigned getLinkage() const;

  /// ApplyToFields - Target the visitor to the fields of the AnchorDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const;
    
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.anchor.type";
  }
    
#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const AnchorDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// AnchoredDesc - This class manages anchors for a variety of top level
/// descriptors.
class AnchoredDesc : public DebugInfoDesc {
  // Anchor for all descriptors of the same type.
  DebugInfoDesc *Anchor;
protected:
  explicit AnchoredDesc(unsigned T);
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

  /// ApplyToFields - Target the visitor to the fields of the CompileUnitDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.compile_unit";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.compile_unit.type";
  }

  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  const char *getAnchorString() const {
    return "llvm.dbg.compile_units";
  }
    
#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const CompileUnitDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// TypeDesc - This class packages debug information associated with a type.
///
class TypeDesc : public DebugInfoDesc {
  enum {
    FlagPrivate    = 1 << 0,
    FlagProtected  = 1 << 1
  };
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Type name (may be empty.)
  DebugInfoDesc *File;                  // Defined compile unit (may be NULL.)
  unsigned Line;                        // Defined line# (may be zero.)
  uint64_t Size;                        // Type bit size (may be zero.)
  uint64_t Align;                       // Type bit alignment (may be zero.)
  uint64_t Offset;                      // Type bit offset (may be zero.)
  unsigned Flags;                       // Miscellaneous flags.
public:
  explicit TypeDesc(unsigned T);

  // Accessors
  DebugInfoDesc *getContext() const  { return Context; }
  const std::string &getName() const { return Name; }
  CompileUnitDesc *getFile() const {
    return static_cast<CompileUnitDesc *>(File);
  }
  unsigned getLine() const   { return Line; }
  uint64_t getSize() const   { return Size; }
  uint64_t getAlign() const  { return Align; }
  uint64_t getOffset() const { return Offset; }
  bool isPrivate() const {
    return (Flags & FlagPrivate) != 0;
  }
  bool isProtected() const {
    return (Flags & FlagProtected) != 0;
  }
  void setContext(DebugInfoDesc *C)  { Context = C; }
  void setName(const std::string &N) { Name = N; }
  void setFile(CompileUnitDesc *U) {
    File = static_cast<DebugInfoDesc *>(U);
  }
  void setLine(unsigned L)   { Line = L; }
  void setSize(uint64_t S)   { Size = S; }
  void setAlign(uint64_t A)  { Align = A; }
  void setOffset(uint64_t O) { Offset = O; }
  void setIsPrivate()        { Flags |= FlagPrivate; }
  void setIsProtected()      { Flags |= FlagProtected; }
  
  /// ApplyToFields - Target the visitor to the fields of the TypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.type";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.type.type";
  }
  
#ifndef NDEBUG
  virtual void dump();
#endif
};

//===----------------------------------------------------------------------===//
/// BasicTypeDesc - This class packages debug information associated with a
/// basic type (eg. int, bool, double.)
class BasicTypeDesc : public TypeDesc {
  unsigned Encoding;                    // Type encoding.
public:
  BasicTypeDesc();
  
  // Accessors
  unsigned getEncoding() const { return Encoding; }
  void setEncoding(unsigned E) { Encoding = E; }
  
  /// ApplyToFields - Target the visitor to the fields of the BasicTypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.basictype";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.basictype.type";
  }

#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const BasicTypeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// DerivedTypeDesc - This class packages debug information associated with a
/// derived types (eg., typedef, pointer, reference.)
class DerivedTypeDesc : public TypeDesc {
  DebugInfoDesc *FromType;              // Type derived from.
public:
  explicit DerivedTypeDesc(unsigned T);
  
  // Accessors
  TypeDesc *getFromType() const {
    return static_cast<TypeDesc *>(FromType);
  }
  void setFromType(TypeDesc *F) {
    FromType = static_cast<DebugInfoDesc *>(F);
  }
  
  /// ApplyToFields - Target the visitor to the fields of the DerivedTypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.derivedtype";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.derivedtype.type";
  }

#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const DerivedTypeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// CompositeTypeDesc - This class packages debug information associated with a
/// array/struct types (eg., arrays, struct, union, enums.)
class CompositeTypeDesc : public DerivedTypeDesc {
  std::vector<DebugInfoDesc *> Elements; // Information used to compose type.
public:
  explicit CompositeTypeDesc(unsigned T);
  
  // Accessors
  std::vector<DebugInfoDesc *> &getElements() { return Elements; }
  
  /// ApplyToFields - Target the visitor to the fields of the CompositeTypeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.compositetype";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.compositetype.type";
  }

#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const CompositeTypeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// SubrangeDesc - This class packages debug information associated with integer
/// value ranges.
class SubrangeDesc : public DebugInfoDesc {
  int64_t Lo;                           // Low value of range.
  int64_t Hi;                           // High value of range.
public:
  SubrangeDesc();
  
  // Accessors
  int64_t getLo()                            const { return Lo; }
  int64_t getHi()                            const { return Hi; }
  void setLo(int64_t L)                            { Lo = L; }
  void setHi(int64_t H)                            { Hi = H; }
  
  /// ApplyToFields - Target the visitor to the fields of the SubrangeDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.subrange";
  }
  
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.subrange.type";
  }

#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const SubrangeDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// EnumeratorDesc - This class packages debug information associated with
/// named integer constants.
class EnumeratorDesc : public DebugInfoDesc {
  std::string Name;                     // Enumerator name.
  int64_t Value;                        // Enumerator value.
public:
  EnumeratorDesc();
  
  // Accessors
  const std::string &getName() const { return Name; }
  int64_t getValue() const { return Value; }
  void setName(const std::string &N) { Name = N; }
  void setValue(int64_t V) { Value = V; }
  
  /// ApplyToFields - Target the visitor to the fields of the EnumeratorDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.enumerator";
  }
  
  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.enumerator.type";
  }

#ifndef NDEBUG
  virtual void dump();
#endif

  // Implement isa/cast/dyncast.
  static bool classof(const EnumeratorDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// VariableDesc - This class packages debug information associated with a
/// subprogram variable.
///
class VariableDesc : public DebugInfoDesc {
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Type name (may be empty.)
  DebugInfoDesc *File;                  // Defined compile unit (may be NULL.)
  unsigned Line;                        // Defined line# (may be zero.)
  DebugInfoDesc *TyDesc;                // Type of variable.
public:
  explicit VariableDesc(unsigned T);

  // Accessors
  DebugInfoDesc *getContext() const  { return Context; }
  const std::string &getName() const { return Name; }
  CompileUnitDesc *getFile() const {
    return static_cast<CompileUnitDesc *>(File);
  }
  unsigned getLine() const { return Line; }
  TypeDesc *getType() const {
    return static_cast<TypeDesc*>(TyDesc);
  }
  void setContext(DebugInfoDesc *C)  { Context = C; }
  void setName(const std::string &N) { Name = N; }
  void setFile(CompileUnitDesc *U) {
    File = static_cast<DebugInfoDesc *>(U);
  }
  void setLine(unsigned L) { Line = L; }
  void setType(TypeDesc *T) {
    TyDesc = static_cast<DebugInfoDesc *>(T);
  }
  
  /// ApplyToFields - Target the visitor to the fields of the VariableDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.variable";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.variable.type";
  }

#ifndef NDEBUG
  virtual void dump();
#endif
  
  // Implement isa/cast/dyncast.
  static bool classof(const VariableDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// GlobalDesc - This class is the base descriptor for global functions and
/// variables.
class GlobalDesc : public AnchoredDesc {
  DebugInfoDesc *Context;               // Context debug descriptor.
  std::string Name;                     // Global name.
  std::string FullName;                 // Fully qualified name.
  std::string LinkageName;              // Name for binding to MIPS linkage.
  DebugInfoDesc *File;                  // Defined compile unit (may be NULL.)
  unsigned Line;                        // Defined line# (may be zero.)
  DebugInfoDesc *TyDesc;                // Type debug descriptor.
  bool IsStatic;                        // Is the global a static.
  bool IsDefinition;                    // Is the global defined in context.
protected:
  explicit GlobalDesc(unsigned T);
public:
  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  const std::string &getName()               const { return Name; }
  const std::string &getFullName()           const { return FullName; }
  const std::string &getLinkageName()        const { return LinkageName; }
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
  void setFullName(const std::string &N)           { FullName = N; }
  void setLinkageName(const std::string &N)        { LinkageName = N; }
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
  GlobalVariable *Global;               // llvm global.
public:
  GlobalVariableDesc();

  // Accessors.
  GlobalVariable *getGlobalVariable()        const { return Global; }
  void setGlobalVariable(GlobalVariable *GV)       { Global = GV; }
  
  /// ApplyToFields - Target the visitor to the fields of the
  /// GlobalVariableDesc.
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.global_variable";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.global_variable.type";
  }
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  const char *getAnchorString() const {
    return "llvm.dbg.global_variables";
  }
    
#ifndef NDEBUG
  virtual void dump();
#endif
 
  // Implement isa/cast/dyncast.
  static bool classof(const GlobalVariableDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// SubprogramDesc - This class packages debug information associated with a
/// subprogram/function.
struct SubprogramDesc : public GlobalDesc {
  SubprogramDesc();
  
  /// ApplyToFields - Target the visitor to the fields of the SubprogramDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.subprogram";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.subprogram.type";
  }
  
  /// getAnchorString - Return a string used to label this descriptor's anchor.
  ///
  const char *getAnchorString() const {
    return "llvm.dbg.subprograms";
  }
    
#ifndef NDEBUG
  virtual void dump();
#endif
  
  // Implement isa/cast/dyncast.
  static bool classof(const SubprogramDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

//===----------------------------------------------------------------------===//
/// BlockDesc - This descriptor groups variables and blocks nested in a block.
///
class BlockDesc : public DebugInfoDesc {
  DebugInfoDesc *Context;               // Context debug descriptor.
public:
  BlockDesc();
  
  // Accessors
  DebugInfoDesc *getContext()                const { return Context; }
  void setContext(DebugInfoDesc *C)                { Context = C; }
  
  /// ApplyToFields - Target the visitor to the fields of the BlockDesc.
  ///
  virtual void ApplyToFields(DIVisitor *Visitor);

  /// getDescString - Return a string used to compose global names and labels.
  ///
  virtual const char *getDescString() const {
    return "llvm.dbg.block";
  }

  /// getTypeString - Return a string used to label this descriptor's type.
  ///
  virtual const char *getTypeString() const {
    return "llvm.dbg.block.type";
  }
 
#ifndef NDEBUG
  virtual void dump();
#endif
  
  // Implement isa/cast/dyncast.
  static bool classof(const BlockDesc *) { return true; }
  static bool classof(const DebugInfoDesc *D);
};

} // End llvm namespace

#endif
