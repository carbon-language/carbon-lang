//===--- SemaDeclSpec.h - Declaration Specifier Semantic Analys -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces used for Declaration Specifiers and Declarators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_SEMADECLSPEC_H
#define LLVM_CLANG_PARSE_SEMADECLSPEC_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/Action.h"
#include "clang/Parse/AttributeList.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
  struct LangOptions;
  class IdentifierInfo;
  
/// DeclSpec - This class captures information about "declaration specifiers",
/// which encompasses storage-class-specifiers, type-specifiers,
/// type-qualifiers, and function-specifiers.
class DeclSpec {
public:
  // storage-class-specifier
  enum SCS {
    SCS_unspecified,
    SCS_typedef,
    SCS_extern,
    SCS_static,
    SCS_auto,
    SCS_register,
    SCS_private_extern
  };
  
  // type-specifier
  enum TSW {
    TSW_unspecified,
    TSW_short,
    TSW_long,
    TSW_longlong
  };
  
  enum TSC {
    TSC_unspecified,
    TSC_imaginary,
    TSC_complex
  };
  
  enum TSS {
    TSS_unspecified,
    TSS_signed,
    TSS_unsigned
  };
  
  enum TST {
    TST_unspecified,
    TST_void,
    TST_char,
    TST_int,
    TST_float,
    TST_double,
    TST_bool,         // _Bool
    TST_decimal32,    // _Decimal32
    TST_decimal64,    // _Decimal64
    TST_decimal128,   // _Decimal128
    TST_enum,
    TST_union,
    TST_struct,
    TST_class,        // C++ class type
    TST_typedef,
    TST_typeofType,
    TST_typeofExpr
  };
  
  // type-qualifiers
  enum TQ {   // NOTE: These flags must be kept in sync with QualType::TQ.
    TQ_unspecified = 0,
    TQ_const       = 1,
    TQ_restrict    = 2,
    TQ_volatile    = 4
  };

  /// ParsedSpecifiers - Flags to query which specifiers were applied.  This is
  /// returned by getParsedSpecifiers.
  enum ParsedSpecifiers {
    PQ_None                  = 0,
    PQ_StorageClassSpecifier = 1,
    PQ_TypeSpecifier         = 2,
    PQ_TypeQualifier         = 4,
    PQ_FunctionSpecifier     = 8
  };
  
private:
    
  // storage-class-specifier
  /*SCS*/unsigned StorageClassSpec : 3;
  bool SCS_thread_specified : 1;

  // type-specifier
  /*TSW*/unsigned TypeSpecWidth : 2;
  /*TSC*/unsigned TypeSpecComplex : 2;
  /*TSS*/unsigned TypeSpecSign : 2;
  /*TST*/unsigned TypeSpecType : 5;
  
  // type-qualifiers
  unsigned TypeQualifiers : 3;  // Bitwise OR of TQ.
  
  // function-specifier
  bool FS_inline_specified : 1;
  
  /// TypeRep - This contains action-specific information about a specific TST.
  /// For example, for a typedef or struct, it might contain the declaration for
  /// these.
  void *TypeRep;  
  
  // attributes.
  AttributeList *AttrList;
  
  // List of protocol qualifiers for objective-c classes.  Used for 
  // protocol-qualified interfaces "NString<foo>" and protocol-qualified id
  // "id<foo>".
  llvm::SmallVector<Action::DeclTy *, 8> *ProtocolQualifiers;
  
  // SourceLocation info.  These are null if the item wasn't specified or if
  // the setting was synthesized.
  SourceRange Range;
  
  SourceLocation StorageClassSpecLoc, SCS_threadLoc;
  SourceLocation TSWLoc, TSCLoc, TSSLoc, TSTLoc;
  SourceLocation TQ_constLoc, TQ_restrictLoc, TQ_volatileLoc;
  SourceLocation FS_inlineLoc;
  
  bool BadSpecifier(TST T, const char *&PrevSpec);
  bool BadSpecifier(TQ T, const char *&PrevSpec);
  bool BadSpecifier(TSS T, const char *&PrevSpec);
  bool BadSpecifier(TSC T, const char *&PrevSpec);
  bool BadSpecifier(TSW T, const char *&PrevSpec);
  bool BadSpecifier(SCS T, const char *&PrevSpec);
public:  
  
  DeclSpec()
    : StorageClassSpec(SCS_unspecified),
      SCS_thread_specified(false),
      TypeSpecWidth(TSW_unspecified),
      TypeSpecComplex(TSC_unspecified),
      TypeSpecSign(TSS_unspecified),
      TypeSpecType(TST_unspecified),
      TypeQualifiers(TSS_unspecified),
      FS_inline_specified(false),
      TypeRep(0),
      AttrList(0),
      ProtocolQualifiers(0) {
  }
  ~DeclSpec() {
    delete AttrList;
    delete ProtocolQualifiers;
  }
  // storage-class-specifier
  SCS getStorageClassSpec() const { return (SCS)StorageClassSpec; }
  bool isThreadSpecified() const { return SCS_thread_specified; }
  
  SourceLocation getStorageClassSpecLoc() const { return StorageClassSpecLoc; }
  SourceLocation getThreadSpecLoc() const { return SCS_threadLoc; }
  
  void ClearStorageClassSpecs() {
    StorageClassSpec     = DeclSpec::SCS_unspecified;
    SCS_thread_specified = false;
    StorageClassSpecLoc  = SourceLocation();
    SCS_threadLoc        = SourceLocation();
  }
  
  // type-specifier
  TSW getTypeSpecWidth() const { return (TSW)TypeSpecWidth; }
  TSC getTypeSpecComplex() const { return (TSC)TypeSpecComplex; }
  TSS getTypeSpecSign() const { return (TSS)TypeSpecSign; }
  TST getTypeSpecType() const { return (TST)TypeSpecType; }
  void *getTypeRep() const { return TypeRep; }
  
  const SourceRange &getSourceRange() const { return Range; }
  SourceLocation getTypeSpecWidthLoc() const { return TSWLoc; }
  SourceLocation getTypeSpecComplexLoc() const { return TSCLoc; }
  SourceLocation getTypeSpecSignLoc() const { return TSSLoc; }
  SourceLocation getTypeSpecTypeLoc() const { return TSTLoc; }
  
  /// getSpecifierName - Turn a type-specifier-type into a string like "_Bool"
  /// or "union".
  static const char *getSpecifierName(DeclSpec::TST T);
  static const char *getSpecifierName(DeclSpec::SCS S);
  
  // type-qualifiers

  /// getTypeQualifiers - Return a set of TQs.
  unsigned getTypeQualifiers() const { return TypeQualifiers; }
  SourceLocation getConstSpecLoc() const { return TQ_constLoc; }
  SourceLocation getRestrictSpecLoc() const { return TQ_restrictLoc; }
  SourceLocation getVolatileSpecLoc() const { return TQ_volatileLoc; }

  
  // function-specifier
  bool isInlineSpecified() const { return FS_inline_specified; }
  SourceLocation getInlineSpecLoc() const { return FS_inlineLoc; }
  void ClearFunctionSpecs() {
    FS_inline_specified = false;
    FS_inlineLoc = SourceLocation();
  }
  
  /// hasTypeSpecifier - Return true if any type-specifier has been found.
  bool hasTypeSpecifier() const {
    return getTypeSpecType() != DeclSpec::TST_unspecified ||
           getTypeSpecWidth() != DeclSpec::TSW_unspecified ||
           getTypeSpecComplex() != DeclSpec::TSC_unspecified ||
           getTypeSpecSign() != DeclSpec::TSS_unspecified;
  }
  
  
  /// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
  /// DeclSpec includes.
  ///
  unsigned getParsedSpecifiers() const;
  
  void SetRangeStart(SourceLocation Loc) { Range.setBegin(Loc); }
  void SetRangeEnd(SourceLocation Loc) { Range.setEnd(Loc); }
  
  /// These methods set the specified attribute of the DeclSpec, but return true
  /// and ignore the request if invalid (e.g. "extern" then "auto" is
  /// specified).  The name of the previous specifier is returned in prevspec.
  bool SetStorageClassSpec(SCS S, SourceLocation Loc, const char *&PrevSpec);
  bool SetStorageClassSpecThread(SourceLocation Loc, const char *&PrevSpec);
  bool SetTypeSpecWidth(TSW W, SourceLocation Loc, const char *&PrevSpec);
  bool SetTypeSpecComplex(TSC C, SourceLocation Loc, const char *&PrevSpec);
  bool SetTypeSpecSign(TSS S, SourceLocation Loc, const char *&PrevSpec);
  bool SetTypeSpecType(TST T, SourceLocation Loc, const char *&PrevSpec,
                       void *TypeRep = 0);
  
  bool SetTypeQual(TQ T, SourceLocation Loc, const char *&PrevSpec,
                   const LangOptions &Lang);
  
  bool SetFunctionSpecInline(SourceLocation Loc, const char *&PrevSpec);
  
  /// AddAttributes - contatenates two attribute lists. 
  /// The GCC attribute syntax allows for the following:
  ///
  /// short __attribute__(( unused, deprecated )) 
  /// int __attribute__(( may_alias, aligned(16) )) var;
  ///
  /// This declares 4 attributes using 2 lists. The following syntax is
  /// also allowed and equivalent to the previous declaration.
  ///
  /// short __attribute__((unused)) __attribute__((deprecated)) 
  /// int __attribute__((may_alias)) __attribute__((aligned(16))) var;
  /// 
  void AddAttributes(AttributeList *alist) {
    if (!alist)
      return; // we parsed __attribute__(()) or had a syntax error
      
    if (AttrList) 
      alist->addAttributeList(AttrList); 
    AttrList = alist;
  }
  void SetAttributes(AttributeList *AL) { AttrList = AL; }
  AttributeList *getAttributes() const { return AttrList; }
  
  /// TakeAttributes - Return the current attribute list and remove them from
  /// the DeclSpec so that it doesn't own them.
  AttributeList *TakeAttributes() {
    AttributeList *AL = AttrList;
    AttrList = 0;
    return AL;
  }
  
  llvm::SmallVector<Action::DeclTy *, 8> *getProtocolQualifiers() const {
    return ProtocolQualifiers;
  }
  void setProtocolQualifiers(llvm::SmallVector<Action::DeclTy *, 8> *protos) {
    ProtocolQualifiers = protos;
  }
  unsigned getNumProtocolQualifiers() const {
    return ProtocolQualifiers ?  ProtocolQualifiers->size() : 0;
  }
  /// Finish - This does final analysis of the declspec, issuing diagnostics for
  /// things like "_Imaginary" (lacking an FP type).  After calling this method,
  /// DeclSpec is guaranteed self-consistent, even if an error occurred.
  void Finish(Diagnostic &D, SourceManager& SrcMgr, const LangOptions &Lang);
  
private:
  void Diag(Diagnostic &D, SourceLocation Loc, SourceManager& SrcMgr, 
            unsigned DiagID) {
    D.Report(FullSourceLoc(Loc,SrcMgr), DiagID);
  }
  
  void Diag(Diagnostic &D, SourceLocation Loc, SourceManager& SrcMgr,
            unsigned DiagID, const std::string &info) {
    D.Report(FullSourceLoc(Loc,SrcMgr), DiagID, &info, 1);
  }
};

/// ObjCDeclSpec - This class captures information about 
/// "declaration specifiers" specific to objective-c
class ObjCDeclSpec {
public:
  /// ObjCDeclQualifier - Qualifier used on types in method declarations
  enum ObjCDeclQualifier {
    DQ_None = 0x0,
    DQ_In = 0x1,
    DQ_Inout = 0x2,
    DQ_Out = 0x4,
    DQ_Bycopy = 0x8,
    DQ_Byref = 0x10,
    DQ_Oneway = 0x20
  };
  
  /// PropertyAttributeKind - list of property attributes.
  enum ObjCPropertyAttributeKind { DQ_PR_noattr = 0x0, 
    DQ_PR_readonly = 0x01, 
    DQ_PR_getter = 0x02, 
    DQ_PR_assign = 0x04, 
    DQ_PR_readwrite = 0x08, 
    DQ_PR_retain = 0x10,
    DQ_PR_copy = 0x20, 
    DQ_PR_nonatomic = 0x40,
    DQ_PR_setter = 0x80
  };
  
  
  ObjCDeclSpec() : objcDeclQualifier(DQ_None), PropertyAttributes(DQ_PR_noattr),
  GetterName(0), SetterName(0)
  {}
  ObjCDeclQualifier getObjCDeclQualifier() const { return objcDeclQualifier; }
  void setObjCDeclQualifier(ObjCDeclQualifier DQVal) 
    { objcDeclQualifier = (ObjCDeclQualifier) (objcDeclQualifier | DQVal); }
  
  const ObjCPropertyAttributeKind getPropertyAttributes() const 
    { return ObjCPropertyAttributeKind(PropertyAttributes); }
  void setPropertyAttributes(ObjCPropertyAttributeKind PRVal) { 
    PropertyAttributes = 
      (ObjCPropertyAttributeKind) (PropertyAttributes | PRVal);
  }
  
  const IdentifierInfo *getGetterName() const { return GetterName; }
  IdentifierInfo *getGetterName() { return GetterName; }
  void setGetterName(IdentifierInfo *name) { GetterName = name; }
  
  const IdentifierInfo *getSetterName() const { return SetterName; }
  IdentifierInfo *getSetterName() { return SetterName; }
  void setSetterName(IdentifierInfo *name) { SetterName = name; }
private:
  // FIXME: These two are unrelated and mutially exclusive. So perhaps 
  // we can put them in a union to reflect their mutual exclusiveness
  // (space saving is negligible).
  ObjCDeclQualifier objcDeclQualifier : 6;
  
  // NOTE: VC++ treats enums as signed, avoid using ObjCPropertyAttributeKind
  unsigned PropertyAttributes : 8;
  IdentifierInfo *GetterName;    // getter name of NULL if no getter
  IdentifierInfo *SetterName;    // setter name of NULL if no setter
};
  
/// DeclaratorChunk - One instance of this struct is used for each type in a
/// declarator that is parsed.
///
/// This is intended to be a small value object.
struct DeclaratorChunk {
  enum {
    Pointer, Reference, Array, Function
  } Kind;
  
  /// Loc - The place where this type was defined.
  SourceLocation Loc;
  
  struct PointerTypeInfo {
    /// The type qualifiers: const/volatile/restrict.
    unsigned TypeQuals : 3;
    AttributeList *AttrList;
    void destroy() {
      delete AttrList;
    }
  };

  struct ReferenceTypeInfo {
    /// The type qualifier: restrict. [GNU] C++ extension
    bool HasRestrict;
    AttributeList *AttrList;
    void destroy() {
      delete AttrList;
    }
  };

  struct ArrayTypeInfo {
    /// The type qualifiers for the array: const/volatile/restrict.
    unsigned TypeQuals : 3;
    
    /// True if this dimension included the 'static' keyword.
    bool hasStatic : 1;
    
    /// True if this dimension was [*].  In this case, NumElts is null.
    bool isStar : 1;
    
    /// This is the size of the array, or null if [] or [*] was specified.
    /// Since the parser is multi-purpose, and we don't want to impose a root
    /// expression class on all clients, NumElts is untyped.
    Action::ExprTy *NumElts;
    void destroy() {}
  };
  
  /// ParamInfo - An array of paraminfo objects is allocated whenever a function
  /// declarator is parsed.  There are two interesting styles of arguments here:
  /// K&R-style identifier lists and parameter type lists.  K&R-style identifier
  /// lists will have information about the identifier, but no type information.
  /// Parameter type lists will have type info (if the actions module provides
  /// it), but may have null identifier info: e.g. for 'void foo(int X, int)'.
  struct ParamInfo {
    IdentifierInfo *Ident;
    SourceLocation IdentLoc;
    Action::DeclTy *Param;
    ParamInfo() {}
    ParamInfo(IdentifierInfo *ident, SourceLocation iloc, Action::DeclTy *param)
      : Ident(ident), IdentLoc(iloc), Param(param) {}
  };
  
  struct FunctionTypeInfo {
    /// hasPrototype - This is true if the function had at least one typed
    /// argument.  If the function is () or (a,b,c), then it has no prototype,
    /// and is treated as a K&R-style function.
    bool hasPrototype : 1;
    
    /// isVariadic - If this function has a prototype, and if that proto ends
    /// with ',...)', this is true.
    bool isVariadic : 1;

    /// NumArgs - This is the number of formal arguments provided for the
    /// declarator.
    unsigned NumArgs;

    /// ArgInfo - This is a pointer to a new[]'d array of ParamInfo objects that
    /// describe the arguments for this function declarator.  This is null if
    /// there are no arguments specified.
    ParamInfo *ArgInfo;
    
    void destroy() {
      delete[] ArgInfo;
    }
  };
  
  union {
    PointerTypeInfo   Ptr;
    ReferenceTypeInfo Ref;
    ArrayTypeInfo     Arr;
    FunctionTypeInfo  Fun;
  };
  
  
  /// getPointer - Return a DeclaratorChunk for a pointer.
  ///
  static DeclaratorChunk getPointer(unsigned TypeQuals, SourceLocation Loc,
                                    AttributeList *AL) {
    DeclaratorChunk I;
    I.Kind          = Pointer;
    I.Loc           = Loc;
    I.Ptr.TypeQuals = TypeQuals;
    I.Ptr.AttrList  = AL;
    return I;
  }
  
  /// getReference - Return a DeclaratorChunk for a reference.
  ///
  static DeclaratorChunk getReference(unsigned TypeQuals, SourceLocation Loc,
                                      AttributeList *AL) {
    DeclaratorChunk I;
    I.Kind            = Reference;
    I.Loc             = Loc;
    I.Ref.HasRestrict = (TypeQuals & DeclSpec::TQ_restrict) != 0;
    I.Ref.AttrList  = AL;
    return I;
  }
  
  /// getArray - Return a DeclaratorChunk for an array.
  ///
  static DeclaratorChunk getArray(unsigned TypeQuals, bool isStatic,
                                  bool isStar, void *NumElts,
                                  SourceLocation Loc) {
    DeclaratorChunk I;
    I.Kind          = Array;
    I.Loc           = Loc;
    I.Arr.TypeQuals = TypeQuals;
    I.Arr.hasStatic = isStatic;
    I.Arr.isStar    = isStar;
    I.Arr.NumElts   = NumElts;
    return I;
  }
  
  /// getFunction - Return a DeclaratorChunk for a function.
  static DeclaratorChunk getFunction(bool hasProto, bool isVariadic,
                                     ParamInfo *ArgInfo, unsigned NumArgs,
                                     SourceLocation Loc) {
    DeclaratorChunk I;
    I.Kind             = Function;
    I.Loc              = Loc;
    I.Fun.hasPrototype = hasProto;
    I.Fun.isVariadic   = isVariadic;
    I.Fun.NumArgs      = NumArgs;
    I.Fun.ArgInfo      = 0;
    
    // new[] an argument array if needed.
    if (NumArgs) {
      I.Fun.ArgInfo = new DeclaratorChunk::ParamInfo[NumArgs];
      memcpy(I.Fun.ArgInfo, ArgInfo, sizeof(ArgInfo[0])*NumArgs);
    }
    return I;
  }
};


/// Declarator - Information about one declarator, including the parsed type
/// information and the identifier.  When the declarator is fully formed, this
/// is turned into the appropriate Decl object.
///
/// Declarators come in two types: normal declarators and abstract declarators.
/// Abstract declarators are used when parsing types, and don't have an
/// identifier.  Normal declarators do have ID's.
///
/// Instances of this class should be a transient object that lives on the
/// stack, not objects that are allocated in large quantities on the heap.
class Declarator {
  DeclSpec &DS;
  IdentifierInfo *Identifier;
  SourceLocation IdentifierLoc;
  
public:
  enum TheContext {
    FileContext,         // File scope declaration.
    PrototypeContext,    // Within a function prototype.
    KNRTypeListContext,  // K&R type definition list for formals.
    TypeNameContext,     // Abstract declarator for types.
    MemberContext,       // Struct/Union field.
    BlockContext,        // Declaration within a block in a function.
    ForContext           // Declaration within first part of a for loop.
  };
private:
  /// Context - Where we are parsing this declarator.
  ///
  TheContext Context;
  
  /// DeclTypeInfo - This holds each type that the declarator includes as it is
  /// parsed.  This is pushed from the identifier out, which means that element
  /// #0 will be the most closely bound to the identifier, and
  /// DeclTypeInfo.back() will be the least closely bound.
  llvm::SmallVector<DeclaratorChunk, 8> DeclTypeInfo;

  // InvalidType - Set by Sema::GetTypeForDeclarator().
  bool InvalidType;

  // attributes.
  AttributeList *AttrList;
public:
  Declarator(DeclSpec &ds, TheContext C)
    : DS(ds), Identifier(0), Context(C), InvalidType(false), AttrList(0) {
  }
  
  ~Declarator() {
    clear();
  }

  /// getDeclSpec - Return the declaration-specifier that this declarator was
  /// declared with.
  DeclSpec &getDeclSpec() const { return DS; }
  
  TheContext getContext() const { return Context; }
  
  // getSourceRange - FIXME: This should be implemented.
  const SourceRange getSourceRange() const { return SourceRange(); }
  
  /// clear - Reset the contents of this Declarator.
  void clear() {
    Identifier = 0;
    IdentifierLoc = SourceLocation();
    
    for (unsigned i = 0, e = DeclTypeInfo.size(); i != e; ++i) {
      if (DeclTypeInfo[i].Kind == DeclaratorChunk::Function)
        DeclTypeInfo[i].Fun.destroy();
      else if (DeclTypeInfo[i].Kind == DeclaratorChunk::Pointer)
        DeclTypeInfo[i].Ptr.destroy();
      else if (DeclTypeInfo[i].Kind == DeclaratorChunk::Reference)
        DeclTypeInfo[i].Ref.destroy();
      else if (DeclTypeInfo[i].Kind == DeclaratorChunk::Array)
        DeclTypeInfo[i].Arr.destroy();
      else
        assert(0 && "Unknown decl type!");
    }
    DeclTypeInfo.clear();
    delete AttrList;
    AttrList = 0;
  }
  
  /// mayOmitIdentifier - Return true if the identifier is either optional or
  /// not allowed.  This is true for typenames and prototypes.
  bool mayOmitIdentifier() const {
    return Context == TypeNameContext || Context == PrototypeContext;
  }

  /// mayHaveIdentifier - Return true if the identifier is either optional or
  /// required.  This is true for normal declarators and prototypes, but not
  /// typenames.
  bool mayHaveIdentifier() const {
    return Context != TypeNameContext;
  }
  
  /// isPastIdentifier - Return true if we have parsed beyond the point where
  /// the
  bool isPastIdentifier() const { return IdentifierLoc.isValid(); }
  
  IdentifierInfo *getIdentifier() const { return Identifier; }
  SourceLocation getIdentifierLoc() const { return IdentifierLoc; }
  
  void SetIdentifier(IdentifierInfo *ID, SourceLocation Loc) {
    Identifier = ID;
    IdentifierLoc = Loc;
  }
  
  void AddTypeInfo(const DeclaratorChunk &TI) {
    DeclTypeInfo.push_back(TI);
  }
  
  /// getNumTypeObjects() - Return the number of types applied to this
  /// declarator.
  unsigned getNumTypeObjects() const { return DeclTypeInfo.size(); }
  
  /// Return the specified TypeInfo from this declarator.  TypeInfo #0 is
  /// closest to the identifier.
  const DeclaratorChunk &getTypeObject(unsigned i) const {
    assert(i < DeclTypeInfo.size() && "Invalid type chunk");
    return DeclTypeInfo[i];
  }
  DeclaratorChunk &getTypeObject(unsigned i) {
    assert(i < DeclTypeInfo.size() && "Invalid type chunk");
    return DeclTypeInfo[i];
  }
  
  /// isFunctionDeclarator - Once this declarator is fully parsed and formed,
  /// this method returns true if the identifier is a function declarator.
  bool isFunctionDeclarator() const {
    return !DeclTypeInfo.empty() &&
           DeclTypeInfo[0].Kind == DeclaratorChunk::Function;
  }
  
  /// AddAttributes - simply adds the attribute list to the Declarator.
  /// Unlike AddAttributes on DeclSpec, this routine should never have to
  /// concatenate two lists. The following syntax adds 3 attributes to "var":
  ///
  /// short int var __attribute__((aligned(16),common,deprecated));
  ///
  void AddAttributes(AttributeList *alist) { 
    if (!alist)
      return; // we parsed __attribute__(()) or had a syntax error
    assert((AttrList == 0) && "Declarator already has an attribute list");
    AttrList = alist;
  }
  const AttributeList *getAttributes() const { return AttrList; }
  AttributeList *getAttributes() { return AttrList; }
  
  void setInvalidType(bool flag) { InvalidType = flag; }
  bool getInvalidType() const { return InvalidType; }
};

/// FieldDeclarator - This little struct is used to capture information about
/// structure field declarators, which is basically just a bitfield size.
struct FieldDeclarator {
  Declarator D;
  Action::ExprTy *BitfieldSize;
  explicit FieldDeclarator(DeclSpec &DS) : D(DS, Declarator::MemberContext) {
    BitfieldSize = 0;
  }
};


} // end namespace clang

#endif
