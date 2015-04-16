//===- DIBuilder.h - Debug Information Builder ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DIBuilder that is useful for creating debugging
// information entries in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DIBUILDER_H
#define LLVM_IR_DIBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/TrackingMDRef.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class BasicBlock;
  class Instruction;
  class Function;
  class Module;
  class Value;
  class Constant;
  class LLVMContext;
  class StringRef;

  class DIBuilder {
    Module &M;
    LLVMContext &VMContext;

    TempMDTuple TempEnumTypes;
    TempMDTuple TempRetainTypes;
    TempMDTuple TempSubprograms;
    TempMDTuple TempGVs;
    TempMDTuple TempImportedModules;

    Function *DeclareFn;     // llvm.dbg.declare
    Function *ValueFn;       // llvm.dbg.value

    SmallVector<Metadata *, 4> AllEnumTypes;
    /// Track the RetainTypes, since they can be updated later on.
    SmallVector<TrackingMDNodeRef, 4> AllRetainTypes;
    SmallVector<Metadata *, 4> AllSubprograms;
    SmallVector<Metadata *, 4> AllGVs;
    SmallVector<TrackingMDNodeRef, 4> AllImportedModules;

    /// \brief Track nodes that may be unresolved.
    SmallVector<TrackingMDNodeRef, 4> UnresolvedNodes;
    bool AllowUnresolvedNodes;

    /// Each subprogram's preserved local variables.
    DenseMap<MDNode *, std::vector<TrackingMDNodeRef>> PreservedVariables;

    DIBuilder(const DIBuilder &) = delete;
    void operator=(const DIBuilder &) = delete;

    /// \brief Create a temporary.
    ///
    /// Create an \a temporary node and track it in \a UnresolvedNodes.
    void trackIfUnresolved(MDNode *N);

  public:
    /// \brief Construct a builder for a module.
    ///
    /// If \c AllowUnresolved, collect unresolved nodes attached to the module
    /// in order to resolve cycles during \a finalize().
    explicit DIBuilder(Module &M, bool AllowUnresolved = true);
    enum DebugEmissionKind { FullDebug=1, LineTablesOnly };

    /// finalize - Construct any deferred debug info descriptors.
    void finalize();

    /// createCompileUnit - A CompileUnit provides an anchor for all debugging
    /// information generated during this instance of compilation.
    /// @param Lang     Source programming language, eg. dwarf::DW_LANG_C99
    /// @param File     File name
    /// @param Dir      Directory
    /// @param Producer Identify the producer of debugging information and code.
    ///                 Usually this is a compiler version string.
    /// @param isOptimized A boolean flag which indicates whether optimization
    ///                    is ON or not.
    /// @param Flags    This string lists command line options. This string is
    ///                 directly embedded in debug info output which may be used
    ///                 by a tool analyzing generated debugging information.
    /// @param RV       This indicates runtime version for languages like
    ///                 Objective-C.
    /// @param SplitName The name of the file that we'll split debug info out
    ///                  into.
    /// @param Kind     The kind of debug information to generate.
    /// @param EmitDebugInfo   A boolean flag which indicates whether debug
    ///                        information should be written to the final
    ///                        output or not. When this is false, debug
    ///                        information annotations will be present in
    ///                        the IL but they are not written to the final
    ///                        assembly or object file. This supports tracking
    ///                        source location information in the back end
    ///                        without actually changing the output (e.g.,
    ///                        when using optimization remarks).
    MDCompileUnit *createCompileUnit(unsigned Lang, StringRef File,
                                     StringRef Dir, StringRef Producer,
                                     bool isOptimized, StringRef Flags,
                                     unsigned RV, StringRef SplitName = "",
                                     DebugEmissionKind Kind = FullDebug,
                                     bool EmitDebugInfo = true);

    /// createFile - Create a file descriptor to hold debugging information
    /// for a file.
    MDFile *createFile(StringRef Filename, StringRef Directory);

    /// createEnumerator - Create a single enumerator value.
    MDEnumerator *createEnumerator(StringRef Name, int64_t Val);

    /// \brief Create a DWARF unspecified type.
    MDBasicType *createUnspecifiedType(StringRef Name);

    /// \brief Create C++11 nullptr type.
    MDBasicType *createNullPtrType();

    /// createBasicType - Create debugging information entry for a basic
    /// type.
    /// @param Name        Type name.
    /// @param SizeInBits  Size of the type.
    /// @param AlignInBits Type alignment.
    /// @param Encoding    DWARF encoding code, e.g. dwarf::DW_ATE_float.
    MDBasicType *createBasicType(StringRef Name, uint64_t SizeInBits,
                                 uint64_t AlignInBits, unsigned Encoding);

    /// createQualifiedType - Create debugging information entry for a qualified
    /// type, e.g. 'const int'.
    /// @param Tag         Tag identifing type, e.g. dwarf::TAG_volatile_type
    /// @param FromTy      Base Type.
    MDDerivedType *createQualifiedType(unsigned Tag, MDType *FromTy);

    /// createPointerType - Create debugging information entry for a pointer.
    /// @param PointeeTy   Type pointed by this pointer.
    /// @param SizeInBits  Size.
    /// @param AlignInBits Alignment. (optional)
    /// @param Name        Pointer type name. (optional)
    MDDerivedType *createPointerType(MDType *PointeeTy, uint64_t SizeInBits,
                                     uint64_t AlignInBits = 0,
                                     StringRef Name = "");

    /// \brief Create debugging information entry for a pointer to member.
    /// @param PointeeTy Type pointed to by this pointer.
    /// @param SizeInBits  Size.
    /// @param AlignInBits Alignment. (optional)
    /// @param Class Type for which this pointer points to members of.
    MDDerivedType *createMemberPointerType(MDType *PointeeTy, MDType *Class,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits = 0);

    /// createReferenceType - Create debugging information entry for a c++
    /// style reference or rvalue reference type.
    MDDerivedType *createReferenceType(unsigned Tag, MDType *RTy);

    /// createTypedef - Create debugging information entry for a typedef.
    /// @param Ty          Original type.
    /// @param Name        Typedef name.
    /// @param File        File where this type is defined.
    /// @param LineNo      Line number.
    /// @param Context     The surrounding context for the typedef.
    MDDerivedType *createTypedef(MDType *Ty, StringRef Name, MDFile *File,
                                 unsigned LineNo, MDScope *Context);

    /// createFriend - Create debugging information entry for a 'friend'.
    MDDerivedType *createFriend(MDType *Ty, MDType *FriendTy);

    /// createInheritance - Create debugging information entry to establish
    /// inheritance relationship between two types.
    /// @param Ty           Original type.
    /// @param BaseTy       Base type. Ty is inherits from base.
    /// @param BaseOffset   Base offset.
    /// @param Flags        Flags to describe inheritance attribute,
    ///                     e.g. private
    MDDerivedType *createInheritance(MDType *Ty, MDType *BaseTy,
                                     uint64_t BaseOffset, unsigned Flags);

    /// createMemberType - Create debugging information entry for a member.
    /// @param Scope        Member scope.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    MDDerivedType *createMemberType(MDScope *Scope, StringRef Name,
                                    MDFile *File, unsigned LineNo,
                                    uint64_t SizeInBits, uint64_t AlignInBits,
                                    uint64_t OffsetInBits, unsigned Flags,
                                    MDType *Ty);

    /// createStaticMemberType - Create debugging information entry for a
    /// C++ static data member.
    /// @param Scope      Member scope.
    /// @param Name       Member name.
    /// @param File       File where this member is declared.
    /// @param LineNo     Line number.
    /// @param Ty         Type of the static member.
    /// @param Flags      Flags to encode member attribute, e.g. private.
    /// @param Val        Const initializer of the member.
    MDDerivedType *createStaticMemberType(MDScope *Scope, StringRef Name,
                                          MDFile *File, unsigned LineNo,
                                          MDType *Ty, unsigned Flags,
                                          llvm::Constant *Val);

    /// createObjCIVar - Create debugging information entry for Objective-C
    /// instance variable.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    /// @param PropertyNode Property associated with this ivar.
    MDDerivedType *createObjCIVar(StringRef Name, MDFile *File, unsigned LineNo,
                                  uint64_t SizeInBits, uint64_t AlignInBits,
                                  uint64_t OffsetInBits, unsigned Flags,
                                  MDType *Ty, MDNode *PropertyNode);

    /// createObjCProperty - Create debugging information entry for Objective-C
    /// property.
    /// @param Name         Property name.
    /// @param File         File where this property is defined.
    /// @param LineNumber   Line number.
    /// @param GetterName   Name of the Objective C property getter selector.
    /// @param SetterName   Name of the Objective C property setter selector.
    /// @param PropertyAttributes Objective C property attributes.
    /// @param Ty           Type.
    MDObjCProperty *createObjCProperty(StringRef Name, MDFile *File,
                                       unsigned LineNumber,
                                       StringRef GetterName,
                                       StringRef SetterName,
                                       unsigned PropertyAttributes, MDType *Ty);

    /// createClassType - Create debugging information entry for a class.
    /// @param Scope        Scope in which this class is defined.
    /// @param Name         class name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     class members.
    /// @param VTableHolder Debug info of the base class that contains vtable
    ///                     for this type. This is used in
    ///                     DW_AT_containing_type. See DWARF documentation
    ///                     for more info.
    /// @param TemplateParms Template type parameters.
    /// @param UniqueIdentifier A unique identifier for the class.
    MDCompositeType *createClassType(MDScope *Scope, StringRef Name,
                                     MDFile *File, unsigned LineNumber,
                                     uint64_t SizeInBits, uint64_t AlignInBits,
                                     uint64_t OffsetInBits, unsigned Flags,
                                     MDType *DerivedFrom, DIArray Elements,
                                     MDType *VTableHolder = nullptr,
                                     MDNode *TemplateParms = nullptr,
                                     StringRef UniqueIdentifier = "");

    /// createStructType - Create debugging information entry for a struct.
    /// @param Scope        Scope in which this struct is defined.
    /// @param Name         Struct name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     Struct elements.
    /// @param RunTimeLang  Optional parameter, Objective-C runtime version.
    /// @param UniqueIdentifier A unique identifier for the struct.
    MDCompositeType *createStructType(
        MDScope *Scope, StringRef Name, MDFile *File, unsigned LineNumber,
        uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
        MDType *DerivedFrom, DIArray Elements, unsigned RunTimeLang = 0,
        MDType *VTableHolder = nullptr, StringRef UniqueIdentifier = "");

    /// createUnionType - Create debugging information entry for an union.
    /// @param Scope        Scope in which this union is defined.
    /// @param Name         Union name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     Union elements.
    /// @param RunTimeLang  Optional parameter, Objective-C runtime version.
    /// @param UniqueIdentifier A unique identifier for the union.
    MDCompositeType *createUnionType(MDScope *Scope, StringRef Name,
                                     MDFile *File, unsigned LineNumber,
                                     uint64_t SizeInBits, uint64_t AlignInBits,
                                     unsigned Flags, DIArray Elements,
                                     unsigned RunTimeLang = 0,
                                     StringRef UniqueIdentifier = "");

    /// createTemplateTypeParameter - Create debugging information for template
    /// type parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Type parameter name.
    /// @param Ty           Parameter type.
    MDTemplateTypeParameter *
    createTemplateTypeParameter(MDScope *Scope, StringRef Name, MDType *Ty);

    /// createTemplateValueParameter - Create debugging information for template
    /// value parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Value parameter name.
    /// @param Ty           Parameter type.
    /// @param Val          Constant parameter value.
    MDTemplateValueParameter *createTemplateValueParameter(MDScope *Scope,
                                                           StringRef Name,
                                                           MDType *Ty,
                                                           Constant *Val);

    /// \brief Create debugging information for a template template parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Value parameter name.
    /// @param Ty           Parameter type.
    /// @param Val          The fully qualified name of the template.
    MDTemplateValueParameter *createTemplateTemplateParameter(MDScope *Scope,
                                                              StringRef Name,
                                                              MDType *Ty,
                                                              StringRef Val);

    /// \brief Create debugging information for a template parameter pack.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Value parameter name.
    /// @param Ty           Parameter type.
    /// @param Val          An array of types in the pack.
    MDTemplateValueParameter *createTemplateParameterPack(MDScope *Scope,
                                                          StringRef Name,
                                                          MDType *Ty,
                                                          DIArray Val);

    /// createArrayType - Create debugging information entry for an array.
    /// @param Size         Array size.
    /// @param AlignInBits  Alignment.
    /// @param Ty           Element type.
    /// @param Subscripts   Subscripts.
    MDCompositeType *createArrayType(uint64_t Size, uint64_t AlignInBits,
                                     MDType *Ty, DIArray Subscripts);

    /// createVectorType - Create debugging information entry for a vector type.
    /// @param Size         Array size.
    /// @param AlignInBits  Alignment.
    /// @param Ty           Element type.
    /// @param Subscripts   Subscripts.
    MDCompositeType *createVectorType(uint64_t Size, uint64_t AlignInBits,
                                      MDType *Ty, DIArray Subscripts);

    /// createEnumerationType - Create debugging information entry for an
    /// enumeration.
    /// @param Scope          Scope in which this enumeration is defined.
    /// @param Name           Union name.
    /// @param File           File where this member is defined.
    /// @param LineNumber     Line number.
    /// @param SizeInBits     Member size.
    /// @param AlignInBits    Member alignment.
    /// @param Elements       Enumeration elements.
    /// @param UnderlyingType Underlying type of a C++11/ObjC fixed enum.
    /// @param UniqueIdentifier A unique identifier for the enum.
    MDCompositeType *createEnumerationType(
        MDScope *Scope, StringRef Name, MDFile *File, unsigned LineNumber,
        uint64_t SizeInBits, uint64_t AlignInBits, DIArray Elements,
        MDType *UnderlyingType, StringRef UniqueIdentifier = "");

    /// createSubroutineType - Create subroutine type.
    /// @param File            File in which this subroutine is defined.
    /// @param ParameterTypes  An array of subroutine parameter types. This
    ///                        includes return type at 0th index.
    /// @param Flags           E.g.: LValueReference.
    ///                        These flags are used to emit dwarf attributes.
    MDSubroutineType *createSubroutineType(MDFile *File,
                                           DITypeArray ParameterTypes,
                                           unsigned Flags = 0);

    /// createArtificialType - Create a new MDType* with "artificial" flag set.
    MDType *createArtificialType(MDType *Ty);

    /// createObjectPointerType - Create a new MDType* with the "object pointer"
    /// flag set.
    MDType *createObjectPointerType(MDType *Ty);

    /// \brief Create a permanent forward-declared type.
    MDCompositeType *createForwardDecl(unsigned Tag, StringRef Name,
                                       MDScope *Scope, MDFile *F, unsigned Line,
                                       unsigned RuntimeLang = 0,
                                       uint64_t SizeInBits = 0,
                                       uint64_t AlignInBits = 0,
                                       StringRef UniqueIdentifier = "");

    /// \brief Create a temporary forward-declared type.
    MDCompositeType *createReplaceableCompositeType(
        unsigned Tag, StringRef Name, MDScope *Scope, MDFile *F, unsigned Line,
        unsigned RuntimeLang = 0, uint64_t SizeInBits = 0,
        uint64_t AlignInBits = 0, unsigned Flags = DebugNode::FlagFwdDecl,
        StringRef UniqueIdentifier = "");

    /// retainType - Retain MDType* in a module even if it is not referenced
    /// through debug info anchors.
    void retainType(MDType *T);

    /// createUnspecifiedParameter - Create unspecified parameter type
    /// for a subroutine type.
    MDBasicType *createUnspecifiedParameter();

    /// getOrCreateArray - Get a DIArray, create one if required.
    DIArray getOrCreateArray(ArrayRef<Metadata *> Elements);

    /// getOrCreateTypeArray - Get a DITypeArray, create one if required.
    DITypeArray getOrCreateTypeArray(ArrayRef<Metadata *> Elements);

    /// getOrCreateSubrange - Create a descriptor for a value range.  This
    /// implicitly uniques the values returned.
    MDSubrange *getOrCreateSubrange(int64_t Lo, int64_t Count);

    /// createGlobalVariable - Create a new descriptor for the specified
    /// variable.
    /// @param Context     Variable scope.
    /// @param Name        Name of the variable.
    /// @param LinkageName Mangled  name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    /// @param Decl        Reference to the corresponding declaration.
    MDGlobalVariable *createGlobalVariable(MDScope *Context, StringRef Name,
                                           StringRef LinkageName, MDFile *File,
                                           unsigned LineNo, MDType *Ty,
                                           bool isLocalToUnit,
                                           llvm::Constant *Val,
                                           MDNode *Decl = nullptr);

    /// createTempGlobalVariableFwdDecl - Identical to createGlobalVariable
    /// except that the resulting DbgNode is temporary and meant to be RAUWed.
    MDGlobalVariable *createTempGlobalVariableFwdDecl(
        MDScope *Context, StringRef Name, StringRef LinkageName, MDFile *File,
        unsigned LineNo, MDType *Ty, bool isLocalToUnit, llvm::Constant *Val,
        MDNode *Decl = nullptr);

    /// createLocalVariable - Create a new descriptor for the specified
    /// local variable.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param AlwaysPreserve Boolean. Set to true if debug info for this
    ///                       variable should be preserved in optimized build.
    /// @param Flags       Flags, e.g. artificial variable.
    /// @param ArgNo       If this variable is an argument then this argument's
    ///                    number. 1 indicates 1st argument.
    MDLocalVariable *createLocalVariable(unsigned Tag, MDScope *Scope,
                                         StringRef Name, MDFile *File,
                                         unsigned LineNo, MDType *Ty,
                                         bool AlwaysPreserve = false,
                                         unsigned Flags = 0,
                                         unsigned ArgNo = 0);

    /// createExpression - Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    /// @param Addr        An array of complex address operations.
    MDExpression *createExpression(ArrayRef<uint64_t> Addr = None);
    MDExpression *createExpression(ArrayRef<int64_t> Addr);

    /// createBitPieceExpression - Create a descriptor to describe one part
    /// of aggregate variable that is fragmented across multiple Values.
    ///
    /// @param OffsetInBits Offset of the piece in bits.
    /// @param SizeInBits   Size of the piece in bits.
    MDExpression *createBitPieceExpression(unsigned OffsetInBits,
                                           unsigned SizeInBits);

    /// createFunction - Create a new descriptor for the specified subprogram.
    /// See comments in MDSubprogram* for descriptions of these fields.
    /// @param Scope         Function scope.
    /// @param Name          Function name.
    /// @param LinkageName   Mangled function name.
    /// @param File          File where this variable is defined.
    /// @param LineNo        Line number.
    /// @param Ty            Function type.
    /// @param isLocalToUnit True if this function is not externally visible.
    /// @param isDefinition  True if this is a function definition.
    /// @param ScopeLine     Set to the beginning of the scope this starts
    /// @param Flags         e.g. is this function prototyped or not.
    ///                      These flags are used to emit dwarf attributes.
    /// @param isOptimized   True if optimization is ON.
    /// @param Fn            llvm::Function pointer.
    /// @param TParam        Function template parameters.
    MDSubprogram *
    createFunction(MDScope *Scope, StringRef Name, StringRef LinkageName,
                   MDFile *File, unsigned LineNo, MDSubroutineType *Ty,
                   bool isLocalToUnit, bool isDefinition, unsigned ScopeLine,
                   unsigned Flags = 0, bool isOptimized = false,
                   Function *Fn = nullptr, MDNode *TParam = nullptr,
                   MDNode *Decl = nullptr);

    /// createTempFunctionFwdDecl - Identical to createFunction,
    /// except that the resulting DbgNode is meant to be RAUWed.
    MDSubprogram *createTempFunctionFwdDecl(
        MDScope *Scope, StringRef Name, StringRef LinkageName, MDFile *File,
        unsigned LineNo, MDSubroutineType *Ty, bool isLocalToUnit,
        bool isDefinition, unsigned ScopeLine, unsigned Flags = 0,
        bool isOptimized = false, Function *Fn = nullptr,
        MDNode *TParam = nullptr, MDNode *Decl = nullptr);

    /// FIXME: this is added for dragonegg. Once we update dragonegg
    /// to call resolve function, this will be removed.
    MDSubprogram *
    createFunction(DIScopeRef Scope, StringRef Name, StringRef LinkageName,
                   MDFile *File, unsigned LineNo, MDSubroutineType *Ty,
                   bool isLocalToUnit, bool isDefinition, unsigned ScopeLine,
                   unsigned Flags = 0, bool isOptimized = false,
                   Function *Fn = nullptr, MDNode *TParam = nullptr,
                   MDNode *Decl = nullptr);

    /// createMethod - Create a new descriptor for the specified C++ method.
    /// See comments in MDSubprogram* for descriptions of these fields.
    /// @param Scope         Function scope.
    /// @param Name          Function name.
    /// @param LinkageName   Mangled function name.
    /// @param File          File where this variable is defined.
    /// @param LineNo        Line number.
    /// @param Ty            Function type.
    /// @param isLocalToUnit True if this function is not externally visible..
    /// @param isDefinition  True if this is a function definition.
    /// @param Virtuality    Attributes describing virtualness. e.g. pure
    ///                      virtual function.
    /// @param VTableIndex   Index no of this method in virtual table.
    /// @param VTableHolder  Type that holds vtable.
    /// @param Flags         e.g. is this function prototyped or not.
    ///                      This flags are used to emit dwarf attributes.
    /// @param isOptimized   True if optimization is ON.
    /// @param Fn            llvm::Function pointer.
    /// @param TParam        Function template parameters.
    MDSubprogram *
    createMethod(MDScope *Scope, StringRef Name, StringRef LinkageName,
                 MDFile *File, unsigned LineNo, MDSubroutineType *Ty,
                 bool isLocalToUnit, bool isDefinition, unsigned Virtuality = 0,
                 unsigned VTableIndex = 0, MDType *VTableHolder = nullptr,
                 unsigned Flags = 0, bool isOptimized = false,
                 Function *Fn = nullptr, MDNode *TParam = nullptr);

    /// createNameSpace - This creates new descriptor for a namespace
    /// with the specified parent scope.
    /// @param Scope       Namespace scope
    /// @param Name        Name of this namespace
    /// @param File        Source file
    /// @param LineNo      Line number
    MDNamespace *createNameSpace(MDScope *Scope, StringRef Name, MDFile *File,
                                 unsigned LineNo);

    /// createLexicalBlockFile - This creates a descriptor for a lexical
    /// block with a new file attached. This merely extends the existing
    /// lexical block as it crosses a file.
    /// @param Scope       Lexical block.
    /// @param File        Source file.
    /// @param Discriminator DWARF path discriminator value.
    MDLexicalBlockFile *createLexicalBlockFile(MDScope *Scope, MDFile *File,
                                               unsigned Discriminator = 0);

    /// createLexicalBlock - This creates a descriptor for a lexical block
    /// with the specified parent context.
    /// @param Scope         Parent lexical scope.
    /// @param File          Source file.
    /// @param Line          Line number.
    /// @param Col           Column number.
    MDLexicalBlock *createLexicalBlock(MDScope *Scope, MDFile *File,
                                       unsigned Line, unsigned Col);

    /// \brief Create a descriptor for an imported module.
    /// @param Context The scope this module is imported into
    /// @param NS The namespace being imported here
    /// @param Line Line number
    MDImportedEntity *createImportedModule(MDScope *Context, MDNamespace *NS,
                                           unsigned Line);

    /// \brief Create a descriptor for an imported module.
    /// @param Context The scope this module is imported into
    /// @param NS An aliased namespace
    /// @param Line Line number
    MDImportedEntity *createImportedModule(MDScope *Context,
                                           MDImportedEntity *NS, unsigned Line);

    /// \brief Create a descriptor for an imported function.
    /// @param Context The scope this module is imported into
    /// @param Decl The declaration (or definition) of a function, type, or
    ///             variable
    /// @param Line Line number
    MDImportedEntity *createImportedDeclaration(MDScope *Context,
                                                DebugNode *Decl, unsigned Line,
                                                StringRef Name = "");

    /// insertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage     llvm::Value of the variable
    /// @param VarInfo     Variable's debug info descriptor.
    /// @param Expr         A complex location expression.
    /// @param DL           Debug info location.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *insertDeclare(llvm::Value *Storage, MDLocalVariable *VarInfo,
                               MDExpression *Expr, const MDLocation *DL,
                               BasicBlock *InsertAtEnd);

    /// insertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage      llvm::Value of the variable
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param Expr         A complex location expression.
    /// @param DL           Debug info location.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *insertDeclare(llvm::Value *Storage, MDLocalVariable *VarInfo,
                               MDExpression *Expr, const MDLocation *DL,
                               Instruction *InsertBefore);

    /// insertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param Expr         A complex location expression.
    /// @param DL           Debug info location.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *insertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         MDLocalVariable *VarInfo,
                                         MDExpression *Expr,
                                         const MDLocation *DL,
                                         BasicBlock *InsertAtEnd);

    /// insertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param Expr         A complex location expression.
    /// @param DL           Debug info location.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *insertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         MDLocalVariable *VarInfo,
                                         MDExpression *Expr,
                                         const MDLocation *DL,
                                         Instruction *InsertBefore);

    /// \brief Replace the vtable holder in the given composite type.
    ///
    /// If this creates a self reference, it may orphan some unresolved cycles
    /// in the operands of \c T, so \a DIBuilder needs to track that.
    void replaceVTableHolder(MDCompositeType *&T,
                             MDCompositeType *VTableHolder);

    /// \brief Replace arrays on a composite type.
    ///
    /// If \c T is resolved, but the arrays aren't -- which can happen if \c T
    /// has a self-reference -- \a DIBuilder needs to track the array to
    /// resolve cycles.
    void replaceArrays(MDCompositeType *&T, DIArray Elements,
                       DIArray TParems = DIArray());

    /// \brief Replace a temporary node.
    ///
    /// Call \a MDNode::replaceAllUsesWith() on \c N, replacing it with \c
    /// Replacement.
    ///
    /// If \c Replacement is the same as \c N.get(), instead call \a
    /// MDNode::replaceWithUniqued().  In this case, the uniqued node could
    /// have a different address, so we return the final address.
    template <class NodeTy>
    NodeTy *replaceTemporary(TempMDNode &&N, NodeTy *Replacement) {
      if (N.get() == Replacement)
        return cast<NodeTy>(MDNode::replaceWithUniqued(std::move(N)));

      N->replaceAllUsesWith(Replacement);
      return Replacement;
    }
  };
} // end namespace llvm

#endif
