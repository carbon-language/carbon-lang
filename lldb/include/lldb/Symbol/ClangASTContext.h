//===-- ClangASTContext.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangASTContext_h_
#define liblldb_ClangASTContext_h_

// C Includes
#include <stdint.h>

// C++ Includes
#include <initializer_list>
#include <string>
#include <vector>
#include <utility>

// Other libraries and framework includes
#include "llvm/ADT/SmallVector.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/TemplateBase.h"


// Project includes
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/TypeSystem.h"

namespace lldb_private {

class Declaration;

class ClangASTContext : public TypeSystem
{
public:
    typedef void (*CompleteTagDeclCallback)(void *baton, clang::TagDecl *);
    typedef void (*CompleteObjCInterfaceDeclCallback)(void *baton, clang::ObjCInterfaceDecl *);
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ClangASTContext (const char *triple = NULL);

    ~ClangASTContext();
    
    static ClangASTContext*
    GetASTContext (clang::ASTContext* ast_ctx);

    clang::ASTContext *
    getASTContext();
    
    clang::Builtin::Context *
    getBuiltinContext();

    clang::IdentifierTable *
    getIdentifierTable();

    clang::LangOptions *
    getLanguageOptions();

    clang::SelectorTable *
    getSelectorTable();

    clang::FileManager *
    getFileManager();
    
    clang::SourceManager *
    getSourceManager();

    clang::DiagnosticsEngine *
    getDiagnosticsEngine();
    
    clang::DiagnosticConsumer *
    getDiagnosticConsumer();

    std::shared_ptr<clang::TargetOptions> &getTargetOptions();

    clang::TargetInfo *
    getTargetInfo();

    void
    Clear();

    const char *
    GetTargetTriple ();

    void
    SetTargetTriple (const char *target_triple);

    void
    SetArchitecture (const ArchSpec &arch);

    bool
    HasExternalSource ();

    void
    SetExternalSource (llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> &ast_source_ap);

    void
    RemoveExternalSource ();
    
    bool
    GetCompleteDecl (clang::Decl *decl)
    {
        return ClangASTContext::GetCompleteDecl(getASTContext(), decl);
    }
    
    static bool
    GetCompleteDecl (clang::ASTContext *ast,
                     clang::Decl *decl);

    void SetMetadataAsUserID (const void *object,
                              lldb::user_id_t user_id);

    void SetMetadata (const void *object,
                      ClangASTMetadata &meta_data)
    {
        SetMetadata(getASTContext(), object, meta_data);
    }
    
    static void
    SetMetadata (clang::ASTContext *ast,
                 const void *object,
                 ClangASTMetadata &meta_data);
    
    ClangASTMetadata *
    GetMetadata (const void *object)
    {
        return GetMetadata(getASTContext(), object);
    }
    
    static ClangASTMetadata *
    GetMetadata (clang::ASTContext *ast,
                 const void *object);
    
    //------------------------------------------------------------------
    // Basic Types
    //------------------------------------------------------------------
    ClangASTType
    GetBuiltinTypeForEncodingAndBitSize (lldb::Encoding encoding,
                                          uint32_t bit_size);

    static ClangASTType
    GetBuiltinTypeForEncodingAndBitSize (clang::ASTContext *ast,
                                         lldb::Encoding encoding,
                                         uint32_t bit_size);

    ClangASTType
    GetBasicType (lldb::BasicType type);

    static ClangASTType
    GetBasicType (clang::ASTContext *ast, lldb::BasicType type);
    
    static ClangASTType
    GetBasicType (clang::ASTContext *ast, const ConstString &name);
    
    static lldb::BasicType
    GetBasicTypeEnumeration (const ConstString &name);

    ClangASTType
    GetBuiltinTypeForDWARFEncodingAndBitSize (
        const char *type_name,
        uint32_t dw_ate,
        uint32_t bit_size);

    ClangASTType
    GetCStringType(bool is_const);
    
    static ClangASTType
    GetUnknownAnyType(clang::ASTContext *ast);
    
    ClangASTType
    GetUnknownAnyType()
    {
        return ClangASTContext::GetUnknownAnyType(getASTContext());
    }
    
    
    static clang::DeclContext *
    GetDeclContextForType (clang::QualType type);

    static clang::DeclContext *
    GetDeclContextForType (const ClangASTType& type)
    {
        return GetDeclContextForType(GetQualType(type));
    }
    
    uint32_t
    GetPointerByteSize ();

    static clang::DeclContext *
    GetTranslationUnitDecl (clang::ASTContext *ast);
    
    clang::DeclContext *
    GetTranslationUnitDecl ()
    {
        return GetTranslationUnitDecl (getASTContext());
    }
    
    static bool
    GetClassMethodInfoForDeclContext (clang::DeclContext *decl_ctx,
                                      lldb::LanguageType &language,
                                      bool &is_instance_method,
                                      ConstString &language_object_name);
    
    static ClangASTType
    CopyType(clang::ASTContext *dest_context, 
             ClangASTType source_type);
    
    static clang::Decl *
    CopyDecl (clang::ASTContext *dest_context, 
              clang::ASTContext *source_context,
              clang::Decl *source_decl);

    static bool
    AreTypesSame(ClangASTType type1,
                 ClangASTType type2,
                 bool ignore_qualifiers = false);
    
    static ClangASTType
    GetTypeForDecl (clang::NamedDecl *decl);
    
    static ClangASTType
    GetTypeForDecl (clang::TagDecl *decl);
    
    static ClangASTType
    GetTypeForDecl (clang::ObjCInterfaceDecl *objc_decl);
    
    template <typename RecordDeclType>
    ClangASTType
    GetTypeForIdentifier (const ConstString &type_name)
    {
        ClangASTType clang_type;
        
        if (type_name.GetLength())
        {
            clang::ASTContext *ast = getASTContext();
            if (ast)
            {
                clang::IdentifierInfo &myIdent = ast->Idents.get(type_name.GetCString());
                clang::DeclarationName myName = ast->DeclarationNames.getIdentifier(&myIdent);
                
                clang::DeclContext::lookup_result result = ast->getTranslationUnitDecl()->lookup(myName);
                
                if (!result.empty())
                {
                    clang::NamedDecl *named_decl = result[0];
                    if (const RecordDeclType *record_decl = llvm::dyn_cast<RecordDeclType>(named_decl))
                        clang_type.SetClangType(ast, clang::QualType(record_decl->getTypeForDecl(), 0));
                }
            }
        }
        
        return clang_type;
    }
    
    ClangASTType
    GetOrCreateStructForIdentifier (const ConstString &type_name,
                                    const std::initializer_list< std::pair < const char *, ClangASTType > >& type_fields,
                                    bool packed = false);

    //------------------------------------------------------------------
    // Structure, Unions, Classes
    //------------------------------------------------------------------

    static clang::AccessSpecifier
    ConvertAccessTypeToAccessSpecifier (lldb::AccessType access);

    static clang::AccessSpecifier
    UnifyAccessSpecifiers (clang::AccessSpecifier lhs, clang::AccessSpecifier rhs);

    static uint32_t
    GetNumBaseClasses (const clang::CXXRecordDecl *cxx_record_decl,
                       bool omit_empty_base_classes);

    ClangASTType
    CreateRecordType (clang::DeclContext *decl_ctx,
                      lldb::AccessType access_type,
                      const char *name,
                      int kind,
                      lldb::LanguageType language,
                      ClangASTMetadata *metadata = NULL);
    
    class TemplateParameterInfos
    {
    public:
        bool
        IsValid() const
        {
            if (args.empty())
                return false;
            return args.size() == names.size();
        }

        size_t
        GetSize () const
        {
            if (IsValid())
                return args.size();
            return 0;
        }

        llvm::SmallVector<const char *, 2> names;
        llvm::SmallVector<clang::TemplateArgument, 2> args;
    };

    clang::FunctionTemplateDecl *
    CreateFunctionTemplateDecl (clang::DeclContext *decl_ctx,
                                clang::FunctionDecl *func_decl,
                                const char *name, 
                                const TemplateParameterInfos &infos);
    
    void
    CreateFunctionTemplateSpecializationInfo (clang::FunctionDecl *func_decl, 
                                              clang::FunctionTemplateDecl *Template,
                                              const TemplateParameterInfos &infos);

    clang::ClassTemplateDecl *
    CreateClassTemplateDecl (clang::DeclContext *decl_ctx,
                             lldb::AccessType access_type,
                             const char *class_name, 
                             int kind, 
                             const TemplateParameterInfos &infos);

    clang::ClassTemplateSpecializationDecl *
    CreateClassTemplateSpecializationDecl (clang::DeclContext *decl_ctx,
                                           clang::ClassTemplateDecl *class_template_decl,
                                           int kind,
                                           const TemplateParameterInfos &infos);

    ClangASTType
    CreateClassTemplateSpecializationType (clang::ClassTemplateSpecializationDecl *class_template_specialization_decl);

    static clang::DeclContext *
    GetAsDeclContext (clang::CXXMethodDecl *cxx_method_decl);

    static clang::DeclContext *
    GetAsDeclContext (clang::ObjCMethodDecl *objc_method_decl);

    
    static bool
    CheckOverloadedOperatorKindParameterCount (uint32_t op_kind, 
                                               uint32_t num_params);

    bool
    FieldIsBitfield (clang::FieldDecl* field,
                     uint32_t& bitfield_bit_size);

    static bool
    FieldIsBitfield (clang::ASTContext *ast,
                     clang::FieldDecl* field,
                     uint32_t& bitfield_bit_size);

    static bool
    RecordHasFields (const clang::RecordDecl *record_decl);


    ClangASTType
    CreateObjCClass (const char *name, 
                     clang::DeclContext *decl_ctx, 
                     bool isForwardDecl, 
                     bool isInternal,
                     ClangASTMetadata *metadata = NULL);
    
    bool
    SetTagTypeKind (clang::QualType type, int kind) const;
    
    bool
    SetDefaultAccessForRecordFields (clang::RecordDecl* record_decl,
                                     int default_accessibility,
                                     int *assigned_accessibilities,
                                     size_t num_assigned_accessibilities);

    

    // Returns a mask containing bits from the ClangASTContext::eTypeXXX enumerations


    //------------------------------------------------------------------
    // Namespace Declarations
    //------------------------------------------------------------------

    clang::NamespaceDecl *
    GetUniqueNamespaceDeclaration (const char *name,
                                   clang::DeclContext *decl_ctx);

    //------------------------------------------------------------------
    // Function Types
    //------------------------------------------------------------------

    clang::FunctionDecl *
    CreateFunctionDeclaration (clang::DeclContext *decl_ctx,
                               const char *name,
                               const ClangASTType &function_Type,
                               int storage,
                               bool is_inline);
    
    static ClangASTType
    CreateFunctionType (clang::ASTContext *ast,
                        const ClangASTType &result_type,
                        const ClangASTType *args,
                        unsigned num_args,
                        bool is_variadic,
                        unsigned type_quals);
    
    ClangASTType
    CreateFunctionType (const ClangASTType &result_type,
                        const ClangASTType *args,
                        unsigned num_args,
                        bool is_variadic,
                        unsigned type_quals)
    {
        return ClangASTContext::CreateFunctionType(getASTContext(),
                                                   result_type,
                                                   args,
                                                   num_args,
                                                   is_variadic,
                                                   type_quals);
    }
    
    clang::ParmVarDecl *
    CreateParameterDeclaration (const char *name,
                                const ClangASTType &param_type,
                                int storage);

    void
    SetFunctionParameters (clang::FunctionDecl *function_decl,
                           clang::ParmVarDecl **params,
                           unsigned num_params);

    //------------------------------------------------------------------
    // Array Types
    //------------------------------------------------------------------

    ClangASTType
    CreateArrayType (const ClangASTType &element_type,
                     size_t element_count,
                     bool is_vector);

    //------------------------------------------------------------------
    // Enumeration Types
    //------------------------------------------------------------------
    ClangASTType
    CreateEnumerationType (const char *name, 
                           clang::DeclContext *decl_ctx, 
                           const Declaration &decl, 
                           const ClangASTType &integer_qual_type);
    
    //------------------------------------------------------------------
    // Integer type functions
    //------------------------------------------------------------------
    
    ClangASTType
    GetIntTypeFromBitSize (size_t bit_size, bool is_signed)
    {
        return GetIntTypeFromBitSize (getASTContext(), bit_size, is_signed);
    }
    
    static ClangASTType
    GetIntTypeFromBitSize (clang::ASTContext *ast,
                           size_t bit_size, bool is_signed);
    
    ClangASTType
    GetPointerSizedIntType (bool is_signed)
    {
        return GetPointerSizedIntType (getASTContext(), is_signed);
    }
    
    static ClangASTType
    GetPointerSizedIntType (clang::ASTContext *ast, bool is_signed);
    
    //------------------------------------------------------------------
    // Floating point functions
    //------------------------------------------------------------------
    
    ClangASTType
    GetFloatTypeFromBitSize (size_t bit_size)
    {
        return GetFloatTypeFromBitSize (getASTContext(), bit_size);
    }

    static ClangASTType
    GetFloatTypeFromBitSize (clang::ASTContext *ast,
                             size_t bit_size);

    //------------------------------------------------------------------
    // TypeSystem methods
    //------------------------------------------------------------------
    
    ClangASTContext*
    AsClangASTContext()
    {
        return this;
    }
    
    //----------------------------------------------------------------------
    // Tests
    //----------------------------------------------------------------------
    
    bool
    IsArrayType (void* type,
                 ClangASTType *element_type,
                 uint64_t *size,
                 bool *is_incomplete);
    
    bool
    IsVectorType (void* type,
                  ClangASTType *element_type,
                  uint64_t *size);
    
    bool
    IsAggregateType (void* type);
    
    bool
    IsBeingDefined (void* type);
    
    bool
    IsCharType (void* type);
    
    bool
    IsCompleteType (void* type);
    
    bool
    IsConst(void* type);
    
    bool
    IsCStringType (void* type, uint32_t &length);
    
    static bool
    IsCXXClassType (const ClangASTType& type);
    
    bool
    IsDefined(void* type);
    
    bool
    IsFloatingPointType (void* type, uint32_t &count, bool &is_complex);
    
    bool
    IsFunctionType (void* type, bool *is_variadic_ptr = NULL);
    
    uint32_t
    IsHomogeneousAggregate (void* type, ClangASTType* base_type_ptr);
    
    size_t
    GetNumberOfFunctionArguments (void* type);
    
    ClangASTType
    GetFunctionArgumentAtIndex (void* type, const size_t index);
    
    bool
    IsFunctionPointerType (void* type);
    
    bool
    IsIntegerType (void* type, bool &is_signed);
    
    static bool
    IsObjCClassType (const ClangASTType& type);
    
    static bool
    IsObjCClassTypeAndHasIVars (const ClangASTType& type, bool check_superclass);
    
    static bool
    IsObjCObjectOrInterfaceType (const ClangASTType& type);
    
    static bool
    IsObjCObjectPointerType (const ClangASTType& type, ClangASTType *target_type = NULL);
    
    bool
    IsPolymorphicClass (void* type);
    
    bool
    IsPossibleDynamicType (void* type,
                           ClangASTType *target_type, // Can pass NULL
                           bool check_cplusplus,
                           bool check_objc);
    
    bool
    IsRuntimeGeneratedType (void* type);
    
    bool
    IsPointerType (void* type, ClangASTType *pointee_type = NULL);
    
    bool
    IsPointerOrReferenceType (void* type, ClangASTType *pointee_type = NULL);
    
    bool
    IsReferenceType (void* type, ClangASTType *pointee_type = nullptr, bool* is_rvalue = nullptr);
    
    bool
    IsScalarType (void* type);
    
    bool
    IsTypedefType (void* type);
    
    bool
    IsVoidType (void* type);
    
    static bool
    GetCXXClassName (const ClangASTType& type, std::string &class_name);
    
    static bool
    GetObjCClassName (const ClangASTType& type, std::string &class_name);
    
    
    //----------------------------------------------------------------------
    // Type Completion
    //----------------------------------------------------------------------
    
    bool
    GetCompleteType (void* type);
    
    //----------------------------------------------------------------------
    // Accessors
    //----------------------------------------------------------------------
    
    ConstString
    GetTypeName (void* type);
    
    uint32_t
    GetTypeInfo (void* type, ClangASTType *pointee_or_element_clang_type = NULL);
    
    lldb::LanguageType
    GetMinimumLanguage (void* type);
    
    lldb::TypeClass
    GetTypeClass (void* type);
    
    unsigned
    GetTypeQualifiers(void* type);
    
    //----------------------------------------------------------------------
    // Creating related types
    //----------------------------------------------------------------------
    
    static ClangASTType
    AddConstModifier (const ClangASTType& type);
    
    static ClangASTType
    AddRestrictModifier (const ClangASTType& type);
    
    static ClangASTType
    AddVolatileModifier (const ClangASTType& type);
    
    // Using the current type, create a new typedef to that type using "typedef_name"
    // as the name and "decl_ctx" as the decl context.
    static ClangASTType
    CreateTypedefType (const ClangASTType& type,
                       const char *typedef_name,
                       clang::DeclContext *decl_ctx);
    
    ClangASTType
    GetArrayElementType (void* type, uint64_t *stride = nullptr);
    
    ClangASTType
    GetCanonicalType (void* type);
    
    ClangASTType
    GetFullyUnqualifiedType (void* type);
    
    // Returns -1 if this isn't a function of if the function doesn't have a prototype
    // Returns a value >= 0 if there is a prototype.
    int
    GetFunctionArgumentCount (void* type);
    
    ClangASTType
    GetFunctionArgumentTypeAtIndex (void* type, size_t idx);
    
    ClangASTType
    GetFunctionReturnType (void* type);
    
    size_t
    GetNumMemberFunctions (void* type);
    
    TypeMemberFunctionImpl
    GetMemberFunctionAtIndex (void* type, size_t idx);
    
    static ClangASTType
    GetLValueReferenceType (const ClangASTType& type);
    
    ClangASTType
    GetNonReferenceType (void* type);
    
    ClangASTType
    GetPointeeType (void* type);
    
    ClangASTType
    GetPointerType (void* type);
    
    static ClangASTType
    GetRValueReferenceType (const ClangASTType& type);
    
    // If the current object represents a typedef type, get the underlying type
    ClangASTType
    GetTypedefedType (void* type);
    
    static ClangASTType
    RemoveFastQualifiers (const ClangASTType& type);
    
    //----------------------------------------------------------------------
    // Create related types using the current type's AST
    //----------------------------------------------------------------------
    ClangASTType
    GetBasicTypeFromAST (void* type, lldb::BasicType basic_type);
    
    //----------------------------------------------------------------------
    // Exploring the type
    //----------------------------------------------------------------------
    
    uint64_t
    GetByteSize (void *type, ExecutionContextScope *exe_scope)
    {
        return (GetBitSize (type, exe_scope) + 7) / 8;
    }
    
    uint64_t
    GetBitSize (void* type, ExecutionContextScope *exe_scope);
    
    lldb::Encoding
    GetEncoding (void* type, uint64_t &count);
    
    lldb::Format
    GetFormat (void* type);
    
    size_t
    GetTypeBitAlign (void* type);
    
    uint32_t
    GetNumChildren (void* type, bool omit_empty_base_classes);
    
    lldb::BasicType
    GetBasicTypeEnumeration (void* type);
    
    static lldb::BasicType
    GetBasicTypeEnumeration (void* type, const ConstString &name);
    
    static uint32_t
    GetNumDirectBaseClasses (const ClangASTType& type);
    
    static uint32_t
    GetNumVirtualBaseClasses (const ClangASTType& type);
    
    uint32_t
    GetNumFields (void* type);
    
    static ClangASTType
    GetDirectBaseClassAtIndex (const ClangASTType& type,
                               size_t idx,
                               uint32_t *bit_offset_ptr);
    
    static ClangASTType
    GetVirtualBaseClassAtIndex (const ClangASTType& type,
                                size_t idx,
                                uint32_t *bit_offset_ptr);
    
    ClangASTType
    GetFieldAtIndex (void* type,
                     size_t idx,
                     std::string& name,
                     uint64_t *bit_offset_ptr,
                     uint32_t *bitfield_bit_size_ptr,
                     bool *is_bitfield_ptr);
    
    static uint32_t
    GetNumPointeeChildren (clang::QualType type);
    
    ClangASTType
    GetChildClangTypeAtIndex (void* type,
                              ExecutionContext *exe_ctx,
                              size_t idx,
                              bool transparent_pointers,
                              bool omit_empty_base_classes,
                              bool ignore_array_bounds,
                              std::string& child_name,
                              uint32_t &child_byte_size,
                              int32_t &child_byte_offset,
                              uint32_t &child_bitfield_bit_size,
                              uint32_t &child_bitfield_bit_offset,
                              bool &child_is_base_class,
                              bool &child_is_deref_of_parent,
                              ValueObject *valobj);
    
    // Lookup a child given a name. This function will match base class names
    // and member member names in "clang_type" only, not descendants.
    uint32_t
    GetIndexOfChildWithName (void* type,
                             const char *name,
                             bool omit_empty_base_classes);
    
    // Lookup a child member given a name. This function will match member names
    // only and will descend into "clang_type" children in search for the first
    // member in this class, or any base class that matches "name".
    // TODO: Return all matches for a given name by returning a vector<vector<uint32_t>>
    // so we catch all names that match a given child name, not just the first.
    size_t
    GetIndexOfChildMemberWithName (void* type,
                                   const char *name,
                                   bool omit_empty_base_classes,
                                   std::vector<uint32_t>& child_indexes);
    
    static size_t
    GetNumTemplateArguments (const ClangASTType& type);
    
    static ClangASTType
    GetTemplateArgument (const ClangASTType& type,
                         size_t idx,
                         lldb::TemplateArgumentKind &kind);
    
    
    //----------------------------------------------------------------------
    // Modifying RecordType
    //----------------------------------------------------------------------
    static clang::FieldDecl *
    AddFieldToRecordType (const ClangASTType& type,
                          const char *name,
                          const ClangASTType &field_type,
                          lldb::AccessType access,
                          uint32_t bitfield_bit_size);
    
    static void
    BuildIndirectFields (const ClangASTType& type);
    
    static void
    SetIsPacked (const ClangASTType& type);
    
    static clang::VarDecl *
    AddVariableToRecordType (const ClangASTType& type,
                             const char *name,
                             const ClangASTType &var_type,
                             lldb::AccessType access);
    
    clang::CXXMethodDecl *
    AddMethodToCXXRecordType (void* type,
                              const char *name,
                              const ClangASTType &method_type,
                              lldb::AccessType access,
                              bool is_virtual,
                              bool is_static,
                              bool is_inline,
                              bool is_explicit,
                              bool is_attr_used,
                              bool is_artificial);
    
    // C++ Base Classes
    clang::CXXBaseSpecifier *
    CreateBaseClassSpecifier (void* type,
                              lldb::AccessType access,
                              bool is_virtual,
                              bool base_of_class);
    
    static void
    DeleteBaseClassSpecifiers (clang::CXXBaseSpecifier **base_classes,
                               unsigned num_base_classes);
    
    bool
    SetBaseClassesForClassType (void* type,
                                clang::CXXBaseSpecifier const * const *base_classes,
                                unsigned num_base_classes);
    
    
    static bool
    SetObjCSuperClass (const ClangASTType& type,
                       const ClangASTType &superclass_clang_type);
    
    static bool
    AddObjCClassProperty (const ClangASTType& type,
                          const char *property_name,
                          const ClangASTType &property_clang_type,
                          clang::ObjCIvarDecl *ivar_decl,
                          const char *property_setter_name,
                          const char *property_getter_name,
                          uint32_t property_attributes,
                          ClangASTMetadata *metadata);
    
    static clang::ObjCMethodDecl *
    AddMethodToObjCObjectType (const ClangASTType& type,
                               const char *name,  // the full symbol name as seen in the symbol table (void* type, "-[NString stringWithCString:]")
                               const ClangASTType &method_clang_type,
                               lldb::AccessType access,
                               bool is_artificial);
    
    bool
    SetHasExternalStorage (void* type, bool has_extern);
    
    
    //------------------------------------------------------------------
    // Tag Declarations
    //------------------------------------------------------------------
    static bool
    StartTagDeclarationDefinition (const ClangASTType &type);
    
    static bool
    CompleteTagDeclarationDefinition (const ClangASTType &type);
    
    //----------------------------------------------------------------------
    // Modifying Enumeration types
    //----------------------------------------------------------------------
    bool
    AddEnumerationValueToEnumerationType (void* type,
                                          const ClangASTType &enumerator_qual_type,
                                          const Declaration &decl,
                                          const char *name,
                                          int64_t enum_value,
                                          uint32_t enum_value_bit_size);
    
    
    
    ClangASTType
    GetEnumerationIntegerType (void* type);
    
    
    //------------------------------------------------------------------
    // Pointers & References
    //------------------------------------------------------------------
    
    // Call this function using the class type when you want to make a
    // member pointer type to pointee_type.
    static ClangASTType
    CreateMemberPointerType (const ClangASTType& type, const ClangASTType &pointee_type);
    
    
    // Converts "s" to a floating point value and place resulting floating
    // point bytes in the "dst" buffer.
    size_t
    ConvertStringToFloatValue (void* type,
                               const char *s,
                               uint8_t *dst,
                               size_t dst_size);
    //----------------------------------------------------------------------
    // Dumping types
    //----------------------------------------------------------------------
    void
    DumpValue (void* type,
               ExecutionContext *exe_ctx,
               Stream *s,
               lldb::Format format,
               const DataExtractor &data,
               lldb::offset_t data_offset,
               size_t data_byte_size,
               uint32_t bitfield_bit_size,
               uint32_t bitfield_bit_offset,
               bool show_types,
               bool show_summary,
               bool verbose,
               uint32_t depth);
    
    bool
    DumpTypeValue (void* type,
                   Stream *s,
                   lldb::Format format,
                   const DataExtractor &data,
                   lldb::offset_t data_offset,
                   size_t data_byte_size,
                   uint32_t bitfield_bit_size,
                   uint32_t bitfield_bit_offset,
                   ExecutionContextScope *exe_scope);
    
    void
    DumpSummary (void* type,
                 ExecutionContext *exe_ctx,
                 Stream *s,
                 const DataExtractor &data,
                 lldb::offset_t data_offset,
                 size_t data_byte_size);
    
    virtual void
    DumpTypeDescription (void* type); // Dump to stdout
    
    void
    DumpTypeDescription (void* type, Stream *s);
    
    static clang::EnumDecl *
    GetAsEnumDecl (const ClangASTType& type);
    
    
    static clang::RecordDecl *
    GetAsRecordDecl (const ClangASTType& type);
    
    clang::CXXRecordDecl *
    GetAsCXXRecordDecl (void* type);
    
    static clang::ObjCInterfaceDecl *
    GetAsObjCInterfaceDecl (const ClangASTType& type);
    
    static clang::QualType
    GetQualType (const ClangASTType& type)
    {
        if (type && type.GetTypeSystem()->AsClangASTContext())
            return clang::QualType::getFromOpaquePtr(type.GetOpaqueQualType());
        return clang::QualType();
    }
    static clang::QualType
    GetCanonicalQualType (const ClangASTType& type)
    {
        if (type && type.GetTypeSystem()->AsClangASTContext())
            return clang::QualType::getFromOpaquePtr(type.GetOpaqueQualType()).getCanonicalType();
        return clang::QualType();
    }

protected:
    static clang::QualType
    GetQualType (void *type)
    {
        if (type)
            return clang::QualType::getFromOpaquePtr(type);
        return clang::QualType();
    }
    
    static clang::QualType
    GetCanonicalQualType (void *type)
    {
        if (type)
            return clang::QualType::getFromOpaquePtr(type).getCanonicalType();
        return clang::QualType();
    }

    
    //------------------------------------------------------------------
    // Classes that inherit from ClangASTContext can see and modify these
    //------------------------------------------------------------------
    std::string                                     m_target_triple;
    std::unique_ptr<clang::ASTContext>              m_ast_ap;
    std::unique_ptr<clang::LangOptions>             m_language_options_ap;
    std::unique_ptr<clang::FileManager>             m_file_manager_ap;
    std::unique_ptr<clang::FileSystemOptions>       m_file_system_options_ap;
    std::unique_ptr<clang::SourceManager>           m_source_manager_ap;
    std::unique_ptr<clang::DiagnosticsEngine>       m_diagnostics_engine_ap;
    std::unique_ptr<clang::DiagnosticConsumer>      m_diagnostic_consumer_ap;
    std::shared_ptr<clang::TargetOptions>           m_target_options_rp;
    std::unique_ptr<clang::TargetInfo>              m_target_info_ap;
    std::unique_ptr<clang::IdentifierTable>         m_identifier_table_ap;
    std::unique_ptr<clang::SelectorTable>           m_selector_table_ap;
    std::unique_ptr<clang::Builtin::Context>        m_builtins_ap;
    CompleteTagDeclCallback                         m_callback_tag_decl;
    CompleteObjCInterfaceDeclCallback               m_callback_objc_decl;
    void *                                          m_callback_baton;
    uint32_t                                        m_pointer_byte_size;
private:
    //------------------------------------------------------------------
    // For ClangASTContext only
    //------------------------------------------------------------------
    ClangASTContext(const ClangASTContext&);
    const ClangASTContext& operator=(const ClangASTContext&);
};

} // namespace lldb_private

#endif  // liblldb_ClangASTContext_h_
