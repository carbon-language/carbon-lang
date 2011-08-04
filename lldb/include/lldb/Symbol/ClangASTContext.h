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
// C++ Includes
#include <string>
#include <vector>
#include <memory>
#include <stdint.h>

// Other libraries and framework includes
#include "llvm/ADT/OwningPtr.h"

// Project includes
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/ClangASTType.h"

namespace lldb_private {

class Declaration;

class ClangASTContext
{
public:
    enum {
        eTypeHasChildren        = (1u <<  0),
        eTypeHasValue           = (1u <<  1),
        eTypeIsArray            = (1u <<  2),
        eTypeIsBlock            = (1u <<  3),
        eTypeIsBuiltIn          = (1u <<  4),
        eTypeIsClass            = (1u <<  5),
        eTypeIsCPlusPlus        = (1u <<  6),
        eTypeIsEnumeration      = (1u <<  7),
        eTypeIsFuncPrototype    = (1u <<  8),
        eTypeIsMember           = (1u <<  9),
        eTypeIsObjC             = (1u << 10),
        eTypeIsPointer          = (1u << 11),
        eTypeIsReference        = (1u << 12),
        eTypeIsStructUnion      = (1u << 13),
        eTypeIsTemplate         = (1u << 14),
        eTypeIsTypedef          = (1u << 15),
        eTypeIsVector           = (1u << 16),
        eTypeIsScalar           = (1u << 17)
    };

    typedef void (*CompleteTagDeclCallback)(void *baton, clang::TagDecl *);
    typedef void (*CompleteObjCInterfaceDeclCallback)(void *baton, clang::ObjCInterfaceDecl *);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ClangASTContext (const char *triple = NULL);

    ~ClangASTContext();

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

    clang::Diagnostic *
    getDiagnostic();
    
    clang::DiagnosticClient *
    getDiagnosticClient();

    clang::TargetOptions *
    getTargetOptions();

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
    SetExternalSource (llvm::OwningPtr<clang::ExternalASTSource> &ast_source_ap);

    void
    RemoveExternalSource ();

    bool
    GetCompleteType (lldb::clang_type_t clang_type);
    
    static bool
    GetCompleteType (clang::ASTContext *ast,
                     lldb::clang_type_t clang_type);

    bool
    GetCompleteDecl (clang::Decl *decl)
    {
        return ClangASTContext::GetCompleteDecl(getASTContext(), decl);
    }
    
    static bool
    GetCompleteDecl (clang::ASTContext *ast,
                     clang::Decl *decl);
    
    //------------------------------------------------------------------
    // Basic Types
    //------------------------------------------------------------------

    lldb::clang_type_t
    GetBuiltinTypeForEncodingAndBitSize (lldb::Encoding encoding,
                                          uint32_t bit_size);
    
    static lldb::clang_type_t
    GetBuiltinTypeForEncodingAndBitSize (clang::ASTContext *ast,
                                         lldb::Encoding encoding,
                                         uint32_t bit_size);

    lldb::clang_type_t
    GetBuiltinTypeForDWARFEncodingAndBitSize (
        const char *type_name,
        uint32_t dw_ate,
        uint32_t bit_size);

    static lldb::clang_type_t
    GetBuiltInType_void(clang::ASTContext *ast);
    
    lldb::clang_type_t
    GetBuiltInType_void()
    {
        return GetBuiltInType_void(getASTContext());
    }
    
    lldb::clang_type_t
    GetBuiltInType_bool();

    lldb::clang_type_t
    GetBuiltInType_objc_id();

    lldb::clang_type_t
    GetBuiltInType_objc_Class();
    
    static lldb::clang_type_t
    GetUnknownAnyType(clang::ASTContext *ast);
    
    lldb::clang_type_t
    GetUnknownAnyType()
    {
        return ClangASTContext::GetUnknownAnyType(getASTContext());
    }

    lldb::clang_type_t
    GetBuiltInType_objc_selector();

    lldb::clang_type_t
    GetCStringType(bool is_const);

    lldb::clang_type_t
    GetVoidPtrType(bool is_const);
    
    static lldb::clang_type_t
    GetVoidPtrType(clang::ASTContext *ast, bool is_const);
    
    static lldb::clang_type_t
    CopyType(clang::ASTContext *dest_context, 
             clang::ASTContext *source_context,
             lldb::clang_type_t clang_type);
    
    static clang::Decl *
    CopyDecl (clang::ASTContext *dest_context, 
              clang::ASTContext *source_context,
              clang::Decl *source_decl);

    static bool
    AreTypesSame(clang::ASTContext *ast,
                 lldb::clang_type_t type1,
                 lldb::clang_type_t type2);
    
    bool
    AreTypesSame(lldb::clang_type_t type1,
                 lldb::clang_type_t type2)
    {
        return ClangASTContext::AreTypesSame(getASTContext(), type1, type2);
    }
    
    
    lldb::clang_type_t
    GetTypeForDecl (clang::TagDecl *decl);
    
    lldb::clang_type_t
    GetTypeForDecl (clang::ObjCInterfaceDecl *objc_decl);

    //------------------------------------------------------------------
    // CVR modifiers
    //------------------------------------------------------------------

    static lldb::clang_type_t
    AddConstModifier (lldb::clang_type_t clang_type);

    static lldb::clang_type_t
    AddRestrictModifier (lldb::clang_type_t clang_type);

    static lldb::clang_type_t
    AddVolatileModifier (lldb::clang_type_t clang_type);

    //------------------------------------------------------------------
    // Structure, Unions, Classes
    //------------------------------------------------------------------

    lldb::clang_type_t
    CreateRecordType (const char *name,
                      int kind,
                      clang::DeclContext *decl_ctx,
                      lldb::LanguageType language);

    static bool
    AddFieldToRecordType (clang::ASTContext *ast,
                          lldb::clang_type_t record_qual_type,
                          const char *name,
                          lldb::clang_type_t field_type,
                          lldb::AccessType access,
                          uint32_t bitfield_bit_size);
    
    bool
    AddFieldToRecordType (lldb::clang_type_t record_qual_type,
                          const char *name,
                          lldb::clang_type_t field_type,
                          lldb::AccessType access,
                          uint32_t bitfield_bit_size)
    {
        return ClangASTContext::AddFieldToRecordType (getASTContext(),
                                                      record_qual_type,
                                                      name,
                                                      field_type,
                                                      access,
                                                      bitfield_bit_size);
    }
    
    static clang::CXXMethodDecl *
    AddMethodToCXXRecordType (clang::ASTContext *ast,
                              lldb::clang_type_t record_opaque_type,
                              const char *name,
                              lldb::clang_type_t method_type,
                              lldb::AccessType access,
                              bool is_virtual,
                              bool is_static,
                              bool is_inline,
                              bool is_explicit);
    
    clang::CXXMethodDecl *
    AddMethodToCXXRecordType (lldb::clang_type_t record_opaque_type,
                              const char *name,
                              lldb::clang_type_t method_type,
                              lldb::AccessType access,
                              bool is_virtual,
                              bool is_static,
                              bool is_inline,
                              bool is_explicit)
    
    {
        return ClangASTContext::AddMethodToCXXRecordType (getASTContext(),
                                                          record_opaque_type,
                                                          name,
                                                          method_type,
                                                          access,
                                                          is_virtual,
                                                          is_static,
                                                          is_inline,
                                                          is_explicit);
    }
    
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

    void
    SetDefaultAccessForRecordFields (lldb::clang_type_t clang_type,
                                     int default_accessibility,
                                     int *assigned_accessibilities,
                                     size_t num_assigned_accessibilities);

    lldb::clang_type_t
    CreateObjCClass (const char *name, 
                     clang::DeclContext *decl_ctx, 
                     bool isForwardDecl, 
                     bool isInternal);
    
    static bool
    AddObjCClassIVar (clang::ASTContext *ast,
                      lldb::clang_type_t class_opaque_type, 
                      const char *name, 
                      lldb::clang_type_t ivar_opaque_type, 
                      lldb::AccessType access, 
                      uint32_t bitfield_bit_size, 
                      bool isSynthesized);
    
    bool
    AddObjCClassIVar (lldb::clang_type_t class_opaque_type, 
                      const char *name, 
                      lldb::clang_type_t ivar_opaque_type, 
                      lldb::AccessType access, 
                      uint32_t bitfield_bit_size, 
                      bool isSynthesized)
    {
        return ClangASTContext::AddObjCClassIVar (getASTContext(),
                                                  class_opaque_type,
                                                  name,
                                                  ivar_opaque_type,
                                                  access,
                                                  bitfield_bit_size,
                                                  isSynthesized);
    }

    bool
    SetObjCSuperClass (lldb::clang_type_t class_clang_type,
                       lldb::clang_type_t superclass_clang_type);

    static bool
    ObjCTypeHasIVars (lldb::clang_type_t class_clang_type, bool check_superclass);

    static bool
    ObjCDeclHasIVars (clang::ObjCInterfaceDecl *class_interface_decl, 
                      bool check_superclass);


    static clang::ObjCMethodDecl *
    AddMethodToObjCObjectType (clang::ASTContext *ast,
                               lldb::clang_type_t class_opaque_type, 
                               const char *name,  // the full symbol name as seen in the symbol table ("-[NString stringWithCString:]")
                               lldb::clang_type_t method_opaque_type,
                               lldb::AccessType access);

    clang::ObjCMethodDecl *
    AddMethodToObjCObjectType (lldb::clang_type_t class_opaque_type, 
                               const char *name,  // the full symbol name as seen in the symbol table ("-[NString stringWithCString:]")
                               lldb::clang_type_t method_opaque_type,
                               lldb::AccessType access)
    {
        return AddMethodToObjCObjectType (getASTContext(),
                                          class_opaque_type,
                                          name,
                                          method_opaque_type,
                                          access);
    }

    static bool
    SetHasExternalStorage (lldb::clang_type_t clang_type, bool has_extern);

    //------------------------------------------------------------------
    // Aggregate Types
    //------------------------------------------------------------------
    static bool
    IsAggregateType (lldb::clang_type_t clang_type);

    // Returns a mask containing bits from the ClangASTContext::eTypeXXX enumerations
    static uint32_t
    GetTypeInfo (lldb::clang_type_t clang_type, 
                     clang::ASTContext *ast,                // The AST for clang_type (can be NULL)
                     lldb::clang_type_t *pointee_or_element_type);  // (can be NULL)

    static uint32_t
    GetNumChildren (clang::ASTContext *ast,
                    lldb::clang_type_t clang_type,
                    bool omit_empty_base_classes);

    static uint32_t
    GetNumPointeeChildren (lldb::clang_type_t clang_type);

    lldb::clang_type_t
    GetChildClangTypeAtIndex (ExecutionContext *exe_ctx,
                              const char *parent_name,
                              lldb::clang_type_t  parent_clang_type,
                              uint32_t idx,
                              bool transparent_pointers,
                              bool omit_empty_base_classes,
                              bool ignore_array_bounds,
                              std::string& child_name,
                              uint32_t &child_byte_size,
                              int32_t &child_byte_offset,
                              uint32_t &child_bitfield_bit_size,
                              uint32_t &child_bitfield_bit_offset,
                              bool &child_is_base_class,
                              bool &child_is_deref_of_parent);
 
    static lldb::clang_type_t
    GetChildClangTypeAtIndex (ExecutionContext *exe_ctx,
                              clang::ASTContext *ast,
                              const char *parent_name,
                              lldb::clang_type_t  parent_clang_type,
                              uint32_t idx,
                              bool transparent_pointers,
                              bool omit_empty_base_classes,
                              bool ignore_array_bounds,
                              std::string& child_name,
                              uint32_t &child_byte_size,
                              int32_t &child_byte_offset,
                              uint32_t &child_bitfield_bit_size,
                              uint32_t &child_bitfield_bit_offset,
                              bool &child_is_base_class,
                              bool &child_is_deref_of_parent);
    
    // Lookup a child given a name. This function will match base class names
    // and member member names in "clang_type" only, not descendants.
    static uint32_t
    GetIndexOfChildWithName (clang::ASTContext *ast,
                             lldb::clang_type_t clang_type,
                             const char *name,
                             bool omit_empty_base_classes);

    // Lookup a child member given a name. This function will match member names
    // only and will descend into "clang_type" children in search for the first
    // member in this class, or any base class that matches "name".
    // TODO: Return all matches for a given name by returning a vector<vector<uint32_t>>
    // so we catch all names that match a given child name, not just the first.
    static size_t
    GetIndexOfChildMemberWithName (clang::ASTContext *ast,
                                   lldb::clang_type_t clang_type,
                                   const char *name,
                                   bool omit_empty_base_classes,
                                   std::vector<uint32_t>& child_indexes);

    //------------------------------------------------------------------
    // clang::TagType
    //------------------------------------------------------------------

    bool
    SetTagTypeKind (lldb::clang_type_t  tag_qual_type,
                    int kind);

    //------------------------------------------------------------------
    // C++ Base Classes
    //------------------------------------------------------------------

    clang::CXXBaseSpecifier *
    CreateBaseClassSpecifier (lldb::clang_type_t  base_class_type,
                              lldb::AccessType access,
                              bool is_virtual,
                              bool base_of_class);
    
    static void
    DeleteBaseClassSpecifiers (clang::CXXBaseSpecifier **base_classes, 
                               unsigned num_base_classes);

    bool
    SetBaseClassesForClassType (lldb::clang_type_t  class_clang_type,
                                clang::CXXBaseSpecifier const * const *base_classes,
                                unsigned num_base_classes);

    //------------------------------------------------------------------
    // DeclContext Functions
    //------------------------------------------------------------------

    static clang::DeclContext *
    GetDeclContextForType (lldb::clang_type_t  qual_type);

    //------------------------------------------------------------------
    // Namespace Declarations
    //------------------------------------------------------------------

    clang::NamespaceDecl *
    GetUniqueNamespaceDeclaration (const char *name,
                                   const Declaration &decl,
                                   clang::DeclContext *decl_ctx);

    //------------------------------------------------------------------
    // Function Types
    //------------------------------------------------------------------

    clang::FunctionDecl *
    CreateFunctionDeclaration (const char *name,
                               lldb::clang_type_t  function_Type,
                               int storage,
                               bool is_inline);
    
    static lldb::clang_type_t
    CreateFunctionType (clang::ASTContext *ast,
                        lldb::clang_type_t result_type,
                        lldb::clang_type_t *args,
                        unsigned num_args,
                        bool is_variadic,
                        unsigned type_quals);
    
    lldb::clang_type_t
    CreateFunctionType (lldb::clang_type_t result_type,
                        lldb::clang_type_t *args,
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
                               lldb::clang_type_t param_type,
                               int storage);

    void
    SetFunctionParameters (clang::FunctionDecl *function_decl,
                           clang::ParmVarDecl **params,
                           unsigned num_params);

    //------------------------------------------------------------------
    // Array Types
    //------------------------------------------------------------------

    lldb::clang_type_t
    CreateArrayType (lldb::clang_type_t  element_type,
                     size_t element_count,
                     uint32_t bit_stride);

    //------------------------------------------------------------------
    // Tag Declarations
    //------------------------------------------------------------------
    bool
    StartTagDeclarationDefinition (lldb::clang_type_t  qual_type);

    bool
    CompleteTagDeclarationDefinition (lldb::clang_type_t  qual_type);

    //------------------------------------------------------------------
    // Enumeration Types
    //------------------------------------------------------------------
    lldb::clang_type_t
    CreateEnumerationType (const char *name, 
                           clang::DeclContext *decl_ctx, 
                           const Declaration &decl, 
                           lldb::clang_type_t integer_qual_type);

    static lldb::clang_type_t
    GetEnumerationIntegerType (lldb::clang_type_t enum_clang_type);
    
    bool
    AddEnumerationValueToEnumerationType (lldb::clang_type_t  enum_qual_type,
                                          lldb::clang_type_t  enumerator_qual_type,
                                          const Declaration &decl,
                                          const char *name,
                                          int64_t enum_value,
                                          uint32_t enum_value_bit_size);
    
    //------------------------------------------------------------------
    // Pointers & References
    //------------------------------------------------------------------
    lldb::clang_type_t
    CreatePointerType (lldb::clang_type_t clang_type);

    static lldb::clang_type_t
    CreatePointerType (clang::ASTContext *ast, 
                       lldb::clang_type_t clang_type);

    static lldb::clang_type_t
    CreateLValueReferenceType (clang::ASTContext *ast_context,
                               lldb::clang_type_t clang_type);
    
    static lldb::clang_type_t
    CreateRValueReferenceType (clang::ASTContext *ast_context,
                               lldb::clang_type_t clang_type);
    
    lldb::clang_type_t
    CreateLValueReferenceType (lldb::clang_type_t clang_type)
    {
        return ClangASTContext::CreateLValueReferenceType(getASTContext(), clang_type);
    }

    lldb::clang_type_t
    CreateRValueReferenceType (lldb::clang_type_t clang_type)
    {
        return ClangASTContext::CreateRValueReferenceType(getASTContext(), clang_type);
    }

    lldb::clang_type_t
    CreateMemberPointerType (lldb::clang_type_t  clang_pointee_type,
                             lldb::clang_type_t  clang_class_type);

    uint32_t
    GetPointerBitSize ();

    static bool
    IsIntegerType (lldb::clang_type_t clang_type, bool &is_signed);
    
    static bool
    IsPointerType (lldb::clang_type_t clang_type, lldb::clang_type_t *target_type = NULL);

    static bool
    IsPointerOrReferenceType (lldb::clang_type_t clang_type, lldb::clang_type_t *target_type = NULL);
    
    static bool
    IsPossibleCPlusPlusDynamicType (clang::ASTContext *ast,
                                    lldb::clang_type_t clang_type, 
                                    lldb::clang_type_t *target_type = NULL);

    static bool
    IsPossibleDynamicType (clang::ASTContext *ast, 
                           lldb::clang_type_t clang_type, 
                           lldb::clang_type_t *dynamic_pointee_type = NULL);

    static bool
    IsCStringType (lldb::clang_type_t clang_type, uint32_t &length);

    static bool
    IsFunctionPointerType (lldb::clang_type_t clang_type);
    
    static bool
    IsArrayType (lldb::clang_type_t clang_type, lldb::clang_type_t *member_type = NULL, uint64_t *size = NULL);

    //------------------------------------------------------------------
    // Typedefs
    //------------------------------------------------------------------
    lldb::clang_type_t
    CreateTypedefType (const char *name,
                       lldb::clang_type_t clang_type,
                       clang::DeclContext *decl_ctx);

    //------------------------------------------------------------------
    // Type names
    //------------------------------------------------------------------
    static bool
    IsFloatingPointType (lldb::clang_type_t clang_type, uint32_t &count, bool &is_complex);
    
    // true iff this is one of the types that can "fit"
    // in a Scalar object
    static bool
    IsScalarType (lldb::clang_type_t clang_type);
    
    static bool
    IsPointerToScalarType (lldb::clang_type_t clang_type);

    static bool
    IsArrayOfScalarType (lldb::clang_type_t clang_type);
    
    static bool
    GetCXXClassName (lldb::clang_type_t clang_type, 
                     std::string &class_name);

    static bool
    IsCXXClassType (lldb::clang_type_t clang_type);
    
    static bool
    IsObjCClassType (lldb::clang_type_t clang_type);

    static bool
    IsCharType (lldb::clang_type_t clang_type);

    static size_t
    GetArraySize (lldb::clang_type_t clang_type);

    //static bool
    //ConvertFloatValueToString (clang::ASTContext *ast, 
    //                           lldb::clang_type_t clang_type, 
    //                           const uint8_t* bytes, 
    //                           size_t byte_size, 
    //                           int apint_byte_order, 
    //                           std::string &float_str);
    
    static size_t
    ConvertStringToFloatValue (clang::ASTContext *ast, 
                               lldb::clang_type_t clang_type, 
                               const char *s, 
                               uint8_t *dst, 
                               size_t dst_size);
    
    //------------------------------------------------------------------
    // Qualifiers
    //------------------------------------------------------------------
    static unsigned
    GetTypeQualifiers(lldb::clang_type_t clang_type);
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangASTContext can see and modify these
    //------------------------------------------------------------------
    std::string                             m_target_triple;
    std::auto_ptr<clang::ASTContext>        m_ast_ap;
    std::auto_ptr<clang::LangOptions>       m_language_options_ap;
    std::auto_ptr<clang::FileManager>       m_file_manager_ap;
    std::auto_ptr<clang::FileSystemOptions> m_file_system_options_ap;
    std::auto_ptr<clang::SourceManager>     m_source_manager_ap;
    std::auto_ptr<clang::Diagnostic>        m_diagnostic_ap;
    std::auto_ptr<clang::DiagnosticClient>  m_diagnostic_client_ap;
    std::auto_ptr<clang::TargetOptions>     m_target_options_ap;
    std::auto_ptr<clang::TargetInfo>        m_target_info_ap;
    std::auto_ptr<clang::IdentifierTable>   m_identifier_table_ap;
    std::auto_ptr<clang::SelectorTable>     m_selector_table_ap;
    std::auto_ptr<clang::Builtin::Context>  m_builtins_ap;
    CompleteTagDeclCallback                 m_callback_tag_decl;
    CompleteObjCInterfaceDeclCallback       m_callback_objc_decl;
    void *                                  m_callback_baton;
private:
    //------------------------------------------------------------------
    // For ClangASTContext only
    //------------------------------------------------------------------
    ClangASTContext(const ClangASTContext&);
    const ClangASTContext& operator=(const ClangASTContext&);
};

} // namespace lldb_private

#endif  // liblldb_ClangASTContext_h_
