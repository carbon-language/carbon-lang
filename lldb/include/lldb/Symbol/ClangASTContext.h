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
// Project includes
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Symbol/ClangASTType.h"

namespace lldb_private {

class Declaration;

class ClangASTContext
{
public:
    // Define access values that can be used for all functions in this
    // class since Clang uses different values for all of the different
    // access values (C++ AccessSpecifier enums differ from ObjC AccessControl).
    // The SymbolFile classes that use these methods to created types
    // will then be able to use one enumeration for all access and we can
    // translate them correctly into the correct Clang versions depending on
    // what the access is applied to.
    enum AccessType
    {
        eAccessNone,
        eAccessPublic,
        eAccessPrivate,
        eAccessProtected,
        eAccessPackage
    };
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ClangASTContext(const char *target_triple);

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

    clang::SourceManager *
    getSourceManager();

    clang::Diagnostic *
    getDiagnostic();

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

    //------------------------------------------------------------------
    // Basic Types
    //------------------------------------------------------------------

    void *
    GetBuiltinTypeForEncodingAndBitSize (lldb::Encoding encoding,
                                          uint32_t bit_size);
    
    static void *
    GetBuiltinTypeForEncodingAndBitSize (clang::ASTContext *ast_context,
                                         lldb::Encoding encoding,
                                         uint32_t bit_size);

    void *
    GetBuiltinTypeForDWARFEncodingAndBitSize (
        const char *type_name,
        uint32_t dw_ate,
        uint32_t bit_size);

    void *
    GetBuiltInType_void();

    void *
    GetBuiltInType_objc_id();

    void *
    GetBuiltInType_objc_Class();

    void *
    GetBuiltInType_objc_selector();

    void *
    GetCStringType(bool is_const);

    void *
    GetVoidPtrType(bool is_const);
    
    static void *
    GetVoidPtrType(clang::ASTContext *ast_context, bool is_const);
    
    static void *
    CopyType(clang::ASTContext *dest_context, 
             clang::ASTContext *source_context,
             void *clang_type);
    
    static bool
    AreTypesSame(clang::ASTContext *ast_context,
                 void *type1,
                 void *type2);
    
    bool
    AreTypesSame(void *type1,
                 void *type2)
    {
        return ClangASTContext::AreTypesSame(m_ast_context_ap.get(), type1, type2);
    }
    
    //------------------------------------------------------------------
    // CVR modifiers
    //------------------------------------------------------------------

    static void *
    AddConstModifier (void *clang_type);

    static void *
    AddRestrictModifier (void *clang_type);

    static void *
    AddVolatileModifier (void *clang_type);

    //------------------------------------------------------------------
    // Structure, Unions, Classes
    //------------------------------------------------------------------

    void *
    CreateRecordType (const char *name,
                      int kind,
                      clang::DeclContext *decl_ctx,
                      lldb::LanguageType language);

    bool
    AddFieldToRecordType (void * record_qual_type,
                          const char *name,
                          void * field_type,
                          AccessType access,
                          uint32_t bitfield_bit_size);
    
    bool
    FieldIsBitfield (clang::FieldDecl* field,
                     uint32_t& bitfield_bit_size);

    static bool
    FieldIsBitfield (clang::ASTContext *ast_context,
                     clang::FieldDecl* field,
                     uint32_t& bitfield_bit_size);

    static bool
    RecordHasFields (const clang::RecordDecl *record_decl);

    void
    SetDefaultAccessForRecordFields (void *clang_type,
                                     int default_accessibility,
                                     int *assigned_accessibilities,
                                     size_t num_assigned_accessibilities);

    void *
    CreateObjCClass (const char *name, 
                     clang::DeclContext *decl_ctx, 
                     bool isForwardDecl, 
                     bool isInternal);
    
    bool
    AddObjCClassIVar (void *class_opaque_type, 
                      const char *name, 
                      void *ivar_opaque_type, 
                      AccessType access, 
                      uint32_t bitfield_bit_size, 
                      bool isSynthesized);

    bool
    SetObjCSuperClass (void *class_clang_type,
                       void *superclass_clang_type);

    static bool
    ObjCTypeHasIVars (void *class_clang_type, bool check_superclass);

    static bool
    ObjCDeclHasIVars (clang::ObjCInterfaceDecl *class_interface_decl, 
                      bool check_superclass);

    
    //------------------------------------------------------------------
    // Aggregate Types
    //------------------------------------------------------------------
    static bool
    IsAggregateType (void *clang_type);

    static uint32_t
    GetNumChildren (void *clang_type,
                    bool omit_empty_base_classes);

    void *
    GetChildClangTypeAtIndex (const char *parent_name,
                              void * parent_clang_type,
                              uint32_t idx,
                              bool transparent_pointers,
                              bool omit_empty_base_classes,
                              std::string& child_name,
                              uint32_t &child_byte_size,
                              int32_t &child_byte_offset,
                              uint32_t &child_bitfield_bit_size,
                              uint32_t &child_bitfield_bit_offset);
    
    static void *
    GetChildClangTypeAtIndex (clang::ASTContext *ast_context,
                              const char *parent_name,
                              void * parent_clang_type,
                              uint32_t idx,
                              bool transparent_pointers,
                              bool omit_empty_base_classes,
                              std::string& child_name,
                              uint32_t &child_byte_size,
                              int32_t &child_byte_offset,
                              uint32_t &child_bitfield_bit_size,
                              uint32_t &child_bitfield_bit_offset);
    
    // Lookup a child given a name. This function will match base class names
    // and member member names in "clang_type" only, not descendants.
    static uint32_t
    GetIndexOfChildWithName (clang::ASTContext *ast_context,
                             void *clang_type,
                             const char *name,
                             bool omit_empty_base_classes);

    // Lookup a child member given a name. This function will match member names
    // only and will descend into "clang_type" children in search for the first
    // member in this class, or any base class that matches "name".
    // TODO: Return all matches for a given name by returning a vector<vector<uint32_t>>
    // so we catch all names that match a given child name, not just the first.
    static size_t
    GetIndexOfChildMemberWithName (clang::ASTContext *ast_context,
                                   void *clang_type,
                                   const char *name,
                                   bool omit_empty_base_classes,
                                   std::vector<uint32_t>& child_indexes);

    //------------------------------------------------------------------
    // clang::TagType
    //------------------------------------------------------------------

    bool
    SetTagTypeKind (void * tag_qual_type,
                    int kind);

    //------------------------------------------------------------------
    // C++ Base Classes
    //------------------------------------------------------------------

    clang::CXXBaseSpecifier *
    CreateBaseClassSpecifier (void * base_class_type,
                              AccessType access,
                              bool is_virtual,
                              bool base_of_class);
    
    static void
    DeleteBaseClassSpecifiers (clang::CXXBaseSpecifier **base_classes, 
                               unsigned num_base_classes);

    bool
    SetBaseClassesForClassType (void * class_clang_type,
                                clang::CXXBaseSpecifier const * const *base_classes,
                                unsigned num_base_classes);

    //------------------------------------------------------------------
    // DeclContext Functions
    //------------------------------------------------------------------

    static clang::DeclContext *
    GetDeclContextForType (void * qual_type);

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
                               void * function_Type,
                               int storage,
                               bool is_inline);
    
    void *
    CreateFunctionType (void * result_type,
                        void **args,
                        unsigned num_args,
                        bool isVariadic,
                        unsigned TypeQuals);
    
    clang::ParmVarDecl *
    CreateParmeterDeclaration (const char *name,
                               void * return_type,
                               int storage);

    void
    SetFunctionParameters (clang::FunctionDecl *function_decl,
                           clang::ParmVarDecl **params,
                           unsigned num_params);

    //------------------------------------------------------------------
    // Array Types
    //------------------------------------------------------------------

    void *
    CreateArrayType (void * element_type,
                     size_t element_count,
                     uint32_t bit_stride);

    //------------------------------------------------------------------
    // Tag Declarations
    //------------------------------------------------------------------
    bool
    StartTagDeclarationDefinition (void * qual_type);

    bool
    CompleteTagDeclarationDefinition (void * qual_type);

    //------------------------------------------------------------------
    // Enumeration Types
    //------------------------------------------------------------------
    void *
    CreateEnumerationType (const Declaration &decl, const char *name);

    bool
    AddEnumerationValueToEnumerationType (void * enum_qual_type,
                                          void * enumerator_qual_type,
                                          const Declaration &decl,
                                          const char *name,
                                          int64_t enum_value,
                                          uint32_t enum_value_bit_size);
    
    //------------------------------------------------------------------
    // Pointers & References
    //------------------------------------------------------------------
    void *
    CreatePointerType (void *clang_type);

    void *
    CreateLValueReferenceType (void *clang_type);

    void *
    CreateRValueReferenceType (void *clang_type);

    void *
    CreateMemberPointerType (void * clang_pointee_type,
                             void * clang_class_type);

    size_t
    GetPointerBitSize ();

    static bool
    IsIntegerType (void *clang_type, bool &is_signed);
    
    static bool
    IsPointerType (void *clang_type, void **target_type = NULL);

    static bool
    IsPointerOrReferenceType (void *clang_type, void **target_type = NULL);

    static bool
    IsCStringType (void *clang_type, uint32_t &length);
    
    static bool
    IsArrayType (void *clang_type, void **member_type = NULL, uint64_t *size = NULL);

    //------------------------------------------------------------------
    // Typedefs
    //------------------------------------------------------------------
    void *
    CreateTypedefType (const char *name,
                       void *clang_type,
                       clang::DeclContext *decl_ctx);

    //------------------------------------------------------------------
    // Type names
    //------------------------------------------------------------------
    static std::string
    GetTypeName(void *clang_type);
    
    static bool
    IsFloatingPointType (void *clang_type, uint32_t &count, bool &is_complex);

    //static bool
    //ConvertFloatValueToString (clang::ASTContext *ast_context, 
    //                           void *clang_type, 
    //                           const uint8_t* bytes, 
    //                           size_t byte_size, 
    //                           int apint_byte_order, 
    //                           std::string &float_str);
    
    static size_t
    ConvertStringToFloatValue (clang::ASTContext *ast_context, 
                               void *clang_type, 
                               const char *s, 
                               uint8_t *dst, 
                               size_t dst_size);
    
protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangASTContext can see and modify these
    //------------------------------------------------------------------
    std::string                             m_target_triple;
    std::auto_ptr<clang::ASTContext>        m_ast_context_ap;
    std::auto_ptr<clang::LangOptions>       m_language_options_ap;
    std::auto_ptr<clang::SourceManager>     m_source_manager_ap;
    std::auto_ptr<clang::Diagnostic>        m_diagnostic_ap;
    std::auto_ptr<clang::TargetOptions>     m_target_options_ap;
    std::auto_ptr<clang::TargetInfo>        m_target_info_ap;
    std::auto_ptr<clang::IdentifierTable>   m_identifier_table_ap;
    std::auto_ptr<clang::SelectorTable>     m_selector_table_ap;
    std::auto_ptr<clang::Builtin::Context>  m_builtins_ap;

private:
    //------------------------------------------------------------------
    // For ClangASTContext only
    //------------------------------------------------------------------
    ClangASTContext(const ClangASTContext&);
    const ClangASTContext& operator=(const ClangASTContext&);
};

} // namespace lldb_private

#endif  // liblldb_ClangASTContext_h_
