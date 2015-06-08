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

namespace lldb_private {

class Declaration;

class ClangASTContext
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

    static uint32_t
    GetIndexForRecordBase (const clang::RecordDecl *record_decl,
                           const clang::CXXBaseSpecifier *base_spec,
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
protected:
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
