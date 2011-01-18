//===-- ClangASTContext.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangASTContext.h"

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes

// Clang headers like to use NDEBUG inside of them to enable/disable debug 
// releated features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#ifndef NDEBUG
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/LangStandard.h"

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

#include "lldb/Core/dwarf.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Log.h"

#include <stdio.h>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace clang;


static bool
GetCompleteQualType (clang::ASTContext *ast, clang::QualType qual_type)
{
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Record:
    case clang::Type::Enum:
        {
            clang::TagType *tag_type = dyn_cast<clang::TagType>(qual_type.getTypePtr());
            if (tag_type)
            {
                clang::TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    if (tag_decl->getDefinition())
                        return true;

                    if (tag_decl->hasExternalLexicalStorage())
                    {
                        ExternalASTSource *external_ast_source = ast->getExternalSource();
                        if (external_ast_source)
                        {
                            external_ast_source->CompleteType(tag_decl);
                            return !tag_type->isIncompleteType();
                        }
                    }
                    return false;
                }
            }

        }
        break;

    case clang::Type::ObjCObject:
    case clang::Type::ObjCInterface:
        {
            clang::ObjCObjectType *objc_class_type = dyn_cast<clang::ObjCObjectType>(qual_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                // We currently can't complete objective C types through the newly added ASTContext
                // because it only supports TagDecl objects right now...
                bool is_forward_decl = class_interface_decl->isForwardDecl();
                if (is_forward_decl && class_interface_decl->hasExternalLexicalStorage())
                {
                    ExternalASTSource *external_ast_source = ast->getExternalSource();
                    if (external_ast_source)
                    {
                        external_ast_source->CompleteType (class_interface_decl);
                        is_forward_decl = class_interface_decl->isForwardDecl();
                    }
                }
                return is_forward_decl;
            }
        }
        break;

    case clang::Type::Typedef:
        return GetCompleteQualType (ast, cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType());

    default:
        break;
    }

    return true;
}


static AccessSpecifier
ConvertAccessTypeToAccessSpecifier (AccessType access)
{
    switch (access)
    {
    default:               break;
    case eAccessNone:      return AS_none;
    case eAccessPublic:    return AS_public;
    case eAccessPrivate:   return AS_private;
    case eAccessProtected: return AS_protected;
    }
    return AS_none;
}

static ObjCIvarDecl::AccessControl
ConvertAccessTypeToObjCIvarAccessControl (AccessType access)
{
    switch (access)
    {
    default:               break;
    case eAccessNone:      return ObjCIvarDecl::None;
    case eAccessPublic:    return ObjCIvarDecl::Public;
    case eAccessPrivate:   return ObjCIvarDecl::Private;
    case eAccessProtected: return ObjCIvarDecl::Protected;
    case eAccessPackage:   return ObjCIvarDecl::Package;
    }
    return ObjCIvarDecl::None;
}


static void
ParseLangArgs
(
    LangOptions &Opts,
    InputKind IK
)
{
    // FIXME: Cleanup per-file based stuff.

    // Set some properties which depend soley on the input kind; it would be nice
    // to move these to the language standard, and have the driver resolve the
    // input kind + language standard.
    if (IK == IK_Asm) {
        Opts.AsmPreprocessor = 1;
    } else if (IK == IK_ObjC ||
               IK == IK_ObjCXX ||
               IK == IK_PreprocessedObjC ||
               IK == IK_PreprocessedObjCXX) {
        Opts.ObjC1 = Opts.ObjC2 = 1;
    }

    LangStandard::Kind LangStd = LangStandard::lang_unspecified;

    if (LangStd == LangStandard::lang_unspecified) {
        // Based on the base language, pick one.
        switch (IK) {
            case IK_None:
            case IK_AST:
                assert (!"Invalid input kind!");
            case IK_OpenCL:
                LangStd = LangStandard::lang_opencl;
                break;
            case IK_Asm:
            case IK_C:
            case IK_PreprocessedC:
            case IK_ObjC:
            case IK_PreprocessedObjC:
                LangStd = LangStandard::lang_gnu99;
                break;
            case IK_CXX:
            case IK_PreprocessedCXX:
            case IK_ObjCXX:
            case IK_PreprocessedObjCXX:
                LangStd = LangStandard::lang_gnucxx98;
                break;
        }
    }

    const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
    Opts.BCPLComment = Std.hasBCPLComments();
    Opts.C99 = Std.isC99();
    Opts.CPlusPlus = Std.isCPlusPlus();
    Opts.CPlusPlus0x = Std.isCPlusPlus0x();
    Opts.Digraphs = Std.hasDigraphs();
    Opts.GNUMode = Std.isGNUMode();
    Opts.GNUInline = !Std.isC99();
    Opts.HexFloats = Std.hasHexFloats();
    Opts.ImplicitInt = Std.hasImplicitInt();

    // OpenCL has some additional defaults.
    if (LangStd == LangStandard::lang_opencl) {
        Opts.OpenCL = 1;
        Opts.AltiVec = 1;
        Opts.CXXOperatorNames = 1;
        Opts.LaxVectorConversions = 1;
    }

    // OpenCL and C++ both have bool, true, false keywords.
    Opts.Bool = Opts.OpenCL || Opts.CPlusPlus;

//    if (Opts.CPlusPlus)
//        Opts.CXXOperatorNames = !Args.hasArg(OPT_fno_operator_names);
//
//    if (Args.hasArg(OPT_fobjc_gc_only))
//        Opts.setGCMode(LangOptions::GCOnly);
//    else if (Args.hasArg(OPT_fobjc_gc))
//        Opts.setGCMode(LangOptions::HybridGC);
//
//    if (Args.hasArg(OPT_print_ivar_layout))
//        Opts.ObjCGCBitmapPrint = 1;
//
//    if (Args.hasArg(OPT_faltivec))
//        Opts.AltiVec = 1;
//
//    if (Args.hasArg(OPT_pthread))
//        Opts.POSIXThreads = 1;
//
//    llvm::StringRef Vis = getLastArgValue(Args, OPT_fvisibility,
//                                          "default");
//    if (Vis == "default")
        Opts.setVisibilityMode(DefaultVisibility);
//    else if (Vis == "hidden")
//        Opts.setVisibilityMode(LangOptions::Hidden);
//    else if (Vis == "protected")
//        Opts.setVisibilityMode(LangOptions::Protected);
//    else
//        Diags.Report(diag::err_drv_invalid_value)
//        << Args.getLastArg(OPT_fvisibility)->getAsString(Args) << Vis;

//    Opts.OverflowChecking = Args.hasArg(OPT_ftrapv);

    // Mimicing gcc's behavior, trigraphs are only enabled if -trigraphs
    // is specified, or -std is set to a conforming mode.
    Opts.Trigraphs = !Opts.GNUMode;
//    if (Args.hasArg(OPT_trigraphs))
//        Opts.Trigraphs = 1;
//
//    Opts.DollarIdents = Args.hasFlag(OPT_fdollars_in_identifiers,
//                                     OPT_fno_dollars_in_identifiers,
//                                     !Opts.AsmPreprocessor);
//    Opts.PascalStrings = Args.hasArg(OPT_fpascal_strings);
//    Opts.Microsoft = Args.hasArg(OPT_fms_extensions);
//    Opts.WritableStrings = Args.hasArg(OPT_fwritable_strings);
//    if (Args.hasArg(OPT_fno_lax_vector_conversions))
//        Opts.LaxVectorConversions = 0;
//    Opts.Exceptions = Args.hasArg(OPT_fexceptions);
//    Opts.RTTI = !Args.hasArg(OPT_fno_rtti);
//    Opts.Blocks = Args.hasArg(OPT_fblocks);
//    Opts.CharIsSigned = !Args.hasArg(OPT_fno_signed_char);
//    Opts.ShortWChar = Args.hasArg(OPT_fshort_wchar);
//    Opts.Freestanding = Args.hasArg(OPT_ffreestanding);
//    Opts.NoBuiltin = Args.hasArg(OPT_fno_builtin) || Opts.Freestanding;
//    Opts.AssumeSaneOperatorNew = !Args.hasArg(OPT_fno_assume_sane_operator_new);
//    Opts.HeinousExtensions = Args.hasArg(OPT_fheinous_gnu_extensions);
//    Opts.AccessControl = Args.hasArg(OPT_faccess_control);
//    Opts.ElideConstructors = !Args.hasArg(OPT_fno_elide_constructors);
//    Opts.MathErrno = !Args.hasArg(OPT_fno_math_errno);
//    Opts.InstantiationDepth = getLastArgIntValue(Args, OPT_ftemplate_depth, 99,
//                                                 Diags);
//    Opts.NeXTRuntime = !Args.hasArg(OPT_fgnu_runtime);
//    Opts.ObjCConstantStringClass = getLastArgValue(Args,
//                                                   OPT_fconstant_string_class);
//    Opts.ObjCNonFragileABI = Args.hasArg(OPT_fobjc_nonfragile_abi);
//    Opts.CatchUndefined = Args.hasArg(OPT_fcatch_undefined_behavior);
//    Opts.EmitAllDecls = Args.hasArg(OPT_femit_all_decls);
//    Opts.PICLevel = getLastArgIntValue(Args, OPT_pic_level, 0, Diags);
//    Opts.Static = Args.hasArg(OPT_static_define);
    Opts.OptimizeSize = 0;

    // FIXME: Eliminate this dependency.
//    unsigned Opt =
//    Args.hasArg(OPT_Os) ? 2 : getLastArgIntValue(Args, OPT_O, 0, Diags);
//    Opts.Optimize = Opt != 0;
    unsigned Opt = 0;

    // This is the __NO_INLINE__ define, which just depends on things like the
    // optimization level and -fno-inline, not actually whether the backend has
    // inlining enabled.
    //
    // FIXME: This is affected by other options (-fno-inline).
    Opts.NoInline = !Opt;

//    unsigned SSP = getLastArgIntValue(Args, OPT_stack_protector, 0, Diags);
//    switch (SSP) {
//        default:
//            Diags.Report(diag::err_drv_invalid_value)
//            << Args.getLastArg(OPT_stack_protector)->getAsString(Args) << SSP;
//            break;
//        case 0: Opts.setStackProtectorMode(LangOptions::SSPOff); break;
//        case 1: Opts.setStackProtectorMode(LangOptions::SSPOn);  break;
//        case 2: Opts.setStackProtectorMode(LangOptions::SSPReq); break;
//    }
}


ClangASTContext::ClangASTContext (const char *target_triple) :   
    m_target_triple(),
    m_ast_ap(),
    m_language_options_ap(),
    m_source_manager_ap(),
    m_diagnostic_ap(),
    m_target_options_ap(),
    m_target_info_ap(),
    m_identifier_table_ap(),
    m_selector_table_ap(),
    m_builtins_ap(),
    m_callback_tag_decl (NULL),
    m_callback_objc_decl (NULL),
    m_callback_baton (NULL)

{
    if (target_triple && target_triple[0])
        m_target_triple.assign (target_triple);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ClangASTContext::~ClangASTContext()
{
    m_builtins_ap.reset();
    m_selector_table_ap.reset();
    m_identifier_table_ap.reset();
    m_target_info_ap.reset();
    m_target_options_ap.reset();
    m_diagnostic_ap.reset();
    m_source_manager_ap.reset();
    m_language_options_ap.reset();
    m_ast_ap.reset();
}


void
ClangASTContext::Clear()
{
    m_ast_ap.reset();
    m_language_options_ap.reset();
    m_source_manager_ap.reset();
    m_diagnostic_ap.reset();
    m_target_options_ap.reset();
    m_target_info_ap.reset();
    m_identifier_table_ap.reset();
    m_selector_table_ap.reset();
    m_builtins_ap.reset();
}

const char *
ClangASTContext::GetTargetTriple ()
{
    return m_target_triple.c_str();
}

void
ClangASTContext::SetTargetTriple (const char *target_triple)
{
    Clear();
    m_target_triple.assign(target_triple);
}

bool
ClangASTContext::HasExternalSource ()
{
    ASTContext *ast = getASTContext();
    if (ast)
        return ast->getExternalSource () != NULL;
    return false;
}

void
ClangASTContext::SetExternalSource (llvm::OwningPtr<ExternalASTSource> &ast_source_ap)
{
    ASTContext *ast = getASTContext();
    if (ast)
    {
        ast->setExternalSource (ast_source_ap);
        ast->getTranslationUnitDecl()->setHasExternalLexicalStorage(true);
        //ast->getTranslationUnitDecl()->setHasExternalVisibleStorage(true);
    }
}

void
ClangASTContext::RemoveExternalSource ()
{
    ASTContext *ast = getASTContext();
    
    if (ast)
    {
        llvm::OwningPtr<ExternalASTSource> empty_ast_source_ap;
        ast->setExternalSource (empty_ast_source_ap);
        ast->getTranslationUnitDecl()->setHasExternalLexicalStorage(false);
        //ast->getTranslationUnitDecl()->setHasExternalVisibleStorage(false);
    }
}



ASTContext *
ClangASTContext::getASTContext()
{
    if (m_ast_ap.get() == NULL)
    {
        m_ast_ap.reset(new ASTContext (*getLanguageOptions(),
                                       *getSourceManager(),
                                       *getTargetInfo(),
                                       *getIdentifierTable(),
                                       *getSelectorTable(),
                                       *getBuiltinContext(),
                                       0));
        
        if ((m_callback_tag_decl || m_callback_objc_decl) && m_callback_baton)
        {
            m_ast_ap->getTranslationUnitDecl()->setHasExternalLexicalStorage();
            //m_ast_ap->getTranslationUnitDecl()->setHasExternalVisibleStorage();
        }
        
        m_ast_ap->getDiagnostics().setClient(getDiagnosticClient(), false);
    }
    return m_ast_ap.get();
}

Builtin::Context *
ClangASTContext::getBuiltinContext()
{
    if (m_builtins_ap.get() == NULL)
        m_builtins_ap.reset (new Builtin::Context(*getTargetInfo()));
    return m_builtins_ap.get();
}

IdentifierTable *
ClangASTContext::getIdentifierTable()
{
    if (m_identifier_table_ap.get() == NULL)
        m_identifier_table_ap.reset(new IdentifierTable (*ClangASTContext::getLanguageOptions(), NULL));
    return m_identifier_table_ap.get();
}

LangOptions *
ClangASTContext::getLanguageOptions()
{
    if (m_language_options_ap.get() == NULL)
    {
        m_language_options_ap.reset(new LangOptions());
        ParseLangArgs(*m_language_options_ap, IK_ObjCXX);
//        InitializeLangOptions(*m_language_options_ap, IK_ObjCXX);
    }
    return m_language_options_ap.get();
}

SelectorTable *
ClangASTContext::getSelectorTable()
{
    if (m_selector_table_ap.get() == NULL)
        m_selector_table_ap.reset (new SelectorTable());
    return m_selector_table_ap.get();
}

clang::FileManager *
ClangASTContext::getFileManager()
{
    if (m_file_manager_ap.get() == NULL)
    {
        clang::FileSystemOptions file_system_options;
        m_file_manager_ap.reset(new clang::FileManager(file_system_options));
    }
    return m_file_manager_ap.get();
}

clang::SourceManager *
ClangASTContext::getSourceManager()
{
    if (m_source_manager_ap.get() == NULL)
        m_source_manager_ap.reset(new clang::SourceManager(*getDiagnostic(), *getFileManager()));
    return m_source_manager_ap.get();
}

Diagnostic *
ClangASTContext::getDiagnostic()
{
    if (m_diagnostic_ap.get() == NULL)
    {
        llvm::IntrusiveRefCntPtr<DiagnosticIDs> diag_id_sp(new DiagnosticIDs());
        m_diagnostic_ap.reset(new Diagnostic(diag_id_sp));
    }
    return m_diagnostic_ap.get();
}

class NullDiagnosticClient : public DiagnosticClient
{
public:
    NullDiagnosticClient ()
    {
        m_log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    }
    
    void HandleDiagnostic (Diagnostic::Level DiagLevel, const DiagnosticInfo &info)
    {
        if (m_log)
        {
            llvm::SmallVectorImpl<char> diag_str(10);
            info.FormatDiagnostic(diag_str);
            diag_str.push_back('\0');
            m_log->Printf("Compiler diagnostic: %s\n", diag_str.data());
        }
    }
private:
    LogSP m_log;
};

DiagnosticClient *
ClangASTContext::getDiagnosticClient()
{
    if (m_diagnostic_client_ap.get() == NULL)
        m_diagnostic_client_ap.reset(new NullDiagnosticClient);
    
    return m_diagnostic_client_ap.get();
}

TargetOptions *
ClangASTContext::getTargetOptions()
{
    if (m_target_options_ap.get() == NULL && !m_target_triple.empty())
    {
        m_target_options_ap.reset (new TargetOptions());
        if (m_target_options_ap.get())
            m_target_options_ap->Triple = m_target_triple;
    }
    return m_target_options_ap.get();
}


TargetInfo *
ClangASTContext::getTargetInfo()
{
    // target_triple should be something like "x86_64-apple-darwin10"
    if (m_target_info_ap.get() == NULL && !m_target_triple.empty())
        m_target_info_ap.reset (TargetInfo::CreateTargetInfo(*getDiagnostic(), *getTargetOptions()));
    return m_target_info_ap.get();
}

#pragma mark Basic Types

static inline bool
QualTypeMatchesBitSize(const uint64_t bit_size, ASTContext *ast, QualType qual_type)
{
    uint64_t qual_type_bit_size = ast->getTypeSize(qual_type);
    if (qual_type_bit_size == bit_size)
        return true;
    return false;
}

clang_type_t
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (Encoding encoding, uint32_t bit_size)
{
    ASTContext *ast = getASTContext();

    assert (ast != NULL);

    return GetBuiltinTypeForEncodingAndBitSize (ast, encoding, bit_size);
}

clang_type_t
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (ASTContext *ast, Encoding encoding, uint32_t bit_size)
{
    if (!ast)
        return NULL;
    
    switch (encoding)
    {
    case eEncodingInvalid:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
            return ast->VoidPtrTy.getAsOpaquePtr();
        break;
        
    case eEncodingUint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
            return ast->UnsignedCharTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
            return ast->UnsignedShortTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
            return ast->UnsignedIntTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
            return ast->UnsignedLongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
            return ast->UnsignedLongLongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
            return ast->UnsignedInt128Ty.getAsOpaquePtr();
        break;
        
    case eEncodingSint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
            return ast->CharTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
            return ast->ShortTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
            return ast->IntTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
            return ast->LongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
            return ast->LongLongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
            return ast->Int128Ty.getAsOpaquePtr();
        break;
        
    case eEncodingIEEE754:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
            return ast->FloatTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
            return ast->DoubleTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
            return ast->LongDoubleTy.getAsOpaquePtr();
        break;
        
    case eEncodingVector:
    default:
        break;
    }
    
    return NULL;
}

clang_type_t
ClangASTContext::GetBuiltinTypeForDWARFEncodingAndBitSize (const char *type_name, uint32_t dw_ate, uint32_t bit_size)
{
    ASTContext *ast = getASTContext();

    #define streq(a,b) strcmp(a,b) == 0
    assert (ast != NULL);
    if (ast)
    {
        switch (dw_ate)
        {
        default:
            break;

        case DW_ATE_address:
            if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
                return ast->VoidPtrTy.getAsOpaquePtr();
            break;

        case DW_ATE_boolean:
            if (QualTypeMatchesBitSize (bit_size, ast, ast->BoolTy))
                return ast->BoolTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                return ast->UnsignedCharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                return ast->UnsignedShortTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                return ast->UnsignedIntTy.getAsOpaquePtr();
            break;

        case DW_ATE_lo_user:
            // This has been seen to mean DW_AT_complex_integer
            if (strcmp(type_name, "complex") == 0)
            {
                clang_type_t complex_int_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("int", DW_ATE_signed, bit_size/2);
                return ast->getComplexType (QualType::getFromOpaquePtr(complex_int_clang_type)).getAsOpaquePtr();
            }
            break;
            
        case DW_ATE_complex_float:
            if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatComplexTy))
                return ast->FloatComplexTy.getAsOpaquePtr();
            else if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleComplexTy))
                return ast->DoubleComplexTy.getAsOpaquePtr();
            else if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleComplexTy))
                return ast->LongDoubleComplexTy.getAsOpaquePtr();
            else 
            {
                clang_type_t complex_float_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("float", DW_ATE_float, bit_size/2);
                return ast->getComplexType (QualType::getFromOpaquePtr(complex_float_clang_type)).getAsOpaquePtr();
            }
            break;

        case DW_ATE_float:
            if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
                return ast->FloatTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
                return ast->DoubleTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
                return ast->LongDoubleTy.getAsOpaquePtr();
            break;

        case DW_ATE_signed:
            if (type_name)
            {
                if (strstr(type_name, "long long"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                        return ast->LongLongTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "long"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                        return ast->LongTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "short"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                        return ast->ShortTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                        return ast->CharTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                        return ast->SignedCharTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "int"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                        return ast->IntTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                        return ast->Int128Ty.getAsOpaquePtr();
                }
                else if (streq(type_name, "wchar_t"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->WCharTy))
                        return ast->WCharTy.getAsOpaquePtr();
                }
            }
            // We weren't able to match up a type name, just search by size
            if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                return ast->CharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                return ast->ShortTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                return ast->IntTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                return ast->LongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                return ast->LongLongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                return ast->Int128Ty.getAsOpaquePtr();
            break;

        case DW_ATE_signed_char:
            if (type_name)
            {
                if (streq(type_name, "signed char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                        return ast->SignedCharTy.getAsOpaquePtr();
                }
            }
            if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                return ast->CharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                return ast->SignedCharTy.getAsOpaquePtr();
            break;

        case DW_ATE_unsigned:
            if (type_name)
            {
                if (strstr(type_name, "long long"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                        return ast->UnsignedLongLongTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "long"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                        return ast->UnsignedLongTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "short"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                        return ast->UnsignedShortTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                        return ast->UnsignedCharTy.getAsOpaquePtr();
                }
                else if (strstr(type_name, "int"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                        return ast->UnsignedIntTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                        return ast->UnsignedInt128Ty.getAsOpaquePtr();
                }
            }
            // We weren't able to match up a type name, just search by size
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                return ast->UnsignedCharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                return ast->UnsignedShortTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                return ast->UnsignedIntTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                return ast->UnsignedLongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                return ast->UnsignedLongLongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                return ast->UnsignedInt128Ty.getAsOpaquePtr();
            break;

        case DW_ATE_unsigned_char:
            if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                return ast->UnsignedCharTy.getAsOpaquePtr();
            break;

        case DW_ATE_imaginary_float:
            break;
        }
    }
    // This assert should fire for anything that we don't catch above so we know
    // to fix any issues we run into.
    assert (!"error: ClangASTContext::GetClangTypeForDWARFEncodingAndSize() contains an unhandled encoding. Fix this ASAP!");
    return NULL;
}

clang_type_t
ClangASTContext::GetBuiltInType_void(ASTContext *ast)
{
    return ast->VoidTy.getAsOpaquePtr();
}

clang_type_t
ClangASTContext::GetBuiltInType_bool()
{
    return getASTContext()->BoolTy.getAsOpaquePtr();
}

clang_type_t
ClangASTContext::GetBuiltInType_objc_id()
{
    return getASTContext()->getPointerType(getASTContext()->ObjCBuiltinIdTy).getAsOpaquePtr();
}

clang_type_t
ClangASTContext::GetBuiltInType_objc_Class()
{
    return getASTContext()->ObjCBuiltinClassTy.getAsOpaquePtr();
}

clang_type_t
ClangASTContext::GetBuiltInType_objc_selector()
{
    return getASTContext()->getPointerType(getASTContext()->ObjCBuiltinSelTy).getAsOpaquePtr();
}

clang_type_t
ClangASTContext::GetCStringType (bool is_const)
{
    QualType char_type(getASTContext()->CharTy);
    
    if (is_const)
        char_type.addConst();
    
    return getASTContext()->getPointerType(char_type).getAsOpaquePtr();
}

clang_type_t
ClangASTContext::GetVoidPtrType (bool is_const)
{
    return GetVoidPtrType(getASTContext(), is_const);
}

clang_type_t
ClangASTContext::GetVoidPtrType (ASTContext *ast, bool is_const)
{
    QualType void_ptr_type(ast->VoidPtrTy);
    
    if (is_const)
        void_ptr_type.addConst();
    
    return void_ptr_type.getAsOpaquePtr();
}

clang_type_t
ClangASTContext::CopyType (ASTContext *dst_ast, 
                           ASTContext *src_ast,
                           clang_type_t clang_type)
{
    FileSystemOptions file_system_options;
    FileManager file_manager (file_system_options);
    ASTImporter importer(*dst_ast, file_manager,
                         *src_ast, file_manager,
                         false);
    
    QualType src (QualType::getFromOpaquePtr(clang_type));
    QualType dst (importer.Import(src));
    
    return dst.getAsOpaquePtr();
}


clang::Decl *
ClangASTContext::CopyDecl (ASTContext *dst_ast, 
                           ASTContext *src_ast,
                           clang::Decl *source_decl)
{    
    FileSystemOptions file_system_options;
    FileManager file_manager (file_system_options);
    ASTImporter importer(*dst_ast, file_manager,
                         *src_ast, file_manager,
                         false);
    
    return importer.Import(source_decl);
}

bool
ClangASTContext::AreTypesSame(ASTContext *ast,
             clang_type_t type1,
             clang_type_t type2)
{
    return ast->hasSameType(QualType::getFromOpaquePtr(type1),
                                    QualType::getFromOpaquePtr(type2));
}

#pragma mark CVR modifiers

clang_type_t
ClangASTContext::AddConstModifier (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType result(QualType::getFromOpaquePtr(clang_type));
        result.addConst();
        return result.getAsOpaquePtr();
    }
    return NULL;
}

clang_type_t
ClangASTContext::AddRestrictModifier (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType result(QualType::getFromOpaquePtr(clang_type));
        result.getQualifiers().setRestrict (true);
        return result.getAsOpaquePtr();
    }
    return NULL;
}

clang_type_t
ClangASTContext::AddVolatileModifier (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType result(QualType::getFromOpaquePtr(clang_type));
        result.getQualifiers().setVolatile (true);
        return result.getAsOpaquePtr();
    }
    return NULL;
}


clang_type_t
ClangASTContext::GetTypeForDecl (TagDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    if (m_ast_ap.get())
        return m_ast_ap->getTagDeclType(decl).getAsOpaquePtr();
    return NULL;
}

clang_type_t
ClangASTContext::GetTypeForDecl (ObjCInterfaceDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    if (m_ast_ap.get())
        return m_ast_ap->getObjCInterfaceType(decl).getAsOpaquePtr();
    return NULL;
}

#pragma mark Structure, Unions, Classes

clang_type_t
ClangASTContext::CreateRecordType (const char *name, int kind, DeclContext *decl_ctx, LanguageType language)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);
    
    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();


    if (language == eLanguageTypeObjC)
    {
        bool isForwardDecl = true;
        bool isInternal = false;
        return CreateObjCClass (name, decl_ctx, isForwardDecl, isInternal);
    }

    // NOTE: Eventually CXXRecordDecl will be merged back into RecordDecl and
    // we will need to update this code. I was told to currently always use
    // the CXXRecordDecl class since we often don't know from debug information
    // if something is struct or a class, so we default to always use the more
    // complete definition just in case.
    CXXRecordDecl *decl = CXXRecordDecl::Create(*ast,
                                                (TagDecl::TagKind)kind,
                                                decl_ctx,
                                                SourceLocation(),
                                                name && name[0] ? &ast->Idents.get(name) : NULL);

    return ast->getTagDeclType(decl).getAsOpaquePtr();
}

bool
ClangASTContext::SetHasExternalStorage (clang_type_t clang_type, bool has_extern)
{
    if (clang_type == NULL)
        return false;

    QualType qual_type (QualType::getFromOpaquePtr(clang_type));

    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Record:
        {
            CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
            if (cxx_record_decl)
            {
                cxx_record_decl->setHasExternalLexicalStorage (has_extern);
                //cxx_record_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
        break;

    case clang::Type::Enum:
        {
            EnumDecl *enum_decl = cast<EnumType>(qual_type)->getDecl();
            if (enum_decl)
            {
                enum_decl->setHasExternalLexicalStorage (has_extern);
                //enum_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
        break;

    case clang::Type::ObjCObject:
    case clang::Type::ObjCInterface:
        {
            ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            
                if (class_interface_decl)
                {
                    class_interface_decl->setHasExternalLexicalStorage (has_extern);
                    //class_interface_decl->setHasExternalVisibleStorage (has_extern);
                    return true;
                }
            }
        }
        break;

    case clang::Type::Typedef:
        return ClangASTContext::SetHasExternalStorage (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), has_extern);

    default:
        break;
    }
    return false;
}

static bool
IsOperator (const char *name, OverloadedOperatorKind &op_kind)
{
    if (name == NULL || name[0] == '\0')
        return false;
    
#define OPERATOR_PREFIX "operator"
#define OPERATOR_PREFIX_LENGTH (sizeof (OPERATOR_PREFIX) - 1)
    
    const char *post_op_name = NULL;

    bool no_space = true;
    
    if (::strncmp(name, OPERATOR_PREFIX, OPERATOR_PREFIX_LENGTH))
        return false;
    
    post_op_name = name + OPERATOR_PREFIX_LENGTH;
    
    if (post_op_name[0] == ' ')
    {
        post_op_name++;
        no_space = false;
    }
    
#undef OPERATOR_PREFIX
#undef OPERATOR_PREFIX_LENGTH
    
    // This is an operator, set the overloaded operator kind to invalid
    // in case this is a conversion operator...
    op_kind = NUM_OVERLOADED_OPERATORS;

    switch (post_op_name[0])
    {
    default:
        if (no_space)
            return false;
        break;
    case 'n':
        if (no_space)
            return false;
        if  (strcmp (post_op_name, "new") == 0)  
            op_kind = OO_New;
        else if (strcmp (post_op_name, "new[]") == 0)  
            op_kind = OO_Array_New;
        break;

    case 'd':
        if (no_space)
            return false;
        if (strcmp (post_op_name, "delete") == 0)
            op_kind = OO_Delete;
        else if (strcmp (post_op_name, "delete[]") == 0)  
            op_kind = OO_Array_Delete;
        break;
    
    case '+':
        if (post_op_name[1] == '\0')
            op_kind = OO_Plus;
        else if (post_op_name[2] == '\0')
        {
            if (post_op_name[1] == '=')
                op_kind = OO_PlusEqual;
            else if (post_op_name[1] == '+')
                op_kind = OO_PlusPlus;
        }
        break;

    case '-':
        if (post_op_name[1] == '\0')
            op_kind = OO_Minus;
        else if (post_op_name[2] == '\0')
        {
            switch (post_op_name[1])
            {
            case '=': op_kind = OO_MinusEqual; break;
            case '-': op_kind = OO_MinusMinus; break;
            case '>': op_kind = OO_Arrow; break;
            }
        }
        else if (post_op_name[3] == '\0')
        {
            if (post_op_name[2] == '*')
                op_kind = OO_ArrowStar; break;
        }
        break;
        
    case '*':
        if (post_op_name[1] == '\0')
            op_kind = OO_Star;
        else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
            op_kind = OO_StarEqual;
        break;
    
    case '/':
        if (post_op_name[1] == '\0')
            op_kind = OO_Slash;
        else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
            op_kind = OO_SlashEqual;
        break;
    
    case '%':
        if (post_op_name[1] == '\0')
            op_kind = OO_Percent;
        else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
            op_kind = OO_PercentEqual;
        break;


    case '^':
        if (post_op_name[1] == '\0')
            op_kind = OO_Caret;
        else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
            op_kind = OO_CaretEqual;
        break;

    case '&':
        if (post_op_name[1] == '\0')
            op_kind = OO_Amp;
        else if (post_op_name[2] == '\0')
        {
            switch (post_op_name[1])
            {
            case '=': op_kind = OO_AmpEqual; break;
            case '&': op_kind = OO_AmpAmp; break;
            }   
        }
        break;

    case '|':
        if (post_op_name[1] == '\0')
            op_kind = OO_Pipe;
        else if (post_op_name[2] == '\0')
        {
            switch (post_op_name[1])
            {
            case '=': op_kind = OO_PipeEqual; break;
            case '|': op_kind = OO_PipePipe; break;
            }   
        }
        break;
    
    case '~':
        if (post_op_name[1] == '\0')
            op_kind = OO_Tilde;
        break;
    
    case '!':
        if (post_op_name[1] == '\0')
            op_kind = OO_Exclaim;
        else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
            op_kind = OO_ExclaimEqual;
        break;

    case '=':
        if (post_op_name[1] == '\0')
            op_kind = OO_Equal;
        else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
            op_kind = OO_EqualEqual;
        break;
    
    case '<':
        if (post_op_name[1] == '\0')
            op_kind = OO_Less;
        else if (post_op_name[2] == '\0')
        {
            switch (post_op_name[1])
            {
            case '<': op_kind = OO_LessLess; break;
            case '=': op_kind = OO_LessEqual; break;
            }   
        }
        else if (post_op_name[3] == '\0')
        {
            if (post_op_name[2] == '=')
                op_kind = OO_LessLessEqual;
        }
        break;

    case '>':
        if (post_op_name[1] == '\0')
            op_kind = OO_Greater;
        else if (post_op_name[2] == '\0')
        {
            switch (post_op_name[1])
            {
            case '>': op_kind = OO_GreaterGreater; break;
            case '=': op_kind = OO_GreaterEqual; break;
            }   
        }
        else if (post_op_name[1] == '>' && 
                 post_op_name[2] == '=' && 
                 post_op_name[3] == '\0')
        {
                op_kind = OO_GreaterGreaterEqual;
        }
        break;
        
    case ',':
        if (post_op_name[1] == '\0')
            op_kind = OO_Comma;
        break;
    
    case '(':
        if (post_op_name[1] == ')' && post_op_name[2] == '\0')
            op_kind = OO_Call;
        break;
    
    case '[':
        if (post_op_name[1] == ']' && post_op_name[2] == '\0')
            op_kind = OO_Subscript;
        break;
    }

    return true;
}

CXXMethodDecl *
ClangASTContext::AddMethodToCXXRecordType
(
    ASTContext *ast,
    clang_type_t record_opaque_type,
    const char *name,
    clang_type_t method_opaque_type,
    lldb::AccessType access,
    bool is_virtual,
    bool is_static,
    bool is_inline,
    bool is_explicit
)
{
    if (!record_opaque_type || !method_opaque_type || !name)
        return NULL;
    
    assert(ast);
    
    IdentifierTable *identifier_table = &ast->Idents;
    
    assert(identifier_table);
    
    QualType record_qual_type(QualType::getFromOpaquePtr(record_opaque_type));

    CXXRecordDecl *cxx_record_decl = record_qual_type->getAsCXXRecordDecl();
    
    if (cxx_record_decl == NULL)
        return NULL;
    
    QualType method_qual_type (QualType::getFromOpaquePtr (method_opaque_type));
    
    CXXMethodDecl *cxx_method_decl = NULL;
    
    DeclarationName decl_name (&identifier_table->get(name));

    const bool is_implicitly_declared = false;
    
    clang::FunctionType *function_Type = dyn_cast<FunctionType>(method_qual_type.getTypePtr());
    
    if (function_Type == NULL)
        return NULL;

    FunctionProtoType *method_function_prototype (dyn_cast<FunctionProtoType>(function_Type));
    
    if (!method_function_prototype)
        return NULL;
    
    unsigned int num_params = method_function_prototype->getNumArgs();
    
    if (name[0] == '~')
    {
        cxx_method_decl = CXXDestructorDecl::Create (*ast,
                                                     cxx_record_decl,
                                                     DeclarationNameInfo (ast->DeclarationNames.getCXXDestructorName (ast->getCanonicalType (record_qual_type)), SourceLocation()),
                                                     method_qual_type,
                                                     NULL,
                                                     is_inline,
                                                     is_implicitly_declared);
    }
    else if (decl_name == cxx_record_decl->getDeclName())
    {
        cxx_method_decl = CXXConstructorDecl::Create (*ast,
                                                      cxx_record_decl,
                                                      DeclarationNameInfo (ast->DeclarationNames.getCXXConstructorName (ast->getCanonicalType (record_qual_type)), SourceLocation()),
                                                      method_qual_type,
                                                      NULL, // TypeSourceInfo *
                                                      is_explicit, 
                                                      is_inline,
                                                      is_implicitly_declared);
    }
    else
    {   
    
        OverloadedOperatorKind op_kind = NUM_OVERLOADED_OPERATORS;
        if (IsOperator (name, op_kind))
        {
            if (op_kind != NUM_OVERLOADED_OPERATORS)
            {
                cxx_method_decl = CXXMethodDecl::Create (*ast,
                                                         cxx_record_decl,
                                                         DeclarationNameInfo (ast->DeclarationNames.getCXXOperatorName (op_kind), SourceLocation()),
                                                         method_qual_type,
                                                         NULL, // TypeSourceInfo *
                                                         is_static,
                                                         SC_None,
                                                         is_inline);
            }
            else if (num_params == 0)
            {
                // Conversion operators don't take params...
                cxx_method_decl = CXXConversionDecl::Create (*ast,
                                                             cxx_record_decl,
                                                             DeclarationNameInfo (ast->DeclarationNames.getCXXConversionFunctionName (ast->getCanonicalType (function_Type->getResultType())), SourceLocation()),
                                                             method_qual_type,
                                                             NULL, // TypeSourceInfo *
                                                             is_inline,
                                                             is_explicit);
            }
        }
        
        if (cxx_method_decl == NULL)
        {
            cxx_method_decl = CXXMethodDecl::Create (*ast,
                                                     cxx_record_decl,
                                                     DeclarationNameInfo (decl_name, SourceLocation()),
                                                     method_qual_type,
                                                     NULL, // TypeSourceInfo *
                                                     is_static,
                                                     SC_None,
                                                     is_inline);
        }
    }

    AccessSpecifier access_specifier = ConvertAccessTypeToAccessSpecifier (access);
    
    cxx_method_decl->setAccess (access_specifier);
    cxx_method_decl->setVirtualAsWritten (is_virtual);
    
    // Populate the method decl with parameter decls
    
    ParmVarDecl *params[num_params];
    
    for (int param_index = 0;
         param_index < num_params;
         ++param_index)
    {
        params[param_index] = ParmVarDecl::Create (*ast,
                                                   cxx_method_decl,
                                                   SourceLocation(),
                                                   NULL, // anonymous
                                                   method_function_prototype->getArgType(param_index), 
                                                   NULL,
                                                   SC_None, 
                                                   SC_None,
                                                   NULL); 
    }
    
    cxx_method_decl->setParams (params, num_params);
    
    cxx_record_decl->addDecl (cxx_method_decl);
    
    return cxx_method_decl;
}

bool
ClangASTContext::AddFieldToRecordType 
(
    ASTContext *ast,
    clang_type_t record_clang_type, 
    const char *name, 
    clang_type_t field_type, 
    AccessType access, 
    uint32_t bitfield_bit_size
)
{
    if (record_clang_type == NULL || field_type == NULL)
        return false;

    IdentifierTable *identifier_table = &ast->Idents;

    assert (ast != NULL);
    assert (identifier_table != NULL);

    QualType record_qual_type(QualType::getFromOpaquePtr(record_clang_type));

    clang::Type *clang_type = record_qual_type.getTypePtr();
    if (clang_type)
    {
        const RecordType *record_type = dyn_cast<RecordType>(clang_type);

        if (record_type)
        {
            RecordDecl *record_decl = record_type->getDecl();

            clang::Expr *bit_width = NULL;
            if (bitfield_bit_size != 0)
            {
                APInt bitfield_bit_size_apint(ast->getTypeSize(ast->IntTy), bitfield_bit_size);
                bit_width = new (*ast)IntegerLiteral (*ast, bitfield_bit_size_apint, ast->IntTy, SourceLocation());
            }
            FieldDecl *field = FieldDecl::Create (*ast,
                                                  record_decl,
                                                  SourceLocation(),
                                                  name ? &identifier_table->get(name) : NULL, // Identifier
                                                  QualType::getFromOpaquePtr(field_type), // Field type
                                                  NULL,       // DeclaratorInfo *
                                                  bit_width,  // BitWidth
                                                  false);     // Mutable

            field->setAccess (ConvertAccessTypeToAccessSpecifier (access));

            if (field)
            {
                record_decl->addDecl(field);
            }
        }
        else
        {
            ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(clang_type);
            if (objc_class_type)
            {
                bool is_synthesized = false;
                ClangASTContext::AddObjCClassIVar (ast,
                                                   record_clang_type,
                                                   name,
                                                   field_type,
                                                   access,
                                                   bitfield_bit_size,
                                                   is_synthesized);
            }
        }
    }
    return false;
}

bool
ClangASTContext::FieldIsBitfield (FieldDecl* field, uint32_t& bitfield_bit_size)
{
    return FieldIsBitfield(getASTContext(), field, bitfield_bit_size);
}

bool
ClangASTContext::FieldIsBitfield
(
    ASTContext *ast,
    FieldDecl* field,
    uint32_t& bitfield_bit_size
)
{
    if (ast == NULL || field == NULL)
        return false;

    if (field->isBitField())
    {
        Expr* bit_width_expr = field->getBitWidth();
        if (bit_width_expr)
        {
            llvm::APSInt bit_width_apsint;
            if (bit_width_expr->isIntegerConstantExpr(bit_width_apsint, *ast))
            {
                bitfield_bit_size = bit_width_apsint.getLimitedValue(UINT32_MAX);
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTContext::RecordHasFields (const RecordDecl *record_decl)
{
    if (record_decl == NULL)
        return false;

    if (!record_decl->field_empty())
        return true;

    // No fields, lets check this is a CXX record and check the base classes
    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
    if (cxx_record_decl)
    {
        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
             base_class != base_class_end;
             ++base_class)
        {
            const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
            if (RecordHasFields(base_class_decl))
                return true;
        }
    }
    return false;
}

void
ClangASTContext::SetDefaultAccessForRecordFields (clang_type_t clang_type, int default_accessibility, int *assigned_accessibilities, size_t num_assigned_accessibilities)
{
    if (clang_type)
    {
        QualType qual_type(QualType::getFromOpaquePtr(clang_type));

        RecordType *record_type = dyn_cast<RecordType>(qual_type.getTypePtr());
        if (record_type)
        {
            RecordDecl *record_decl = record_type->getDecl();
            if (record_decl)
            {
                uint32_t field_idx;
                RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(), field_idx = 0;
                     field != field_end;
                     ++field, ++field_idx)
                {
                    // If no accessibility was assigned, assign the correct one
                    if (field_idx < num_assigned_accessibilities && assigned_accessibilities[field_idx] == clang::AS_none)
                        field->setAccess ((AccessSpecifier)default_accessibility);
                }
            }
        }
    }
}

#pragma mark C++ Base Classes

CXXBaseSpecifier *
ClangASTContext::CreateBaseClassSpecifier (clang_type_t base_class_type, AccessType access, bool is_virtual, bool base_of_class)
{
    if (base_class_type)
        return new CXXBaseSpecifier (SourceRange(), 
                                     is_virtual, 
                                     base_of_class, 
                                     ConvertAccessTypeToAccessSpecifier (access), 
                                     getASTContext()->CreateTypeSourceInfo (QualType::getFromOpaquePtr(base_class_type)),
                                     SourceLocation());
    return NULL;
}

void
ClangASTContext::DeleteBaseClassSpecifiers (CXXBaseSpecifier **base_classes, unsigned num_base_classes)
{
    for (unsigned i=0; i<num_base_classes; ++i)
    {
        delete base_classes[i];
        base_classes[i] = NULL;
    }
}

bool
ClangASTContext::SetBaseClassesForClassType (clang_type_t class_clang_type, CXXBaseSpecifier const * const *base_classes, unsigned num_base_classes)
{
    if (class_clang_type)
    {
        CXXRecordDecl *cxx_record_decl = QualType::getFromOpaquePtr(class_clang_type)->getAsCXXRecordDecl();
        if (cxx_record_decl)
        {
            cxx_record_decl->setBases(base_classes, num_base_classes);
            return true;
        }
    }
    return false;
}
#pragma mark Objective C Classes

clang_type_t
ClangASTContext::CreateObjCClass 
(
    const char *name, 
    DeclContext *decl_ctx, 
    bool isForwardDecl, 
    bool isInternal
)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);
    assert (name && name[0]);
    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();

    // NOTE: Eventually CXXRecordDecl will be merged back into RecordDecl and
    // we will need to update this code. I was told to currently always use
    // the CXXRecordDecl class since we often don't know from debug information
    // if something is struct or a class, so we default to always use the more
    // complete definition just in case.
    ObjCInterfaceDecl *decl = ObjCInterfaceDecl::Create (*ast,
                                                         decl_ctx,
                                                         SourceLocation(),
                                                         &ast->Idents.get(name),
                                                         SourceLocation(),
                                                         isForwardDecl,
                                                         isInternal);
    
    return ast->getObjCInterfaceType(decl).getAsOpaquePtr();
}

bool
ClangASTContext::SetObjCSuperClass (clang_type_t class_opaque_type, clang_type_t super_opaque_type)
{
    if (class_opaque_type && super_opaque_type)
    {
        QualType class_qual_type(QualType::getFromOpaquePtr(class_opaque_type));
        QualType super_qual_type(QualType::getFromOpaquePtr(super_opaque_type));
        clang::Type *class_type = class_qual_type.getTypePtr();
        clang::Type *super_type = super_qual_type.getTypePtr();
        if (class_type && super_type)
        {
            ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(class_type);
            ObjCObjectType *objc_super_type = dyn_cast<ObjCObjectType>(super_type);
            if (objc_class_type && objc_super_type)
            {
                ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                ObjCInterfaceDecl *super_interface_decl = objc_super_type->getInterface();
                if (class_interface_decl && super_interface_decl)
                {
                    class_interface_decl->setSuperClass(super_interface_decl);
                    return true;
                }
            }
        }
    }
    return false;
}


bool
ClangASTContext::AddObjCClassIVar 
(
    ASTContext *ast,
    clang_type_t class_opaque_type, 
    const char *name, 
    clang_type_t ivar_opaque_type, 
    AccessType access, 
    uint32_t bitfield_bit_size, 
    bool is_synthesized
)
{
    if (class_opaque_type == NULL || ivar_opaque_type == NULL)
        return false;

    IdentifierTable *identifier_table = &ast->Idents;

    assert (ast != NULL);
    assert (identifier_table != NULL);

    QualType class_qual_type(QualType::getFromOpaquePtr(class_opaque_type));

    clang::Type *class_type = class_qual_type.getTypePtr();
    if (class_type)
    {
        ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(class_type);

        if (objc_class_type)
        {
            ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            
            if (class_interface_decl)
            {
                clang::Expr *bit_width = NULL;
                if (bitfield_bit_size != 0)
                {
                    APInt bitfield_bit_size_apint(ast->getTypeSize(ast->IntTy), bitfield_bit_size);
                    bit_width = new (*ast)IntegerLiteral (*ast, bitfield_bit_size_apint, ast->IntTy, SourceLocation());
                }
                
                ObjCIvarDecl *field = ObjCIvarDecl::Create (*ast,
                                                            class_interface_decl,
                                                            SourceLocation(),
                                                            &identifier_table->get(name), // Identifier
                                                            QualType::getFromOpaquePtr(ivar_opaque_type), // Field type
                                                            NULL, // TypeSourceInfo *
                                                            ConvertAccessTypeToObjCIvarAccessControl (access),
                                                            bit_width,
                                                            is_synthesized);
                
                if (field)
                {
                    class_interface_decl->addDecl(field);
                    return true;
                }
            }
        }
    }
    return false;
}


bool
ClangASTContext::ObjCTypeHasIVars (clang_type_t class_opaque_type, bool check_superclass)
{
    QualType class_qual_type(QualType::getFromOpaquePtr(class_opaque_type));

    clang::Type *class_type = class_qual_type.getTypePtr();
    if (class_type)
    {
        ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(class_type);

        if (objc_class_type)
            return ObjCDeclHasIVars (objc_class_type->getInterface(), check_superclass);
    }
    return false;            
}

bool
ClangASTContext::ObjCDeclHasIVars (ObjCInterfaceDecl *class_interface_decl, bool check_superclass)
{
    while (class_interface_decl)
    {
        if (class_interface_decl->ivar_size() > 0)
            return true;
        
        if (check_superclass)
            class_interface_decl = class_interface_decl->getSuperClass();
        else
            break;
    }
    return false;            
}

ObjCMethodDecl *
ClangASTContext::AddMethodToObjCObjectType
(
    ASTContext *ast,
    clang_type_t class_opaque_type, 
    const char *name,  // the full symbol name as seen in the symbol table ("-[NString stringWithCString:]")
    clang_type_t method_opaque_type,
    lldb::AccessType access
)
{
    if (class_opaque_type == NULL || method_opaque_type == NULL)
        return NULL;

    IdentifierTable *identifier_table = &ast->Idents;

    assert (ast != NULL);
    assert (identifier_table != NULL);

    QualType class_qual_type(QualType::getFromOpaquePtr(class_opaque_type));

    clang::Type *class_type = class_qual_type.getTypePtr();
    if (class_type == NULL)
        return NULL;

    ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(class_type);

    if (objc_class_type == NULL)
        return NULL;

    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            
    if (class_interface_decl == NULL)
        return NULL;
    
    const char *selector_start = ::strchr (name, ' ');
    if (selector_start == NULL)
        return NULL;
    
    selector_start++;
    if (!(::isalpha (selector_start[0]) || selector_start[0] == '_'))
        return NULL;
    llvm::SmallVector<IdentifierInfo *, 12> selector_idents;

    size_t len = 0;
    const char *start;
    //printf ("name = '%s'\n", name);

    unsigned num_selectors_with_args = 0;
    for (start = selector_start;
         start && *start != '\0' && *start != ']';
         start += len)
    {
        len = ::strcspn(start, ":]");
        bool has_arg = (start[len] == ':');
        if (has_arg)
            ++num_selectors_with_args;
        selector_idents.push_back (&identifier_table->get (StringRef (start, len)));
        if (has_arg)
            len += 1;
    }

    
    if (selector_idents.size() == 0)
        return 0;

    clang::Selector method_selector = ast->Selectors.getSelector (num_selectors_with_args ? selector_idents.size() : 0, 
                                                                          selector_idents.data());
    
    QualType method_qual_type (QualType::getFromOpaquePtr (method_opaque_type));

    // Populate the method decl with parameter decls
    clang::Type *method_type(method_qual_type.getTypePtr());
    
    if (method_type == NULL)
        return NULL;
    
    FunctionProtoType *method_function_prototype (dyn_cast<FunctionProtoType>(method_type));
    
    if (!method_function_prototype)
        return NULL;
    

    bool is_variadic = false;
    bool is_synthesized = false;
    bool is_defined = false;
    ObjCMethodDecl::ImplementationControl imp_control = ObjCMethodDecl::None;

    const unsigned num_args = method_function_prototype->getNumArgs();

    ObjCMethodDecl *objc_method_decl = ObjCMethodDecl::Create (*ast,
                                                               SourceLocation(), // beginLoc,
                                                               SourceLocation(), // endLoc, 
                                                               method_selector,
                                                               method_function_prototype->getResultType(),
                                                               NULL, // TypeSourceInfo *ResultTInfo,
                                                               GetDeclContextForType (class_opaque_type),
                                                               name[0] == '-',
                                                               is_variadic,
                                                               is_synthesized,
                                                               is_defined,
                                                               imp_control,
                                                               num_args);


    if (objc_method_decl == NULL)
        return NULL;

    if (num_args > 0)
    {
        llvm::SmallVector<ParmVarDecl *, 12> params;
            
        for (int param_index = 0; param_index < num_args; ++param_index)
        {
            params.push_back (ParmVarDecl::Create (*ast,
                                                   objc_method_decl,
                                                   SourceLocation(),
                                                   NULL, // anonymous
                                                   method_function_prototype->getArgType(param_index), 
                                                   NULL,
                                                   SC_Auto, 
                                                   SC_Auto,
                                                   NULL));
        }
        
        objc_method_decl->setMethodParams(*ast, params.data(), params.size(), num_args);
    }
    
    class_interface_decl->addDecl (objc_method_decl);


    return objc_method_decl;
}


uint32_t
ClangASTContext::GetTypeInfo 
(
    clang_type_t clang_type, 
    clang::ASTContext *ast, 
    clang_type_t *pointee_or_element_clang_type
)
{
    if (clang_type == NULL)
        return 0;
        
    if (pointee_or_element_clang_type)
        *pointee_or_element_clang_type = NULL;
    
    QualType qual_type (QualType::getFromOpaquePtr(clang_type));

    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Builtin:
        switch (cast<clang::BuiltinType>(qual_type)->getKind())
        {
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCClass:
            if (ast && pointee_or_element_clang_type)
                *pointee_or_element_clang_type = ast->ObjCBuiltinClassTy.getAsOpaquePtr();
            return eTypeIsBuiltIn | eTypeIsPointer | eTypeHasValue;

        default: 
            break;
        }
        return eTypeIsBuiltIn | eTypeHasValue;

    case clang::Type::BlockPointer:                     
        if (pointee_or_element_clang_type)
            *pointee_or_element_clang_type = qual_type->getPointeeType().getAsOpaquePtr();
        return eTypeIsPointer | eTypeHasChildren | eTypeIsBlock;

    case clang::Type::Complex:                          return eTypeIsBuiltIn | eTypeHasValue;

    case clang::Type::ConstantArray:
    case clang::Type::DependentSizedArray:
    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:                    
        if (pointee_or_element_clang_type)
            *pointee_or_element_clang_type = cast<ArrayType>(qual_type.getTypePtr())->getElementType().getAsOpaquePtr();
        return eTypeHasChildren | eTypeIsArray;

    case clang::Type::DependentName:                    return 0;
    case clang::Type::DependentSizedExtVector:          return eTypeHasChildren | eTypeIsVector;
    case clang::Type::DependentTemplateSpecialization:  return eTypeIsTemplate;
    case clang::Type::Decltype:                         return 0;

    case clang::Type::Enum:
        if (pointee_or_element_clang_type)
            *pointee_or_element_clang_type = cast<EnumType>(qual_type)->getDecl()->getIntegerType().getAsOpaquePtr();
        return eTypeIsEnumeration | eTypeHasValue;

    case clang::Type::Elaborated:                       return 0;
    case clang::Type::ExtVector:                        return eTypeHasChildren | eTypeIsVector;
    case clang::Type::FunctionProto:                    return eTypeIsFuncPrototype | eTypeHasValue;
    case clang::Type::FunctionNoProto:                  return eTypeIsFuncPrototype | eTypeHasValue;
    case clang::Type::InjectedClassName:                return 0;

    case clang::Type::LValueReference:                  
    case clang::Type::RValueReference:                  
        if (pointee_or_element_clang_type)
            *pointee_or_element_clang_type = cast<ReferenceType>(qual_type.getTypePtr())->getPointeeType().getAsOpaquePtr();
        return eTypeHasChildren | eTypeIsReference | eTypeHasValue;

    case clang::Type::MemberPointer:                    return eTypeIsPointer   | eTypeIsMember | eTypeHasValue;

    case clang::Type::ObjCObjectPointer:
        if (pointee_or_element_clang_type)
            *pointee_or_element_clang_type = qual_type->getPointeeType().getAsOpaquePtr();
        return eTypeHasChildren | eTypeIsObjC | eTypeIsClass | eTypeIsPointer | eTypeHasValue;

    case clang::Type::ObjCObject:                       return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
    case clang::Type::ObjCInterface:                    return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;

    case clang::Type::Pointer:                      	
        if (pointee_or_element_clang_type)
            *pointee_or_element_clang_type = qual_type->getPointeeType().getAsOpaquePtr();
        return eTypeHasChildren | eTypeIsPointer | eTypeHasValue;

    case clang::Type::Record:
        if (qual_type->getAsCXXRecordDecl())
            return eTypeHasChildren | eTypeIsClass | eTypeIsCPlusPlus;
        else
            return eTypeHasChildren | eTypeIsStructUnion;
        break;
    case clang::Type::SubstTemplateTypeParm:            return eTypeIsTemplate;
    case clang::Type::TemplateTypeParm:                 return eTypeIsTemplate;
    case clang::Type::TemplateSpecialization:           return eTypeIsTemplate;

    case clang::Type::Typedef:                         
        return eTypeIsTypedef | ClangASTContext::GetTypeInfo (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                                                                  ast, 
                                                                  pointee_or_element_clang_type);

    case clang::Type::TypeOfExpr:                       return 0;
    case clang::Type::TypeOf:                           return 0;
    case clang::Type::UnresolvedUsing:                  return 0;
    case clang::Type::Vector:                           return eTypeHasChildren | eTypeIsVector;
    default:                                            return 0;
    }
    return 0;
}


#pragma mark Aggregate Types

bool
ClangASTContext::IsAggregateType (clang_type_t clang_type)
{
    if (clang_type == NULL)
        return false;

    QualType qual_type (QualType::getFromOpaquePtr(clang_type));

    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:
    case clang::Type::ConstantArray:
    case clang::Type::ExtVector:
    case clang::Type::Vector:
    case clang::Type::Record:
    case clang::Type::ObjCObject:
    case clang::Type::ObjCInterface:
        return true;

    case clang::Type::Typedef:
        return ClangASTContext::IsAggregateType (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());

    default:
        break;
    }
    // The clang type does have a value
    return false;
}

uint32_t
ClangASTContext::GetNumChildren (clang::ASTContext *ast, clang_type_t clang_type, bool omit_empty_base_classes)
{
    if (clang_type == NULL)
        return 0;

    uint32_t num_children = 0;
    QualType qual_type(QualType::getFromOpaquePtr(clang_type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Builtin:
        switch (cast<clang::BuiltinType>(qual_type)->getKind())
        {
        case clang::BuiltinType::ObjCId:    // child is Class
        case clang::BuiltinType::ObjCClass: // child is Class
            num_children = 1;
            break;

        default:
            break;
        }
        break;

    case clang::Type::Complex: return 0;

    case clang::Type::Record:
        if (ClangASTContext::GetCompleteType (ast, clang_type))
        {
            const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
            const RecordDecl *record_decl = record_type->getDecl();
            assert(record_decl);
            const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
            if (cxx_record_decl)
            {
                if (omit_empty_base_classes)
                {
                    // Check each base classes to see if it or any of its
                    // base classes contain any fields. This can help
                    // limit the noise in variable views by not having to
                    // show base classes that contain no members.
                    CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());

                        // Skip empty base classes
                        if (RecordHasFields(base_class_decl) == false)
                            continue;

                        num_children++;
                    }
                }
                else
                {
                    // Include all base classes
                    num_children += cxx_record_decl->getNumBases();
                }

            }
            RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                ++num_children;
        }
        break;

    case clang::Type::ObjCObject:
    case clang::Type::ObjCInterface:
        {
            ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            
                if (class_interface_decl)
                {
            
                    ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                    if (superclass_interface_decl)
                    {
                        if (omit_empty_base_classes)
                        {
                            if (ClangASTContext::ObjCDeclHasIVars (superclass_interface_decl, true))
                                ++num_children;
                        }
                        else
                            ++num_children;
                    }
                    
                    num_children += class_interface_decl->ivar_size();
                }
            }
        }
        break;
        
    case clang::Type::ObjCObjectPointer:
        {
            ObjCObjectPointerType *pointer_type = cast<ObjCObjectPointerType>(qual_type.getTypePtr());
            QualType pointee_type = pointer_type->getPointeeType();
            uint32_t num_pointee_children = ClangASTContext::GetNumChildren (ast,
                                                                             pointee_type.getAsOpaquePtr(), 
                                                                             omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
        break;

    case clang::Type::ConstantArray:
        num_children = cast<ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
        break;

    case clang::Type::Pointer:
        {
            PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
            QualType pointee_type (pointer_type->getPointeeType());
            uint32_t num_pointee_children = ClangASTContext::GetNumChildren (ast,
                                                                             pointee_type.getAsOpaquePtr(), 
                                                                             omit_empty_base_classes);
            if (num_pointee_children == 0)
            {
                // We have a pointer to a pointee type that claims it has no children.
                // We will want to look at
                num_children = ClangASTContext::GetNumPointeeChildren (pointee_type.getAsOpaquePtr());
            }
            else
                num_children = num_pointee_children;
        }
        break;

    case clang::Type::LValueReference:
    case clang::Type::RValueReference:
        {
            ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
            QualType pointee_type = reference_type->getPointeeType();
            uint32_t num_pointee_children = ClangASTContext::GetNumChildren (ast,
                                                                             pointee_type.getAsOpaquePtr(), 
                                                                             omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
        break;


    case clang::Type::Typedef:
        num_children = ClangASTContext::GetNumChildren (ast,
                                                        cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), 
                                                        omit_empty_base_classes);
        break;

    default:
        break;
    }
    return num_children;
}

// If a pointer to a pointee type (the clang_type arg) says that it has no 
// children, then we either need to trust it, or override it and return a 
// different result. For example, an "int *" has one child that is an integer, 
// but a function pointer doesn't have any children. Likewise if a Record type
// claims it has no children, then there really is nothing to show.
uint32_t
ClangASTContext::GetNumPointeeChildren (clang_type_t clang_type)
{
    if (clang_type == NULL)
        return 0;

    QualType qual_type(QualType::getFromOpaquePtr(clang_type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Builtin:                  
        switch (cast<clang::BuiltinType>(qual_type)->getKind())
        {
        case clang::BuiltinType::Void:
        case clang::BuiltinType::NullPtr:  
            return 0;
        case clang::BuiltinType::Bool:
        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::UChar:
        case clang::BuiltinType::WChar_U:
        case clang::BuiltinType::Char16:
        case clang::BuiltinType::Char32:
        case clang::BuiltinType::UShort:
        case clang::BuiltinType::UInt:
        case clang::BuiltinType::ULong:
        case clang::BuiltinType::ULongLong:
        case clang::BuiltinType::UInt128:
        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::SChar:
        case clang::BuiltinType::WChar_S:
        case clang::BuiltinType::Short:
        case clang::BuiltinType::Int:
        case clang::BuiltinType::Long:
        case clang::BuiltinType::LongLong:
        case clang::BuiltinType::Int128:
        case clang::BuiltinType::Float:
        case clang::BuiltinType::Double:
        case clang::BuiltinType::LongDouble:
        case clang::BuiltinType::Dependent:
        case clang::BuiltinType::Overload:
        case clang::BuiltinType::UndeducedAuto:
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCClass:
        case clang::BuiltinType::ObjCSel:
            return 1;
        }
        break;

    case clang::Type::Complex:                  return 1;
    case clang::Type::Pointer:                  return 1;
    case clang::Type::BlockPointer:             return 0;   // If block pointers don't have debug info, then no children for them
    case clang::Type::LValueReference:          return 1;
    case clang::Type::RValueReference:          return 1;
    case clang::Type::MemberPointer:            return 0;
    case clang::Type::ConstantArray:            return 0;
    case clang::Type::IncompleteArray:          return 0;
    case clang::Type::VariableArray:            return 0;
    case clang::Type::DependentSizedArray:      return 0;
    case clang::Type::DependentSizedExtVector:  return 0;
    case clang::Type::Vector:                   return 0;
    case clang::Type::ExtVector:                return 0;
    case clang::Type::FunctionProto:            return 0;   // When we function pointers, they have no children...
    case clang::Type::FunctionNoProto:          return 0;   // When we function pointers, they have no children...
    case clang::Type::UnresolvedUsing:          return 0;
    case clang::Type::Paren:                    return 0;
    case clang::Type::Typedef:                  return ClangASTContext::GetNumPointeeChildren (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
    case clang::Type::TypeOfExpr:               return 0;
    case clang::Type::TypeOf:                   return 0;
    case clang::Type::Decltype:                 return 0;
    case clang::Type::Record:                   return 0;
    case clang::Type::Enum:                     return 1;
    case clang::Type::Elaborated:               return 1;
    case clang::Type::TemplateTypeParm:         return 1;
    case clang::Type::SubstTemplateTypeParm:    return 1;
    case clang::Type::TemplateSpecialization:   return 1;
    case clang::Type::InjectedClassName:        return 0;
    case clang::Type::DependentName:            return 1;
    case clang::Type::DependentTemplateSpecialization:  return 1;
    case clang::Type::ObjCObject:               return 0;
    case clang::Type::ObjCInterface:            return 0;
    case clang::Type::ObjCObjectPointer:        return 1;
    default: 
        break;
    }
    return 0;
}

clang_type_t
ClangASTContext::GetChildClangTypeAtIndex
(
    const char *parent_name,
    clang_type_t parent_clang_type,
    uint32_t idx,
    bool transparent_pointers,
    bool omit_empty_base_classes,
    std::string& child_name,
    uint32_t &child_byte_size,
    int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class
)
{
    if (parent_clang_type)

        return GetChildClangTypeAtIndex (getASTContext(),
                                         parent_name,
                                         parent_clang_type,
                                         idx,
                                         transparent_pointers,
                                         omit_empty_base_classes,
                                         child_name,
                                         child_byte_size,
                                         child_byte_offset,
                                         child_bitfield_bit_size,
                                         child_bitfield_bit_offset,
                                         child_is_base_class);
    return NULL;
}

clang_type_t
ClangASTContext::GetChildClangTypeAtIndex
(
    ASTContext *ast,
    const char *parent_name,
    clang_type_t parent_clang_type,
    uint32_t idx,
    bool transparent_pointers,
    bool omit_empty_base_classes,
    std::string& child_name,
    uint32_t &child_byte_size,
    int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset,
    bool &child_is_base_class
)
{
    if (parent_clang_type == NULL)
        return NULL;

    if (idx < ClangASTContext::GetNumChildren (ast, parent_clang_type, omit_empty_base_classes))
    {
        uint32_t bit_offset;
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;
        child_is_base_class = false;
        QualType parent_qual_type(QualType::getFromOpaquePtr(parent_clang_type));
        const clang::Type::TypeClass parent_type_class = parent_qual_type->getTypeClass();
        switch (parent_type_class)
        {
        case clang::Type::Builtin:
            switch (cast<clang::BuiltinType>(parent_qual_type)->getKind())
            {
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
                child_name = "isa";
                child_byte_size = ast->getTypeSize(ast->ObjCBuiltinClassTy) / CHAR_BIT;
                return ast->ObjCBuiltinClassTy.getAsOpaquePtr();
                
            default:
                break;
            }
            break;

        case clang::Type::Record:
            if (ClangASTContext::GetCompleteType (ast, parent_clang_type))
            {
                const RecordType *record_type = cast<RecordType>(parent_qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const ASTRecordLayout &record_layout = ast->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;

                const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const CXXRecordDecl *base_class_decl = NULL;

                        // Skip empty base classes
                        if (omit_empty_base_classes)
                        {
                            base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                            if (RecordHasFields(base_class_decl) == false)
                                continue;
                        }

                        if (idx == child_idx)
                        {
                            if (base_class_decl == NULL)
                                base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());


                            if (base_class->isVirtual())
                                bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity();
                            else
                                bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity();

                            // Base classes should be a multiple of 8 bits in size
                            assert (bit_offset % 8 == 0);
                            child_byte_offset = bit_offset/8;
                            std::string base_class_type_name(base_class->getType().getAsString());

                            child_name.assign(base_class_type_name.c_str());

                            uint64_t clang_type_info_bit_size = ast->getTypeSize(base_class->getType());

                            // Base classes biut sizes should be a multiple of 8 bits in size
                            assert (clang_type_info_bit_size % 8 == 0);
                            child_byte_size = clang_type_info_bit_size / 8;
                            child_is_base_class = true;
                            return base_class->getType().getAsOpaquePtr();
                        }
                        // We don't increment the child index in the for loop since we might
                        // be skipping empty base classes
                        ++child_idx;
                    }
                }
                // Make sure index is in range...
                uint32_t field_idx = 0;
                RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
                {
                    if (idx == child_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        child_name.assign(field->getNameAsString().c_str());

                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(field->getType());
                        assert(field_idx < record_layout.getFieldCount());

                        child_byte_size = field_type_info.first / 8;

                        // Figure out the field offset within the current struct/union/class type
                        bit_offset = record_layout.getFieldOffset (field_idx);
                        child_byte_offset = bit_offset / 8;
                        if (ClangASTContext::FieldIsBitfield (ast, *field, child_bitfield_bit_size))
                            child_bitfield_bit_offset = bit_offset % 8;

                        return field->getType().getAsOpaquePtr();
                    }
                }
            }
            break;

        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            {
                ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(parent_qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
                    if (class_interface_decl)
                    {
                
                        const ASTRecordLayout &interface_layout = ast->getASTObjCInterfaceLayout(class_interface_decl);
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                if (ClangASTContext::GetNumChildren(ast, ast->getObjCInterfaceType(superclass_interface_decl).getAsOpaquePtr(), omit_empty_base_classes) > 0)
                                {
                                    if (idx == 0)
                                    {
                                        QualType ivar_qual_type(ast->getObjCInterfaceType(superclass_interface_decl));
                                        

                                        child_name.assign(superclass_interface_decl->getNameAsString().c_str());

                                        std::pair<uint64_t, unsigned> ivar_type_info = ast->getTypeInfo(ivar_qual_type.getTypePtr());

                                        child_byte_size = ivar_type_info.first / 8;
                                        child_byte_offset = 0;
                                        child_is_base_class = true;

                                        return ivar_qual_type.getAsOpaquePtr();
                                    }

                                    ++child_idx;
                                }
                            }
                            else
                                ++child_idx;
                        }
    
                        const uint32_t superclass_idx = child_idx;

                        if (idx < (child_idx + class_interface_decl->ivar_size()))
                        {
                            ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
                            {
                                if (child_idx == idx)
                                {
                                    const ObjCIvarDecl* ivar_decl = *ivar_pos;
                                    
                                    QualType ivar_qual_type(ivar_decl->getType());

                                    child_name.assign(ivar_decl->getNameAsString().c_str());

                                    std::pair<uint64_t, unsigned> ivar_type_info = ast->getTypeInfo(ivar_qual_type.getTypePtr());

                                    child_byte_size = ivar_type_info.first / 8;

                                    // Figure out the field offset within the current struct/union/class type
                                    bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                    child_byte_offset = bit_offset / 8;

                                    return ivar_qual_type.getAsOpaquePtr();
                                }
                                ++child_idx;
                            }
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            {
                ObjCObjectPointerType *pointer_type = cast<ObjCObjectPointerType>(parent_qual_type.getTypePtr());
                QualType pointee_type = pointer_type->getPointeeType();

                if (transparent_pointers && ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetChildClangTypeAtIndex (ast,
                                                     parent_name,
                                                     pointer_type->getPointeeType().getAsOpaquePtr(),
                                                     idx,
                                                     transparent_pointers,
                                                     omit_empty_base_classes,
                                                     child_name,
                                                     child_byte_size,
                                                     child_byte_offset,
                                                     child_bitfield_bit_size,
                                                     child_bitfield_bit_offset,
                                                     child_is_base_class);
                }
                else
                {
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }

                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
                        assert(clang_type_info.first % 8 == 0);
                        child_byte_size = clang_type_info.first / 8;
                        child_byte_offset = 0;
                        return pointee_type.getAsOpaquePtr();
                    }
                }
            }
            break;

        case clang::Type::ConstantArray:
            {
                const ConstantArrayType *array = cast<ConstantArrayType>(parent_qual_type.getTypePtr());
                const uint64_t element_count = array->getSize().getLimitedValue();

                if (idx < element_count)
                {
                    if (GetCompleteQualType (ast, array->getElementType()))
                    {
                        std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());

                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%u]", idx);

                        child_name.assign(element_name);
                        assert(field_type_info.first % 8 == 0);
                        child_byte_size = field_type_info.first / 8;
                        child_byte_offset = idx * child_byte_size;
                        return array->getElementType().getAsOpaquePtr();
                    }
                }
            }
            break;

        case clang::Type::Pointer:
            {
                PointerType *pointer_type = cast<PointerType>(parent_qual_type.getTypePtr());
                QualType pointee_type = pointer_type->getPointeeType();
                
                // Don't dereference "void *" pointers
                if (pointee_type->isVoidType())
                    return NULL;

                if (transparent_pointers && ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetChildClangTypeAtIndex (ast,
                                                     parent_name,
                                                     pointer_type->getPointeeType().getAsOpaquePtr(),
                                                     idx,
                                                     transparent_pointers,
                                                     omit_empty_base_classes,
                                                     child_name,
                                                     child_byte_size,
                                                     child_byte_offset,
                                                     child_bitfield_bit_size,
                                                     child_bitfield_bit_offset,
                                                     child_is_base_class);
                }
                else
                {
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }

                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
                        assert(clang_type_info.first % 8 == 0);
                        child_byte_size = clang_type_info.first / 8;
                        child_byte_offset = 0;
                        return pointee_type.getAsOpaquePtr();
                    }
                }
            }
            break;

        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                ReferenceType *reference_type = cast<ReferenceType>(parent_qual_type.getTypePtr());
                QualType pointee_type(reference_type->getPointeeType());
                clang_type_t pointee_clang_type = pointee_type.getAsOpaquePtr();
                if (transparent_pointers && ClangASTContext::IsAggregateType (pointee_clang_type))
                {
                    return GetChildClangTypeAtIndex (ast,
                                                     parent_name,
                                                     pointee_clang_type,
                                                     idx,
                                                     transparent_pointers,
                                                     omit_empty_base_classes,
                                                     child_name,
                                                     child_byte_size,
                                                     child_byte_offset,
                                                     child_bitfield_bit_size,
                                                     child_bitfield_bit_offset,
                                                     child_is_base_class);
                }
                else
                {
                    if (parent_name)
                    {
                        child_name.assign(1, '&');
                        child_name += parent_name;
                    }

                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
                        assert(clang_type_info.first % 8 == 0);
                        child_byte_size = clang_type_info.first / 8;
                        child_byte_offset = 0;
                        return pointee_type.getAsOpaquePtr();
                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetChildClangTypeAtIndex (ast,
                                             parent_name,
                                             cast<TypedefType>(parent_qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                                             idx,
                                             transparent_pointers,
                                             omit_empty_base_classes,
                                             child_name,
                                             child_byte_size,
                                             child_byte_offset,
                                             child_bitfield_bit_size,
                                             child_bitfield_bit_offset,
                                             child_is_base_class);
            break;

        default:
            break;
        }
    }
    return NULL;
}

static inline bool
BaseSpecifierIsEmpty (const CXXBaseSpecifier *b)
{
    return ClangASTContext::RecordHasFields(b->getType()->getAsCXXRecordDecl()) == false;
}

static uint32_t
GetNumBaseClasses (const CXXRecordDecl *cxx_record_decl, bool omit_empty_base_classes)
{
    uint32_t num_bases = 0;
    if (cxx_record_decl)
    {
        if (omit_empty_base_classes)
        {
            CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
            for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                 base_class != base_class_end;
                 ++base_class)
            {
                // Skip empty base classes
                if (omit_empty_base_classes)
                {
                    if (BaseSpecifierIsEmpty (base_class))
                        continue;
                }
                ++num_bases;
            }
        }
        else
            num_bases = cxx_record_decl->getNumBases();
    }
    return num_bases;
}


static uint32_t
GetIndexForRecordBase
(
    const RecordDecl *record_decl,
    const CXXBaseSpecifier *base_spec,
    bool omit_empty_base_classes
)
{
    uint32_t child_idx = 0;

    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);

//    const char *super_name = record_decl->getNameAsCString();
//    const char *base_name = base_spec->getType()->getAs<RecordType>()->getDecl()->getNameAsCString();
//    printf ("GetIndexForRecordChild (%s, %s)\n", super_name, base_name);
//
    if (cxx_record_decl)
    {
        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
             base_class != base_class_end;
             ++base_class)
        {
            if (omit_empty_base_classes)
            {
                if (BaseSpecifierIsEmpty (base_class))
                    continue;
            }

//            printf ("GetIndexForRecordChild (%s, %s) base[%u] = %s\n", super_name, base_name,
//                    child_idx,
//                    base_class->getType()->getAs<RecordType>()->getDecl()->getNameAsCString());
//
//
            if (base_class == base_spec)
                return child_idx;
            ++child_idx;
        }
    }

    return UINT32_MAX;
}


static uint32_t
GetIndexForRecordChild
(
    const RecordDecl *record_decl,
    NamedDecl *canonical_decl,
    bool omit_empty_base_classes
)
{
    uint32_t child_idx = GetNumBaseClasses (dyn_cast<CXXRecordDecl>(record_decl), omit_empty_base_classes);

//    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
//
////    printf ("GetIndexForRecordChild (%s, %s)\n", record_decl->getNameAsCString(), canonical_decl->getNameAsCString());
//    if (cxx_record_decl)
//    {
//        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
//        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
//             base_class != base_class_end;
//             ++base_class)
//        {
//            if (omit_empty_base_classes)
//            {
//                if (BaseSpecifierIsEmpty (base_class))
//                    continue;
//            }
//
////            printf ("GetIndexForRecordChild (%s, %s) base[%u] = %s\n",
////                    record_decl->getNameAsCString(),
////                    canonical_decl->getNameAsCString(),
////                    child_idx,
////                    base_class->getType()->getAs<RecordType>()->getDecl()->getNameAsCString());
//
//
//            CXXRecordDecl *curr_base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
//            if (curr_base_class_decl == canonical_decl)
//            {
//                return child_idx;
//            }
//            ++child_idx;
//        }
//    }
//
//    const uint32_t num_bases = child_idx;
    RecordDecl::field_iterator field, field_end;
    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
         field != field_end;
         ++field, ++child_idx)
    {
//            printf ("GetIndexForRecordChild (%s, %s) field[%u] = %s\n",
//                    record_decl->getNameAsCString(),
//                    canonical_decl->getNameAsCString(),
//                    child_idx - num_bases,
//                    field->getNameAsCString());

        if (field->getCanonicalDecl() == canonical_decl)
            return child_idx;
    }

    return UINT32_MAX;
}

// Look for a child member (doesn't include base classes, but it does include
// their members) in the type hierarchy. Returns an index path into "clang_type"
// on how to reach the appropriate member.
//
//    class A
//    {
//    public:
//        int m_a;
//        int m_b;
//    };
//
//    class B
//    {
//    };
//
//    class C :
//        public B,
//        public A
//    {
//    };
//
// If we have a clang type that describes "class C", and we wanted to looked
// "m_b" in it:
//
// With omit_empty_base_classes == false we would get an integer array back with:
// { 1,  1 }
// The first index 1 is the child index for "class A" within class C
// The second index 1 is the child index for "m_b" within class A
//
// With omit_empty_base_classes == true we would get an integer array back with:
// { 0,  1 }
// The first index 0 is the child index for "class A" within class C (since class B doesn't have any members it doesn't count)
// The second index 1 is the child index for "m_b" within class A

size_t
ClangASTContext::GetIndexOfChildMemberWithName
(
    ASTContext *ast,
    clang_type_t clang_type,
    const char *name,
    bool omit_empty_base_classes,
    std::vector<uint32_t>& child_indexes
)
{
    if (clang_type && name && name[0])
    {
        QualType qual_type(QualType::getFromOpaquePtr(clang_type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::Record:
            if (ClangASTContext::GetCompleteType (ast, clang_type))
            {
                const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();

                assert(record_decl);
                uint32_t child_idx = 0;

                const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);

                // Try and find a field that matches NAME
                RecordDecl::field_iterator field, field_end;
                StringRef name_sref(name);
                for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                     field != field_end;
                     ++field, ++child_idx)
                {
                    if (field->getName().equals (name_sref))
                    {
                        // We have to add on the number of base classes to this index!
                        child_indexes.push_back (child_idx + GetNumBaseClasses (cxx_record_decl, omit_empty_base_classes));
                        return child_indexes.size();
                    }
                }

                if (cxx_record_decl)
                {
                    const RecordDecl *parent_record_decl = cxx_record_decl;

                    //printf ("parent = %s\n", parent_record_decl->getNameAsCString());

                    //const Decl *root_cdecl = cxx_record_decl->getCanonicalDecl();
                    // Didn't find things easily, lets let clang do its thang...
                    IdentifierInfo & ident_ref = ast->Idents.get(name, name + strlen (name));
                    DeclarationName decl_name(&ident_ref);

                    CXXBasePaths paths;
                    if (cxx_record_decl->lookupInBases(CXXRecordDecl::FindOrdinaryMember,
                                                       decl_name.getAsOpaquePtr(),
                                                       paths))
                    {
                        CXXBasePaths::const_paths_iterator path, path_end = paths.end();
                        for (path = paths.begin(); path != path_end; ++path)
                        {
                            const size_t num_path_elements = path->size();
                            for (size_t e=0; e<num_path_elements; ++e)
                            {
                                CXXBasePathElement elem = (*path)[e];

                                child_idx = GetIndexForRecordBase (parent_record_decl, elem.Base, omit_empty_base_classes);
                                if (child_idx == UINT32_MAX)
                                {
                                    child_indexes.clear();
                                    return 0;
                                }
                                else
                                {
                                    child_indexes.push_back (child_idx);
                                    parent_record_decl = cast<RecordDecl>(elem.Base->getType()->getAs<RecordType>()->getDecl());
                                }
                            }
                            DeclContext::lookup_iterator named_decl_pos;
                            for (named_decl_pos = path->Decls.first;
                                 named_decl_pos != path->Decls.second && parent_record_decl;
                                 ++named_decl_pos)
                            {
                                //printf ("path[%zu] = %s\n", child_indexes.size(), (*named_decl_pos)->getNameAsCString());

                                child_idx = GetIndexForRecordChild (parent_record_decl, *named_decl_pos, omit_empty_base_classes);
                                if (child_idx == UINT32_MAX)
                                {
                                    child_indexes.clear();
                                    return 0;
                                }
                                else
                                {
                                    child_indexes.push_back (child_idx);
                                }
                            }
                        }
                        return child_indexes.size();
                    }
                }

            }
            break;

        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            {
                StringRef name_sref(name);
                ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
                    if (class_interface_decl)
                    {
                        ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        
                        for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                        {
                            const ObjCIvarDecl* ivar_decl = *ivar_pos;
                            
                            if (ivar_decl->getName().equals (name_sref))
                            {
                                if ((!omit_empty_base_classes && superclass_interface_decl) || 
                                    ( omit_empty_base_classes && ObjCDeclHasIVars (superclass_interface_decl, true)))
                                    ++child_idx;

                                child_indexes.push_back (child_idx);
                                return child_indexes.size();
                            }
                        }

                        if (superclass_interface_decl)
                        {
                            // The super class index is always zero for ObjC classes,
                            // so we push it onto the child indexes in case we find
                            // an ivar in our superclass...
                            child_indexes.push_back (0);
                            
                            if (GetIndexOfChildMemberWithName (ast,
                                                               ast->getObjCInterfaceType(superclass_interface_decl).getAsOpaquePtr(),
                                                               name,
                                                               omit_empty_base_classes,
                                                               child_indexes))
                            {
                                // We did find an ivar in a superclass so just
                                // return the results!
                                return child_indexes.size();
                            }
                            
                            // We didn't find an ivar matching "name" in our 
                            // superclass, pop the superclass zero index that
                            // we pushed on above.
                            child_indexes.pop_back();
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            {
                return GetIndexOfChildMemberWithName (ast,
                                                      cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType().getAsOpaquePtr(),
                                                      name,
                                                      omit_empty_base_classes,
                                                      child_indexes);
            }
            break;


        case clang::Type::ConstantArray:
            {
//                const ConstantArrayType *array = cast<ConstantArrayType>(parent_qual_type.getTypePtr());
//                const uint64_t element_count = array->getSize().getLimitedValue();
//
//                if (idx < element_count)
//                {
//                    std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());
//
//                    char element_name[32];
//                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);
//
//                    child_name.assign(element_name);
//                    assert(field_type_info.first % 8 == 0);
//                    child_byte_size = field_type_info.first / 8;
//                    child_byte_offset = idx * child_byte_size;
//                    return array->getElementType().getAsOpaquePtr();
//                }
            }
            break;

//        case clang::Type::MemberPointerType:
//            {
//                MemberPointerType *mem_ptr_type = cast<MemberPointerType>(qual_type.getTypePtr());
//                QualType pointee_type = mem_ptr_type->getPointeeType();
//
//                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
//                {
//                    return GetIndexOfChildWithName (ast,
//                                                    mem_ptr_type->getPointeeType().getAsOpaquePtr(),
//                                                    name);
//                }
//            }
//            break;
//
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                QualType pointee_type = reference_type->getPointeeType();

                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetIndexOfChildMemberWithName (ast,
                                                          reference_type->getPointeeType().getAsOpaquePtr(),
                                                          name,
                                                          omit_empty_base_classes,
                                                          child_indexes);
                }
            }
            break;

        case clang::Type::Pointer:
            {
                PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
                QualType pointee_type = pointer_type->getPointeeType();

                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetIndexOfChildMemberWithName (ast,
                                                          pointer_type->getPointeeType().getAsOpaquePtr(),
                                                          name,
                                                          omit_empty_base_classes,
                                                          child_indexes);
                }
                else
                {
//                    if (parent_name)
//                    {
//                        child_name.assign(1, '*');
//                        child_name += parent_name;
//                    }
//
//                    // We have a pointer to an simple type
//                    if (idx == 0)
//                    {
//                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
//                        assert(clang_type_info.first % 8 == 0);
//                        child_byte_size = clang_type_info.first / 8;
//                        child_byte_offset = 0;
//                        return pointee_type.getAsOpaquePtr();
//                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetIndexOfChildMemberWithName (ast,
                                                  cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                                                  name,
                                                  omit_empty_base_classes,
                                                  child_indexes);

        default:
            break;
        }
    }
    return 0;
}


// Get the index of the child of "clang_type" whose name matches. This function
// doesn't descend into the children, but only looks one level deep and name
// matches can include base class names.

uint32_t
ClangASTContext::GetIndexOfChildWithName
(
    ASTContext *ast,
    clang_type_t clang_type,
    const char *name,
    bool omit_empty_base_classes
)
{
    if (clang_type && name && name[0])
    {
        QualType qual_type(QualType::getFromOpaquePtr(clang_type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();

        switch (type_class)
        {
        case clang::Type::Record:
            if (ClangASTContext::GetCompleteType (ast, clang_type))
            {
                const RecordType *record_type = cast<RecordType>(qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();

                assert(record_decl);
                uint32_t child_idx = 0;

                const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);

                if (cxx_record_decl)
                {
                    CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        // Skip empty base classes
                        CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
                        if (omit_empty_base_classes && RecordHasFields(base_class_decl) == false)
                            continue;

                        if (base_class->getType().getAsString().compare (name) == 0)
                            return child_idx;
                        ++child_idx;
                    }
                }

                // Try and find a field that matches NAME
                RecordDecl::field_iterator field, field_end;
                StringRef name_sref(name);
                for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                     field != field_end;
                     ++field, ++child_idx)
                {
                    if (field->getName().equals (name_sref))
                        return child_idx;
                }

            }
            break;

        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            {
                StringRef name_sref(name);
                ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
                    if (class_interface_decl)
                    {
                        ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        
                        for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
                        {
                            const ObjCIvarDecl* ivar_decl = *ivar_pos;
                            
                            if (ivar_decl->getName().equals (name_sref))
                            {
                                if ((!omit_empty_base_classes && superclass_interface_decl) || 
                                    ( omit_empty_base_classes && ObjCDeclHasIVars (superclass_interface_decl, true)))
                                    ++child_idx;

                                return child_idx;
                            }
                        }

                        if (superclass_interface_decl)
                        {
                            if (superclass_interface_decl->getName().equals (name_sref))
                                return 0;
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            {
                return GetIndexOfChildWithName (ast,
                                                cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType().getAsOpaquePtr(),
                                                name,
                                                omit_empty_base_classes);
            }
            break;

        case clang::Type::ConstantArray:
            {
//                const ConstantArrayType *array = cast<ConstantArrayType>(parent_qual_type.getTypePtr());
//                const uint64_t element_count = array->getSize().getLimitedValue();
//
//                if (idx < element_count)
//                {
//                    std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());
//
//                    char element_name[32];
//                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);
//
//                    child_name.assign(element_name);
//                    assert(field_type_info.first % 8 == 0);
//                    child_byte_size = field_type_info.first / 8;
//                    child_byte_offset = idx * child_byte_size;
//                    return array->getElementType().getAsOpaquePtr();
//                }
            }
            break;

//        case clang::Type::MemberPointerType:
//            {
//                MemberPointerType *mem_ptr_type = cast<MemberPointerType>(qual_type.getTypePtr());
//                QualType pointee_type = mem_ptr_type->getPointeeType();
//
//                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
//                {
//                    return GetIndexOfChildWithName (ast,
//                                                    mem_ptr_type->getPointeeType().getAsOpaquePtr(),
//                                                    name);
//                }
//            }
//            break;
//
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                QualType pointee_type = reference_type->getPointeeType();

                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetIndexOfChildWithName (ast,
                                                    reference_type->getPointeeType().getAsOpaquePtr(),
                                                    name,
                                                    omit_empty_base_classes);
                }
            }
            break;

        case clang::Type::Pointer:
            {
                PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
                QualType pointee_type = pointer_type->getPointeeType();

                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetIndexOfChildWithName (ast,
                                                    pointer_type->getPointeeType().getAsOpaquePtr(),
                                                    name,
                                                    omit_empty_base_classes);
                }
                else
                {
//                    if (parent_name)
//                    {
//                        child_name.assign(1, '*');
//                        child_name += parent_name;
//                    }
//
//                    // We have a pointer to an simple type
//                    if (idx == 0)
//                    {
//                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
//                        assert(clang_type_info.first % 8 == 0);
//                        child_byte_size = clang_type_info.first / 8;
//                        child_byte_offset = 0;
//                        return pointee_type.getAsOpaquePtr();
//                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetIndexOfChildWithName (ast,
                                            cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                                            name,
                                            omit_empty_base_classes);

        default:
            break;
        }
    }
    return UINT32_MAX;
}

#pragma mark TagType

bool
ClangASTContext::SetTagTypeKind (clang_type_t tag_clang_type, int kind)
{
    if (tag_clang_type)
    {
        QualType tag_qual_type(QualType::getFromOpaquePtr(tag_clang_type));
        clang::Type *clang_type = tag_qual_type.getTypePtr();
        if (clang_type)
        {
            TagType *tag_type = dyn_cast<TagType>(clang_type);
            if (tag_type)
            {
                TagDecl *tag_decl = dyn_cast<TagDecl>(tag_type->getDecl());
                if (tag_decl)
                {
                    tag_decl->setTagKind ((TagDecl::TagKind)kind);
                    return true;
                }
            }
        }
    }
    return false;
}


#pragma mark DeclContext Functions

DeclContext *
ClangASTContext::GetDeclContextForType (clang_type_t clang_type)
{
    if (clang_type == NULL)
        return NULL;

    QualType qual_type(QualType::getFromOpaquePtr(clang_type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::FunctionNoProto:          break;
    case clang::Type::FunctionProto:            break;
    case clang::Type::IncompleteArray:          break;
    case clang::Type::VariableArray:            break;
    case clang::Type::ConstantArray:            break;
    case clang::Type::ExtVector:                break;
    case clang::Type::Vector:                   break;
    case clang::Type::Builtin:                  break;
    case clang::Type::BlockPointer:             break;
    case clang::Type::Pointer:                  break;
    case clang::Type::LValueReference:          break;
    case clang::Type::RValueReference:          break;
    case clang::Type::MemberPointer:            break;
    case clang::Type::Complex:                  break;
    case clang::Type::ObjCObject:               break;
    case clang::Type::ObjCInterface:            return cast<ObjCObjectType>(qual_type.getTypePtr())->getInterface();
    case clang::Type::ObjCObjectPointer:        return ClangASTContext::GetDeclContextForType (cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType().getAsOpaquePtr());
    case clang::Type::Record:                   return cast<RecordType>(qual_type)->getDecl();
    case clang::Type::Enum:                     return cast<EnumType>(qual_type)->getDecl();
    case clang::Type::Typedef:                  return ClangASTContext::GetDeclContextForType (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());

    case clang::Type::TypeOfExpr:               break;
    case clang::Type::TypeOf:                   break;
    case clang::Type::Decltype:                 break;
    //case clang::Type::QualifiedName:          break;
    case clang::Type::TemplateSpecialization:   break;
    }
    // No DeclContext in this type...
    return NULL;
}

#pragma mark Namespace Declarations

NamespaceDecl *
ClangASTContext::GetUniqueNamespaceDeclaration (const char *name, const Declaration &decl, DeclContext *decl_ctx)
{
    // TODO: Do something intelligent with the Declaration object passed in
    // like maybe filling in the SourceLocation with it...
    if (name)
    {
        ASTContext *ast = getASTContext();
        if (decl_ctx == NULL)
            decl_ctx = ast->getTranslationUnitDecl();
        return NamespaceDecl::Create(*ast, decl_ctx, SourceLocation(), &ast->Idents.get(name));
    }
    return NULL;
}


#pragma mark Function Types

FunctionDecl *
ClangASTContext::CreateFunctionDeclaration (const char *name, clang_type_t function_clang_type, int storage, bool is_inline)
{
    if (name)
    {
        ASTContext *ast = getASTContext();
        assert (ast != NULL);

        if (name && name[0])
        {
            return FunctionDecl::Create(*ast,
                                        ast->getTranslationUnitDecl(),
                                        SourceLocation(),
                                        DeclarationName (&ast->Idents.get(name)),
                                        QualType::getFromOpaquePtr(function_clang_type),
                                        NULL,
                                        (FunctionDecl::StorageClass)storage,
                                        (FunctionDecl::StorageClass)storage,
                                        is_inline);
        }
        else
        {
            return FunctionDecl::Create(*ast,
                                        ast->getTranslationUnitDecl(),
                                        SourceLocation(),
                                        DeclarationName (),
                                        QualType::getFromOpaquePtr(function_clang_type),
                                        NULL,
                                        (FunctionDecl::StorageClass)storage,
                                        (FunctionDecl::StorageClass)storage,
                                        is_inline);
        }
    }
    return NULL;
}

clang_type_t
ClangASTContext::CreateFunctionType (ASTContext *ast,
                                     clang_type_t result_type, 
                                     clang_type_t *args, 
                                     unsigned num_args, 
                                     bool is_variadic, 
                                     unsigned type_quals)
{
    assert (ast != NULL);
    std::vector<QualType> qual_type_args;
    for (unsigned i=0; i<num_args; ++i)
        qual_type_args.push_back (QualType::getFromOpaquePtr(args[i]));

    // TODO: Detect calling convention in DWARF?
    FunctionProtoType::ExtProtoInfo proto_info;
    proto_info.Variadic = is_variadic;
    proto_info.HasExceptionSpec = false;
    proto_info.HasAnyExceptionSpec = false;
    proto_info.TypeQuals = type_quals;
    proto_info.NumExceptions = 0;
    proto_info.Exceptions = NULL;
    
    return ast->getFunctionType(QualType::getFromOpaquePtr(result_type),
                                        qual_type_args.empty() ? NULL : &qual_type_args.front(),
                                        qual_type_args.size(),
                                        proto_info).getAsOpaquePtr();    // NoReturn);
}

ParmVarDecl *
ClangASTContext::CreateParameterDeclaration (const char *name, clang_type_t param_type, int storage)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);
    return ParmVarDecl::Create(*ast,
                                ast->getTranslationUnitDecl(),
                                SourceLocation(),
                                name && name[0] ? &ast->Idents.get(name) : NULL,
                                QualType::getFromOpaquePtr(param_type),
                                NULL,
                                (VarDecl::StorageClass)storage,
                                (VarDecl::StorageClass)storage,
                                0);
}

void
ClangASTContext::SetFunctionParameters (FunctionDecl *function_decl, ParmVarDecl **params, unsigned num_params)
{
    if (function_decl)
        function_decl->setParams (params, num_params);
}


#pragma mark Array Types

clang_type_t
ClangASTContext::CreateArrayType (clang_type_t element_type, size_t element_count, uint32_t bit_stride)
{
    if (element_type)
    {
        ASTContext *ast = getASTContext();
        assert (ast != NULL);
        llvm::APInt ap_element_count (64, element_count);
        return ast->getConstantArrayType(QualType::getFromOpaquePtr(element_type),
                                                 ap_element_count,
                                                 ArrayType::Normal,
                                                 0).getAsOpaquePtr(); // ElemQuals
    }
    return NULL;
}


#pragma mark TagDecl

bool
ClangASTContext::StartTagDeclarationDefinition (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        clang::Type *t = qual_type.getTypePtr();
        if (t)
        {
            TagType *tag_type = dyn_cast<TagType>(t);
            if (tag_type)
            {
                TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    tag_decl->startDefinition();
                    return true;
                }
            }
        }
    }
    return false;
}

bool
ClangASTContext::CompleteTagDeclarationDefinition (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        
        CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        
        if (cxx_record_decl)
        {
            cxx_record_decl->completeDefinition();
            
            return true;
        }
        
        ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(qual_type);
        
        if (objc_class_type)
        {
            ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            
            class_interface_decl->setForwardDecl(false);
        }
        
        const EnumType *enum_type = dyn_cast<EnumType>(qual_type.getTypePtr());
        
        if (enum_type)
        {
            EnumDecl *enum_decl = enum_type->getDecl();
            
            if (enum_decl)
            {
                /// TODO This really needs to be fixed.
                
                unsigned NumPositiveBits = 1;
                unsigned NumNegativeBits = 0;
                
                ASTContext *ast = getASTContext();

                QualType promotion_qual_type;
                // If the enum integer type is less than an integer in bit width,
                // then we must promote it to an integer size.
                if (ast->getTypeSize(enum_decl->getIntegerType()) < ast->getTypeSize(ast->IntTy))
                {
                    if (enum_decl->getIntegerType()->isSignedIntegerType())
                        promotion_qual_type = ast->IntTy;
                    else
                        promotion_qual_type = ast->UnsignedIntTy;
                }
                else
                    promotion_qual_type = enum_decl->getIntegerType();

                enum_decl->completeDefinition(enum_decl->getIntegerType(), promotion_qual_type, NumPositiveBits, NumNegativeBits);
                return true;
            }
        }
    }
    return false;
}


#pragma mark Enumeration Types

clang_type_t
ClangASTContext::CreateEnumerationType 
(
    const char *name, 
    DeclContext *decl_ctx, 
    const Declaration &decl, 
    clang_type_t integer_qual_type
)
{
    // TODO: Do something intelligent with the Declaration object passed in
    // like maybe filling in the SourceLocation with it...
    ASTContext *ast = getASTContext();
    assert (ast != NULL);

    // TODO: ask about these...
//    const bool IsScoped = false;
//    const bool IsFixed = false;

    EnumDecl *enum_decl = EnumDecl::Create (*ast,
                                            decl_ctx,
                                            SourceLocation(),
                                            name && name[0] ? &ast->Idents.get(name) : NULL,
                                            SourceLocation(),
                                            NULL, 
                                            false,  // IsScoped
                                            false,  // IsScopedUsingClassTag
                                            false); // IsFixed
    
    
    if (enum_decl)
    {
        // TODO: check if we should be setting the promotion type too?
        enum_decl->setIntegerType(QualType::getFromOpaquePtr (integer_qual_type));
        
        enum_decl->setAccess(AS_public); // TODO respect what's in the debug info
        
        return ast->getTagDeclType(enum_decl).getAsOpaquePtr();
    }
    return NULL;
}

clang_type_t
ClangASTContext::GetEnumerationIntegerType (clang_type_t enum_clang_type)
{
    QualType enum_qual_type (QualType::getFromOpaquePtr(enum_clang_type));

    clang::Type *clang_type = enum_qual_type.getTypePtr();
    if (clang_type)
    {
        const EnumType *enum_type = dyn_cast<EnumType>(clang_type);
        if (enum_type)
        {
            EnumDecl *enum_decl = enum_type->getDecl();
            if (enum_decl)
                return enum_decl->getIntegerType().getAsOpaquePtr();
        }
    }
    return NULL;
}
bool
ClangASTContext::AddEnumerationValueToEnumerationType
(
    clang_type_t enum_clang_type,
    clang_type_t enumerator_clang_type,
    const Declaration &decl,
    const char *name,
    int64_t enum_value,
    uint32_t enum_value_bit_size
)
{
    if (enum_clang_type && enumerator_clang_type && name)
    {
        // TODO: Do something intelligent with the Declaration object passed in
        // like maybe filling in the SourceLocation with it...
        ASTContext *ast = getASTContext();
        IdentifierTable *identifier_table = getIdentifierTable();

        assert (ast != NULL);
        assert (identifier_table != NULL);
        QualType enum_qual_type (QualType::getFromOpaquePtr(enum_clang_type));

        clang::Type *clang_type = enum_qual_type.getTypePtr();
        if (clang_type)
        {
            const EnumType *enum_type = dyn_cast<EnumType>(clang_type);

            if (enum_type)
            {
                llvm::APSInt enum_llvm_apsint(enum_value_bit_size, false);
                enum_llvm_apsint = enum_value;
                EnumConstantDecl *enumerator_decl =
                    EnumConstantDecl::Create (*ast,
                                              enum_type->getDecl(),
                                              SourceLocation(),
                                              name ? &identifier_table->get(name) : NULL,    // Identifier
                                              QualType::getFromOpaquePtr(enumerator_clang_type),
                                              NULL,
                                              enum_llvm_apsint);
                
                if (enumerator_decl)
                {
                    enum_type->getDecl()->addDecl(enumerator_decl);
                    return true;
                }
            }
        }
    }
    return false;
}

#pragma mark Pointers & References

clang_type_t
ClangASTContext::CreatePointerType (clang_type_t clang_type)
{
    return CreatePointerType (getASTContext(), clang_type);
}

clang_type_t
ClangASTContext::CreatePointerType (clang::ASTContext *ast, clang_type_t clang_type)
{
    if (ast && clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            return ast->getObjCObjectPointerType(qual_type).getAsOpaquePtr();

        default:
            return ast->getPointerType(qual_type).getAsOpaquePtr();
        }
    }
    return NULL;
}

clang_type_t
ClangASTContext::CreateLValueReferenceType (clang::ASTContext *ast,
                                            clang_type_t clang_type)
{
    if (clang_type)
        return ast->getLValueReferenceType (QualType::getFromOpaquePtr(clang_type)).getAsOpaquePtr();
    return NULL;
}

clang_type_t
ClangASTContext::CreateRValueReferenceType (clang::ASTContext *ast,
                                            clang_type_t clang_type)
{
    if (clang_type)
        return ast->getRValueReferenceType (QualType::getFromOpaquePtr(clang_type)).getAsOpaquePtr();
    return NULL;
}

clang_type_t
ClangASTContext::CreateMemberPointerType (clang_type_t clang_pointee_type, clang_type_t clang_class_type)
{
    if (clang_pointee_type && clang_pointee_type)
        return getASTContext()->getMemberPointerType(QualType::getFromOpaquePtr(clang_pointee_type),
                                                     QualType::getFromOpaquePtr(clang_class_type).getTypePtr()).getAsOpaquePtr();
    return NULL;
}

size_t
ClangASTContext::GetPointerBitSize ()
{
    ASTContext *ast = getASTContext();
    return ast->getTypeSize(ast->VoidPtrTy);
}

bool
ClangASTContext::IsPointerOrReferenceType (clang_type_t clang_type, clang_type_t*target_type)
{
    if (clang_type == NULL)
        return false;

    QualType qual_type (QualType::getFromOpaquePtr(clang_type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Builtin:
        switch (cast<clang::BuiltinType>(qual_type)->getKind())
        {
        default:
            break;
        case clang::BuiltinType::ObjCId:
        case clang::BuiltinType::ObjCClass:
            return true;
        }
        return false;
    case clang::Type::ObjCObjectPointer:
        if (target_type)
            *target_type = cast<ObjCObjectPointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
        return true;
    case clang::Type::BlockPointer:
        if (target_type)
            *target_type = cast<BlockPointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
        return true;
    case clang::Type::Pointer:
        if (target_type)
            *target_type = cast<PointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
        return true;
    case clang::Type::MemberPointer:
        if (target_type)
            *target_type = cast<MemberPointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
        return true;
    case clang::Type::LValueReference:
        if (target_type)
            *target_type = cast<LValueReferenceType>(qual_type)->desugar().getAsOpaquePtr();
        return true;
    case clang::Type::RValueReference:
        if (target_type)
            *target_type = cast<LValueReferenceType>(qual_type)->desugar().getAsOpaquePtr();
        return true;
    case clang::Type::Typedef:
        return ClangASTContext::IsPointerOrReferenceType (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
    default:
        break;
    }
    return false;
}

bool
ClangASTContext::IsIntegerType (clang_type_t clang_type, bool &is_signed)
{
    if (!clang_type)
        return false;
    
    QualType qual_type (QualType::getFromOpaquePtr(clang_type));
    const BuiltinType *builtin_type = dyn_cast<BuiltinType>(qual_type->getCanonicalTypeInternal());
    
    if (builtin_type)
    {
        if (builtin_type->isInteger())
            is_signed = builtin_type->isSignedInteger();
        
        return true;
    }
    
    return false;
}

bool
ClangASTContext::IsPointerType (clang_type_t clang_type, clang_type_t*target_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::Builtin:
            switch (cast<clang::BuiltinType>(qual_type)->getKind())
            {
            default:
                break;
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
                return true;
            }
            return false;
        case clang::Type::ObjCObjectPointer:
            if (target_type)
                *target_type = cast<ObjCObjectPointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
            return true;
        case clang::Type::BlockPointer:
            if (target_type)
                *target_type = cast<BlockPointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
            return true;
        case clang::Type::Pointer:
            if (target_type)
                *target_type = cast<PointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
            return true;
        case clang::Type::MemberPointer:
            if (target_type)
                *target_type = cast<MemberPointerType>(qual_type)->getPointeeType().getAsOpaquePtr();
            return true;
        case clang::Type::Typedef:
            return ClangASTContext::IsPointerOrReferenceType (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), target_type);
        default:
            break;
        }
    }
    return false;
}

bool
ClangASTContext::IsFloatingPointType (clang_type_t clang_type, uint32_t &count, bool &is_complex)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));

        if (const BuiltinType *BT = dyn_cast<BuiltinType>(qual_type->getCanonicalTypeInternal()))
        {
            clang::BuiltinType::Kind kind = BT->getKind();
            if (kind >= BuiltinType::Float && kind <= BuiltinType::LongDouble)
            {
                count = 1;
                is_complex = false;
                return true;
            }
        }
        else if (const ComplexType *CT = dyn_cast<ComplexType>(qual_type->getCanonicalTypeInternal()))
        {
            if (IsFloatingPointType(CT->getElementType().getAsOpaquePtr(), count, is_complex))
            {
                count = 2;
                is_complex = true;
                return true;
            }
        }
        else if (const VectorType *VT = dyn_cast<VectorType>(qual_type->getCanonicalTypeInternal()))
        {
            if (IsFloatingPointType(VT->getElementType().getAsOpaquePtr(), count, is_complex))
            {
                count = VT->getNumElements();
                is_complex = false;
                return true;
            }
        }
    }
    return false;
}


bool
ClangASTContext::GetCXXClassName (clang_type_t clang_type, std::string &class_name)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        
        CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        if (cxx_record_decl)
        {
            class_name.assign (cxx_record_decl->getIdentifier()->getNameStart());
            return true;
        }
    }
    class_name.clear();
    return false;
}


bool
ClangASTContext::IsCXXClassType (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        if (qual_type->getAsCXXRecordDecl() != NULL)
            return true;
    }
    return false;
}

bool 
ClangASTContext::IsObjCClassType (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        if (qual_type->isObjCObjectOrInterfaceType())
            return true;
    }
    return false;
}


bool 
ClangASTContext::IsCharType (clang_type_t clang_type)
{
    if (clang_type)
        return QualType::getFromOpaquePtr(clang_type)->isCharType();
    return false;
}

bool
ClangASTContext::IsCStringType (clang_type_t clang_type, uint32_t &length)
{
    clang_type_t pointee_or_element_clang_type = NULL;
    Flags type_flags (ClangASTContext::GetTypeInfo (clang_type, NULL, &pointee_or_element_clang_type));
    
    if (pointee_or_element_clang_type == NULL)
        return false;

    if (type_flags.AnySet (eTypeIsArray | eTypeIsPointer))
    {
        QualType pointee_or_element_qual_type (QualType::getFromOpaquePtr (pointee_or_element_clang_type));
        
        if (pointee_or_element_qual_type.getUnqualifiedType()->isCharType())
        {
            QualType qual_type (QualType::getFromOpaquePtr(clang_type));
            if (type_flags.Test (eTypeIsArray))
            {
                // We know the size of the array and it could be a C string
                // since it is an array of characters
                length = cast<ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
                return true;
            }
            else
            {
                length = 0;
                return true;
            }

        }
    }
    return false;
}

bool
ClangASTContext::IsFunctionPointerType (clang_type_t clang_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        
        if (qual_type->isFunctionPointerType())
            return true;
    
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
        case clang::Type::Typedef:
            return ClangASTContext::IsFunctionPointerType (cast<TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());

        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                if (reference_type)
                    return ClangASTContext::IsFunctionPointerType (reference_type->getPointeeType().getAsOpaquePtr());
            }
            break;
        }
    }
    return false;
}

size_t
ClangASTContext::GetArraySize (clang_type_t clang_type)
{
    if (clang_type)
    {
        ConstantArrayType *array = cast<ConstantArrayType>(QualType::getFromOpaquePtr(clang_type).getTypePtr());
        if (array)
            return array->getSize().getLimitedValue();
    }
    return 0;
}

bool
ClangASTContext::IsArrayType (clang_type_t clang_type, clang_type_t*member_type, uint64_t *size)
{
    if (!clang_type)
        return false;
    
    QualType qual_type (QualType::getFromOpaquePtr(clang_type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::ConstantArray:
        if (member_type)
            *member_type = cast<ConstantArrayType>(qual_type)->getElementType().getAsOpaquePtr();
        if (size)
            *size = cast<ConstantArrayType>(qual_type)->getSize().getLimitedValue(ULONG_LONG_MAX);
        return true;
    case clang::Type::IncompleteArray:
        if (member_type)
            *member_type = cast<IncompleteArrayType>(qual_type)->getElementType().getAsOpaquePtr();
        if (size)
            *size = 0;
        return true;
    case clang::Type::VariableArray:
        if (member_type)
            *member_type = cast<VariableArrayType>(qual_type)->getElementType().getAsOpaquePtr();
        if (size)
            *size = 0;
    case clang::Type::DependentSizedArray:
        if (member_type)
            *member_type = cast<DependentSizedArrayType>(qual_type)->getElementType().getAsOpaquePtr();
        if (size)
            *size = 0;
        return true;
    }
    return false;
}


#pragma mark Typedefs

clang_type_t
ClangASTContext::CreateTypedefType (const char *name, clang_type_t clang_type, DeclContext *decl_ctx)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        ASTContext *ast = getASTContext();
        IdentifierTable *identifier_table = getIdentifierTable();
        assert (ast != NULL);
        assert (identifier_table != NULL);
        if (decl_ctx == NULL)
            decl_ctx = ast->getTranslationUnitDecl();
        TypedefDecl *decl = TypedefDecl::Create (*ast,
                                                 decl_ctx,
                                                 SourceLocation(),
                                                 name ? &identifier_table->get(name) : NULL, // Identifier
                                                 ast->CreateTypeSourceInfo(qual_type));
        
        decl->setAccess(AS_public); // TODO respect proper access specifier

        // Get a uniqued QualType for the typedef decl type
        return ast->getTypedefType (decl).getAsOpaquePtr();
    }
    return NULL;
}


std::string
ClangASTContext::GetTypeName (clang_type_t opaque_qual_type)
{
    std::string return_name;
    
    QualType qual_type(QualType::getFromOpaquePtr(opaque_qual_type));

    const TypedefType *typedef_type = qual_type->getAs<TypedefType>();
    if (typedef_type)
    {
        const TypedefDecl *typedef_decl = typedef_type->getDecl();
        return_name = typedef_decl->getQualifiedNameAsString();
    }
    else
    {
        return_name = qual_type.getAsString();
    }

    return return_name;
}

// Disable this for now since I can't seem to get a nicely formatted float
// out of the APFloat class without just getting the float, double or quad
// and then using a formatted print on it which defeats the purpose. We ideally
// would like to get perfect string values for any kind of float semantics
// so we can support remote targets. The code below also requires a patch to
// llvm::APInt.
//bool
//ClangASTContext::ConvertFloatValueToString (ASTContext *ast, clang_type_t clang_type, const uint8_t* bytes, size_t byte_size, int apint_byte_order, std::string &float_str)
//{
//  uint32_t count = 0;
//  bool is_complex = false;
//  if (ClangASTContext::IsFloatingPointType (clang_type, count, is_complex))
//  {
//      unsigned num_bytes_per_float = byte_size / count;
//      unsigned num_bits_per_float = num_bytes_per_float * 8;
//
//      float_str.clear();
//      uint32_t i;
//      for (i=0; i<count; i++)
//      {
//          APInt ap_int(num_bits_per_float, bytes + i * num_bytes_per_float, (APInt::ByteOrder)apint_byte_order);
//          bool is_ieee = false;
//          APFloat ap_float(ap_int, is_ieee);
//          char s[1024];
//          unsigned int hex_digits = 0;
//          bool upper_case = false;
//
//          if (ap_float.convertToHexString(s, hex_digits, upper_case, APFloat::rmNearestTiesToEven) > 0)
//          {
//              if (i > 0)
//                  float_str.append(", ");
//              float_str.append(s);
//              if (i == 1 && is_complex)
//                  float_str.append(1, 'i');
//          }
//      }
//      return !float_str.empty();
//  }
//  return false;
//}

size_t
ClangASTContext::ConvertStringToFloatValue (ASTContext *ast, clang_type_t clang_type, const char *s, uint8_t *dst, size_t dst_size)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        uint32_t count = 0;
        bool is_complex = false;
        if (ClangASTContext::IsFloatingPointType (clang_type, count, is_complex))
        {
            // TODO: handle complex and vector types
            if (count != 1)
                return false;

            StringRef s_sref(s);
            APFloat ap_float(ast->getFloatTypeSemantics(qual_type), s_sref);

            const uint64_t bit_size = ast->getTypeSize (qual_type);
            const uint64_t byte_size = bit_size / 8;
            if (dst_size >= byte_size)
            {
                if (bit_size == sizeof(float)*8)
                {
                    float float32 = ap_float.convertToFloat();
                    ::memcpy (dst, &float32, byte_size);
                    return byte_size;
                }
                else if (bit_size >= 64)
                {
                    llvm::APInt ap_int(ap_float.bitcastToAPInt());
                    ::memcpy (dst, ap_int.getRawData(), byte_size);
                    return byte_size;
                }
            }
        }
    }
    return 0;
}

unsigned
ClangASTContext::GetTypeQualifiers(clang_type_t clang_type)
{
    assert (clang_type);
    
    QualType qual_type (QualType::getFromOpaquePtr(clang_type));
    
    return qual_type.getQualifiers().getCVRQualifiers();
}

bool
ClangASTContext::GetCompleteType (clang::ASTContext *ast, lldb::clang_type_t clang_type)
{
    if (clang_type == NULL)
        return false;

    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(clang_type));
    return GetCompleteQualType (ast, qual_type);
}


bool
ClangASTContext::GetCompleteType (clang_type_t clang_type)
{   
    return ClangASTContext::GetCompleteType (getASTContext(), clang_type);
}

