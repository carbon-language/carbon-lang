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

#if !defined(NDEBUG) && !defined(LLVM_NDEBUG_OFF)
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
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

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Expression/ASTDumper.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

#include <stdio.h>

#include <mutex>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace clang;

clang::AccessSpecifier
ClangASTContext::ConvertAccessTypeToAccessSpecifier (AccessType access)
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
            case IK_LLVM_IR:
                assert (!"Invalid input kind!");
            case IK_OpenCL:
                LangStd = LangStandard::lang_opencl;
                break;
            case IK_CUDA:
                LangStd = LangStandard::lang_cuda;
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
    Opts.LineComment = Std.hasLineComments();
    Opts.C99 = Std.isC99();
    Opts.CPlusPlus = Std.isCPlusPlus();
    Opts.CPlusPlus11 = Std.isCPlusPlus11();
    Opts.Digraphs = Std.hasDigraphs();
    Opts.GNUMode = Std.isGNUMode();
    Opts.GNUInline = !Std.isC99();
    Opts.HexFloats = Std.hasHexFloats();
    Opts.ImplicitInt = Std.hasImplicitInt();
    
    Opts.WChar = true;

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
        Opts.setValueVisibilityMode(DefaultVisibility);
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
    Opts.NoInlineDefine = !Opt;

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
    m_diagnostics_engine_ap(),
    m_target_options_rp(),
    m_target_info_ap(),
    m_identifier_table_ap(),
    m_selector_table_ap(),
    m_builtins_ap(),
    m_callback_tag_decl (NULL),
    m_callback_objc_decl (NULL),
    m_callback_baton (NULL),
    m_pointer_byte_size (0)

{
    if (target_triple && target_triple[0])
        SetTargetTriple (target_triple);
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
    m_target_options_rp.reset();
    m_diagnostics_engine_ap.reset();
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
    m_diagnostics_engine_ap.reset();
    m_target_options_rp.reset();
    m_target_info_ap.reset();
    m_identifier_table_ap.reset();
    m_selector_table_ap.reset();
    m_builtins_ap.reset();
    m_pointer_byte_size = 0;
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

void
ClangASTContext::SetArchitecture (const ArchSpec &arch)
{
    SetTargetTriple(arch.GetTriple().str().c_str());
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
                                       getTargetInfo(),
                                       *getIdentifierTable(),
                                       *getSelectorTable(),
                                       *getBuiltinContext(),
                                       0));
        
        if ((m_callback_tag_decl || m_callback_objc_decl) && m_callback_baton)
        {
            m_ast_ap->getTranslationUnitDecl()->setHasExternalLexicalStorage();
            //m_ast_ap->getTranslationUnitDecl()->setHasExternalVisibleStorage();
        }
        
        m_ast_ap->getDiagnostics().setClient(getDiagnosticConsumer(), false);
    }
    return m_ast_ap.get();
}

Builtin::Context *
ClangASTContext::getBuiltinContext()
{
    if (m_builtins_ap.get() == NULL)
        m_builtins_ap.reset (new Builtin::Context());
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
        m_source_manager_ap.reset(new clang::SourceManager(*getDiagnosticsEngine(), *getFileManager()));
    return m_source_manager_ap.get();
}

clang::DiagnosticsEngine *
ClangASTContext::getDiagnosticsEngine()
{
    if (m_diagnostics_engine_ap.get() == NULL)
    {
        llvm::IntrusiveRefCntPtr<DiagnosticIDs> diag_id_sp(new DiagnosticIDs());
        m_diagnostics_engine_ap.reset(new DiagnosticsEngine(diag_id_sp, new DiagnosticOptions()));
    }
    return m_diagnostics_engine_ap.get();
}

class NullDiagnosticConsumer : public DiagnosticConsumer
{
public:
    NullDiagnosticConsumer ()
    {
        m_log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    }
    
    void HandleDiagnostic (DiagnosticsEngine::Level DiagLevel, const Diagnostic &info)
    {
        if (m_log)
        {
            llvm::SmallVector<char, 32> diag_str(10);
            info.FormatDiagnostic(diag_str);
            diag_str.push_back('\0');
            m_log->Printf("Compiler diagnostic: %s\n", diag_str.data());
        }
    }
    
    DiagnosticConsumer *clone (DiagnosticsEngine &Diags) const
    {
        return new NullDiagnosticConsumer ();
    }
private:
    Log * m_log;
};

DiagnosticConsumer *
ClangASTContext::getDiagnosticConsumer()
{
    if (m_diagnostic_consumer_ap.get() == NULL)
        m_diagnostic_consumer_ap.reset(new NullDiagnosticConsumer);
    
    return m_diagnostic_consumer_ap.get();
}

TargetOptions *
ClangASTContext::getTargetOptions()
{
    if (m_target_options_rp.getPtr() == NULL && !m_target_triple.empty())
    {
        m_target_options_rp.reset ();
        m_target_options_rp = new TargetOptions();
        if (m_target_options_rp.getPtr() != NULL)
            m_target_options_rp->Triple = m_target_triple;
    }
    return m_target_options_rp.getPtr();
}


TargetInfo *
ClangASTContext::getTargetInfo()
{
    // target_triple should be something like "x86_64-apple-macosx"
    if (m_target_info_ap.get() == NULL && !m_target_triple.empty())
        m_target_info_ap.reset (TargetInfo::CreateTargetInfo(*getDiagnosticsEngine(), getTargetOptions()));
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
ClangASTType
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (Encoding encoding, uint32_t bit_size)
{
    return ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (getASTContext(), encoding, bit_size);
}

ClangASTType
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (ASTContext *ast, Encoding encoding, uint32_t bit_size)
{
    if (!ast)
        return ClangASTType();
    
    switch (encoding)
    {
    case eEncodingInvalid:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
            return ClangASTType (ast, ast->VoidPtrTy.getAsOpaquePtr());
        break;
        
    case eEncodingUint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
            return ClangASTType (ast, ast->UnsignedCharTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
            return ClangASTType (ast, ast->UnsignedShortTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
            return ClangASTType (ast, ast->UnsignedIntTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
            return ClangASTType (ast, ast->UnsignedLongTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
            return ClangASTType (ast, ast->UnsignedLongLongTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
            return ClangASTType (ast, ast->UnsignedInt128Ty.getAsOpaquePtr());
        break;
        
    case eEncodingSint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
            return ClangASTType (ast, ast->CharTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
            return ClangASTType (ast, ast->ShortTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
            return ClangASTType (ast, ast->IntTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
            return ClangASTType (ast, ast->LongTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
            return ClangASTType (ast, ast->LongLongTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
            return ClangASTType (ast, ast->Int128Ty.getAsOpaquePtr());
        break;
        
    case eEncodingIEEE754:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
            return ClangASTType (ast, ast->FloatTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
            return ClangASTType (ast, ast->DoubleTy.getAsOpaquePtr());
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
            return ClangASTType (ast, ast->LongDoubleTy.getAsOpaquePtr());
        break;
        
    case eEncodingVector:
        // Sanity check that bit_size is a multiple of 8's.
        if (bit_size && !(bit_size & 0x7u))
            return ClangASTType (ast, ast->getExtVectorType (ast->UnsignedCharTy, bit_size/8).getAsOpaquePtr());
        break;
    }
    
    return ClangASTType();
}



lldb::BasicType
ClangASTContext::GetBasicTypeEnumeration (const ConstString &name)
{
    if (name)
    {
        typedef UniqueCStringMap<lldb::BasicType> TypeNameToBasicTypeMap;
        static TypeNameToBasicTypeMap g_type_map;
        static std::once_flag g_once_flag;
        std::call_once(g_once_flag, [](){
            // "void"
            g_type_map.Append(ConstString("void").GetCString(), eBasicTypeVoid);
            
            // "char"
            g_type_map.Append(ConstString("char").GetCString(), eBasicTypeChar);
            g_type_map.Append(ConstString("signed char").GetCString(), eBasicTypeSignedChar);
            g_type_map.Append(ConstString("unsigned char").GetCString(), eBasicTypeUnsignedChar);
            g_type_map.Append(ConstString("wchar_t").GetCString(), eBasicTypeWChar);
            g_type_map.Append(ConstString("signed wchar_t").GetCString(), eBasicTypeSignedWChar);
            g_type_map.Append(ConstString("unsigned wchar_t").GetCString(), eBasicTypeUnsignedWChar);
            // "short"
            g_type_map.Append(ConstString("short").GetCString(), eBasicTypeShort);
            g_type_map.Append(ConstString("short int").GetCString(), eBasicTypeShort);
            g_type_map.Append(ConstString("unsigned short").GetCString(), eBasicTypeUnsignedShort);
            g_type_map.Append(ConstString("unsigned short int").GetCString(), eBasicTypeUnsignedShort);
            
            // "int"
            g_type_map.Append(ConstString("int").GetCString(), eBasicTypeInt);
            g_type_map.Append(ConstString("signed int").GetCString(), eBasicTypeInt);
            g_type_map.Append(ConstString("unsigned int").GetCString(), eBasicTypeUnsignedInt);
            g_type_map.Append(ConstString("unsigned").GetCString(), eBasicTypeUnsignedInt);
            
            // "long"
            g_type_map.Append(ConstString("long").GetCString(), eBasicTypeLong);
            g_type_map.Append(ConstString("long int").GetCString(), eBasicTypeLong);
            g_type_map.Append(ConstString("unsigned long").GetCString(), eBasicTypeUnsignedLong);
            g_type_map.Append(ConstString("unsigned long int").GetCString(), eBasicTypeUnsignedLong);
            
            // "long long"
            g_type_map.Append(ConstString("long long").GetCString(), eBasicTypeLongLong);
            g_type_map.Append(ConstString("long long int").GetCString(), eBasicTypeLongLong);
            g_type_map.Append(ConstString("unsigned long long").GetCString(), eBasicTypeUnsignedLongLong);
            g_type_map.Append(ConstString("unsigned long long int").GetCString(), eBasicTypeUnsignedLongLong);
            
            // "int128"
            g_type_map.Append(ConstString("__int128_t").GetCString(), eBasicTypeInt128);
            g_type_map.Append(ConstString("__uint128_t").GetCString(), eBasicTypeUnsignedInt128);
            
            // Miscelaneous
            g_type_map.Append(ConstString("bool").GetCString(), eBasicTypeBool);
            g_type_map.Append(ConstString("float").GetCString(), eBasicTypeFloat);
            g_type_map.Append(ConstString("double").GetCString(), eBasicTypeDouble);
            g_type_map.Append(ConstString("long double").GetCString(), eBasicTypeLongDouble);
            g_type_map.Append(ConstString("id").GetCString(), eBasicTypeObjCID);
            g_type_map.Append(ConstString("SEL").GetCString(), eBasicTypeObjCSel);
            g_type_map.Append(ConstString("nullptr").GetCString(), eBasicTypeNullPtr);
            g_type_map.Sort();
        });
        
        return g_type_map.Find(name.GetCString(), eBasicTypeInvalid);
    }
    return eBasicTypeInvalid;
}

ClangASTType
ClangASTContext::GetBasicType (ASTContext *ast, const ConstString &name)
{
    if (ast)
    {
        lldb::BasicType basic_type = ClangASTContext::GetBasicTypeEnumeration (name);
        return ClangASTContext::GetBasicType (ast, basic_type);
    }
    return ClangASTType();
}

uint32_t
ClangASTContext::GetPointerByteSize ()
{
    if (m_pointer_byte_size == 0)
        m_pointer_byte_size = GetBasicType(lldb::eBasicTypeVoid).GetPointerType().GetByteSize();
    return m_pointer_byte_size;
}

ClangASTType
ClangASTContext::GetBasicType (lldb::BasicType basic_type)
{
    return GetBasicType (getASTContext(), basic_type);
}

ClangASTType
ClangASTContext::GetBasicType (ASTContext *ast, lldb::BasicType basic_type)
{
    if (ast)
    {
        clang_type_t clang_type = NULL;
        
        switch (basic_type)
        {
            case eBasicTypeInvalid:
            case eBasicTypeOther:
                break;
            case eBasicTypeVoid:
                clang_type = ast->VoidTy.getAsOpaquePtr();
                break;
            case eBasicTypeChar:
                clang_type = ast->CharTy.getAsOpaquePtr();
                break;
            case eBasicTypeSignedChar:
                clang_type = ast->SignedCharTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedChar:
                clang_type = ast->UnsignedCharTy.getAsOpaquePtr();
                break;
            case eBasicTypeWChar:
                clang_type = ast->getWCharType().getAsOpaquePtr();
                break;
            case eBasicTypeSignedWChar:
                clang_type = ast->getSignedWCharType().getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedWChar:
                clang_type = ast->getUnsignedWCharType().getAsOpaquePtr();
                break;
            case eBasicTypeChar16:
                clang_type = ast->Char16Ty.getAsOpaquePtr();
                break;
            case eBasicTypeChar32:
                clang_type = ast->Char32Ty.getAsOpaquePtr();
                break;
            case eBasicTypeShort:
                clang_type = ast->ShortTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedShort:
                clang_type = ast->UnsignedShortTy.getAsOpaquePtr();
                break;
            case eBasicTypeInt:
                clang_type = ast->IntTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedInt:
                clang_type = ast->UnsignedIntTy.getAsOpaquePtr();
                break;
            case eBasicTypeLong:
                clang_type = ast->LongTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedLong:
                clang_type = ast->UnsignedLongTy.getAsOpaquePtr();
                break;
            case eBasicTypeLongLong:
                clang_type = ast->LongLongTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedLongLong:
                clang_type = ast->UnsignedLongLongTy.getAsOpaquePtr();
                break;
            case eBasicTypeInt128:
                clang_type = ast->Int128Ty.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedInt128:
                clang_type = ast->UnsignedInt128Ty.getAsOpaquePtr();
                break;
            case eBasicTypeBool:
                clang_type = ast->BoolTy.getAsOpaquePtr();
                break;
            case eBasicTypeHalf:
                clang_type = ast->HalfTy.getAsOpaquePtr();
                break;
            case eBasicTypeFloat:
                clang_type = ast->FloatTy.getAsOpaquePtr();
                break;
            case eBasicTypeDouble:
                clang_type = ast->DoubleTy.getAsOpaquePtr();
                break;
            case eBasicTypeLongDouble:
                clang_type = ast->LongDoubleTy.getAsOpaquePtr();
                break;
            case eBasicTypeFloatComplex:
                clang_type = ast->FloatComplexTy.getAsOpaquePtr();
                break;
            case eBasicTypeDoubleComplex:
                clang_type = ast->DoubleComplexTy.getAsOpaquePtr();
                break;
            case eBasicTypeLongDoubleComplex:
                clang_type = ast->LongDoubleComplexTy.getAsOpaquePtr();
                break;
            case eBasicTypeObjCID:
                clang_type = ast->getObjCIdType().getAsOpaquePtr();
                break;
            case eBasicTypeObjCClass:
                clang_type = ast->getObjCClassType().getAsOpaquePtr();
                break;
            case eBasicTypeObjCSel:
                clang_type = ast->getObjCSelType().getAsOpaquePtr();
                break;
            case eBasicTypeNullPtr:
                clang_type = ast->NullPtrTy.getAsOpaquePtr();
                break;
        }
        
        if (clang_type)
            return ClangASTType (ast, clang_type);
    }
    return ClangASTType();
}


ClangASTType
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
                    return ClangASTType (ast, ast->VoidPtrTy.getAsOpaquePtr());
                break;
                
            case DW_ATE_boolean:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->BoolTy))
                    return ClangASTType (ast, ast->BoolTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return ClangASTType (ast, ast->UnsignedCharTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return ClangASTType (ast, ast->UnsignedShortTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                    return ClangASTType (ast, ast->UnsignedIntTy.getAsOpaquePtr());
                break;
                
            case DW_ATE_lo_user:
                // This has been seen to mean DW_AT_complex_integer
                if (type_name)
                {
                    if (::strstr(type_name, "complex"))
                    {
                        ClangASTType complex_int_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("int", DW_ATE_signed, bit_size/2);
                        return ClangASTType (ast, ast->getComplexType (complex_int_clang_type.GetQualType()).getAsOpaquePtr());
                    }
                }
                break;
                
            case DW_ATE_complex_float:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatComplexTy))
                    return ClangASTType (ast, ast->FloatComplexTy.getAsOpaquePtr());
                else if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleComplexTy))
                    return ClangASTType (ast, ast->DoubleComplexTy.getAsOpaquePtr());
                else if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleComplexTy))
                    return ClangASTType (ast, ast->LongDoubleComplexTy.getAsOpaquePtr());
                else 
                {
                    ClangASTType complex_float_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("float", DW_ATE_float, bit_size/2);
                    return ClangASTType (ast, ast->getComplexType (complex_float_clang_type.GetQualType()).getAsOpaquePtr());
                }
                break;
                
            case DW_ATE_float:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
                    return ClangASTType (ast, ast->FloatTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
                    return ClangASTType (ast, ast->DoubleTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
                    return ClangASTType (ast, ast->LongDoubleTy.getAsOpaquePtr());
                break;
                
            case DW_ATE_signed:
                if (type_name)
                {
                    if (streq(type_name, "wchar_t") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->WCharTy))
                        return ClangASTType (ast, ast->WCharTy.getAsOpaquePtr());
                    if (streq(type_name, "void") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->VoidTy))
                        return ClangASTType (ast, ast->VoidTy.getAsOpaquePtr());
                    if (strstr(type_name, "long long") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                        return ClangASTType (ast, ast->LongLongTy.getAsOpaquePtr());
                    if (strstr(type_name, "long") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                        return ClangASTType (ast, ast->LongTy.getAsOpaquePtr());
                    if (strstr(type_name, "short") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                        return ClangASTType (ast, ast->ShortTy.getAsOpaquePtr());
                    if (strstr(type_name, "char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                            return ClangASTType (ast, ast->CharTy.getAsOpaquePtr());
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                            return ClangASTType (ast, ast->SignedCharTy.getAsOpaquePtr());
                    }
                    if (strstr(type_name, "int"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                            return ClangASTType (ast, ast->IntTy.getAsOpaquePtr());
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                            return ClangASTType (ast, ast->Int128Ty.getAsOpaquePtr());
                    }
                }
                // We weren't able to match up a type name, just search by size
                if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                    return ClangASTType (ast, ast->CharTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                    return ClangASTType (ast, ast->ShortTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                    return ClangASTType (ast, ast->IntTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                    return ClangASTType (ast, ast->LongTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                    return ClangASTType (ast, ast->LongLongTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                    return ClangASTType (ast, ast->Int128Ty.getAsOpaquePtr());
                break;
                
            case DW_ATE_signed_char:
                if (type_name)
                {
                    if (streq(type_name, "signed char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                            return ClangASTType (ast, ast->SignedCharTy.getAsOpaquePtr());
                    }
                }
                if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                    return ClangASTType (ast, ast->CharTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                    return ClangASTType (ast, ast->SignedCharTy.getAsOpaquePtr());
                break;
                
            case DW_ATE_unsigned:
                if (type_name)
                {
                    if (strstr(type_name, "long long"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                            return ClangASTType (ast, ast->UnsignedLongLongTy.getAsOpaquePtr());
                    }
                    else if (strstr(type_name, "long"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                            return ClangASTType (ast, ast->UnsignedLongTy.getAsOpaquePtr());
                    }
                    else if (strstr(type_name, "short"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                            return ClangASTType (ast, ast->UnsignedShortTy.getAsOpaquePtr());
                    }
                    else if (strstr(type_name, "char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                            return ClangASTType (ast, ast->UnsignedCharTy.getAsOpaquePtr());
                    }
                    else if (strstr(type_name, "int"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                            return ClangASTType (ast, ast->UnsignedIntTy.getAsOpaquePtr());
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                            return ClangASTType (ast, ast->UnsignedInt128Ty.getAsOpaquePtr());
                    }
                }
                // We weren't able to match up a type name, just search by size
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return ClangASTType (ast, ast->UnsignedCharTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return ClangASTType (ast, ast->UnsignedShortTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                    return ClangASTType (ast, ast->UnsignedIntTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                    return ClangASTType (ast, ast->UnsignedLongTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                    return ClangASTType (ast, ast->UnsignedLongLongTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                    return ClangASTType (ast, ast->UnsignedInt128Ty.getAsOpaquePtr());
                break;
                
            case DW_ATE_unsigned_char:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return ClangASTType (ast, ast->UnsignedCharTy.getAsOpaquePtr());
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return ClangASTType (ast, ast->UnsignedShortTy.getAsOpaquePtr());
                break;
                
            case DW_ATE_imaginary_float:
                break;
                
            case DW_ATE_UTF:
                if (type_name)
                {
                    if (streq(type_name, "char16_t"))
                    {
                        return ClangASTType (ast, ast->Char16Ty.getAsOpaquePtr());
                    }
                    else if (streq(type_name, "char32_t"))
                    {
                        return ClangASTType (ast, ast->Char32Ty.getAsOpaquePtr());
                    }
                }
                break;
        }
    }
    // This assert should fire for anything that we don't catch above so we know
    // to fix any issues we run into.
    if (type_name)
    {
        Host::SystemLog (Host::eSystemLogError, "error: need to add support for DW_TAG_base_type '%s' encoded with DW_ATE = 0x%x, bit_size = %u\n", type_name, dw_ate, bit_size);
    }
    else
    {
        Host::SystemLog (Host::eSystemLogError, "error: need to add support for DW_TAG_base_type encoded with DW_ATE = 0x%x, bit_size = %u\n", dw_ate, bit_size);
    }
    return ClangASTType ();
}

ClangASTType
ClangASTContext::GetUnknownAnyType(clang::ASTContext *ast)
{
    if (ast)
        return ClangASTType (ast, ast->UnknownAnyTy.getAsOpaquePtr());
    return ClangASTType();
}

ClangASTType
ClangASTContext::GetCStringType (bool is_const)
{
    ASTContext *ast = getASTContext();
    QualType char_type(ast->CharTy);
    
    if (is_const)
        char_type.addConst();
    
    return ClangASTType (ast, ast->getPointerType(char_type).getAsOpaquePtr());
}

clang::DeclContext *
ClangASTContext::GetTranslationUnitDecl (clang::ASTContext *ast)
{
    return ast->getTranslationUnitDecl();
}

ClangASTType
ClangASTContext::CopyType (ASTContext *dst_ast, 
                           ClangASTType src)
{
    FileSystemOptions file_system_options;
    ASTContext *src_ast = src.GetASTContext();
    FileManager file_manager (file_system_options);
    ASTImporter importer(*dst_ast, file_manager,
                         *src_ast, file_manager,
                         false);
    
    QualType dst (importer.Import(src.GetQualType()));
    
    return ClangASTType (dst_ast, dst.getAsOpaquePtr());
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
ClangASTContext::AreTypesSame (ClangASTType type1,
                               ClangASTType type2,
                               bool ignore_qualifiers)
{
    ASTContext *ast = type1.GetASTContext();
    if (ast != type2.GetASTContext())
        return false;

    if (type1.GetOpaqueQualType() == type2.GetOpaqueQualType())
        return true;

    QualType type1_qual = type1.GetQualType();
    QualType type2_qual = type2.GetQualType();
    
    if (ignore_qualifiers)
    {
        type1_qual = type1_qual.getUnqualifiedType();
        type2_qual = type2_qual.getUnqualifiedType();
    }
    
    return ast->hasSameType (type1_qual, type2_qual);
}


ClangASTType
ClangASTContext::GetTypeForDecl (TagDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    ASTContext *ast = m_ast_ap.get();
    if (ast)
        return ClangASTType (ast, ast->getTagDeclType(decl).getAsOpaquePtr());
    return ClangASTType();
}

ClangASTType
ClangASTContext::GetTypeForDecl (ObjCInterfaceDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    ASTContext *ast = m_ast_ap.get();
    if (ast)
        return ClangASTType (ast, ast->getObjCInterfaceType(decl).getAsOpaquePtr());
    return ClangASTType();
}

#pragma mark Structure, Unions, Classes

ClangASTType
ClangASTContext::CreateRecordType (DeclContext *decl_ctx,
                                   AccessType access_type,
                                   const char *name,
                                   int kind,
                                   LanguageType language,
                                   ClangASTMetadata *metadata)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);
     
    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();


    if (language == eLanguageTypeObjC || language == eLanguageTypeObjC_plus_plus)
    {
        bool isForwardDecl = true;
        bool isInternal = false;
        return CreateObjCClass (name, decl_ctx, isForwardDecl, isInternal, metadata);
    }

    // NOTE: Eventually CXXRecordDecl will be merged back into RecordDecl and
    // we will need to update this code. I was told to currently always use
    // the CXXRecordDecl class since we often don't know from debug information
    // if something is struct or a class, so we default to always use the more
    // complete definition just in case.
    
    bool is_anonymous = (!name) || (!name[0]);
    
    CXXRecordDecl *decl = CXXRecordDecl::Create (*ast,
                                                 (TagDecl::TagKind)kind,
                                                 decl_ctx,
                                                 SourceLocation(),
                                                 SourceLocation(),
                                                 is_anonymous ? NULL : &ast->Idents.get(name));
    
    if (is_anonymous)
        decl->setAnonymousStructOrUnion(true);
    
    if (decl)
    {
        if (metadata)
            SetMetadata(ast, decl, *metadata);

        if (access_type != eAccessNone)
            decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));
    
        if (decl_ctx)
            decl_ctx->addDecl (decl);

        return ClangASTType(ast, ast->getTagDeclType(decl).getAsOpaquePtr());
    }
    return ClangASTType();
}

static TemplateParameterList *
CreateTemplateParameterList (ASTContext *ast, 
                             const ClangASTContext::TemplateParameterInfos &template_param_infos,
                             llvm::SmallVector<NamedDecl *, 8> &template_param_decls)
{
    const bool parameter_pack = false;
    const bool is_typename = false;
    const unsigned depth = 0;
    const size_t num_template_params = template_param_infos.GetSize();
    for (size_t i=0; i<num_template_params; ++i)
    {
        const char *name = template_param_infos.names[i];
        
        IdentifierInfo *identifier_info = NULL;
        if (name && name[0])
            identifier_info = &ast->Idents.get(name);
        if (template_param_infos.args[i].getKind() == TemplateArgument::Integral)
        {
            template_param_decls.push_back (NonTypeTemplateParmDecl::Create (*ast,
                                                                             ast->getTranslationUnitDecl(), // Is this the right decl context?, SourceLocation StartLoc,
                                                                             SourceLocation(), 
                                                                             SourceLocation(), 
                                                                             depth, 
                                                                             i,
                                                                             identifier_info,
                                                                             template_param_infos.args[i].getIntegralType(), 
                                                                             parameter_pack, 
                                                                             NULL));
            
        }
        else
        {
            template_param_decls.push_back (TemplateTypeParmDecl::Create (*ast, 
                                                                          ast->getTranslationUnitDecl(), // Is this the right decl context?
                                                                          SourceLocation(),
                                                                          SourceLocation(),
                                                                          depth, 
                                                                          i,
                                                                          identifier_info,
                                                                          is_typename,
                                                                          parameter_pack));
        }
    }

    TemplateParameterList *template_param_list = TemplateParameterList::Create (*ast,
                                                                                SourceLocation(),
                                                                                SourceLocation(),
                                                                                &template_param_decls.front(),
                                                                                template_param_decls.size(),
                                                                                SourceLocation());
    return template_param_list;
}

clang::FunctionTemplateDecl *
ClangASTContext::CreateFunctionTemplateDecl (clang::DeclContext *decl_ctx,
                                             clang::FunctionDecl *func_decl,
                                             const char *name, 
                                             const TemplateParameterInfos &template_param_infos)
{
//    /// \brief Create a function template node.
    ASTContext *ast = getASTContext();
    
    llvm::SmallVector<NamedDecl *, 8> template_param_decls;

    TemplateParameterList *template_param_list = CreateTemplateParameterList (ast,
                                                                              template_param_infos, 
                                                                              template_param_decls);
    FunctionTemplateDecl *func_tmpl_decl = FunctionTemplateDecl::Create (*ast,
                                                                         decl_ctx,
                                                                         func_decl->getLocation(),
                                                                         func_decl->getDeclName(),
                                                                         template_param_list,
                                                                         func_decl);
    
    for (size_t i=0, template_param_decl_count = template_param_decls.size();
         i < template_param_decl_count;
         ++i)
    {
        // TODO: verify which decl context we should put template_param_decls into..
        template_param_decls[i]->setDeclContext (func_decl); 
    }

    return func_tmpl_decl;
}

void
ClangASTContext::CreateFunctionTemplateSpecializationInfo (FunctionDecl *func_decl, 
                                                           clang::FunctionTemplateDecl *func_tmpl_decl,
                                                           const TemplateParameterInfos &infos)
{
    TemplateArgumentList template_args (TemplateArgumentList::OnStack,
                                        infos.args.data(), 
                                        infos.args.size());

    func_decl->setFunctionTemplateSpecialization (func_tmpl_decl,
                                                  &template_args,
                                                  NULL);
}


ClassTemplateDecl *
ClangASTContext::CreateClassTemplateDecl (DeclContext *decl_ctx,
                                          lldb::AccessType access_type,
                                          const char *class_name, 
                                          int kind, 
                                          const TemplateParameterInfos &template_param_infos)
{
    ASTContext *ast = getASTContext();
    
    ClassTemplateDecl *class_template_decl = NULL;
    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();
    
    IdentifierInfo &identifier_info = ast->Idents.get(class_name);
    DeclarationName decl_name (&identifier_info);

    clang::DeclContext::lookup_result result = decl_ctx->lookup(decl_name);
    
    for (NamedDecl *decl : result)
    {
        class_template_decl = dyn_cast<clang::ClassTemplateDecl>(decl);
        if (class_template_decl)
            return class_template_decl;
    }

    llvm::SmallVector<NamedDecl *, 8> template_param_decls;

    TemplateParameterList *template_param_list = CreateTemplateParameterList (ast,
                                                                              template_param_infos, 
                                                                              template_param_decls);

    CXXRecordDecl *template_cxx_decl = CXXRecordDecl::Create (*ast,
                                                              (TagDecl::TagKind)kind,
                                                              decl_ctx,  // What decl context do we use here? TU? The actual decl context?
                                                              SourceLocation(),
                                                              SourceLocation(),
                                                              &identifier_info);

    for (size_t i=0, template_param_decl_count = template_param_decls.size();
         i < template_param_decl_count;
         ++i)
    {
        template_param_decls[i]->setDeclContext (template_cxx_decl);
    }

    // With templated classes, we say that a class is templated with
    // specializations, but that the bare class has no functions.
    //template_cxx_decl->startDefinition();
    //template_cxx_decl->completeDefinition();
    
    class_template_decl = ClassTemplateDecl::Create (*ast,
                                                     decl_ctx,  // What decl context do we use here? TU? The actual decl context?
                                                     SourceLocation(),
                                                     decl_name,
                                                     template_param_list,
                                                     template_cxx_decl,
                                                     NULL);
    
    if (class_template_decl)
    {
        if (access_type != eAccessNone)
            class_template_decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));
        
        //if (TagDecl *ctx_tag_decl = dyn_cast<TagDecl>(decl_ctx))
        //    CompleteTagDeclarationDefinition(GetTypeForDecl(ctx_tag_decl));
        
        decl_ctx->addDecl (class_template_decl);
        
#ifdef LLDB_CONFIGURATION_DEBUG
        VerifyDecl(class_template_decl);
#endif
    }

    return class_template_decl;
}


ClassTemplateSpecializationDecl *
ClangASTContext::CreateClassTemplateSpecializationDecl (DeclContext *decl_ctx,
                                                        ClassTemplateDecl *class_template_decl,
                                                        int kind,
                                                        const TemplateParameterInfos &template_param_infos)
{
    ASTContext *ast = getASTContext();
    ClassTemplateSpecializationDecl *class_template_specialization_decl = ClassTemplateSpecializationDecl::Create (*ast, 
                                                                                                                   (TagDecl::TagKind)kind,
                                                                                                                   decl_ctx,
                                                                                                                   SourceLocation(), 
                                                                                                                   SourceLocation(),
                                                                                                                   class_template_decl,
                                                                                                                   &template_param_infos.args.front(),
                                                                                                                   template_param_infos.args.size(),
                                                                                                                   NULL);
    
    class_template_specialization_decl->setSpecializationKind(TSK_ExplicitSpecialization);
    
    return class_template_specialization_decl;
}

ClangASTType
ClangASTContext::CreateClassTemplateSpecializationType (ClassTemplateSpecializationDecl *class_template_specialization_decl)
{
    if (class_template_specialization_decl)
    {
        ASTContext *ast = getASTContext();
        if (ast)
            return ClangASTType(ast, ast->getTagDeclType(class_template_specialization_decl).getAsOpaquePtr());
    }
    return ClangASTType();
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

static inline bool
check_op_param (uint32_t op_kind, bool unary, bool binary, uint32_t num_params)
{
    // Special-case call since it can take any number of operands
    if(op_kind == OO_Call)
        return true;
    
    // The parameter count doens't include "this"
    if (num_params == 0)
        return unary;
    if (num_params == 1)
        return binary;
    else 
    return false;
}

bool
ClangASTContext::CheckOverloadedOperatorKindParameterCount (uint32_t op_kind, uint32_t num_params)
{
    switch (op_kind)
    {
    default:
        break;
    // C++ standard allows any number of arguments to new/delete
    case OO_New:
    case OO_Array_New:
    case OO_Delete:
    case OO_Array_Delete:
        return true;
    }
    
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) case OO_##Name: return check_op_param (op_kind, Unary, Binary, num_params);
    switch (op_kind)
    {
#include "clang/Basic/OperatorKinds.def"
        default: break;
    }
    return false;
}

clang::AccessSpecifier
ClangASTContext::UnifyAccessSpecifiers (clang::AccessSpecifier lhs, clang::AccessSpecifier rhs)
{
    clang::AccessSpecifier ret = lhs;
    
    // Make the access equal to the stricter of the field and the nested field's access
    switch (ret)
    {
        case clang::AS_none:
            break;
        case clang::AS_private:
            break;
        case clang::AS_protected:
            if (rhs == AS_private)
                ret = AS_private;
            break;
        case clang::AS_public:
            ret = rhs;
            break;
    }
    
    return ret;
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

#pragma mark Objective C Classes

ClangASTType
ClangASTContext::CreateObjCClass
(
    const char *name, 
    DeclContext *decl_ctx, 
    bool isForwardDecl, 
    bool isInternal,
    ClangASTMetadata *metadata
)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);
    assert (name && name[0]);
    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();

    ObjCInterfaceDecl *decl = ObjCInterfaceDecl::Create (*ast,
                                                         decl_ctx,
                                                         SourceLocation(),
                                                         &ast->Idents.get(name),
                                                         NULL,
                                                         SourceLocation(),
                                                         /*isForwardDecl,*/
                                                         isInternal);
    
    if (decl && metadata)
        SetMetadata(ast, decl, *metadata);
    
    return ClangASTType (ast, ast->getObjCInterfaceType(decl));
}

static inline bool
BaseSpecifierIsEmpty (const CXXBaseSpecifier *b)
{
    return ClangASTContext::RecordHasFields(b->getType()->getAsCXXRecordDecl()) == false;
}

uint32_t
ClangASTContext::GetNumBaseClasses (const CXXRecordDecl *cxx_record_decl, bool omit_empty_base_classes)
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


#pragma mark Namespace Declarations

NamespaceDecl *
ClangASTContext::GetUniqueNamespaceDeclaration (const char *name, DeclContext *decl_ctx)
{
    NamespaceDecl *namespace_decl = NULL;
    ASTContext *ast = getASTContext();
    TranslationUnitDecl *translation_unit_decl = ast->getTranslationUnitDecl ();
    if (decl_ctx == NULL)
        decl_ctx = translation_unit_decl;
    
    if (name)
    {
        IdentifierInfo &identifier_info = ast->Idents.get(name);
        DeclarationName decl_name (&identifier_info);
        clang::DeclContext::lookup_result result = decl_ctx->lookup(decl_name);
        for (NamedDecl *decl : result)
        {
            namespace_decl = dyn_cast<clang::NamespaceDecl>(decl);
            if (namespace_decl)
                return namespace_decl;
        }

        namespace_decl = NamespaceDecl::Create(*ast, 
                                               decl_ctx, 
                                               false, 
                                               SourceLocation(), 
                                               SourceLocation(),
                                               &identifier_info,
                                               NULL);
        
        decl_ctx->addDecl (namespace_decl);        
    }
    else
    {
        if (decl_ctx == translation_unit_decl)
        {
            namespace_decl = translation_unit_decl->getAnonymousNamespace();
            if (namespace_decl)
                return namespace_decl;
            
            namespace_decl = NamespaceDecl::Create(*ast, 
                                                   decl_ctx,
                                                   false,
                                                   SourceLocation(),
                                                   SourceLocation(),
                                                   NULL,
                                                   NULL);
            translation_unit_decl->setAnonymousNamespace (namespace_decl);
            translation_unit_decl->addDecl (namespace_decl);
            assert (namespace_decl == translation_unit_decl->getAnonymousNamespace());
        }
        else
        {
            NamespaceDecl *parent_namespace_decl = cast<NamespaceDecl>(decl_ctx);
            if (parent_namespace_decl)
            {
                namespace_decl = parent_namespace_decl->getAnonymousNamespace();
                if (namespace_decl)
                    return namespace_decl;
                namespace_decl = NamespaceDecl::Create(*ast, 
                                                       decl_ctx, 
                                                       false,
                                                       SourceLocation(), 
                                                       SourceLocation(), 
                                                       NULL,
                                                       NULL);
                parent_namespace_decl->setAnonymousNamespace (namespace_decl);
                parent_namespace_decl->addDecl (namespace_decl);
                assert (namespace_decl == parent_namespace_decl->getAnonymousNamespace());
            }
            else
            {
                // BAD!!!
            }
        }
        

        if (namespace_decl)
        {
            // If we make it here, we are creating the anonymous namespace decl
            // for the first time, so we need to do the using directive magic
            // like SEMA does
            UsingDirectiveDecl* using_directive_decl = UsingDirectiveDecl::Create (*ast, 
                                                                                   decl_ctx, 
                                                                                   SourceLocation(),
                                                                                   SourceLocation(),
                                                                                   NestedNameSpecifierLoc(),
                                                                                   SourceLocation(),
                                                                                   namespace_decl,
                                                                                   decl_ctx);
            using_directive_decl->setImplicit();
            decl_ctx->addDecl(using_directive_decl);
        }
    }
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(namespace_decl);
#endif
    return namespace_decl;
}


#pragma mark Function Types

FunctionDecl *
ClangASTContext::CreateFunctionDeclaration (DeclContext *decl_ctx,
                                            const char *name,
                                            const ClangASTType &function_clang_type,
                                            int storage,
                                            bool is_inline)
{
    FunctionDecl *func_decl = NULL;
    ASTContext *ast = getASTContext();
    if (decl_ctx == NULL)
        decl_ctx = ast->getTranslationUnitDecl();

    
    const bool hasWrittenPrototype = true;
    const bool isConstexprSpecified = false;

    if (name && name[0])
    {
        func_decl = FunctionDecl::Create (*ast,
                                          decl_ctx,
                                          SourceLocation(),
                                          SourceLocation(),
                                          DeclarationName (&ast->Idents.get(name)),
                                          function_clang_type.GetQualType(),
                                          NULL,
                                          (FunctionDecl::StorageClass)storage,
                                          is_inline,
                                          hasWrittenPrototype,
                                          isConstexprSpecified);
    }
    else
    {
        func_decl = FunctionDecl::Create (*ast,
                                          decl_ctx,
                                          SourceLocation(),
                                          SourceLocation(),
                                          DeclarationName (),
                                          function_clang_type.GetQualType(),
                                          NULL,
                                          (FunctionDecl::StorageClass)storage,
                                          is_inline,
                                          hasWrittenPrototype,
                                          isConstexprSpecified);
    }
    if (func_decl)
        decl_ctx->addDecl (func_decl);
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(func_decl);
#endif
    
    return func_decl;
}

ClangASTType
ClangASTContext::CreateFunctionType (ASTContext *ast,
                                     const ClangASTType& result_type,
                                     const ClangASTType *args,
                                     unsigned num_args, 
                                     bool is_variadic, 
                                     unsigned type_quals)
{
    assert (ast != NULL);
    std::vector<QualType> qual_type_args;
    for (unsigned i=0; i<num_args; ++i)
        qual_type_args.push_back (args[i].GetQualType());

    // TODO: Detect calling convention in DWARF?
    FunctionProtoType::ExtProtoInfo proto_info;
    proto_info.Variadic = is_variadic;
    proto_info.ExceptionSpecType = EST_None;
    proto_info.TypeQuals = type_quals;
    proto_info.RefQualifier = RQ_None;
    proto_info.NumExceptions = 0;
    proto_info.Exceptions = NULL;
    
    return ClangASTType (ast, ast->getFunctionType (result_type.GetQualType(),
                                                    qual_type_args,
                                                    proto_info).getAsOpaquePtr());
}

ParmVarDecl *
ClangASTContext::CreateParameterDeclaration (const char *name, const ClangASTType &param_type, int storage)
{
    ASTContext *ast = getASTContext();
    assert (ast != NULL);
    return ParmVarDecl::Create(*ast,
                                ast->getTranslationUnitDecl(),
                                SourceLocation(),
                                SourceLocation(),
                                name && name[0] ? &ast->Idents.get(name) : NULL,
                                param_type.GetQualType(),
                                NULL,
                                (VarDecl::StorageClass)storage,
                                0);
}

void
ClangASTContext::SetFunctionParameters (FunctionDecl *function_decl, ParmVarDecl **params, unsigned num_params)
{
    if (function_decl)
        function_decl->setParams (ArrayRef<ParmVarDecl*>(params, num_params));
}


#pragma mark Array Types

ClangASTType
ClangASTContext::CreateArrayType (const ClangASTType &element_type,
                                  size_t element_count,
                                  bool is_vector)
{
    if (element_type.IsValid())
    {
        ASTContext *ast = getASTContext();
        assert (ast != NULL);

        if (is_vector)
        {
            return ClangASTType (ast, ast->getExtVectorType(element_type.GetQualType(), element_count).getAsOpaquePtr());
        }
        else
        {
        
            llvm::APInt ap_element_count (64, element_count);
            if (element_count == 0)
            {
                return ClangASTType (ast, ast->getIncompleteArrayType (element_type.GetQualType(),
                                                                       ArrayType::Normal,
                                                                       0).getAsOpaquePtr());
            }
            else
            {
                return ClangASTType (ast, ast->getConstantArrayType (element_type.GetQualType(),
                                                                     ap_element_count,
                                                                     ArrayType::Normal,
                                                                     0).getAsOpaquePtr());
            }
        }
    }
    return ClangASTType();
}



#pragma mark Enumeration Types

ClangASTType
ClangASTContext::CreateEnumerationType 
(
    const char *name, 
    DeclContext *decl_ctx, 
    const Declaration &decl, 
    const ClangASTType &integer_clang_type
)
{
    // TODO: Do something intelligent with the Declaration object passed in
    // like maybe filling in the SourceLocation with it...
    ASTContext *ast = getASTContext();

    // TODO: ask about these...
//    const bool IsScoped = false;
//    const bool IsFixed = false;

    EnumDecl *enum_decl = EnumDecl::Create (*ast,
                                            decl_ctx,
                                            SourceLocation(),
                                            SourceLocation(),
                                            name && name[0] ? &ast->Idents.get(name) : NULL,
                                            NULL, 
                                            false,  // IsScoped
                                            false,  // IsScopedUsingClassTag
                                            false); // IsFixed
    
    
    if (enum_decl)
    {
        // TODO: check if we should be setting the promotion type too?
        enum_decl->setIntegerType(integer_clang_type.GetQualType());
        
        enum_decl->setAccess(AS_public); // TODO respect what's in the debug info
        
        return ClangASTType (ast, ast->getTagDeclType(enum_decl).getAsOpaquePtr());
    }
    return ClangASTType();
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


ClangASTType
ClangASTContext::GetFloatTypeFromBitSize (clang::ASTContext *ast,
                                          size_t bit_size)
{
    if (ast)
    {
        if (bit_size == ast->getTypeSize(ast->FloatTy))
            return ClangASTType(ast, ast->FloatTy.getAsOpaquePtr());
        else if (bit_size == ast->getTypeSize(ast->DoubleTy))
            return ClangASTType(ast, ast->DoubleTy.getAsOpaquePtr());
        else if (bit_size == ast->getTypeSize(ast->LongDoubleTy))
            return ClangASTType(ast, ast->LongDoubleTy.getAsOpaquePtr());
        else if (bit_size == ast->getTypeSize(ast->HalfTy))
            return ClangASTType(ast, ast->HalfTy.getAsOpaquePtr());
    }
    return ClangASTType();
}

bool
ClangASTContext::GetCompleteDecl (clang::ASTContext *ast,
                                  clang::Decl *decl)
{
    if (!decl)
        return false;
    
    ExternalASTSource *ast_source = ast->getExternalSource();
    
    if (!ast_source)
        return false;
        
    if (clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(decl))
    {
        if (tag_decl->isCompleteDefinition())
            return true;
        
        if (!tag_decl->hasExternalLexicalStorage())
            return false;
        
        ast_source->CompleteType(tag_decl);
        
        return !tag_decl->getTypeForDecl()->isIncompleteType();
    }
    else if (clang::ObjCInterfaceDecl *objc_interface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl))
    {
        if (objc_interface_decl->getDefinition())
            return true;
        
        if (!objc_interface_decl->hasExternalLexicalStorage())
            return false;
        
        ast_source->CompleteType(objc_interface_decl);
        
        return !objc_interface_decl->getTypeForDecl()->isIncompleteType();
    }
    else
    {
        return false;
    }
}

void
ClangASTContext::SetMetadataAsUserID (const void *object,
                                      user_id_t user_id)
{
    ClangASTMetadata meta_data;
    meta_data.SetUserID (user_id);
    SetMetadata (object, meta_data);
}

void
ClangASTContext::SetMetadata (clang::ASTContext *ast,
                              const void *object,
                              ClangASTMetadata &metadata)
{
    ClangExternalASTSourceCommon *external_source =
        static_cast<ClangExternalASTSourceCommon*>(ast->getExternalSource());
    
    if (external_source)
        external_source->SetMetadata(object, metadata);
}

ClangASTMetadata *
ClangASTContext::GetMetadata (clang::ASTContext *ast,
                              const void *object)
{
    ClangExternalASTSourceCommon *external_source =
        static_cast<ClangExternalASTSourceCommon*>(ast->getExternalSource());
    
    if (external_source && external_source->HasMetadata(object))
        return external_source->GetMetadata(object);
    else
        return NULL;
}

clang::DeclContext *
ClangASTContext::GetAsDeclContext (clang::CXXMethodDecl *cxx_method_decl)
{
    return llvm::dyn_cast<clang::DeclContext>(cxx_method_decl);
}

clang::DeclContext *
ClangASTContext::GetAsDeclContext (clang::ObjCMethodDecl *objc_method_decl)
{
    return llvm::dyn_cast<clang::DeclContext>(objc_method_decl);
}


bool
ClangASTContext::GetClassMethodInfoForDeclContext (clang::DeclContext *decl_ctx,
                                                   lldb::LanguageType &language,
                                                   bool &is_instance_method,
                                                   ConstString &language_object_name)
{
    language_object_name.Clear();
    language = eLanguageTypeUnknown;
    is_instance_method = false;

    if (decl_ctx)
    {
        if (clang::CXXMethodDecl *method_decl = llvm::dyn_cast<clang::CXXMethodDecl>(decl_ctx))
        {
            if (method_decl->isStatic())
            {
                is_instance_method = false;
            }
            else
            {
                language_object_name.SetCString("this");
                is_instance_method = true;
            }
            language = eLanguageTypeC_plus_plus;
            return true;
        }
        else if (clang::ObjCMethodDecl *method_decl = llvm::dyn_cast<clang::ObjCMethodDecl>(decl_ctx))
        {
            // Both static and instance methods have a "self" object in objective C
            language_object_name.SetCString("self");
            if (method_decl->isInstanceMethod())
            {
                is_instance_method = true;
            }
            else
            {
                is_instance_method = false;
            }
            language = eLanguageTypeObjC;
            return true;
        }
        else if (clang::FunctionDecl *function_decl = llvm::dyn_cast<clang::FunctionDecl>(decl_ctx))
        {
            ClangASTMetadata *metadata = GetMetadata (&decl_ctx->getParentASTContext(), function_decl);
            if (metadata && metadata->HasObjectPtr())
            {
                language_object_name.SetCString (metadata->GetObjectPtrName());
                language = eLanguageTypeObjC;
                is_instance_method = true;
            }
            return true;
        }
    }
    return false;
}

