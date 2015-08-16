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
#include <mutex> // std::once
#include <string>

// Other libraries and framework includes

// Clang headers like to use NDEBUG inside of them to enable/disable debug 
// related features using "#ifndef NDEBUG" preprocessor blocks to do one thing
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
#include "clang/AST/VTableBuilder.h"
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

#include "llvm/Support/Signals.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ThreadSafeDenseMap.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Expression/ASTDumper.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

#include <stdio.h>

#include <mutex>


//#define ENABLE_DEBUG_PRINTF // COMMENT OUT THIS LINE PRIOR TO CHECKIN

#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace clang;

typedef lldb_private::ThreadSafeDenseMap<clang::ASTContext *, ClangASTContext*> ClangASTMap;

static ClangASTMap &
GetASTMap()
{
    static ClangASTMap *g_map_ptr = nullptr;
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        g_map_ptr = new ClangASTMap(); // leaked on purpose to avoid spins
    });
    return *g_map_ptr;
}


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
ParseLangArgs (LangOptions &Opts, InputKind IK, const char* triple)
{
    // FIXME: Cleanup per-file based stuff.

    // Set some properties which depend solely on the input kind; it would be nice
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
            case IK_PreprocessedCuda:
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
      Opts.CharIsSigned = ArchSpec(triple).CharIsSignedByDefault();
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
    m_callback_tag_decl (nullptr),
    m_callback_objc_decl (nullptr),
    m_callback_baton (nullptr),
    m_pointer_byte_size (0),
    m_ast_owned(false)

{
    if (target_triple && target_triple[0])
        SetTargetTriple (target_triple);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ClangASTContext::~ClangASTContext()
{
    if (m_ast_ap.get())
    {
        GetASTMap().Erase(m_ast_ap.get());
        if (!m_ast_owned)
            m_ast_ap.release();
    }

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
        return ast->getExternalSource () != nullptr;
    return false;
}

void
ClangASTContext::SetExternalSource (llvm::IntrusiveRefCntPtr<ExternalASTSource> &ast_source_ap)
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
        llvm::IntrusiveRefCntPtr<ExternalASTSource> empty_ast_source_ap;
        ast->setExternalSource (empty_ast_source_ap);
        ast->getTranslationUnitDecl()->setHasExternalLexicalStorage(false);
        //ast->getTranslationUnitDecl()->setHasExternalVisibleStorage(false);
    }
}

void
ClangASTContext::setASTContext(clang::ASTContext *ast_ctx)
{
    if (!m_ast_owned) {
        m_ast_ap.release();
    }
    m_ast_owned = false;
    m_ast_ap.reset(ast_ctx);
    GetASTMap().Insert(ast_ctx, this);
}

ASTContext *
ClangASTContext::getASTContext()
{
    if (m_ast_ap.get() == nullptr)
    {
        m_ast_owned = true;
        m_ast_ap.reset(new ASTContext (*getLanguageOptions(),
                                       *getSourceManager(),
                                       *getIdentifierTable(),
                                       *getSelectorTable(),
                                       *getBuiltinContext()));
        
        m_ast_ap->getDiagnostics().setClient(getDiagnosticConsumer(), false);

        // This can be NULL if we don't know anything about the architecture or if the
        // target for an architecture isn't enabled in the llvm/clang that we built
        TargetInfo *target_info = getTargetInfo();
        if (target_info)
            m_ast_ap->InitBuiltinTypes(*target_info);
        
        if ((m_callback_tag_decl || m_callback_objc_decl) && m_callback_baton)
        {
            m_ast_ap->getTranslationUnitDecl()->setHasExternalLexicalStorage();
            //m_ast_ap->getTranslationUnitDecl()->setHasExternalVisibleStorage();
        }
        
        GetASTMap().Insert(m_ast_ap.get(), this);
    }
    return m_ast_ap.get();
}

ClangASTContext*
ClangASTContext::GetASTContext (clang::ASTContext* ast)
{
    ClangASTContext *clang_ast = GetASTMap().Lookup(ast);
    return clang_ast;
}

Builtin::Context *
ClangASTContext::getBuiltinContext()
{
    if (m_builtins_ap.get() == nullptr)
        m_builtins_ap.reset (new Builtin::Context());
    return m_builtins_ap.get();
}

IdentifierTable *
ClangASTContext::getIdentifierTable()
{
    if (m_identifier_table_ap.get() == nullptr)
        m_identifier_table_ap.reset(new IdentifierTable (*ClangASTContext::getLanguageOptions(), nullptr));
    return m_identifier_table_ap.get();
}

LangOptions *
ClangASTContext::getLanguageOptions()
{
    if (m_language_options_ap.get() == nullptr)
    {
        m_language_options_ap.reset(new LangOptions());
        ParseLangArgs(*m_language_options_ap, IK_ObjCXX, GetTargetTriple());
//        InitializeLangOptions(*m_language_options_ap, IK_ObjCXX);
    }
    return m_language_options_ap.get();
}

SelectorTable *
ClangASTContext::getSelectorTable()
{
    if (m_selector_table_ap.get() == nullptr)
        m_selector_table_ap.reset (new SelectorTable());
    return m_selector_table_ap.get();
}

clang::FileManager *
ClangASTContext::getFileManager()
{
    if (m_file_manager_ap.get() == nullptr)
    {
        clang::FileSystemOptions file_system_options;
        m_file_manager_ap.reset(new clang::FileManager(file_system_options));
    }
    return m_file_manager_ap.get();
}

clang::SourceManager *
ClangASTContext::getSourceManager()
{
    if (m_source_manager_ap.get() == nullptr)
        m_source_manager_ap.reset(new clang::SourceManager(*getDiagnosticsEngine(), *getFileManager()));
    return m_source_manager_ap.get();
}

clang::DiagnosticsEngine *
ClangASTContext::getDiagnosticsEngine()
{
    if (m_diagnostics_engine_ap.get() == nullptr)
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
    if (m_diagnostic_consumer_ap.get() == nullptr)
        m_diagnostic_consumer_ap.reset(new NullDiagnosticConsumer);
    
    return m_diagnostic_consumer_ap.get();
}

std::shared_ptr<TargetOptions> &
ClangASTContext::getTargetOptions() {
    if (m_target_options_rp.get() == nullptr && !m_target_triple.empty())
    {
        m_target_options_rp = std::make_shared<TargetOptions>();
        if (m_target_options_rp.get() != nullptr)
            m_target_options_rp->Triple = m_target_triple;
    }
    return m_target_options_rp;
}


TargetInfo *
ClangASTContext::getTargetInfo()
{
    // target_triple should be something like "x86_64-apple-macosx"
    if (m_target_info_ap.get() == nullptr && !m_target_triple.empty())
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
CompilerType
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (Encoding encoding, uint32_t bit_size)
{
    return ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (getASTContext(), encoding, bit_size);
}

CompilerType
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (ASTContext *ast, Encoding encoding, uint32_t bit_size)
{
    if (!ast)
        return CompilerType();
    switch (encoding)
    {
    case eEncodingInvalid:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
            return CompilerType (ast, ast->VoidPtrTy);
        break;
        
    case eEncodingUint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
            return CompilerType (ast, ast->UnsignedCharTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
            return CompilerType (ast, ast->UnsignedShortTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
            return CompilerType (ast, ast->UnsignedIntTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
            return CompilerType (ast, ast->UnsignedLongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
            return CompilerType (ast, ast->UnsignedLongLongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
            return CompilerType (ast, ast->UnsignedInt128Ty);
        break;
        
    case eEncodingSint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
            return CompilerType (ast, ast->CharTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
            return CompilerType (ast, ast->ShortTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
            return CompilerType (ast, ast->IntTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
            return CompilerType (ast, ast->LongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
            return CompilerType (ast, ast->LongLongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
            return CompilerType (ast, ast->Int128Ty);
        break;
        
    case eEncodingIEEE754:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
            return CompilerType (ast, ast->FloatTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
            return CompilerType (ast, ast->DoubleTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
            return CompilerType (ast, ast->LongDoubleTy);
        break;
        
    case eEncodingVector:
        // Sanity check that bit_size is a multiple of 8's.
        if (bit_size && !(bit_size & 0x7u))
            return CompilerType (ast, ast->getExtVectorType (ast->UnsignedCharTy, bit_size/8));
        break;
    }
    
    return CompilerType();
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
            
            // Miscellaneous
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

CompilerType
ClangASTContext::GetBasicType (ASTContext *ast, const ConstString &name)
{
    if (ast)
    {
        lldb::BasicType basic_type = ClangASTContext::GetBasicTypeEnumeration (name);
        return ClangASTContext::GetBasicType (ast, basic_type);
    }
    return CompilerType();
}

uint32_t
ClangASTContext::GetPointerByteSize ()
{
    if (m_pointer_byte_size == 0)
        m_pointer_byte_size = GetBasicType(lldb::eBasicTypeVoid).GetPointerType().GetByteSize(nullptr);
    return m_pointer_byte_size;
}

CompilerType
ClangASTContext::GetBasicType (lldb::BasicType basic_type)
{
    return GetBasicType (getASTContext(), basic_type);
}

CompilerType
ClangASTContext::GetBasicType (ASTContext *ast, lldb::BasicType basic_type)
{
    if (ast)
    {
        clang_type_t clang_type = nullptr;
        
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
            return CompilerType (GetASTContext(ast), clang_type);
    }
    return CompilerType();
}


CompilerType
ClangASTContext::GetBuiltinTypeForDWARFEncodingAndBitSize (const char *type_name, uint32_t dw_ate, uint32_t bit_size)
{
    ASTContext *ast = getASTContext();
    
#define streq(a,b) strcmp(a,b) == 0
    assert (ast != nullptr);
    if (ast)
    {
        switch (dw_ate)
        {
            default:
                break;
                
            case DW_ATE_address:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
                    return CompilerType (ast, ast->VoidPtrTy);
                break;
                
            case DW_ATE_boolean:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->BoolTy))
                    return CompilerType (ast, ast->BoolTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return CompilerType (ast, ast->UnsignedCharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return CompilerType (ast, ast->UnsignedShortTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                    return CompilerType (ast, ast->UnsignedIntTy);
                break;
                
            case DW_ATE_lo_user:
                // This has been seen to mean DW_AT_complex_integer
                if (type_name)
                {
                    if (::strstr(type_name, "complex"))
                    {
                        CompilerType complex_int_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("int", DW_ATE_signed, bit_size/2);
                        return CompilerType (ast, ast->getComplexType (GetQualType(complex_int_clang_type)));
                    }
                }
                break;
                
            case DW_ATE_complex_float:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatComplexTy))
                    return CompilerType (ast, ast->FloatComplexTy);
                else if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleComplexTy))
                    return CompilerType (ast, ast->DoubleComplexTy);
                else if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleComplexTy))
                    return CompilerType (ast, ast->LongDoubleComplexTy);
                else 
                {
                    CompilerType complex_float_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("float", DW_ATE_float, bit_size/2);
                    return CompilerType (ast, ast->getComplexType (GetQualType(complex_float_clang_type)));
                }
                break;
                
            case DW_ATE_float:
                if (streq(type_name, "float") && QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
                    return CompilerType (ast, ast->FloatTy);
                if (streq(type_name, "double") && QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
                    return CompilerType (ast, ast->DoubleTy);
                if (streq(type_name, "long double") && QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
                    return CompilerType (ast, ast->LongDoubleTy);
                // Fall back to not requiring a name match
                if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
                    return CompilerType (ast, ast->FloatTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
                    return CompilerType (ast, ast->DoubleTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
                    return CompilerType (ast, ast->LongDoubleTy);
                break;
                
            case DW_ATE_signed:
                if (type_name)
                {
                    if (streq(type_name, "wchar_t") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->WCharTy) &&
                        (getTargetInfo() && TargetInfo::isTypeSigned (getTargetInfo()->getWCharType())))
                        return CompilerType (ast, ast->WCharTy);
                    if (streq(type_name, "void") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->VoidTy))
                        return CompilerType (ast, ast->VoidTy);
                    if (strstr(type_name, "long long") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                        return CompilerType (ast, ast->LongLongTy);
                    if (strstr(type_name, "long") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                        return CompilerType (ast, ast->LongTy);
                    if (strstr(type_name, "short") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                        return CompilerType (ast, ast->ShortTy);
                    if (strstr(type_name, "char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                            return CompilerType (ast, ast->CharTy);
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                            return CompilerType (ast, ast->SignedCharTy);
                    }
                    if (strstr(type_name, "int"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                            return CompilerType (ast, ast->IntTy);
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                            return CompilerType (ast, ast->Int128Ty);
                    }
                }
                // We weren't able to match up a type name, just search by size
                if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                    return CompilerType (ast, ast->CharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                    return CompilerType (ast, ast->ShortTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                    return CompilerType (ast, ast->IntTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                    return CompilerType (ast, ast->LongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                    return CompilerType (ast, ast->LongLongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                    return CompilerType (ast, ast->Int128Ty);
                break;

            case DW_ATE_signed_char:
                if (ast->getLangOpts().CharIsSigned && type_name && streq(type_name, "char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                        return CompilerType (ast, ast->CharTy);
                }
                if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                    return CompilerType (ast, ast->SignedCharTy);
                break;
                
            case DW_ATE_unsigned:
                if (type_name)
                {
                    if (streq(type_name, "wchar_t"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->WCharTy))
                        {
                            if (!(getTargetInfo() && TargetInfo::isTypeSigned (getTargetInfo()->getWCharType())))
                                return CompilerType (ast, ast->WCharTy);
                        }
                    }
                    if (strstr(type_name, "long long"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                            return CompilerType (ast, ast->UnsignedLongLongTy);
                    }
                    else if (strstr(type_name, "long"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                            return CompilerType (ast, ast->UnsignedLongTy);
                    }
                    else if (strstr(type_name, "short"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                            return CompilerType (ast, ast->UnsignedShortTy);
                    }
                    else if (strstr(type_name, "char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                            return CompilerType (ast, ast->UnsignedCharTy);
                    }
                    else if (strstr(type_name, "int"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                            return CompilerType (ast, ast->UnsignedIntTy);
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                            return CompilerType (ast, ast->UnsignedInt128Ty);
                    }
                }
                // We weren't able to match up a type name, just search by size
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return CompilerType (ast, ast->UnsignedCharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return CompilerType (ast, ast->UnsignedShortTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                    return CompilerType (ast, ast->UnsignedIntTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                    return CompilerType (ast, ast->UnsignedLongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                    return CompilerType (ast, ast->UnsignedLongLongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                    return CompilerType (ast, ast->UnsignedInt128Ty);
                break;

            case DW_ATE_unsigned_char:
                if (!ast->getLangOpts().CharIsSigned && type_name && streq(type_name, "char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                        return CompilerType (ast, ast->CharTy);
                }
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return CompilerType (ast, ast->UnsignedCharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return CompilerType (ast, ast->UnsignedShortTy);
                break;
                
            case DW_ATE_imaginary_float:
                break;
                
            case DW_ATE_UTF:
                if (type_name)
                {
                    if (streq(type_name, "char16_t"))
                    {
                        return CompilerType (ast, ast->Char16Ty);
                    }
                    else if (streq(type_name, "char32_t"))
                    {
                        return CompilerType (ast, ast->Char32Ty);
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
    return CompilerType ();
}

CompilerType
ClangASTContext::GetUnknownAnyType(clang::ASTContext *ast)
{
    if (ast)
        return CompilerType (ast, ast->UnknownAnyTy);
    return CompilerType();
}

CompilerType
ClangASTContext::GetCStringType (bool is_const)
{
    ASTContext *ast = getASTContext();
    QualType char_type(ast->CharTy);
    
    if (is_const)
        char_type.addConst();
    
    return CompilerType (ast, ast->getPointerType(char_type));
}

clang::DeclContext *
ClangASTContext::GetTranslationUnitDecl (clang::ASTContext *ast)
{
    return ast->getTranslationUnitDecl();
}

CompilerType
ClangASTContext::CopyType (ASTContext *dst_ast, 
                           CompilerType src)
{
    FileSystemOptions file_system_options;
    ClangASTContext *src_ast = src.GetTypeSystem()->AsClangASTContext();
    if (src_ast == nullptr)
        return CompilerType();
    FileManager file_manager (file_system_options);
    ASTImporter importer(*dst_ast, file_manager,
                         *src_ast->getASTContext(), file_manager,
                         false);
    
    QualType dst (importer.Import(GetQualType(src)));
    
    return CompilerType (dst_ast, dst);
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
ClangASTContext::AreTypesSame (CompilerType type1,
                               CompilerType type2,
                               bool ignore_qualifiers)
{
    TypeSystem *ast = type1.GetTypeSystem();
    if (!ast->AsClangASTContext() || ast != type2.GetTypeSystem())
        return false;

    if (type1.GetOpaqueQualType() == type2.GetOpaqueQualType())
        return true;

    QualType type1_qual = GetQualType(type1);
    QualType type2_qual = GetQualType(type2);
    
    if (ignore_qualifiers)
    {
        type1_qual = type1_qual.getUnqualifiedType();
        type2_qual = type2_qual.getUnqualifiedType();
    }
    
    return ast->AsClangASTContext()->getASTContext()->hasSameType (type1_qual, type2_qual);
}

CompilerType
ClangASTContext::GetTypeForDecl (clang::NamedDecl *decl)
{
    if (clang::ObjCInterfaceDecl *interface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl))
        return GetTypeForDecl(interface_decl);
    if (clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(decl))
        return GetTypeForDecl(tag_decl);
    return CompilerType();
}


CompilerType
ClangASTContext::GetTypeForDecl (TagDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    ASTContext *ast = &decl->getASTContext();
    if (ast)
        return CompilerType (ast, ast->getTagDeclType(decl));
    return CompilerType();
}

CompilerType
ClangASTContext::GetTypeForDecl (ObjCInterfaceDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    ASTContext *ast = &decl->getASTContext();
    if (ast)
        return CompilerType (ast, ast->getObjCInterfaceType(decl));
    return CompilerType();
}

#pragma mark Structure, Unions, Classes

CompilerType
ClangASTContext::CreateRecordType (DeclContext *decl_ctx,
                                   AccessType access_type,
                                   const char *name,
                                   int kind,
                                   LanguageType language,
                                   ClangASTMetadata *metadata)
{
    ASTContext *ast = getASTContext();
    assert (ast != nullptr);
     
    if (decl_ctx == nullptr)
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
                                                 is_anonymous ? nullptr : &ast->Idents.get(name));
    
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

        return CompilerType(ast, ast->getTagDeclType(decl));
    }
    return CompilerType();
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
        
        IdentifierInfo *identifier_info = nullptr;
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
                                                                             nullptr));
            
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
                                                  nullptr);
}


ClassTemplateDecl *
ClangASTContext::CreateClassTemplateDecl (DeclContext *decl_ctx,
                                          lldb::AccessType access_type,
                                          const char *class_name, 
                                          int kind, 
                                          const TemplateParameterInfos &template_param_infos)
{
    ASTContext *ast = getASTContext();
    
    ClassTemplateDecl *class_template_decl = nullptr;
    if (decl_ctx == nullptr)
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
                                                     nullptr);
    
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
                                                                                                                   nullptr);
    
    class_template_specialization_decl->setSpecializationKind(TSK_ExplicitSpecialization);
    
    return class_template_specialization_decl;
}

CompilerType
ClangASTContext::CreateClassTemplateSpecializationType (ClassTemplateSpecializationDecl *class_template_specialization_decl)
{
    if (class_template_specialization_decl)
    {
        ASTContext *ast = getASTContext();
        if (ast)
            return CompilerType(ast, ast->getTagDeclType(class_template_specialization_decl));
    }
    return CompilerType();
}

static inline bool
check_op_param (uint32_t op_kind, bool unary, bool binary, uint32_t num_params)
{
    // Special-case call since it can take any number of operands
    if(op_kind == OO_Call)
        return true;
    
    // The parameter count doesn't include "this"
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
    if (ast == nullptr || field == nullptr)
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
    if (record_decl == nullptr)
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

CompilerType
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
    assert (ast != nullptr);
    assert (name && name[0]);
    if (decl_ctx == nullptr)
        decl_ctx = ast->getTranslationUnitDecl();

    ObjCInterfaceDecl *decl = ObjCInterfaceDecl::Create (*ast,
                                                         decl_ctx,
                                                         SourceLocation(),
                                                         &ast->Idents.get(name),
                                                         nullptr,
                                                         nullptr,
                                                         SourceLocation(),
                                                         /*isForwardDecl,*/
                                                         isInternal);
    
    if (decl && metadata)
        SetMetadata(ast, decl, *metadata);
    
    return CompilerType (ast, ast->getObjCInterfaceType(decl));
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
    NamespaceDecl *namespace_decl = nullptr;
    ASTContext *ast = getASTContext();
    TranslationUnitDecl *translation_unit_decl = ast->getTranslationUnitDecl ();
    if (decl_ctx == nullptr)
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
                                               nullptr);
        
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
                                                   nullptr,
                                                   nullptr);
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
                                                       nullptr,
                                                       nullptr);
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
                                            const CompilerType &function_clang_type,
                                            int storage,
                                            bool is_inline)
{
    FunctionDecl *func_decl = nullptr;
    ASTContext *ast = getASTContext();
    if (decl_ctx == nullptr)
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
                                          GetQualType(function_clang_type),
                                          nullptr,
                                          (clang::StorageClass)storage,
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
                                          GetQualType(function_clang_type),
                                          nullptr,
                                          (clang::StorageClass)storage,
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

CompilerType
ClangASTContext::CreateFunctionType (ASTContext *ast,
                                     const CompilerType& result_type,
                                     const CompilerType *args,
                                     unsigned num_args, 
                                     bool is_variadic, 
                                     unsigned type_quals)
{
    assert (ast != nullptr);
    std::vector<QualType> qual_type_args;
    for (unsigned i=0; i<num_args; ++i)
        qual_type_args.push_back (GetQualType(args[i]));

    // TODO: Detect calling convention in DWARF?
    FunctionProtoType::ExtProtoInfo proto_info;
    proto_info.Variadic = is_variadic;
    proto_info.ExceptionSpec = EST_None;
    proto_info.TypeQuals = type_quals;
    proto_info.RefQualifier = RQ_None;

    return CompilerType (ast, ast->getFunctionType (GetQualType(result_type),
                                                    qual_type_args,
                                                    proto_info));
}

ParmVarDecl *
ClangASTContext::CreateParameterDeclaration (const char *name, const CompilerType &param_type, int storage)
{
    ASTContext *ast = getASTContext();
    assert (ast != nullptr);
    return ParmVarDecl::Create(*ast,
                                ast->getTranslationUnitDecl(),
                                SourceLocation(),
                                SourceLocation(),
                                name && name[0] ? &ast->Idents.get(name) : nullptr,
                                GetQualType(param_type),
                                nullptr,
                                (clang::StorageClass)storage,
                                nullptr);
}

void
ClangASTContext::SetFunctionParameters (FunctionDecl *function_decl, ParmVarDecl **params, unsigned num_params)
{
    if (function_decl)
        function_decl->setParams (ArrayRef<ParmVarDecl*>(params, num_params));
}


#pragma mark Array Types

CompilerType
ClangASTContext::CreateArrayType (const CompilerType &element_type,
                                  size_t element_count,
                                  bool is_vector)
{
    if (element_type.IsValid())
    {
        ASTContext *ast = getASTContext();
        assert (ast != nullptr);

        if (is_vector)
        {
            return CompilerType (ast, ast->getExtVectorType(GetQualType(element_type), element_count));
        }
        else
        {
        
            llvm::APInt ap_element_count (64, element_count);
            if (element_count == 0)
            {
                return CompilerType (ast, ast->getIncompleteArrayType (GetQualType(element_type),
                                                                       ArrayType::Normal,
                                                                       0));
            }
            else
            {
                return CompilerType (ast, ast->getConstantArrayType (GetQualType(element_type),
                                                                     ap_element_count,
                                                                     ArrayType::Normal,
                                                                     0));
            }
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetOrCreateStructForIdentifier (const ConstString &type_name,
                                                 const std::initializer_list< std::pair < const char *, CompilerType > >& type_fields,
                                                 bool packed)
{
    CompilerType type;
    if ((type = GetTypeForIdentifier<clang::CXXRecordDecl>(type_name)).IsValid())
        return type;
    type = CreateRecordType(nullptr, lldb::eAccessPublic, type_name.GetCString(), clang::TTK_Struct, lldb::eLanguageTypeC);
    StartTagDeclarationDefinition(type);
    for (const auto& field : type_fields)
        AddFieldToRecordType(type, field.first, field.second, lldb::eAccessPublic, 0);
    if (packed)
        SetIsPacked(type);
    CompleteTagDeclarationDefinition(type);
    return type;
}

#pragma mark Enumeration Types

CompilerType
ClangASTContext::CreateEnumerationType
(
 const char *name,
 DeclContext *decl_ctx,
 const Declaration &decl,
 const CompilerType &integer_clang_type
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
                                            name && name[0] ? &ast->Idents.get(name) : nullptr,
                                            nullptr,
                                            false,  // IsScoped
                                            false,  // IsScopedUsingClassTag
                                            false); // IsFixed
    
    
    if (enum_decl)
    {
        // TODO: check if we should be setting the promotion type too?
        enum_decl->setIntegerType(GetQualType(integer_clang_type));
        
        enum_decl->setAccess(AS_public); // TODO respect what's in the debug info
        
        return CompilerType (ast, ast->getTagDeclType(enum_decl));
    }
    return CompilerType();
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

CompilerType
ClangASTContext::GetIntTypeFromBitSize (clang::ASTContext *ast,
                                        size_t bit_size, bool is_signed)
{
    if (ast)
    {
        if (is_signed)
        {
            if (bit_size == ast->getTypeSize(ast->SignedCharTy))
                return CompilerType(ast, ast->SignedCharTy);
            
            if (bit_size == ast->getTypeSize(ast->ShortTy))
                return CompilerType(ast, ast->ShortTy);
            
            if (bit_size == ast->getTypeSize(ast->IntTy))
                return CompilerType(ast, ast->IntTy);
            
            if (bit_size == ast->getTypeSize(ast->LongTy))
                return CompilerType(ast, ast->LongTy);
            
            if (bit_size == ast->getTypeSize(ast->LongLongTy))
                return CompilerType(ast, ast->LongLongTy);
            
            if (bit_size == ast->getTypeSize(ast->Int128Ty))
                return CompilerType(ast, ast->Int128Ty);
        }
        else
        {
            if (bit_size == ast->getTypeSize(ast->UnsignedCharTy))
                return CompilerType(ast, ast->UnsignedCharTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedShortTy))
                return CompilerType(ast, ast->UnsignedShortTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedIntTy))
                return CompilerType(ast, ast->UnsignedIntTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedLongTy))
                return CompilerType(ast, ast->UnsignedLongTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedLongLongTy))
                return CompilerType(ast, ast->UnsignedLongLongTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedInt128Ty))
                return CompilerType(ast, ast->UnsignedInt128Ty);
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetPointerSizedIntType (clang::ASTContext *ast, bool is_signed)
{
    if (ast)
        return GetIntTypeFromBitSize(ast, ast->getTypeSize(ast->VoidPtrTy), is_signed);
    return CompilerType();
}

CompilerType
ClangASTContext::GetFloatTypeFromBitSize (clang::ASTContext *ast,
                                          size_t bit_size)
{
    if (ast)
    {
        if (bit_size == ast->getTypeSize(ast->FloatTy))
            return CompilerType(ast, ast->FloatTy);
        else if (bit_size == ast->getTypeSize(ast->DoubleTy))
            return CompilerType(ast, ast->DoubleTy);
        else if (bit_size == ast->getTypeSize(ast->LongDoubleTy))
            return CompilerType(ast, ast->LongDoubleTy);
        else if (bit_size == ast->getTypeSize(ast->HalfTy))
            return CompilerType(ast, ast->HalfTy);
    }
    return CompilerType();
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
        ClangExternalASTSourceCommon::Lookup(ast->getExternalSource());
    
    if (external_source)
        external_source->SetMetadata(object, metadata);
}

ClangASTMetadata *
ClangASTContext::GetMetadata (clang::ASTContext *ast,
                              const void *object)
{
    ClangExternalASTSourceCommon *external_source =
        ClangExternalASTSourceCommon::Lookup(ast->getExternalSource());
    
    if (external_source && external_source->HasMetadata(object))
        return external_source->GetMetadata(object);
    else
        return nullptr;
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


bool
ClangASTContext::SetTagTypeKind (clang::QualType tag_qual_type, int kind) const
{
    const clang::Type *clang_type = tag_qual_type.getTypePtr();
    if (clang_type)
    {
        const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(clang_type);
        if (tag_type)
        {
            clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(tag_type->getDecl());
            if (tag_decl)
            {
                tag_decl->setTagKind ((clang::TagDecl::TagKind)kind);
                return true;
            }
        }
    }
    return false;
}


bool
ClangASTContext::SetDefaultAccessForRecordFields (clang::RecordDecl* record_decl,
                                                  int default_accessibility,
                                                  int *assigned_accessibilities,
                                                  size_t num_assigned_accessibilities)
{
    if (record_decl)
    {
        uint32_t field_idx;
        clang::RecordDecl::field_iterator field, field_end;
        for (field = record_decl->field_begin(), field_end = record_decl->field_end(), field_idx = 0;
             field != field_end;
             ++field, ++field_idx)
        {
            // If no accessibility was assigned, assign the correct one
            if (field_idx < num_assigned_accessibilities && assigned_accessibilities[field_idx] == clang::AS_none)
                field->setAccess ((clang::AccessSpecifier)default_accessibility);
        }
        return true;
    }
    return false;
}

clang::DeclContext *
ClangASTContext::GetDeclContextForType (clang::QualType type)
{
    if (type.isNull())
        return nullptr;
    
    clang::QualType qual_type = type.getCanonicalType();
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::UnaryTransform:           break;
        case clang::Type::FunctionNoProto:          break;
        case clang::Type::FunctionProto:            break;
        case clang::Type::IncompleteArray:          break;
        case clang::Type::VariableArray:            break;
        case clang::Type::ConstantArray:            break;
        case clang::Type::DependentSizedArray:      break;
        case clang::Type::ExtVector:                break;
        case clang::Type::DependentSizedExtVector:  break;
        case clang::Type::Vector:                   break;
        case clang::Type::Builtin:                  break;
        case clang::Type::BlockPointer:             break;
        case clang::Type::Pointer:                  break;
        case clang::Type::LValueReference:          break;
        case clang::Type::RValueReference:          break;
        case clang::Type::MemberPointer:            break;
        case clang::Type::Complex:                  break;
        case clang::Type::ObjCObject:               break;
        case clang::Type::ObjCInterface:            return llvm::cast<clang::ObjCObjectType>(qual_type.getTypePtr())->getInterface();
        case clang::Type::ObjCObjectPointer:        return GetDeclContextForType (llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
        case clang::Type::Record:                   return llvm::cast<clang::RecordType>(qual_type)->getDecl();
        case clang::Type::Enum:                     return llvm::cast<clang::EnumType>(qual_type)->getDecl();
        case clang::Type::Typedef:                  return GetDeclContextForType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType());
        case clang::Type::Elaborated:               return GetDeclContextForType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType());
        case clang::Type::Paren:                    return GetDeclContextForType (llvm::cast<clang::ParenType>(qual_type)->desugar());
        case clang::Type::TypeOfExpr:               break;
        case clang::Type::TypeOf:                   break;
        case clang::Type::Decltype:                 break;
            //case clang::Type::QualifiedName:          break;
        case clang::Type::TemplateSpecialization:   break;
        case clang::Type::DependentTemplateSpecialization:  break;
        case clang::Type::TemplateTypeParm:         break;
        case clang::Type::SubstTemplateTypeParm:    break;
        case clang::Type::SubstTemplateTypeParmPack:break;
        case clang::Type::PackExpansion:            break;
        case clang::Type::UnresolvedUsing:          break;
        case clang::Type::Attributed:               break;
        case clang::Type::Auto:                     break;
        case clang::Type::InjectedClassName:        break;
        case clang::Type::DependentName:            break;
        case clang::Type::Atomic:                   break;
        case clang::Type::Adjusted:                 break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:                  break;
    }
    // No DeclContext in this type...
    return nullptr;
}

static bool
GetCompleteQualType (clang::ASTContext *ast, clang::QualType qual_type, bool allow_completion = true)
{
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
        {
            const clang::ArrayType *array_type = llvm::dyn_cast<clang::ArrayType>(qual_type.getTypePtr());
            
            if (array_type)
                return GetCompleteQualType (ast, array_type->getElementType(), allow_completion);
        }
            break;
            
        case clang::Type::Record:
        case clang::Type::Enum:
        {
            const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
            if (tag_type)
            {
                clang::TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    if (tag_decl->isCompleteDefinition())
                        return true;
                    
                    if (!allow_completion)
                        return false;
                    
                    if (tag_decl->hasExternalLexicalStorage())
                    {
                        if (ast)
                        {
                            clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                            if (external_ast_source)
                            {
                                external_ast_source->CompleteType(tag_decl);
                                return !tag_type->isIncompleteType();
                            }
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
            const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                // We currently can't complete objective C types through the newly added ASTContext
                // because it only supports TagDecl objects right now...
                if (class_interface_decl)
                {
                    if (class_interface_decl->getDefinition())
                        return true;
                    
                    if (!allow_completion)
                        return false;
                    
                    if (class_interface_decl->hasExternalLexicalStorage())
                    {
                        if (ast)
                        {
                            clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                            if (external_ast_source)
                            {
                                external_ast_source->CompleteType (class_interface_decl);
                                return !objc_class_type->isIncompleteType();
                            }
                        }
                    }
                    return false;
                }
            }
        }
            break;
            
        case clang::Type::Typedef:
            return GetCompleteQualType (ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType(), allow_completion);
            
        case clang::Type::Elaborated:
            return GetCompleteQualType (ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType(), allow_completion);
            
        case clang::Type::Paren:
            return GetCompleteQualType (ast, llvm::cast<clang::ParenType>(qual_type)->desugar(), allow_completion);
            
        default:
            break;
    }
    
    return true;
}

static clang::ObjCIvarDecl::AccessControl
ConvertAccessTypeToObjCIvarAccessControl (AccessType access)
{
    switch (access)
    {
        case eAccessNone:      return clang::ObjCIvarDecl::None;
        case eAccessPublic:    return clang::ObjCIvarDecl::Public;
        case eAccessPrivate:   return clang::ObjCIvarDecl::Private;
        case eAccessProtected: return clang::ObjCIvarDecl::Protected;
        case eAccessPackage:   return clang::ObjCIvarDecl::Package;
    }
    return clang::ObjCIvarDecl::None;
}


//----------------------------------------------------------------------
// Tests
//----------------------------------------------------------------------

bool
ClangASTContext::IsAggregateType (void* type)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
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
        case clang::Type::Elaborated:
            return IsAggregateType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
        case clang::Type::Typedef:
            return IsAggregateType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
        case clang::Type::Paren:
            return IsAggregateType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());
        default:
            break;
    }
    // The clang type does have a value
    return false;
}

bool
ClangASTContext::IsArrayType (void* type,
                              CompilerType *element_type_ptr,
                              uint64_t *size,
                              bool *is_incomplete)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        default:
            break;
            
        case clang::Type::ConstantArray:
            if (element_type_ptr)
                element_type_ptr->SetClangType (getASTContext(), llvm::cast<clang::ConstantArrayType>(qual_type)->getElementType());
            if (size)
                *size = llvm::cast<clang::ConstantArrayType>(qual_type)->getSize().getLimitedValue(ULLONG_MAX);
            return true;
            
        case clang::Type::IncompleteArray:
            if (element_type_ptr)
                element_type_ptr->SetClangType (getASTContext(), llvm::cast<clang::IncompleteArrayType>(qual_type)->getElementType());
            if (size)
                *size = 0;
            if (is_incomplete)
                *is_incomplete = true;
            return true;
            
        case clang::Type::VariableArray:
            if (element_type_ptr)
                element_type_ptr->SetClangType (getASTContext(), llvm::cast<clang::VariableArrayType>(qual_type)->getElementType());
            if (size)
                *size = 0;
            return true;
            
        case clang::Type::DependentSizedArray:
            if (element_type_ptr)
                element_type_ptr->SetClangType (getASTContext(), llvm::cast<clang::DependentSizedArrayType>(qual_type)->getElementType());
            if (size)
                *size = 0;
            return true;
            
        case clang::Type::Typedef:
            return IsArrayType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
        case clang::Type::Elaborated:
            return IsArrayType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
        case clang::Type::Paren:
            return IsArrayType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
    }
    if (element_type_ptr)
        element_type_ptr->Clear();
    if (size)
        *size = 0;
    if (is_incomplete)
        *is_incomplete = false;
    return 0;
}

bool
ClangASTContext::IsVectorType (void* type,
                               CompilerType *element_type,
                               uint64_t *size)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Vector:
        {
            const clang::VectorType *vector_type = qual_type->getAs<clang::VectorType>();
            if (vector_type)
            {
                if (size)
                    *size = vector_type->getNumElements();
                if (element_type)
                    *element_type = CompilerType(getASTContext(), vector_type->getElementType());
            }
            return true;
        }
            break;
        case clang::Type::ExtVector:
        {
            const clang::ExtVectorType *ext_vector_type = qual_type->getAs<clang::ExtVectorType>();
            if (ext_vector_type)
            {
                if (size)
                    *size = ext_vector_type->getNumElements();
                if (element_type)
                    *element_type = CompilerType(getASTContext(), ext_vector_type->getElementType());
            }
            return true;
        }
        default:
            break;
    }
    return false;
}

bool
ClangASTContext::IsRuntimeGeneratedType (void* type)
{
    clang::DeclContext* decl_ctx = ClangASTContext::GetASTContext(getASTContext())->GetDeclContextForType(GetQualType(type));
    if (!decl_ctx)
        return false;
    
    if (!llvm::isa<clang::ObjCInterfaceDecl>(decl_ctx))
        return false;
    
    clang::ObjCInterfaceDecl *result_iface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl_ctx);
    
    ClangASTMetadata* ast_metadata = ClangASTContext::GetMetadata(getASTContext(), result_iface_decl);
    if (!ast_metadata)
        return false;
    return (ast_metadata->GetISAPtr() != 0);
}

bool
ClangASTContext::IsCharType (void* type)
{
    return GetQualType(type).getUnqualifiedType()->isCharType();
}


bool
ClangASTContext::IsCompleteType (void* type)
{
    const bool allow_completion = false;
    return GetCompleteQualType (getASTContext(), GetQualType(type), allow_completion);
}

bool
ClangASTContext::IsConst(void* type)
{
    return GetQualType(type).isConstQualified();
}

bool
ClangASTContext::IsCStringType (void* type, uint32_t &length)
{
    CompilerType pointee_or_element_clang_type;
    length = 0;
    Flags type_flags (GetTypeInfo (type, &pointee_or_element_clang_type));
    
    if (!pointee_or_element_clang_type.IsValid())
        return false;
    
    if (type_flags.AnySet (eTypeIsArray | eTypeIsPointer))
    {
        if (pointee_or_element_clang_type.IsCharType())
        {
            if (type_flags.Test (eTypeIsArray))
            {
                // We know the size of the array and it could be a C string
                // since it is an array of characters
                length = llvm::cast<clang::ConstantArrayType>(GetCanonicalQualType(type).getTypePtr())->getSize().getLimitedValue();
            }
            return true;
            
        }
    }
    return false;
}

bool
ClangASTContext::IsFunctionType (void* type, bool *is_variadic_ptr)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        if (qual_type->isFunctionType())
        {
            if (is_variadic_ptr)
            {
                const clang::FunctionProtoType *function_proto_type = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
                if (function_proto_type)
                    *is_variadic_ptr = function_proto_type->isVariadic();
                else
                    *is_variadic_ptr = false;
            }
            return true;
        }
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
            case clang::Type::Typedef:
                return IsFunctionType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), nullptr);
            case clang::Type::Elaborated:
                return IsFunctionType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), nullptr);
            case clang::Type::Paren:
                return IsFunctionType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), nullptr);
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    if (reference_type)
                        return IsFunctionType(reference_type->getPointeeType().getAsOpaquePtr(), nullptr);
                }
                break;
        }
    }
    return false;
}

// Used to detect "Homogeneous Floating-point Aggregates"
uint32_t
ClangASTContext::IsHomogeneousAggregate (void* type, CompilerType* base_type_ptr)
{
    if (!type)
        return 0;
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType (type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    if (cxx_record_decl->getNumBases() ||
                        cxx_record_decl->isDynamicClass())
                        return 0;
                }
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        // We are looking for a structure that contains only floating point types
                        clang::RecordDecl::field_iterator field_pos, field_end = record_decl->field_end();
                        uint32_t num_fields = 0;
                        bool is_hva = false;
                        bool is_hfa = false;
                        clang::QualType base_qual_type;
                        for (field_pos = record_decl->field_begin(); field_pos != field_end; ++field_pos)
                        {
                            clang::QualType field_qual_type = field_pos->getType();
                            if (field_qual_type->isFloatingType())
                            {
                                if (field_qual_type->isComplexType())
                                    return 0;
                                else
                                {
                                    if (num_fields == 0)
                                        base_qual_type = field_qual_type;
                                    else
                                    {
                                        if (is_hva)
                                            return 0;
                                        is_hfa = true;
                                        if (field_qual_type.getTypePtr() != base_qual_type.getTypePtr())
                                            return 0;
                                    }
                                }
                            }
                            else if (field_qual_type->isVectorType() || field_qual_type->isExtVectorType())
                            {
                                const clang::VectorType *array = field_qual_type.getTypePtr()->getAs<clang::VectorType>();
                                if (array && array->getNumElements() <= 4)
                                {
                                    if (num_fields == 0)
                                        base_qual_type = array->getElementType();
                                    else
                                    {
                                        if (is_hfa)
                                            return 0;
                                        is_hva = true;
                                        if (field_qual_type.getTypePtr() != base_qual_type.getTypePtr())
                                            return 0;
                                    }
                                }
                                else
                                    return 0;
                            }
                            else
                                return 0;
                            ++num_fields;
                        }
                        if (base_type_ptr)
                            *base_type_ptr = CompilerType (getASTContext(), base_qual_type);
                        return num_fields;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return IsHomogeneousAggregate(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), base_type_ptr);
            
        case clang::Type::Elaborated:
            return IsHomogeneousAggregate(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), base_type_ptr);
        default:
            break;
    }
    return 0;
}

size_t
ClangASTContext::GetNumberOfFunctionArguments (void* type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return func->getNumParams();
    }
    return 0;
}

CompilerType
ClangASTContext::GetFunctionArgumentAtIndex (void* type, const size_t index)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
        {
            if (index < func->getNumParams())
                return CompilerType(getASTContext(), func->getParamType(index));
        }
    }
    return CompilerType();
}

bool
ClangASTContext::IsFunctionPointerType (void* type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        if (qual_type->isFunctionPointerType())
            return true;
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
            case clang::Type::Typedef:
                return IsFunctionPointerType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
            case clang::Type::Elaborated:
                return IsFunctionPointerType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
            case clang::Type::Paren:
                return IsFunctionPointerType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    if (reference_type)
                        return IsFunctionPointerType(reference_type->getPointeeType().getAsOpaquePtr());
                }
                break;
        }
    }
    return false;
    
}

bool
ClangASTContext::IsIntegerType (void* type, bool &is_signed)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::BuiltinType *builtin_type = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal());
    
    if (builtin_type)
    {
        if (builtin_type->isInteger())
        {
            is_signed = builtin_type->isSignedInteger();
            return true;
        }
    }
    
    return false;
}

bool
ClangASTContext::IsPointerType (void* type, CompilerType *pointee_type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                    break;
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                    return true;
            }
                return false;
            case clang::Type::ObjCObjectPointer:
                if (pointee_type)
                    pointee_type->SetClangType (getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetClangType (getASTContext(), llvm::cast<clang::BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetClangType (getASTContext(), llvm::cast<clang::PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetClangType (getASTContext(), llvm::cast<clang::MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Typedef:
                return IsPointerType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Elaborated:
                return IsPointerType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Paren:
                return IsPointerType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTContext::IsPointerOrReferenceType (void* type, CompilerType *pointee_type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
                {
                    default:
                        break;
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                        return true;
                }
                return false;
            case clang::Type::ObjCObjectPointer:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::RValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::Typedef:
                return IsPointerOrReferenceType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Elaborated:
                return IsPointerOrReferenceType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Paren:
                return IsPointerOrReferenceType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTContext::IsReferenceType (void* type, CompilerType *pointee_type, bool* is_rvalue)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::LValueReferenceType>(qual_type)->desugar());
                if (is_rvalue)
                    *is_rvalue = false;
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetClangType(getASTContext(), llvm::cast<clang::RValueReferenceType>(qual_type)->desugar());
                if (is_rvalue)
                    *is_rvalue = true;
                return true;
            case clang::Type::Typedef:
                return IsReferenceType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), pointee_type, is_rvalue);
            case clang::Type::Elaborated:
                return IsReferenceType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), pointee_type, is_rvalue);
            case clang::Type::Paren:
                return IsReferenceType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), pointee_type, is_rvalue);
                
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}

bool
ClangASTContext::IsFloatingPointType (void* type, uint32_t &count, bool &is_complex)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        if (const clang::BuiltinType *BT = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal()))
        {
            clang::BuiltinType::Kind kind = BT->getKind();
            if (kind >= clang::BuiltinType::Float && kind <= clang::BuiltinType::LongDouble)
            {
                count = 1;
                is_complex = false;
                return true;
            }
        }
        else if (const clang::ComplexType *CT = llvm::dyn_cast<clang::ComplexType>(qual_type->getCanonicalTypeInternal()))
        {
            if (IsFloatingPointType (CT->getElementType().getAsOpaquePtr(), count, is_complex))
            {
                count = 2;
                is_complex = true;
                return true;
            }
        }
        else if (const clang::VectorType *VT = llvm::dyn_cast<clang::VectorType>(qual_type->getCanonicalTypeInternal()))
        {
            if (IsFloatingPointType (VT->getElementType().getAsOpaquePtr(), count, is_complex))
            {
                count = VT->getNumElements();
                is_complex = false;
                return true;
            }
        }
    }
    count = 0;
    is_complex = false;
    return false;
}


bool
ClangASTContext::IsDefined(void* type)
{
    if (!type)
        return false;
    
    clang::QualType qual_type(GetQualType(type));
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
    if (tag_type)
    {
        clang::TagDecl *tag_decl = tag_type->getDecl();
        if (tag_decl)
            return tag_decl->isCompleteDefinition();
        return false;
    }
    else
    {
        const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
        if (objc_class_type)
        {
            clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            if (class_interface_decl)
                return class_interface_decl->getDefinition() != nullptr;
            return false;
        }
    }
    return true;
}

bool
ClangASTContext::IsObjCClassType (const CompilerType& type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        const clang::ObjCObjectPointerType *obj_pointer_type = llvm::dyn_cast<clang::ObjCObjectPointerType>(qual_type);
        
        if (obj_pointer_type)
            return obj_pointer_type->isObjCClassType();
    }
    return false;
}

bool
ClangASTContext::IsObjCObjectOrInterfaceType (const CompilerType& type)
{
    if (type)
        return GetCanonicalQualType(type)->isObjCObjectOrInterfaceType();
    return false;
}

bool
ClangASTContext::IsPolymorphicClass (void* type)
{
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType(type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                        if (cxx_record_decl)
                            return cxx_record_decl->isPolymorphic();
                    }
                }
                break;
                
            default:
                break;
        }
    }
    return false;
}

bool
ClangASTContext::IsPossibleDynamicType (void* type, CompilerType *dynamic_pointee_type,
                                           bool check_cplusplus,
                                           bool check_objc)
{
    clang::QualType pointee_qual_type;
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        bool success = false;
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                if (check_objc && llvm::cast<clang::BuiltinType>(qual_type)->getKind() == clang::BuiltinType::ObjCId)
                {
                    if (dynamic_pointee_type)
                        dynamic_pointee_type->SetClangType(this, type);
                    return true;
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (check_objc)
                {
                    if (dynamic_pointee_type)
                        dynamic_pointee_type->SetClangType(getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                    return true;
                }
                break;
                
            case clang::Type::Pointer:
                pointee_qual_type = llvm::cast<clang::PointerType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                pointee_qual_type = llvm::cast<clang::ReferenceType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::Typedef:
                return IsPossibleDynamicType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);
                
            case clang::Type::Elaborated:
                return IsPossibleDynamicType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);
                
            case clang::Type::Paren:
                return IsPossibleDynamicType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);
            default:
                break;
        }
        
        if (success)
        {
            // Check to make sure what we are pointing too is a possible dynamic C++ type
            // We currently accept any "void *" (in case we have a class that has been
            // watered down to an opaque pointer) and virtual C++ classes.
            const clang::Type::TypeClass pointee_type_class = pointee_qual_type.getCanonicalType()->getTypeClass();
            switch (pointee_type_class)
            {
                case clang::Type::Builtin:
                    switch (llvm::cast<clang::BuiltinType>(pointee_qual_type)->getKind())
                {
                    case clang::BuiltinType::UnknownAny:
                    case clang::BuiltinType::Void:
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetClangType(getASTContext(), pointee_qual_type);
                        return true;
                        
                    case clang::BuiltinType::NullPtr:
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
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                    case clang::BuiltinType::ObjCSel:
                    case clang::BuiltinType::BoundMember:
                    case clang::BuiltinType::Half:
                    case clang::BuiltinType::ARCUnbridgedCast:
                    case clang::BuiltinType::PseudoObject:
                    case clang::BuiltinType::BuiltinFn:
                    case clang::BuiltinType::OCLEvent:
                    case clang::BuiltinType::OCLImage1d:
                    case clang::BuiltinType::OCLImage1dArray:
                    case clang::BuiltinType::OCLImage1dBuffer:
                    case clang::BuiltinType::OCLImage2d:
                    case clang::BuiltinType::OCLImage2dArray:
                    case clang::BuiltinType::OCLImage3d:
                    case clang::BuiltinType::OCLSampler:
                        break;
                }
                    break;
                    
                case clang::Type::Record:
                    if (check_cplusplus)
                    {
                        clang::CXXRecordDecl *cxx_record_decl = pointee_qual_type->getAsCXXRecordDecl();
                        if (cxx_record_decl)
                        {
                            bool is_complete = cxx_record_decl->isCompleteDefinition();
                            
                            if (is_complete)
                                success = cxx_record_decl->isDynamicClass();
                            else
                            {
                                ClangASTMetadata *metadata = ClangASTContext::GetMetadata (getASTContext(), cxx_record_decl);
                                if (metadata)
                                    success = metadata->GetIsDynamicCXXType();
                                else
                                {
                                    is_complete = CompilerType(getASTContext(), pointee_qual_type).GetCompleteType();
                                    if (is_complete)
                                        success = cxx_record_decl->isDynamicClass();
                                    else
                                        success = false;
                                }
                            }
                            
                            if (success)
                            {
                                if (dynamic_pointee_type)
                                    dynamic_pointee_type->SetClangType(getASTContext(), pointee_qual_type);
                                return true;
                            }
                        }
                    }
                    break;
                    
                case clang::Type::ObjCObject:
                case clang::Type::ObjCInterface:
                    if (check_objc)
                    {
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetClangType(getASTContext(), pointee_qual_type);
                        return true;
                    }
                    break;
                    
                default:
                    break;
            }
        }
    }
    if (dynamic_pointee_type)
        dynamic_pointee_type->Clear();
    return false;
}


bool
ClangASTContext::IsScalarType (void* type)
{
    if (!type)
        return false;
    
    return (GetTypeInfo (type, nullptr) & eTypeIsScalar) != 0;
}

bool
ClangASTContext::IsTypedefType (void* type)
{
    if (!type)
        return false;
    return GetQualType(type)->getTypeClass() == clang::Type::Typedef;
}

bool
ClangASTContext::IsVoidType (void* type)
{
    if (!type)
        return false;
    return GetCanonicalQualType(type)->isVoidType();
}

bool
ClangASTContext::GetCXXClassName (const CompilerType& type, std::string &class_name)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
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
ClangASTContext::IsCXXClassType (const CompilerType& type)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    if (qual_type->getAsCXXRecordDecl() != nullptr)
        return true;
    return false;
}

bool
ClangASTContext::IsBeingDefined (void* type)
{
    if (!type)
        return false;
    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type);
    if (tag_type)
        return tag_type->isBeingDefined();
    return false;
}

bool
ClangASTContext::IsObjCObjectPointerType (const CompilerType& type, CompilerType *class_type_ptr)
{
    if (!type)
        return false;

    clang::QualType qual_type (GetCanonicalQualType(type));
    
    if (qual_type->isObjCObjectPointerType())
    {
        if (class_type_ptr)
        {
            if (!qual_type->isObjCClassType() &&
                !qual_type->isObjCIdType())
            {
                const clang::ObjCObjectPointerType *obj_pointer_type = llvm::dyn_cast<clang::ObjCObjectPointerType>(qual_type);
                if (obj_pointer_type == nullptr)
                    class_type_ptr->Clear();
                else
                    class_type_ptr->SetClangType (type.GetTypeSystem(), clang::QualType(obj_pointer_type->getInterfaceType(), 0).getAsOpaquePtr());
            }
        }
        return true;
    }
    if (class_type_ptr)
        class_type_ptr->Clear();
    return false;
}

bool
ClangASTContext::GetObjCClassName (const CompilerType& type, std::string &class_name)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::ObjCObjectType *object_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
    if (object_type)
    {
        const clang::ObjCInterfaceDecl *interface = object_type->getInterface();
        if (interface)
        {
            class_name = interface->getNameAsString();
            return true;
        }
    }
    return false;
}


//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

bool
ClangASTContext::GetCompleteType (void* type)
{
    if (!type)
        return false;
    const bool allow_completion = true;
    return GetCompleteQualType (getASTContext(), GetQualType(type), allow_completion);
}

ConstString
ClangASTContext::GetTypeName (void* type)
{
    std::string type_name;
    if (type)
    {
        clang::PrintingPolicy printing_policy (getASTContext()->getPrintingPolicy());
        clang::QualType qual_type(GetQualType(type));
        printing_policy.SuppressTagKeyword = true;
        printing_policy.LangOpts.WChar = true;
        const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
        if (typedef_type)
        {
            const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
            type_name = typedef_decl->getQualifiedNameAsString();
        }
        else
        {
            type_name = qual_type.getAsString(printing_policy);
        }
    }
    return ConstString(type_name);
}

uint32_t
ClangASTContext::GetTypeInfo (void* type, CompilerType *pointee_or_element_clang_type)
{
    if (!type)
        return 0;
    
    if (pointee_or_element_clang_type)
        pointee_or_element_clang_type->Clear();
    
    clang::QualType qual_type (GetQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
        {
            const clang::BuiltinType *builtin_type = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal());
            
            uint32_t builtin_type_flags = eTypeIsBuiltIn | eTypeHasValue;
            switch (builtin_type->getKind())
            {
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                    if (pointee_or_element_clang_type)
                        pointee_or_element_clang_type->SetClangType(getASTContext(), getASTContext()->ObjCBuiltinClassTy);
                    builtin_type_flags |= eTypeIsPointer | eTypeIsObjC;
                    break;
                    
                case clang::BuiltinType::ObjCSel:
                    if (pointee_or_element_clang_type)
                        pointee_or_element_clang_type->SetClangType(getASTContext(), getASTContext()->CharTy);
                    builtin_type_flags |= eTypeIsPointer | eTypeIsObjC;
                    break;
                    
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
                    builtin_type_flags |= eTypeIsScalar;
                    if (builtin_type->isInteger())
                    {
                        builtin_type_flags |= eTypeIsInteger;
                        if (builtin_type->isSignedInteger())
                            builtin_type_flags |= eTypeIsSigned;
                    }
                    else if (builtin_type->isFloatingPoint())
                        builtin_type_flags |= eTypeIsFloat;
                    break;
                default:
                    break;
            }
            return builtin_type_flags;
        }
            
        case clang::Type::BlockPointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(getASTContext(), qual_type->getPointeeType());
            return eTypeIsPointer | eTypeHasChildren | eTypeIsBlock;
            
        case clang::Type::Complex:
        {
            uint32_t complex_type_flags = eTypeIsBuiltIn | eTypeHasValue | eTypeIsComplex;
            const clang::ComplexType *complex_type = llvm::dyn_cast<clang::ComplexType>(qual_type->getCanonicalTypeInternal());
            if (complex_type)
            {
                clang::QualType complex_element_type (complex_type->getElementType());
                if (complex_element_type->isIntegerType())
                    complex_type_flags |= eTypeIsFloat;
                else if (complex_element_type->isFloatingType())
                    complex_type_flags |= eTypeIsInteger;
            }
            return complex_type_flags;
        }
            break;
            
        case clang::Type::ConstantArray:
        case clang::Type::DependentSizedArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(getASTContext(), llvm::cast<clang::ArrayType>(qual_type.getTypePtr())->getElementType());
            return eTypeHasChildren | eTypeIsArray;
            
        case clang::Type::DependentName:                    return 0;
        case clang::Type::DependentSizedExtVector:          return eTypeHasChildren | eTypeIsVector;
        case clang::Type::DependentTemplateSpecialization:  return eTypeIsTemplate;
        case clang::Type::Decltype:                         return 0;
            
        case clang::Type::Enum:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(getASTContext(), llvm::cast<clang::EnumType>(qual_type)->getDecl()->getIntegerType());
            return eTypeIsEnumeration | eTypeHasValue;
            
        case clang::Type::Elaborated:
            return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::Paren:
            return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTypeInfo (pointee_or_element_clang_type);
            
        case clang::Type::FunctionProto:                    return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::FunctionNoProto:                  return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::InjectedClassName:                return 0;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(getASTContext(), llvm::cast<clang::ReferenceType>(qual_type.getTypePtr())->getPointeeType());
            return eTypeHasChildren | eTypeIsReference | eTypeHasValue;
            
        case clang::Type::MemberPointer:                    return eTypeIsPointer   | eTypeIsMember | eTypeHasValue;
            
        case clang::Type::ObjCObjectPointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(getASTContext(), qual_type->getPointeeType());
            return eTypeHasChildren | eTypeIsObjC | eTypeIsClass | eTypeIsPointer | eTypeHasValue;
            
        case clang::Type::ObjCObject:                       return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
        case clang::Type::ObjCInterface:                    return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
            
        case clang::Type::Pointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetClangType(getASTContext(), qual_type->getPointeeType());
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
            return eTypeIsTypedef | CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::TypeOfExpr:                       return 0;
        case clang::Type::TypeOf:                           return 0;
        case clang::Type::UnresolvedUsing:                  return 0;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
        {
            uint32_t vector_type_flags = eTypeHasChildren | eTypeIsVector;
            const clang::VectorType *vector_type = llvm::dyn_cast<clang::VectorType>(qual_type->getCanonicalTypeInternal());
            if (vector_type)
            {
                if (vector_type->isIntegerType())
                    vector_type_flags |= eTypeIsFloat;
                else if (vector_type->isFloatingType())
                    vector_type_flags |= eTypeIsInteger;
            }
            return vector_type_flags;
        }
        default:                                            return 0;
    }
    return 0;
}



lldb::LanguageType
ClangASTContext::GetMinimumLanguage (void* type)
{
    if (!type)
        return lldb::eLanguageTypeC;
    
    // If the type is a reference, then resolve it to what it refers to first:
    clang::QualType qual_type (GetCanonicalQualType(type).getNonReferenceType());
    if (qual_type->isAnyPointerType())
    {
        if (qual_type->isObjCObjectPointerType())
            return lldb::eLanguageTypeObjC;
        
        clang::QualType pointee_type (qual_type->getPointeeType());
        if (pointee_type->getPointeeCXXRecordDecl() != nullptr)
            return lldb::eLanguageTypeC_plus_plus;
        if (pointee_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type->isObjCClassType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type.getTypePtr() == getASTContext()->ObjCBuiltinIdTy.getTypePtr())
            return lldb::eLanguageTypeObjC;
    }
    else
    {
        if (qual_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (qual_type->getAsCXXRecordDecl())
            return lldb::eLanguageTypeC_plus_plus;
        switch (qual_type->getTypeClass())
        {
            default:
                break;
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                case clang::BuiltinType::Void:
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
                    break;
                    
                case clang::BuiltinType::NullPtr:
                    return eLanguageTypeC_plus_plus;
                    
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                case clang::BuiltinType::ObjCSel:
                    return eLanguageTypeObjC;
                    
                case clang::BuiltinType::Dependent:
                case clang::BuiltinType::Overload:
                case clang::BuiltinType::BoundMember:
                case clang::BuiltinType::UnknownAny:
                    break;
            }
                break;
            case clang::Type::Typedef:
                return CompilerType(getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetMinimumLanguage();
        }
    }
    return lldb::eLanguageTypeC;
}

lldb::TypeClass
ClangASTContext::GetTypeClass (void* type)
{
    if (!type)
        return lldb::eTypeClassInvalid;
    
    clang::QualType qual_type(GetQualType(type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:           break;
        case clang::Type::FunctionNoProto:          return lldb::eTypeClassFunction;
        case clang::Type::FunctionProto:            return lldb::eTypeClassFunction;
        case clang::Type::IncompleteArray:          return lldb::eTypeClassArray;
        case clang::Type::VariableArray:            return lldb::eTypeClassArray;
        case clang::Type::ConstantArray:            return lldb::eTypeClassArray;
        case clang::Type::DependentSizedArray:      return lldb::eTypeClassArray;
        case clang::Type::DependentSizedExtVector:  return lldb::eTypeClassVector;
        case clang::Type::ExtVector:                return lldb::eTypeClassVector;
        case clang::Type::Vector:                   return lldb::eTypeClassVector;
        case clang::Type::Builtin:                  return lldb::eTypeClassBuiltin;
        case clang::Type::ObjCObjectPointer:        return lldb::eTypeClassObjCObjectPointer;
        case clang::Type::BlockPointer:             return lldb::eTypeClassBlockPointer;
        case clang::Type::Pointer:                  return lldb::eTypeClassPointer;
        case clang::Type::LValueReference:          return lldb::eTypeClassReference;
        case clang::Type::RValueReference:          return lldb::eTypeClassReference;
        case clang::Type::MemberPointer:            return lldb::eTypeClassMemberPointer;
        case clang::Type::Complex:
            if (qual_type->isComplexType())
                return lldb::eTypeClassComplexFloat;
            else
                return lldb::eTypeClassComplexInteger;
        case clang::Type::ObjCObject:               return lldb::eTypeClassObjCObject;
        case clang::Type::ObjCInterface:            return lldb::eTypeClassObjCInterface;
        case clang::Type::Record:
        {
            const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
            const clang::RecordDecl *record_decl = record_type->getDecl();
            if (record_decl->isUnion())
                return lldb::eTypeClassUnion;
            else if (record_decl->isStruct())
                return lldb::eTypeClassStruct;
            else
                return lldb::eTypeClassClass;
        }
            break;
        case clang::Type::Enum:                     return lldb::eTypeClassEnumeration;
        case clang::Type::Typedef:                  return lldb::eTypeClassTypedef;
        case clang::Type::UnresolvedUsing:          break;
        case clang::Type::Paren:
            return CompilerType(getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTypeClass();
        case clang::Type::Elaborated:
            return CompilerType(getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTypeClass();
            
        case clang::Type::Attributed:               break;
        case clang::Type::TemplateTypeParm:         break;
        case clang::Type::SubstTemplateTypeParm:    break;
        case clang::Type::SubstTemplateTypeParmPack:break;
        case clang::Type::Auto:                     break;
        case clang::Type::InjectedClassName:        break;
        case clang::Type::DependentName:            break;
        case clang::Type::DependentTemplateSpecialization: break;
        case clang::Type::PackExpansion:            break;
            
        case clang::Type::TypeOfExpr:               break;
        case clang::Type::TypeOf:                   break;
        case clang::Type::Decltype:                 break;
        case clang::Type::TemplateSpecialization:   break;
        case clang::Type::Atomic:                   break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:                  break;
        case clang::Type::Adjusted:                 break;
    }
    // We don't know hot to display this type...
    return lldb::eTypeClassOther;
    
}

unsigned
ClangASTContext::GetTypeQualifiers(void* type)
{
    if (type)
        return GetQualType(type).getQualifiers().getCVRQualifiers();
    return 0;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

CompilerType
ClangASTContext::AddConstModifier (const CompilerType& type)
{
    if (type && type.GetTypeSystem()->AsClangASTContext())
    {
        clang::QualType result(GetQualType(type));
        result.addConst();
        return CompilerType (type.GetTypeSystem(), result.getAsOpaquePtr());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::AddRestrictModifier (const CompilerType& type)
{
    if (type && type.GetTypeSystem()->AsClangASTContext())
    {
        clang::QualType result(GetQualType(type));
        result.getQualifiers().setRestrict (true);
        return CompilerType (type.GetTypeSystem(), result.getAsOpaquePtr());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::AddVolatileModifier (const CompilerType& type)
{
    if (type && type.GetTypeSystem()->AsClangASTContext())
    {
        clang::QualType result(GetQualType(type));
        result.getQualifiers().setVolatile (true);
        return CompilerType (type.GetTypeSystem(), result.getAsOpaquePtr());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetArrayElementType (void* type, uint64_t *stride)
{
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        
        const clang::Type *array_eletype = qual_type.getTypePtr()->getArrayElementTypeNoTypeQual();
        
        if (!array_eletype)
            return CompilerType();
        
        CompilerType element_type (getASTContext(), array_eletype->getCanonicalTypeUnqualified());
        
        // TODO: the real stride will be >= this value.. find the real one!
        if (stride)
            *stride = element_type.GetByteSize(nullptr);
        
        return element_type;
        
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetCanonicalType (void* type)
{
    if (type)
        return CompilerType (getASTContext(), GetCanonicalQualType(type));
    return CompilerType();
}

static clang::QualType
GetFullyUnqualifiedType_Impl (clang::ASTContext *ast, clang::QualType qual_type)
{
    if (qual_type->isPointerType())
        qual_type = ast->getPointerType(GetFullyUnqualifiedType_Impl(ast, qual_type->getPointeeType()));
    else
        qual_type = qual_type.getUnqualifiedType();
    qual_type.removeLocalConst();
    qual_type.removeLocalRestrict();
    qual_type.removeLocalVolatile();
    return qual_type;
}

CompilerType
ClangASTContext::GetFullyUnqualifiedType (void* type)
{
    if (type)
        return CompilerType(getASTContext(), GetFullyUnqualifiedType_Impl(getASTContext(), GetQualType(type)));
    return CompilerType();
}


int
ClangASTContext::GetFunctionArgumentCount (void* type)
{
    if (type)
    {
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(GetCanonicalQualType(type));
        if (func)
            return func->getNumParams();
    }
    return -1;
}

CompilerType
ClangASTContext::GetFunctionArgumentTypeAtIndex (void* type, size_t idx)
{
    if (type)
    {
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(GetCanonicalQualType(type));
        if (func)
        {
            const uint32_t num_args = func->getNumParams();
            if (idx < num_args)
                return CompilerType(getASTContext(), func->getParamType(idx));
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetFunctionReturnType (void* type)
{
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return CompilerType(getASTContext(), func->getReturnType());
    }
    return CompilerType();
}

size_t
ClangASTContext::GetNumMemberFunctions (void* type)
{
    size_t num_functions = 0;
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        switch (qual_type->getTypeClass()) {
            case clang::Type::Record:
                if (GetCompleteQualType (getASTContext(), qual_type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    assert(record_decl);
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    if (cxx_record_decl)
                        num_functions = std::distance(cxx_record_decl->method_begin(), cxx_record_decl->method_end());
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                        if (class_interface_decl)
                            num_functions = std::distance(class_interface_decl->meth_begin(), class_interface_decl->meth_end());
                    }
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        if (class_interface_decl)
                            num_functions = std::distance(class_interface_decl->meth_begin(), class_interface_decl->meth_end());
                    }
                }
                break;
                
                
            case clang::Type::Typedef:
                return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumMemberFunctions();
                
            case clang::Type::Elaborated:
                return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumMemberFunctions();
                
            case clang::Type::Paren:
                return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumMemberFunctions();
                
            default:
                break;
        }
    }
    return num_functions;
}

TypeMemberFunctionImpl
ClangASTContext::GetMemberFunctionAtIndex (void* type, size_t idx)
{
    std::string name("");
    MemberFunctionKind kind(MemberFunctionKind::eMemberFunctionKindUnknown);
    CompilerType clang_type{};
    clang::ObjCMethodDecl *method_decl(nullptr);
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        switch (qual_type->getTypeClass()) {
            case clang::Type::Record:
                if (GetCompleteQualType (getASTContext(), qual_type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    assert(record_decl);
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    if (cxx_record_decl)
                    {
                        auto method_iter = cxx_record_decl->method_begin();
                        auto method_end = cxx_record_decl->method_end();
                        if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                        {
                            std::advance(method_iter, idx);
                            auto method_decl = method_iter->getCanonicalDecl();
                            if (method_decl)
                            {
                                if (!method_decl->getName().empty())
                                    name.assign(method_decl->getName().data());
                                else
                                    name.clear();
                                if (method_decl->isStatic())
                                    kind = lldb::eMemberFunctionKindStaticMethod;
                                else if (llvm::isa<clang::CXXConstructorDecl>(method_decl))
                                    kind = lldb::eMemberFunctionKindConstructor;
                                else if (llvm::isa<clang::CXXDestructorDecl>(method_decl))
                                    kind = lldb::eMemberFunctionKindDestructor;
                                else
                                    kind = lldb::eMemberFunctionKindInstanceMethod;
                                clang_type = CompilerType(getASTContext(),method_decl->getType());
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                        if (class_interface_decl)
                        {
                            auto method_iter = class_interface_decl->meth_begin();
                            auto method_end = class_interface_decl->meth_end();
                            if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                            {
                                std::advance(method_iter, idx);
                                method_decl = method_iter->getCanonicalDecl();
                                if (method_decl)
                                {
                                    name = method_decl->getSelector().getAsString();
                                    if (method_decl->isClassMethod())
                                        kind = lldb::eMemberFunctionKindStaticMethod;
                                    else
                                        kind = lldb::eMemberFunctionKindInstanceMethod;
                                }
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        if (class_interface_decl)
                        {
                            auto method_iter = class_interface_decl->meth_begin();
                            auto method_end = class_interface_decl->meth_end();
                            if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                            {
                                std::advance(method_iter, idx);
                                method_decl = method_iter->getCanonicalDecl();
                                if (method_decl)
                                {
                                    name = method_decl->getSelector().getAsString();
                                    if (method_decl->isClassMethod())
                                        kind = lldb::eMemberFunctionKindStaticMethod;
                                    else
                                        kind = lldb::eMemberFunctionKindInstanceMethod;
                                }
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::Typedef:
                return GetMemberFunctionAtIndex(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), idx);
                
            case clang::Type::Elaborated:
                return GetMemberFunctionAtIndex(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), idx);
                
            case clang::Type::Paren:
                return GetMemberFunctionAtIndex(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), idx);
                
            default:
                break;
        }
    }
    
    if (kind == eMemberFunctionKindUnknown)
        return TypeMemberFunctionImpl();
    if (method_decl)
        return TypeMemberFunctionImpl(method_decl, name, kind);
    if (type)
        return TypeMemberFunctionImpl(clang_type, name, kind);
    
    return TypeMemberFunctionImpl();
}

CompilerType
ClangASTContext::GetLValueReferenceType (const CompilerType& type)
{
    if (type)
    {
        ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
        if (ast)
            return CompilerType(ast->getASTContext(), ast->getASTContext()->getLValueReferenceType(GetQualType(type)));
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetRValueReferenceType (const CompilerType& type)
{
    if (type)
    {
        ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
        if (ast)
            return CompilerType(ast->getASTContext(), ast->getASTContext()->getRValueReferenceType(GetQualType(type)));
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetNonReferenceType (void* type)
{
    if (type)
        return CompilerType(getASTContext(), GetQualType(type).getNonReferenceType());
    return CompilerType();
}

CompilerType
ClangASTContext::CreateTypedefType (const CompilerType& type,
                                    const char *typedef_name,
                                    clang::DeclContext *decl_ctx)
{
    if (type && typedef_name && typedef_name[0])
    {
        ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
        if (!ast)
            return CompilerType();
        clang::ASTContext* clang_ast = ast->getASTContext();
        clang::QualType qual_type (GetQualType(type));
        if (decl_ctx == nullptr)
            decl_ctx = ast->getASTContext()->getTranslationUnitDecl();
        clang::TypedefDecl *decl = clang::TypedefDecl::Create (*clang_ast,
                                                               decl_ctx,
                                                               clang::SourceLocation(),
                                                               clang::SourceLocation(),
                                                               &clang_ast->Idents.get(typedef_name),
                                                               clang_ast->getTrivialTypeSourceInfo(qual_type));
        
        decl->setAccess(clang::AS_public); // TODO respect proper access specifier
        
        // Get a uniqued clang::QualType for the typedef decl type
        return CompilerType (clang_ast, clang_ast->getTypedefType (decl));
    }
    return CompilerType();
    
}

CompilerType
ClangASTContext::GetPointeeType (void* type)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        return CompilerType (getASTContext(), qual_type.getTypePtr()->getPointeeType());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetPointerType (void* type)
{
    if (type)
    {
        clang::QualType qual_type (GetQualType(type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                return CompilerType(getASTContext(), getASTContext()->getObjCObjectPointerType(qual_type));
                
            default:
                return CompilerType(getASTContext(), getASTContext()->getPointerType(qual_type));
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetTypedefedType (void* type)
{
    if (type)
    {
        const clang::TypedefType *typedef_type = llvm::dyn_cast<clang::TypedefType>(GetQualType(type));
        if (typedef_type)
            return CompilerType (getASTContext(), typedef_type->getDecl()->getUnderlyingType());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::RemoveFastQualifiers (const CompilerType& type)
{
    if (type && type.GetTypeSystem()->AsClangASTContext())
    {
        clang::QualType qual_type(GetQualType(type));
        qual_type.getQualifiers().removeFastQualifiers();
        return CompilerType (type.GetTypeSystem(), qual_type.getAsOpaquePtr());
    }
    return type;
}


//----------------------------------------------------------------------
// Create related types using the current type's AST
//----------------------------------------------------------------------

CompilerType
ClangASTContext::GetBasicTypeFromAST (void* type, lldb::BasicType basic_type)
{
    if (type)
        return ClangASTContext::GetBasicType(getASTContext(), basic_type);
    return CompilerType();
}
//----------------------------------------------------------------------
// Exploring the type
//----------------------------------------------------------------------

uint64_t
ClangASTContext::GetBitSize (void* type, ExecutionContextScope *exe_scope)
{
    if (GetCompleteType (type))
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        switch (qual_type->getTypeClass())
        {
            case clang::Type::ObjCInterface:
            case clang::Type::ObjCObject:
            {
                ExecutionContext exe_ctx (exe_scope);
                Process *process = exe_ctx.GetProcessPtr();
                if (process)
                {
                    ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                    if (objc_runtime)
                    {
                        uint64_t bit_size = 0;
                        if (objc_runtime->GetTypeBitSize(CompilerType(getASTContext(), qual_type), bit_size))
                            return bit_size;
                    }
                }
                else
                {
                    static bool g_printed = false;
                    if (!g_printed)
                    {
                        StreamString s;
                        DumpTypeDescription(&s);
                        
                        llvm::outs() << "warning: trying to determine the size of type ";
                        llvm::outs() << s.GetString() << "\n";
                        llvm::outs() << "without a valid ExecutionContext. this is not reliable. please file a bug against LLDB.\n";
                        llvm::outs() << "backtrace:\n";
                        llvm::sys::PrintStackTrace(llvm::outs());
                        llvm::outs() << "\n";
                        g_printed = true;
                    }
                }
            }
                // fallthrough
            default:
                const uint32_t bit_size = getASTContext()->getTypeSize (qual_type);
                if (bit_size == 0)
                {
                    if (qual_type->isIncompleteArrayType())
                        return getASTContext()->getTypeSize (qual_type->getArrayElementTypeNoTypeQual()->getCanonicalTypeUnqualified());
                }
                if (qual_type->isObjCObjectOrInterfaceType())
                    return bit_size + getASTContext()->getTypeSize(getASTContext()->ObjCBuiltinClassTy);
                return bit_size;
        }
    }
    return 0;
}

size_t
ClangASTContext::GetTypeBitAlign (void* type)
{
    if (GetCompleteType(type))
        return getASTContext()->getTypeAlign(GetQualType(type));
    return 0;
}


lldb::Encoding
ClangASTContext::GetEncoding (void* type, uint64_t &count)
{
    if (!type)
        return lldb::eEncodingInvalid;
    
    count = 1;
    clang::QualType qual_type(GetCanonicalQualType(type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:
            break;
            
        case clang::Type::FunctionNoProto:
        case clang::Type::FunctionProto:
            break;
            
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            break;
            
        case clang::Type::ConstantArray:
            break;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
            // TODO: Set this to more than one???
            break;
            
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            default: assert(0 && "Unknown builtin type!");
            case clang::BuiltinType::Void:
                break;
                
            case clang::BuiltinType::Bool:
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Char16:
            case clang::BuiltinType::Char32:
            case clang::BuiltinType::Short:
            case clang::BuiltinType::Int:
            case clang::BuiltinType::Long:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::Int128:        return lldb::eEncodingSint;
                
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:
            case clang::BuiltinType::UShort:
            case clang::BuiltinType::UInt:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::ULongLong:
            case clang::BuiltinType::UInt128:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::Float:
            case clang::BuiltinType::Double:
            case clang::BuiltinType::LongDouble:    return lldb::eEncodingIEEE754;
                
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCSel:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::NullPtr:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::Kind::ARCUnbridgedCast:
            case clang::BuiltinType::Kind::BoundMember:
            case clang::BuiltinType::Kind::BuiltinFn:
            case clang::BuiltinType::Kind::Dependent:
            case clang::BuiltinType::Kind::Half:
            case clang::BuiltinType::Kind::OCLEvent:
            case clang::BuiltinType::Kind::OCLImage1d:
            case clang::BuiltinType::Kind::OCLImage1dArray:
            case clang::BuiltinType::Kind::OCLImage1dBuffer:
            case clang::BuiltinType::Kind::OCLImage2d:
            case clang::BuiltinType::Kind::OCLImage2dArray:
            case clang::BuiltinType::Kind::OCLImage3d:
            case clang::BuiltinType::Kind::OCLSampler:
            case clang::BuiltinType::Kind::Overload:
            case clang::BuiltinType::Kind::PseudoObject:
            case clang::BuiltinType::Kind::UnknownAny:
                break;
        }
            break;
            // All pointer types are represented as unsigned integer encodings.
            // We may nee to add a eEncodingPointer if we ever need to know the
            // difference
        case clang::Type::ObjCObjectPointer:
        case clang::Type::BlockPointer:
        case clang::Type::Pointer:
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
        case clang::Type::MemberPointer:            return lldb::eEncodingUint;
        case clang::Type::Complex:
        {
            lldb::Encoding encoding = lldb::eEncodingIEEE754;
            if (qual_type->isComplexType())
                encoding = lldb::eEncodingIEEE754;
            else
            {
                const clang::ComplexType *complex_type = qual_type->getAsComplexIntegerType();
                if (complex_type)
                    encoding = CompilerType(getASTContext(), complex_type->getElementType()).GetEncoding(count);
                else
                    encoding = lldb::eEncodingSint;
            }
            count = 2;
            return encoding;
        }
            
        case clang::Type::ObjCInterface:            break;
        case clang::Type::Record:                   break;
        case clang::Type::Enum:                     return lldb::eEncodingSint;
        case clang::Type::Typedef:
            return CompilerType(getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetEncoding(count);
            
        case clang::Type::Elaborated:
            return CompilerType(getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetEncoding(count);
            
        case clang::Type::Paren:
            return CompilerType(getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetEncoding(count);
            
        case clang::Type::DependentSizedArray:
        case clang::Type::DependentSizedExtVector:
        case clang::Type::UnresolvedUsing:
        case clang::Type::Attributed:
        case clang::Type::TemplateTypeParm:
        case clang::Type::SubstTemplateTypeParm:
        case clang::Type::SubstTemplateTypeParmPack:
        case clang::Type::Auto:
        case clang::Type::InjectedClassName:
        case clang::Type::DependentName:
        case clang::Type::DependentTemplateSpecialization:
        case clang::Type::PackExpansion:
        case clang::Type::ObjCObject:
            
        case clang::Type::TypeOfExpr:
        case clang::Type::TypeOf:
        case clang::Type::Decltype:
        case clang::Type::TemplateSpecialization:
        case clang::Type::Atomic:
        case clang::Type::Adjusted:
            break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:
            break;
    }
    count = 0;
    return lldb::eEncodingInvalid;
}

lldb::Format
ClangASTContext::GetFormat (void* type)
{
    if (!type)
        return lldb::eFormatDefault;
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:
            break;
            
        case clang::Type::FunctionNoProto:
        case clang::Type::FunctionProto:
            break;
            
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            break;
            
        case clang::Type::ConstantArray:
            return lldb::eFormatVoid; // no value
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
            break;
            
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
                //default: assert(0 && "Unknown builtin type!");
            case clang::BuiltinType::UnknownAny:
            case clang::BuiltinType::Void:
            case clang::BuiltinType::BoundMember:
                break;
                
            case clang::BuiltinType::Bool:          return lldb::eFormatBoolean;
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:       return lldb::eFormatChar;
            case clang::BuiltinType::Char16:        return lldb::eFormatUnicode16;
            case clang::BuiltinType::Char32:        return lldb::eFormatUnicode32;
            case clang::BuiltinType::UShort:        return lldb::eFormatUnsigned;
            case clang::BuiltinType::Short:         return lldb::eFormatDecimal;
            case clang::BuiltinType::UInt:          return lldb::eFormatUnsigned;
            case clang::BuiltinType::Int:           return lldb::eFormatDecimal;
            case clang::BuiltinType::ULong:         return lldb::eFormatUnsigned;
            case clang::BuiltinType::Long:          return lldb::eFormatDecimal;
            case clang::BuiltinType::ULongLong:     return lldb::eFormatUnsigned;
            case clang::BuiltinType::LongLong:      return lldb::eFormatDecimal;
            case clang::BuiltinType::UInt128:       return lldb::eFormatUnsigned;
            case clang::BuiltinType::Int128:        return lldb::eFormatDecimal;
            case clang::BuiltinType::Float:         return lldb::eFormatFloat;
            case clang::BuiltinType::Double:        return lldb::eFormatFloat;
            case clang::BuiltinType::LongDouble:    return lldb::eFormatFloat;
            case clang::BuiltinType::NullPtr:
            case clang::BuiltinType::Overload:
            case clang::BuiltinType::Dependent:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCSel:
            case clang::BuiltinType::Half:
            case clang::BuiltinType::ARCUnbridgedCast:
            case clang::BuiltinType::PseudoObject:
            case clang::BuiltinType::BuiltinFn:
            case clang::BuiltinType::OCLEvent:
            case clang::BuiltinType::OCLImage1d:
            case clang::BuiltinType::OCLImage1dArray:
            case clang::BuiltinType::OCLImage1dBuffer:
            case clang::BuiltinType::OCLImage2d:
            case clang::BuiltinType::OCLImage2dArray:
            case clang::BuiltinType::OCLImage3d:
            case clang::BuiltinType::OCLSampler:
                return lldb::eFormatHex;
        }
            break;
        case clang::Type::ObjCObjectPointer:        return lldb::eFormatHex;
        case clang::Type::BlockPointer:             return lldb::eFormatHex;
        case clang::Type::Pointer:                  return lldb::eFormatHex;
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:          return lldb::eFormatHex;
        case clang::Type::MemberPointer:            break;
        case clang::Type::Complex:
        {
            if (qual_type->isComplexType())
                return lldb::eFormatComplex;
            else
                return lldb::eFormatComplexInteger;
        }
        case clang::Type::ObjCInterface:            break;
        case clang::Type::Record:                   break;
        case clang::Type::Enum:                     return lldb::eFormatEnum;
        case clang::Type::Typedef:
            return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetFormat();
        case clang::Type::Auto:
            return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Paren:
            return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Elaborated:
            return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetFormat();
        case clang::Type::DependentSizedArray:
        case clang::Type::DependentSizedExtVector:
        case clang::Type::UnresolvedUsing:
        case clang::Type::Attributed:
        case clang::Type::TemplateTypeParm:
        case clang::Type::SubstTemplateTypeParm:
        case clang::Type::SubstTemplateTypeParmPack:
        case clang::Type::InjectedClassName:
        case clang::Type::DependentName:
        case clang::Type::DependentTemplateSpecialization:
        case clang::Type::PackExpansion:
        case clang::Type::ObjCObject:
            
        case clang::Type::TypeOfExpr:
        case clang::Type::TypeOf:
        case clang::Type::Decltype:
        case clang::Type::TemplateSpecialization:
        case clang::Type::Atomic:
        case clang::Type::Adjusted:
            break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:
            break;
    }
    // We don't know hot to display this type...
    return lldb::eFormatBytes;
}

static bool
ObjCDeclHasIVars (clang::ObjCInterfaceDecl *class_interface_decl, bool check_superclass)
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

uint32_t
ClangASTContext::GetNumChildren (void* type, bool omit_empty_base_classes)
{
    if (!type)
        return 0;
    
    uint32_t num_children = 0;
    clang::QualType qual_type(GetQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
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
            if (GetCompleteQualType (getASTContext(), qual_type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    if (omit_empty_base_classes)
                    {
                        // Check each base classes to see if it or any of its
                        // base classes contain any fields. This can help
                        // limit the noise in variable views by not having to
                        // show base classes that contain no members.
                        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            
                            // Skip empty base classes
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
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
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                    ++num_children;
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteQualType (getASTContext(), qual_type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                if (ObjCDeclHasIVars (superclass_interface_decl, true))
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
            const clang::ObjCObjectPointerType *pointer_type = llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr());
            clang::QualType pointee_type = pointer_type->getPointeeType();
            uint32_t num_pointee_children = CompilerType (getASTContext(),pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
        case clang::Type::Vector:
        case clang::Type::ExtVector:
            num_children = llvm::cast<clang::VectorType>(qual_type.getTypePtr())->getNumElements();
            break;
            
        case clang::Type::ConstantArray:
            num_children = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
            break;
            
        case clang::Type::Pointer:
        {
            const clang::PointerType *pointer_type = llvm::cast<clang::PointerType>(qual_type.getTypePtr());
            clang::QualType pointee_type (pointer_type->getPointeeType());
            uint32_t num_pointee_children = CompilerType (getASTContext(),pointee_type).GetNumChildren (omit_empty_base_classes);
            if (num_pointee_children == 0)
            {
                // We have a pointer to a pointee type that claims it has no children.
                // We will want to look at
                num_children = GetNumPointeeChildren (pointee_type);
            }
            else
                num_children = num_pointee_children;
        }
            break;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
        {
            const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
            clang::QualType pointee_type = reference_type->getPointeeType();
            uint32_t num_pointee_children = CompilerType (getASTContext(), pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
            
        case clang::Type::Typedef:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Elaborated:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Paren:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumChildren (omit_empty_base_classes);
            break;
        default:
            break;
    }
    return num_children;
}

lldb::BasicType
ClangASTContext::GetBasicTypeEnumeration (void* type)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        if (type_class == clang::Type::Builtin)
        {
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                case clang::BuiltinType::Void:      return eBasicTypeVoid;
                case clang::BuiltinType::Bool:      return eBasicTypeBool;
                case clang::BuiltinType::Char_S:    return eBasicTypeSignedChar;
                case clang::BuiltinType::Char_U:    return eBasicTypeUnsignedChar;
                case clang::BuiltinType::Char16:    return eBasicTypeChar16;
                case clang::BuiltinType::Char32:    return eBasicTypeChar32;
                case clang::BuiltinType::UChar:     return eBasicTypeUnsignedChar;
                case clang::BuiltinType::SChar:     return eBasicTypeSignedChar;
                case clang::BuiltinType::WChar_S:   return eBasicTypeSignedWChar;
                case clang::BuiltinType::WChar_U:   return eBasicTypeUnsignedWChar;
                case clang::BuiltinType::Short:     return eBasicTypeShort;
                case clang::BuiltinType::UShort:    return eBasicTypeUnsignedShort;
                case clang::BuiltinType::Int:       return eBasicTypeInt;
                case clang::BuiltinType::UInt:      return eBasicTypeUnsignedInt;
                case clang::BuiltinType::Long:      return eBasicTypeLong;
                case clang::BuiltinType::ULong:     return eBasicTypeUnsignedLong;
                case clang::BuiltinType::LongLong:  return eBasicTypeLongLong;
                case clang::BuiltinType::ULongLong: return eBasicTypeUnsignedLongLong;
                case clang::BuiltinType::Int128:    return eBasicTypeInt128;
                case clang::BuiltinType::UInt128:   return eBasicTypeUnsignedInt128;
                    
                case clang::BuiltinType::Half:      return eBasicTypeHalf;
                case clang::BuiltinType::Float:     return eBasicTypeFloat;
                case clang::BuiltinType::Double:    return eBasicTypeDouble;
                case clang::BuiltinType::LongDouble:return eBasicTypeLongDouble;
                    
                case clang::BuiltinType::NullPtr:   return eBasicTypeNullPtr;
                case clang::BuiltinType::ObjCId:    return eBasicTypeObjCID;
                case clang::BuiltinType::ObjCClass: return eBasicTypeObjCClass;
                case clang::BuiltinType::ObjCSel:   return eBasicTypeObjCSel;
                case clang::BuiltinType::Dependent:
                case clang::BuiltinType::Overload:
                case clang::BuiltinType::BoundMember:
                case clang::BuiltinType::PseudoObject:
                case clang::BuiltinType::UnknownAny:
                case clang::BuiltinType::BuiltinFn:
                case clang::BuiltinType::ARCUnbridgedCast:
                case clang::BuiltinType::OCLEvent:
                case clang::BuiltinType::OCLImage1d:
                case clang::BuiltinType::OCLImage1dArray:
                case clang::BuiltinType::OCLImage1dBuffer:
                case clang::BuiltinType::OCLImage2d:
                case clang::BuiltinType::OCLImage2dArray:
                case clang::BuiltinType::OCLImage3d:
                case clang::BuiltinType::OCLSampler:
                    return eBasicTypeOther;
            }
        }
    }
    return eBasicTypeInvalid;
}


#pragma mark Aggregate Types

uint32_t
ClangASTContext::GetNumDirectBaseClasses (const CompilerType& type)
{
    if (!type)
        return 0;
    ClangASTContext *ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumBases();
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            count = GetNumDirectBaseClasses(ast->GetPointeeType(type.GetOpaqueQualType()));
            break;
            
        case clang::Type::ObjCObject:
            if (ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
        case clang::Type::ObjCInterface:
            if (ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::ObjCInterfaceType *objc_interface_type = qual_type->getAs<clang::ObjCInterfaceType>();
                if (objc_interface_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_interface_type->getInterface();
                    
                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            count = GetNumDirectBaseClasses(CompilerType (ast->getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()));
            break;
            
        case clang::Type::Elaborated:
            count = GetNumDirectBaseClasses(CompilerType (ast->getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()));
            break;
            
        case clang::Type::Paren:
            return GetNumDirectBaseClasses(CompilerType (ast->getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()));
            
        default:
            break;
    }
    return count;
}

uint32_t
ClangASTContext::GetNumVirtualBaseClasses (const CompilerType& type)
{
    if (!type)
        return 0;
    ClangASTContext *ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumVBases();
            }
            break;
            
        case clang::Type::Typedef:
            count = GetNumVirtualBaseClasses(CompilerType (ast->getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()));
            break;
            
        case clang::Type::Elaborated:
            count = GetNumVirtualBaseClasses(CompilerType (ast->getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()));
            break;
            
        case clang::Type::Paren:
            count = GetNumVirtualBaseClasses(CompilerType (ast->getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()));
            break;
            
        default:
            break;
    }
    return count;
}

uint32_t
ClangASTContext::GetNumFields (void* type)
{
    if (!type)
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::dyn_cast<clang::RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        uint32_t field_idx = 0;
                        clang::RecordDecl::field_iterator field, field_end;
                        for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                            ++field_idx;
                        count = field_idx;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            count = CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumFields();
            break;
            
        case clang::Type::Elaborated:
            count = CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumFields();
            break;
            
        case clang::Type::Paren:
            count = CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumFields();
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        default:
            break;
    }
    return count;
}

CompilerType
ClangASTContext::GetDirectBaseClassAtIndex (const CompilerType& type, size_t idx, uint32_t *bit_offset_ptr)
{
    if (!type)
        return CompilerType();
    ClangASTContext *ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return CompilerType();
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const clang::ASTRecordLayout &record_layout = ast->getASTContext()->getASTRecordLayout(cxx_record_decl);
                                const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                                if (base_class->isVirtual())
                                    *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                                else
                                    *bit_offset_ptr = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            }
                            return CompilerType (ast, base_class->getType().getAsOpaquePtr());
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            return GetDirectBaseClassAtIndex(ast->GetPointeeType(type.GetOpaqueQualType()), idx, bit_offset_ptr);
            
        case clang::Type::ObjCObject:
            if (idx == 0 && ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (bit_offset_ptr)
                                *bit_offset_ptr = 0;
                            return CompilerType (ast->getASTContext(), ast->getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                        }
                    }
                }
            }
            break;
        case clang::Type::ObjCInterface:
            if (idx == 0 && ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::ObjCObjectType *objc_interface_type = qual_type->getAs<clang::ObjCInterfaceType>();
                if (objc_interface_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_interface_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (bit_offset_ptr)
                                *bit_offset_ptr = 0;
                            return CompilerType (ast->getASTContext(), ast->getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                        }
                    }
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            return GetDirectBaseClassAtIndex (CompilerType (ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr()), idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return GetDirectBaseClassAtIndex (CompilerType (ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr()), idx, bit_offset_ptr);
            
        case clang::Type::Paren:
            return GetDirectBaseClassAtIndex (CompilerType (ast, llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr()), idx, bit_offset_ptr);
            
        default:
            break;
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetVirtualBaseClassAtIndex (const CompilerType& type, size_t idx, uint32_t *bit_offset_ptr)
{
    if (!type)
        return CompilerType();
    ClangASTContext *ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return CompilerType();
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (ast->GetCompleteType(type.GetOpaqueQualType()))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->vbases_begin(), base_class_end = cxx_record_decl->vbases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const clang::ASTRecordLayout &record_layout = ast->getASTContext()->getASTRecordLayout(cxx_record_decl);
                                const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                                *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                                
                            }
                            return CompilerType (ast, base_class->getType().getAsOpaquePtr());
                        }
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return GetVirtualBaseClassAtIndex (CompilerType (ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr()), idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return GetVirtualBaseClassAtIndex (CompilerType (ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr()), idx, bit_offset_ptr);
            
        case clang::Type::Paren:
            return  GetVirtualBaseClassAtIndex (CompilerType (ast, llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr()), idx, bit_offset_ptr);
            
        default:
            break;
    }
    return CompilerType();
}

static clang_type_t
GetObjCFieldAtIndex (clang::ASTContext *ast,
                     clang::ObjCInterfaceDecl *class_interface_decl,
                     size_t idx,
                     std::string& name,
                     uint64_t *bit_offset_ptr,
                     uint32_t *bitfield_bit_size_ptr,
                     bool *is_bitfield_ptr)
{
    if (class_interface_decl)
    {
        if (idx < (class_interface_decl->ivar_size()))
        {
            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
            uint32_t ivar_idx = 0;
            
            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++ivar_idx)
            {
                if (ivar_idx == idx)
                {
                    const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                    
                    clang::QualType ivar_qual_type(ivar_decl->getType());
                    
                    name.assign(ivar_decl->getNameAsString());
                    
                    if (bit_offset_ptr)
                    {
                        const clang::ASTRecordLayout &interface_layout = ast->getASTObjCInterfaceLayout(class_interface_decl);
                        *bit_offset_ptr = interface_layout.getFieldOffset (ivar_idx);
                    }
                    
                    const bool is_bitfield = ivar_pos->isBitField();
                    
                    if (bitfield_bit_size_ptr)
                    {
                        *bitfield_bit_size_ptr = 0;
                        
                        if (is_bitfield && ast)
                        {
                            clang::Expr *bitfield_bit_size_expr = ivar_pos->getBitWidth();
                            llvm::APSInt bitfield_apsint;
                            if (bitfield_bit_size_expr && bitfield_bit_size_expr->EvaluateAsInt(bitfield_apsint, *ast))
                            {
                                *bitfield_bit_size_ptr = bitfield_apsint.getLimitedValue();
                            }
                        }
                    }
                    if (is_bitfield_ptr)
                        *is_bitfield_ptr = is_bitfield;
                    
                    return ivar_qual_type.getAsOpaquePtr();
                }
            }
        }
    }
    return nullptr;
}

CompilerType
ClangASTContext::GetFieldAtIndex (void* type, size_t idx,
                                     std::string& name,
                                     uint64_t *bit_offset_ptr,
                                     uint32_t *bitfield_bit_size_ptr,
                                     bool *is_bitfield_ptr)
{
    if (!type)
        return CompilerType();
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx)
                {
                    if (idx == field_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        name.assign(field->getNameAsString());
                        
                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        if (bit_offset_ptr)
                        {
                            const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(record_decl);
                            *bit_offset_ptr = record_layout.getFieldOffset (field_idx);
                        }
                        
                        const bool is_bitfield = field->isBitField();
                        
                        if (bitfield_bit_size_ptr)
                        {
                            *bitfield_bit_size_ptr = 0;
                            
                            if (is_bitfield)
                            {
                                clang::Expr *bitfield_bit_size_expr = field->getBitWidth();
                                llvm::APSInt bitfield_apsint;
                                if (bitfield_bit_size_expr && bitfield_bit_size_expr->EvaluateAsInt(bitfield_apsint, *getASTContext()))
                                {
                                    *bitfield_bit_size_ptr = bitfield_apsint.getLimitedValue();
                                }
                            }
                        }
                        if (is_bitfield_ptr)
                            *is_bitfield_ptr = is_bitfield;
                        
                        return CompilerType (getASTContext(), field->getType());
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    return CompilerType (this, GetObjCFieldAtIndex(getASTContext(), class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    return CompilerType (this, GetObjCFieldAtIndex(getASTContext(), class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        case clang::Type::Elaborated:
            return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        case clang::Type::Paren:
            return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        default:
            break;
    }
    return CompilerType();
}

// If a pointer to a pointee type (the clang_type arg) says that it has no
// children, then we either need to trust it, or override it and return a
// different result. For example, an "int *" has one child that is an integer,
// but a function pointer doesn't have any children. Likewise if a Record type
// claims it has no children, then there really is nothing to show.
uint32_t
ClangASTContext::GetNumPointeeChildren (clang::QualType type)
{
    if (type.isNull())
        return 0;
    
    clang::QualType qual_type(type.getCanonicalType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            case clang::BuiltinType::UnknownAny:
            case clang::BuiltinType::Void:
            case clang::BuiltinType::NullPtr:
            case clang::BuiltinType::OCLEvent:
            case clang::BuiltinType::OCLImage1d:
            case clang::BuiltinType::OCLImage1dArray:
            case clang::BuiltinType::OCLImage1dBuffer:
            case clang::BuiltinType::OCLImage2d:
            case clang::BuiltinType::OCLImage2dArray:
            case clang::BuiltinType::OCLImage3d:
            case clang::BuiltinType::OCLSampler:
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
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCSel:
            case clang::BuiltinType::BoundMember:
            case clang::BuiltinType::Half:
            case clang::BuiltinType::ARCUnbridgedCast:
            case clang::BuiltinType::PseudoObject:
            case clang::BuiltinType::BuiltinFn:
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
        case clang::Type::Paren:                    return GetNumPointeeChildren (llvm::cast<clang::ParenType>(qual_type)->desugar());
        case clang::Type::Typedef:                  return GetNumPointeeChildren (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType());
        case clang::Type::Elaborated:               return GetNumPointeeChildren (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType());
        case clang::Type::TypeOfExpr:               return 0;
        case clang::Type::TypeOf:                   return 0;
        case clang::Type::Decltype:                 return 0;
        case clang::Type::Record:                   return 0;
        case clang::Type::Enum:                     return 1;
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


CompilerType
ClangASTContext::GetChildClangTypeAtIndex (void* type, ExecutionContext *exe_ctx,
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
                                              ValueObject *valobj)
{
    if (!type)
        return CompilerType();
    
    clang::QualType parent_qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass parent_type_class = parent_qual_type->getTypeClass();
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    
    const bool idx_is_valid = idx < GetNumChildren (type, omit_empty_base_classes);
    uint32_t bit_offset;
    switch (parent_type_class)
    {
        case clang::Type::Builtin:
            if (idx_is_valid)
            {
                switch (llvm::cast<clang::BuiltinType>(parent_qual_type)->getKind())
                {
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                        child_name = "isa";
                        child_byte_size = getASTContext()->getTypeSize(getASTContext()->ObjCBuiltinClassTy) / CHAR_BIT;
                        return CompilerType (getASTContext(), getASTContext()->ObjCBuiltinClassTy);
                        
                    default:
                        break;
                }
            }
            break;
            
        case clang::Type::Record:
            if (idx_is_valid && GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(parent_qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;
                
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const clang::CXXRecordDecl *base_class_decl = nullptr;
                        
                        // Skip empty base classes
                        if (omit_empty_base_classes)
                        {
                            base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                        }
                        
                        if (idx == child_idx)
                        {
                            if (base_class_decl == nullptr)
                                base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            
                            
                            if (base_class->isVirtual())
                            {
                                bool handled = false;
                                if (valobj)
                                {
                                    Error err;
                                    AddressType addr_type = eAddressTypeInvalid;
                                    lldb::addr_t vtable_ptr_addr = valobj->GetCPPVTableAddress(addr_type);
                                    
                                    if (vtable_ptr_addr != LLDB_INVALID_ADDRESS && addr_type == eAddressTypeLoad)
                                    {
                                        
                                        ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
                                        Process *process = exe_ctx.GetProcessPtr();
                                        if (process)
                                        {
                                            clang::VTableContextBase *vtable_ctx = getASTContext()->getVTableContext();
                                            if (vtable_ctx)
                                            {
                                                if (vtable_ctx->isMicrosoft())
                                                {
                                                    clang::MicrosoftVTableContext *msoft_vtable_ctx = static_cast<clang::MicrosoftVTableContext *>(vtable_ctx);
                                                    
                                                    if (vtable_ptr_addr)
                                                    {
                                                        const lldb::addr_t vbtable_ptr_addr = vtable_ptr_addr + record_layout.getVBPtrOffset().getQuantity();
                                                        
                                                        const lldb::addr_t vbtable_ptr = process->ReadPointerFromMemory(vbtable_ptr_addr, err);
                                                        if (vbtable_ptr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            // Get the index into the virtual base table. The index is the index in uint32_t from vbtable_ptr
                                                            const unsigned vbtable_index = msoft_vtable_ctx->getVBTableIndex(cxx_record_decl, base_class_decl);
                                                            const lldb::addr_t base_offset_addr = vbtable_ptr + vbtable_index * 4;
                                                            const uint32_t base_offset = process->ReadUnsignedIntegerFromMemory(base_offset_addr, 4, UINT32_MAX, err);
                                                            if (base_offset != UINT32_MAX)
                                                            {
                                                                handled = true;
                                                                bit_offset = base_offset * 8;
                                                            }
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    clang::ItaniumVTableContext *itanium_vtable_ctx = static_cast<clang::ItaniumVTableContext *>(vtable_ctx);
                                                    if (vtable_ptr_addr)
                                                    {
                                                        const lldb::addr_t vtable_ptr = process->ReadPointerFromMemory(vtable_ptr_addr, err);
                                                        if (vtable_ptr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            clang::CharUnits base_offset_offset = itanium_vtable_ctx->getVirtualBaseOffsetOffset(cxx_record_decl, base_class_decl);
                                                            const lldb::addr_t base_offset_addr = vtable_ptr + base_offset_offset.getQuantity();
                                                            const uint32_t base_offset = process->ReadUnsignedIntegerFromMemory(base_offset_addr, 4, UINT32_MAX, err);
                                                            if (base_offset != UINT32_MAX)
                                                            {
                                                                handled = true;
                                                                bit_offset = base_offset * 8;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                }
                                if (!handled)
                                    bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                            }
                            else
                                bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            
                            // Base classes should be a multiple of 8 bits in size
                            child_byte_offset = bit_offset/8;
                            CompilerType base_class_clang_type(getASTContext(), base_class->getType());
                            child_name = base_class_clang_type.GetTypeName().AsCString("");
                            uint64_t base_class_clang_type_bit_size = base_class_clang_type.GetBitSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                            
                            // Base classes bit sizes should be a multiple of 8 bits in size
                            assert (base_class_clang_type_bit_size % 8 == 0);
                            child_byte_size = base_class_clang_type_bit_size / 8;
                            child_is_base_class = true;
                            return base_class_clang_type;
                        }
                        // We don't increment the child index in the for loop since we might
                        // be skipping empty base classes
                        ++child_idx;
                    }
                }
                // Make sure index is in range...
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
                {
                    if (idx == child_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        child_name.assign(field->getNameAsString().c_str());
                        
                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        CompilerType field_clang_type (getASTContext(), field->getType());
                        assert(field_idx < record_layout.getFieldCount());
                        child_byte_size = field_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        
                        // Figure out the field offset within the current struct/union/class type
                        bit_offset = record_layout.getFieldOffset (field_idx);
                        child_byte_offset = bit_offset / 8;
                        if (ClangASTContext::FieldIsBitfield (getASTContext(), *field, child_bitfield_bit_size))
                            child_bitfield_bit_offset = bit_offset % 8;
                        
                        return field_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (idx_is_valid && GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(parent_qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        const clang::ASTRecordLayout &interface_layout = getASTContext()->getASTObjCInterfaceLayout(class_interface_decl);
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                CompilerType base_class_clang_type (getASTContext(), getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                                if (base_class_clang_type.GetNumChildren(omit_empty_base_classes) > 0)
                                {
                                    if (idx == 0)
                                    {
                                        clang::QualType ivar_qual_type(getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                                        
                                        
                                        child_name.assign(superclass_interface_decl->getNameAsString().c_str());
                                        
                                        clang::TypeInfo ivar_type_info = getASTContext()->getTypeInfo(ivar_qual_type.getTypePtr());
                                        
                                        child_byte_size = ivar_type_info.Width / 8;
                                        child_byte_offset = 0;
                                        child_is_base_class = true;
                                        
                                        return CompilerType (getASTContext(), ivar_qual_type);
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
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
                            {
                                if (child_idx == idx)
                                {
                                    clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                    
                                    clang::QualType ivar_qual_type(ivar_decl->getType());
                                    
                                    child_name.assign(ivar_decl->getNameAsString().c_str());
                                    
                                    clang::TypeInfo  ivar_type_info = getASTContext()->getTypeInfo(ivar_qual_type.getTypePtr());
                                    
                                    child_byte_size = ivar_type_info.Width / 8;
                                    
                                    // Figure out the field offset within the current struct/union/class type
                                    // For ObjC objects, we can't trust the bit offset we get from the Clang AST, since
                                    // that doesn't account for the space taken up by unbacked properties, or from
                                    // the changing size of base classes that are newer than this class.
                                    // So if we have a process around that we can ask about this object, do so.
                                    child_byte_offset = LLDB_INVALID_IVAR_OFFSET;
                                    Process *process = nullptr;
                                    if (exe_ctx)
                                        process = exe_ctx->GetProcessPtr();
                                    if (process)
                                    {
                                        ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                                        if (objc_runtime != nullptr)
                                        {
                                            CompilerType parent_ast_type (getASTContext(), parent_qual_type);
                                            child_byte_offset = objc_runtime->GetByteOffsetForIvar (parent_ast_type, ivar_decl->getNameAsString().c_str());
                                        }
                                    }
                                    
                                    // Setting this to UINT32_MAX to make sure we don't compute it twice...
                                    bit_offset = UINT32_MAX;
                                    
                                    if (child_byte_offset == static_cast<int32_t>(LLDB_INVALID_IVAR_OFFSET))
                                    {
                                        bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                        child_byte_offset = bit_offset / 8;
                                    }
                                    
                                    // Note, the ObjC Ivar Byte offset is just that, it doesn't account for the bit offset
                                    // of a bitfield within its containing object.  So regardless of where we get the byte
                                    // offset from, we still need to get the bit offset for bitfields from the layout.
                                    
                                    if (ClangASTContext::FieldIsBitfield (getASTContext(), ivar_decl, child_bitfield_bit_size))
                                    {
                                        if (bit_offset == UINT32_MAX)
                                            bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                        
                                        child_bitfield_bit_offset = bit_offset % 8;
                                    }
                                    return CompilerType (getASTContext(), ivar_qual_type);
                                }
                                ++child_idx;
                            }
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (idx_is_valid)
            {
                CompilerType pointee_clang_type (GetPointeeType(type));
                
                if (transparent_pointers && pointee_clang_type.IsAggregateType())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        idx,
                                                                        transparent_pointers,
                                                                        omit_empty_base_classes,
                                                                        ignore_array_bounds,
                                                                        child_name,
                                                                        child_byte_size,
                                                                        child_byte_offset,
                                                                        child_bitfield_bit_size,
                                                                        child_bitfield_bit_offset,
                                                                        child_is_base_class,
                                                                        tmp_child_is_deref_of_parent,
                                                                        valobj);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0 && pointee_clang_type.GetCompleteType())
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Vector:
        case clang::Type::ExtVector:
            if (idx_is_valid)
            {
                const clang::VectorType *array = llvm::cast<clang::VectorType>(parent_qual_type.getTypePtr());
                if (array)
                {
                    CompilerType element_type (getASTContext(), array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%" PRIu64 "]", static_cast<uint64_t>(idx));
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                        return element_type;
                    }
                }
            }
            break;
            
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
            if (ignore_array_bounds || idx_is_valid)
            {
                const clang::ArrayType *array = GetQualType(type)->getAsArrayTypeUnsafe();
                if (array)
                {
                    CompilerType element_type (getASTContext(), array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%" PRIu64 "]", static_cast<uint64_t>(idx));
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                        return element_type;
                    }
                }
            }
            break;
            
            
        case clang::Type::Pointer:
            if (idx_is_valid)
            {
                CompilerType pointee_clang_type (GetPointeeType(type));
                
                // Don't dereference "void *" pointers
                if (pointee_clang_type.IsVoidType())
                    return CompilerType();
                
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        idx,
                                                                        transparent_pointers,
                                                                        omit_empty_base_classes,
                                                                        ignore_array_bounds,
                                                                        child_name,
                                                                        child_byte_size,
                                                                        child_byte_offset,
                                                                        child_bitfield_bit_size,
                                                                        child_bitfield_bit_offset,
                                                                        child_is_base_class,
                                                                        tmp_child_is_deref_of_parent,
                                                                        valobj);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (idx_is_valid)
            {
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(parent_qual_type.getTypePtr());
                CompilerType pointee_clang_type (getASTContext(), reference_type->getPointeeType());
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                        idx,
                                                                        transparent_pointers,
                                                                        omit_empty_base_classes,
                                                                        ignore_array_bounds,
                                                                        child_name,
                                                                        child_byte_size,
                                                                        child_byte_offset,
                                                                        child_bitfield_bit_size,
                                                                        child_bitfield_bit_offset,
                                                                        child_is_base_class,
                                                                        tmp_child_is_deref_of_parent,
                                                                        valobj);
                }
                else
                {
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '&');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
        {
            CompilerType typedefed_clang_type (getASTContext(), llvm::cast<clang::TypedefType>(parent_qual_type)->getDecl()->getUnderlyingType());
            return typedefed_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                  idx,
                                                                  transparent_pointers,
                                                                  omit_empty_base_classes,
                                                                  ignore_array_bounds,
                                                                  child_name,
                                                                  child_byte_size,
                                                                  child_byte_offset,
                                                                  child_bitfield_bit_size,
                                                                  child_bitfield_bit_offset,
                                                                  child_is_base_class,
                                                                  child_is_deref_of_parent,
                                                                  valobj);
        }
            break;
            
        case clang::Type::Elaborated:
        {
            CompilerType elaborated_clang_type (getASTContext(), llvm::cast<clang::ElaboratedType>(parent_qual_type)->getNamedType());
            return elaborated_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                                   idx,
                                                                   transparent_pointers,
                                                                   omit_empty_base_classes,
                                                                   ignore_array_bounds,
                                                                   child_name,
                                                                   child_byte_size,
                                                                   child_byte_offset,
                                                                   child_bitfield_bit_size,
                                                                   child_bitfield_bit_offset,
                                                                   child_is_base_class,
                                                                   child_is_deref_of_parent,
                                                                   valobj);
        }
            
        case clang::Type::Paren:
        {
            CompilerType paren_clang_type (getASTContext(), llvm::cast<clang::ParenType>(parent_qual_type)->desugar());
            return paren_clang_type.GetChildClangTypeAtIndex (exe_ctx,
                                                              idx,
                                                              transparent_pointers,
                                                              omit_empty_base_classes,
                                                              ignore_array_bounds,
                                                              child_name,
                                                              child_byte_size,
                                                              child_byte_offset,
                                                              child_bitfield_bit_size,
                                                              child_bitfield_bit_offset,
                                                              child_is_base_class,
                                                              child_is_deref_of_parent,
                                                              valobj);
        }
            
            
        default:
            break;
    }
    return CompilerType();
}

static uint32_t
GetIndexForRecordBase
(
 const clang::RecordDecl *record_decl,
 const clang::CXXBaseSpecifier *base_spec,
 bool omit_empty_base_classes
 )
{
    uint32_t child_idx = 0;
    
    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
    
    //    const char *super_name = record_decl->getNameAsCString();
    //    const char *base_name = base_spec->getType()->getAs<clang::RecordType>()->getDecl()->getNameAsCString();
    //    printf ("GetIndexForRecordChild (%s, %s)\n", super_name, base_name);
    //
    if (cxx_record_decl)
    {
        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
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
            //                    base_class->getType()->getAs<clang::RecordType>()->getDecl()->getNameAsCString());
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
GetIndexForRecordChild (const clang::RecordDecl *record_decl,
                        clang::NamedDecl *canonical_decl,
                        bool omit_empty_base_classes)
{
    uint32_t child_idx = ClangASTContext::GetNumBaseClasses (llvm::dyn_cast<clang::CXXRecordDecl>(record_decl),
                                                             omit_empty_base_classes);
    
    clang::RecordDecl::field_iterator field, field_end;
    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
         field != field_end;
         ++field, ++child_idx)
    {
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
ClangASTContext::GetIndexOfChildMemberWithName (void* type, const char *name,
                                                   bool omit_empty_base_classes,
                                                   std::vector<uint32_t>& child_indexes)
{
    if (type && name && name[0])
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType(type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    
                    // Try and find a field that matches NAME
                    clang::RecordDecl::field_iterator field, field_end;
                    llvm::StringRef name_sref(name);
                    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                         field != field_end;
                         ++field, ++child_idx)
                    {
                        llvm::StringRef field_name = field->getName();
                        if (field_name.empty())
                        {
                            CompilerType field_type(getASTContext(),field->getType());
                            child_indexes.push_back(child_idx);
                            if (field_type.GetIndexOfChildMemberWithName(name,  omit_empty_base_classes, child_indexes))
                                return child_indexes.size();
                            child_indexes.pop_back();
                            
                        }
                        else if (field_name.equals (name_sref))
                        {
                            // We have to add on the number of base classes to this index!
                            child_indexes.push_back (child_idx + ClangASTContext::GetNumBaseClasses (cxx_record_decl, omit_empty_base_classes));
                            return child_indexes.size();
                        }
                    }
                    
                    if (cxx_record_decl)
                    {
                        const clang::RecordDecl *parent_record_decl = cxx_record_decl;
                        
                        //printf ("parent = %s\n", parent_record_decl->getNameAsCString());
                        
                        //const Decl *root_cdecl = cxx_record_decl->getCanonicalDecl();
                        // Didn't find things easily, lets let clang do its thang...
                        clang::IdentifierInfo & ident_ref = getASTContext()->Idents.get(name_sref);
                        clang::DeclarationName decl_name(&ident_ref);
                        
                        clang::CXXBasePaths paths;
                        if (cxx_record_decl->lookupInBases([decl_name](const clang::CXXBaseSpecifier *specifier, clang::CXXBasePath &path) {
                                                               return clang::CXXRecordDecl::FindOrdinaryMember(specifier, path, decl_name);
                                                           },
                                                           paths))
                        {
                            clang::CXXBasePaths::const_paths_iterator path, path_end = paths.end();
                            for (path = paths.begin(); path != path_end; ++path)
                            {
                                const size_t num_path_elements = path->size();
                                for (size_t e=0; e<num_path_elements; ++e)
                                {
                                    clang::CXXBasePathElement elem = (*path)[e];
                                    
                                    child_idx = GetIndexForRecordBase (parent_record_decl, elem.Base, omit_empty_base_classes);
                                    if (child_idx == UINT32_MAX)
                                    {
                                        child_indexes.clear();
                                        return 0;
                                    }
                                    else
                                    {
                                        child_indexes.push_back (child_idx);
                                        parent_record_decl = llvm::cast<clang::RecordDecl>(elem.Base->getType()->getAs<clang::RecordType>()->getDecl());
                                    }
                                }
                                for (clang::NamedDecl *path_decl : path->Decls)
                                {
                                    child_idx = GetIndexForRecordChild (parent_record_decl, path_decl, omit_empty_base_classes);
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
                if (GetCompleteType(type))
                {
                    llvm::StringRef name_sref(name);
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
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
                                
                                CompilerType superclass_clang_type (getASTContext(), getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                                if (superclass_clang_type.GetIndexOfChildMemberWithName (name,
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
                CompilerType objc_object_clang_type (getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                return objc_object_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
            }
                break;
                
                
            case clang::Type::ConstantArray:
            {
                //                const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(parent_qual_type.getTypePtr());
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
                //                MemberPointerType *mem_ptr_type = llvm::cast<MemberPointerType>(qual_type.getTypePtr());
                //                clang::QualType pointee_type = mem_ptr_type->getPointeeType();
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
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                clang::QualType pointee_type(reference_type->getPointeeType());
                CompilerType pointee_clang_type (getASTContext(), pointee_type);
                
                if (pointee_clang_type.IsAggregateType ())
                {
                    return pointee_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
                }
            }
                break;
                
            case clang::Type::Pointer:
            {
                CompilerType pointee_clang_type (GetPointeeType(type));
                
                if (pointee_clang_type.IsAggregateType ())
                {
                    return pointee_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
                }
            }
                break;
                
            case clang::Type::Typedef:
                return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                                      omit_empty_base_classes,
                                                                                                                                                      child_indexes);
                
            case clang::Type::Elaborated:
                return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                         omit_empty_base_classes,
                                                                                                                                         child_indexes);
                
            case clang::Type::Paren:
                return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildMemberWithName (name,
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
ClangASTContext::GetIndexOfChildWithName (void* type, const char *name, bool omit_empty_base_classes)
{
    if (type && name && name[0])
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType(type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    
                    if (cxx_record_decl)
                    {
                        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            // Skip empty base classes
                            clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            if (omit_empty_base_classes && ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                            
                            CompilerType base_class_clang_type (getASTContext(), base_class->getType());
                            std::string base_class_type_name (base_class_clang_type.GetTypeName().AsCString(""));
                            if (base_class_type_name.compare (name) == 0)
                                return child_idx;
                            ++child_idx;
                        }
                    }
                    
                    // Try and find a field that matches NAME
                    clang::RecordDecl::field_iterator field, field_end;
                    llvm::StringRef name_sref(name);
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
                if (GetCompleteType(type))
                {
                    llvm::StringRef name_sref(name);
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
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
                CompilerType pointee_clang_type (getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                return pointee_clang_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
            }
                break;
                
            case clang::Type::ConstantArray:
            {
                //                const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(parent_qual_type.getTypePtr());
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
                //                MemberPointerType *mem_ptr_type = llvm::cast<MemberPointerType>(qual_type.getTypePtr());
                //                clang::QualType pointee_type = mem_ptr_type->getPointeeType();
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
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                CompilerType pointee_type (getASTContext(), reference_type->getPointeeType());
                
                if (pointee_type.IsAggregateType ())
                {
                    return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                }
            }
                break;
                
            case clang::Type::Pointer:
            {
                const clang::PointerType *pointer_type = llvm::cast<clang::PointerType>(qual_type.getTypePtr());
                CompilerType pointee_type (getASTContext(), pointer_type->getPointeeType());
                
                if (pointee_type.IsAggregateType ())
                {
                    return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
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
                
            case clang::Type::Elaborated:
                return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Paren:
                return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Typedef:
                return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            default:
                break;
        }
    }
    return UINT32_MAX;
}


size_t
ClangASTContext::GetNumTemplateArguments (void* type)
{
    if (!type)
        return 0;

    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    const clang::ClassTemplateSpecializationDecl *template_decl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(cxx_record_decl);
                    if (template_decl)
                        return template_decl->getTemplateArgs().size();
                }
            }
            break;
            
        case clang::Type::Typedef:
            return (CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType())).GetNumTemplateArguments();
            
        case clang::Type::Elaborated:
            return (CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType())).GetNumTemplateArguments();
            
        case clang::Type::Paren:
            return (CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar())).GetNumTemplateArguments();
            
        default:
            break;
    }

    return 0;
}

CompilerType
ClangASTContext::GetTemplateArgument (void* type, size_t arg_idx, lldb::TemplateArgumentKind &kind)
{
    if (!type)
        return CompilerType();

    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    const clang::ClassTemplateSpecializationDecl *template_decl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(cxx_record_decl);
                    if (template_decl && arg_idx < template_decl->getTemplateArgs().size())
                    {
                        const clang::TemplateArgument &template_arg = template_decl->getTemplateArgs()[arg_idx];
                        switch (template_arg.getKind())
                        {
                            case clang::TemplateArgument::Null:
                                kind = eTemplateArgumentKindNull;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Type:
                                kind = eTemplateArgumentKindType;
                                return CompilerType(getASTContext(), template_arg.getAsType());
                                
                            case clang::TemplateArgument::Declaration:
                                kind = eTemplateArgumentKindDeclaration;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Integral:
                                kind = eTemplateArgumentKindIntegral;
                                return CompilerType(getASTContext(), template_arg.getIntegralType());
                                
                            case clang::TemplateArgument::Template:
                                kind = eTemplateArgumentKindTemplate;
                                return CompilerType();
                                
                            case clang::TemplateArgument::TemplateExpansion:
                                kind = eTemplateArgumentKindTemplateExpansion;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Expression:
                                kind = eTemplateArgumentKindExpression;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Pack:
                                kind = eTemplateArgumentKindPack;
                                return CompilerType();
                                
                            default:
                                assert (!"Unhandled clang::TemplateArgument::ArgKind");
                                break;
                        }
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return (CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType())).GetTemplateArgument(arg_idx, kind);
            
        case clang::Type::Elaborated:
            return (CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType())).GetTemplateArgument(arg_idx, kind);
            
        case clang::Type::Paren:
            return (CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar())).GetTemplateArgument(arg_idx, kind);
            
        default:
            break;
    }
    kind = eTemplateArgumentKindNull;
    return CompilerType ();
}

static bool
IsOperator (const char *name, clang::OverloadedOperatorKind &op_kind)
{
    if (name == nullptr || name[0] == '\0')
        return false;
    
#define OPERATOR_PREFIX "operator"
#define OPERATOR_PREFIX_LENGTH (sizeof (OPERATOR_PREFIX) - 1)
    
    const char *post_op_name = nullptr;
    
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
    op_kind = clang::NUM_OVERLOADED_OPERATORS;
    
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
                op_kind = clang::OO_New;
            else if (strcmp (post_op_name, "new[]") == 0)
                op_kind = clang::OO_Array_New;
            break;
            
        case 'd':
            if (no_space)
                return false;
            if (strcmp (post_op_name, "delete") == 0)
                op_kind = clang::OO_Delete;
            else if (strcmp (post_op_name, "delete[]") == 0)
                op_kind = clang::OO_Array_Delete;
            break;
            
        case '+':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Plus;
            else if (post_op_name[2] == '\0')
            {
                if (post_op_name[1] == '=')
                    op_kind = clang::OO_PlusEqual;
                else if (post_op_name[1] == '+')
                    op_kind = clang::OO_PlusPlus;
            }
            break;
            
        case '-':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Minus;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_MinusEqual; break;
                    case '-': op_kind = clang::OO_MinusMinus; break;
                    case '>': op_kind = clang::OO_Arrow; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '*')
                    op_kind = clang::OO_ArrowStar; break;
            }
            break;
            
        case '*':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Star;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_StarEqual;
            break;
            
        case '/':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Slash;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_SlashEqual;
            break;
            
        case '%':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Percent;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_PercentEqual;
            break;
            
            
        case '^':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Caret;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_CaretEqual;
            break;
            
        case '&':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Amp;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_AmpEqual; break;
                    case '&': op_kind = clang::OO_AmpAmp; break;
                }
            }
            break;
            
        case '|':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Pipe;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_PipeEqual; break;
                    case '|': op_kind = clang::OO_PipePipe; break;
                }
            }
            break;
            
        case '~':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Tilde;
            break;
            
        case '!':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Exclaim;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_ExclaimEqual;
            break;
            
        case '=':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Equal;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_EqualEqual;
            break;
            
        case '<':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Less;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '<': op_kind = clang::OO_LessLess; break;
                    case '=': op_kind = clang::OO_LessEqual; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '=')
                    op_kind = clang::OO_LessLessEqual;
            }
            break;
            
        case '>':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Greater;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '>': op_kind = clang::OO_GreaterGreater; break;
                    case '=': op_kind = clang::OO_GreaterEqual; break;
                }
            }
            else if (post_op_name[1] == '>' &&
                     post_op_name[2] == '=' &&
                     post_op_name[3] == '\0')
            {
                op_kind = clang::OO_GreaterGreaterEqual;
            }
            break;
            
        case ',':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Comma;
            break;
            
        case '(':
            if (post_op_name[1] == ')' && post_op_name[2] == '\0')
                op_kind = clang::OO_Call;
            break;
            
        case '[':
            if (post_op_name[1] == ']' && post_op_name[2] == '\0')
                op_kind = clang::OO_Subscript;
            break;
    }
    
    return true;
}

clang::EnumDecl *
ClangASTContext::GetAsEnumDecl (const CompilerType& type)
{
    const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(GetCanonicalQualType(type));
    if (enutype)
        return enutype->getDecl();
    return NULL;
}

clang::RecordDecl *
ClangASTContext::GetAsRecordDecl (const CompilerType& type)
{
    const clang::RecordType *record_type = llvm::dyn_cast<clang::RecordType>(GetCanonicalQualType(type));
    if (record_type)
        return record_type->getDecl();
    return nullptr;
}

clang::CXXRecordDecl *
ClangASTContext::GetAsCXXRecordDecl (void* type)
{
    return GetCanonicalQualType(type)->getAsCXXRecordDecl();
}

clang::ObjCInterfaceDecl *
ClangASTContext::GetAsObjCInterfaceDecl (const CompilerType& type)
{
    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(GetCanonicalQualType(type));
    if (objc_class_type)
        return objc_class_type->getInterface();
    return nullptr;
}

clang::FieldDecl *
ClangASTContext::AddFieldToRecordType (const CompilerType& type, const char *name,
                                       const CompilerType &field_clang_type,
                                       AccessType access,
                                       uint32_t bitfield_bit_size)
{
    if (!type.IsValid() || !field_clang_type.IsValid())
        return nullptr;
    ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return nullptr;
    clang::ASTContext* clang_ast = ast->getASTContext();
    
    clang::FieldDecl *field = nullptr;
    
    clang::Expr *bit_width = nullptr;
    if (bitfield_bit_size != 0)
    {
        llvm::APInt bitfield_bit_size_apint(clang_ast->getTypeSize(clang_ast->IntTy), bitfield_bit_size);
        bit_width = new (*clang_ast)clang::IntegerLiteral (*clang_ast, bitfield_bit_size_apint, clang_ast->IntTy, clang::SourceLocation());
    }
    
    clang::RecordDecl *record_decl = ast->GetAsRecordDecl (type);
    if (record_decl)
    {
        field = clang::FieldDecl::Create (*clang_ast,
                                          record_decl,
                                          clang::SourceLocation(),
                                          clang::SourceLocation(),
                                          name ? &clang_ast->Idents.get(name) : nullptr,  // Identifier
                                          GetQualType(field_clang_type),             // Field type
                                          nullptr,                                    // TInfo *
                                          bit_width,                                  // BitWidth
                                          false,                                      // Mutable
                                          clang::ICIS_NoInit);                        // HasInit
        
        if (!name)
        {
            // Determine whether this field corresponds to an anonymous
            // struct or union.
            if (const clang::TagType *TagT = field->getType()->getAs<clang::TagType>()) {
                if (clang::RecordDecl *Rec = llvm::dyn_cast<clang::RecordDecl>(TagT->getDecl()))
                    if (!Rec->getDeclName()) {
                        Rec->setAnonymousStructOrUnion(true);
                        field->setImplicit();
                        
                    }
            }
        }
        
        if (field)
        {
            field->setAccess (ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));
            
            record_decl->addDecl(field);
            
#ifdef LLDB_CONFIGURATION_DEBUG
            VerifyDecl(field);
#endif
        }
    }
    else
    {
        clang::ObjCInterfaceDecl *class_interface_decl = ast->GetAsObjCInterfaceDecl (type);
        
        if (class_interface_decl)
        {
            const bool is_synthesized = false;
            
            field_clang_type.GetCompleteType();
            
            field = clang::ObjCIvarDecl::Create (*clang_ast,
                                                 class_interface_decl,
                                                 clang::SourceLocation(),
                                                 clang::SourceLocation(),
                                                 name ? &clang_ast->Idents.get(name) : nullptr,   // Identifier
                                                 GetQualType(field_clang_type),           // Field type
                                                 nullptr,                                     // TypeSourceInfo *
                                                 ConvertAccessTypeToObjCIvarAccessControl (access),
                                                 bit_width,
                                                 is_synthesized);
            
            if (field)
            {
                class_interface_decl->addDecl(field);
                
#ifdef LLDB_CONFIGURATION_DEBUG
                VerifyDecl(field);
#endif
            }
        }
    }
    return field;
}

void
ClangASTContext::BuildIndirectFields (const CompilerType& type)
{
    ClangASTContext* ast = nullptr;
    if (type)
        ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return;

    clang::RecordDecl *record_decl = ast->GetAsRecordDecl(type);
    
    if (!record_decl)
        return;
    
    typedef llvm::SmallVector <clang::IndirectFieldDecl *, 1> IndirectFieldVector;
    
    IndirectFieldVector indirect_fields;
    clang::RecordDecl::field_iterator field_pos;
    clang::RecordDecl::field_iterator field_end_pos = record_decl->field_end();
    clang::RecordDecl::field_iterator last_field_pos = field_end_pos;
    for (field_pos = record_decl->field_begin(); field_pos != field_end_pos; last_field_pos = field_pos++)
    {
        if (field_pos->isAnonymousStructOrUnion())
        {
            clang::QualType field_qual_type = field_pos->getType();
            
            const clang::RecordType *field_record_type = field_qual_type->getAs<clang::RecordType>();
            
            if (!field_record_type)
                continue;
            
            clang::RecordDecl *field_record_decl = field_record_type->getDecl();
            
            if (!field_record_decl)
                continue;
            
            for (clang::RecordDecl::decl_iterator di = field_record_decl->decls_begin(), de = field_record_decl->decls_end();
                 di != de;
                 ++di)
            {
                if (clang::FieldDecl *nested_field_decl = llvm::dyn_cast<clang::FieldDecl>(*di))
                {
                    clang::NamedDecl **chain = new (*ast->getASTContext()) clang::NamedDecl*[2];
                    chain[0] = *field_pos;
                    chain[1] = nested_field_decl;
                    clang::IndirectFieldDecl *indirect_field = clang::IndirectFieldDecl::Create(*ast->getASTContext(),
                                                                                                record_decl,
                                                                                                clang::SourceLocation(),
                                                                                                nested_field_decl->getIdentifier(),
                                                                                                nested_field_decl->getType(),
                                                                                                chain,
                                                                                                2);
                    
                    indirect_field->setImplicit();
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
                else if (clang::IndirectFieldDecl *nested_indirect_field_decl = llvm::dyn_cast<clang::IndirectFieldDecl>(*di))
                {
                    int nested_chain_size = nested_indirect_field_decl->getChainingSize();
                    clang::NamedDecl **chain = new (*ast->getASTContext()) clang::NamedDecl*[nested_chain_size + 1];
                    chain[0] = *field_pos;
                    
                    int chain_index = 1;
                    for (clang::IndirectFieldDecl::chain_iterator nci = nested_indirect_field_decl->chain_begin(),
                         nce = nested_indirect_field_decl->chain_end();
                         nci < nce;
                         ++nci)
                    {
                        chain[chain_index] = *nci;
                        chain_index++;
                    }
                    
                    clang::IndirectFieldDecl *indirect_field = clang::IndirectFieldDecl::Create(*ast->getASTContext(),
                                                                                                record_decl,
                                                                                                clang::SourceLocation(),
                                                                                                nested_indirect_field_decl->getIdentifier(),
                                                                                                nested_indirect_field_decl->getType(),
                                                                                                chain,
                                                                                                nested_chain_size + 1);
                    
                    indirect_field->setImplicit();
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_indirect_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
            }
        }
    }
    
    // Check the last field to see if it has an incomplete array type as its
    // last member and if it does, the tell the record decl about it
    if (last_field_pos != field_end_pos)
    {
        if (last_field_pos->getType()->isIncompleteArrayType())
            record_decl->hasFlexibleArrayMember();
    }
    
    for (IndirectFieldVector::iterator ifi = indirect_fields.begin(), ife = indirect_fields.end();
         ifi < ife;
         ++ifi)
    {
        record_decl->addDecl(*ifi);
    }
}

void
ClangASTContext::SetIsPacked (const CompilerType& type)
{
    clang::RecordDecl *record_decl = GetAsRecordDecl(type);
    
    if (!record_decl)
        return;
    
    record_decl->addAttr(clang::PackedAttr::CreateImplicit(*type.GetTypeSystem()->AsClangASTContext()->getASTContext()));
}

clang::VarDecl *
ClangASTContext::AddVariableToRecordType (const CompilerType& type, const char *name,
                                          const CompilerType &var_type,
                                          AccessType access)
{
    clang::VarDecl *var_decl = nullptr;
    
    if (!type.IsValid() || !var_type.IsValid())
        return nullptr;
    ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return nullptr;
    
    clang::RecordDecl *record_decl = ast->GetAsRecordDecl (type);
    if (record_decl)
    {
        var_decl = clang::VarDecl::Create (*ast->getASTContext(),                      // ASTContext &
                                           record_decl,                                // DeclContext *
                                           clang::SourceLocation(),                    // clang::SourceLocation StartLoc
                                           clang::SourceLocation(),                    // clang::SourceLocation IdLoc
                                           name ? &ast->getASTContext()->Idents.get(name) : nullptr,  // clang::IdentifierInfo *
                                           GetQualType(var_type),                      // Variable clang::QualType
                                           nullptr,                                    // TypeSourceInfo *
                                           clang::SC_Static);                          // StorageClass
        if (var_decl)
        {
            var_decl->setAccess(ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));
            record_decl->addDecl(var_decl);
            
#ifdef LLDB_CONFIGURATION_DEBUG
            VerifyDecl(var_decl);
#endif
        }
    }
    return var_decl;
}


clang::CXXMethodDecl *
ClangASTContext::AddMethodToCXXRecordType (void* type, const char *name,
                                              const CompilerType &method_clang_type,
                                              lldb::AccessType access,
                                              bool is_virtual,
                                              bool is_static,
                                              bool is_inline,
                                              bool is_explicit,
                                              bool is_attr_used,
                                              bool is_artificial)
{
    if (!type || !method_clang_type.IsValid() || name == nullptr || name[0] == '\0')
        return nullptr;
    
    clang::QualType record_qual_type(GetCanonicalQualType(type));
    
    clang::CXXRecordDecl *cxx_record_decl = record_qual_type->getAsCXXRecordDecl();
    
    if (cxx_record_decl == nullptr)
        return nullptr;
    
    clang::QualType method_qual_type (GetQualType(method_clang_type));
    
    clang::CXXMethodDecl *cxx_method_decl = nullptr;
    
    clang::DeclarationName decl_name (&getASTContext()->Idents.get(name));
    
    const clang::FunctionType *function_type = llvm::dyn_cast<clang::FunctionType>(method_qual_type.getTypePtr());
    
    if (function_type == nullptr)
        return nullptr;
    
    const clang::FunctionProtoType *method_function_prototype (llvm::dyn_cast<clang::FunctionProtoType>(function_type));
    
    if (!method_function_prototype)
        return nullptr;
    
    unsigned int num_params = method_function_prototype->getNumParams();
    
    clang::CXXDestructorDecl *cxx_dtor_decl(nullptr);
    clang::CXXConstructorDecl *cxx_ctor_decl(nullptr);
    
    if (is_artificial)
        return nullptr; // skip everything artificial
    
    if (name[0] == '~')
    {
        cxx_dtor_decl = clang::CXXDestructorDecl::Create (*getASTContext(),
                                                          cxx_record_decl,
                                                          clang::SourceLocation(),
                                                          clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXDestructorName (getASTContext()->getCanonicalType (record_qual_type)), clang::SourceLocation()),
                                                          method_qual_type,
                                                          nullptr,
                                                          is_inline,
                                                          is_artificial);
        cxx_method_decl = cxx_dtor_decl;
    }
    else if (decl_name == cxx_record_decl->getDeclName())
    {
        cxx_ctor_decl = clang::CXXConstructorDecl::Create (*getASTContext(),
                                                           cxx_record_decl,
                                                           clang::SourceLocation(),
                                                           clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXConstructorName (getASTContext()->getCanonicalType (record_qual_type)), clang::SourceLocation()),
                                                           method_qual_type,
                                                           nullptr, // TypeSourceInfo *
                                                           is_explicit,
                                                           is_inline,
                                                           is_artificial,
                                                           false /*is_constexpr*/);
        cxx_method_decl = cxx_ctor_decl;
    }
    else
    {
        clang::StorageClass SC = is_static ? clang::SC_Static : clang::SC_None;
        clang::OverloadedOperatorKind op_kind = clang::NUM_OVERLOADED_OPERATORS;
        
        if (IsOperator (name, op_kind))
        {
            if (op_kind != clang::NUM_OVERLOADED_OPERATORS)
            {
                // Check the number of operator parameters. Sometimes we have
                // seen bad DWARF that doesn't correctly describe operators and
                // if we try to create a method and add it to the class, clang
                // will assert and crash, so we need to make sure things are
                // acceptable.
                if (!ClangASTContext::CheckOverloadedOperatorKindParameterCount (op_kind, num_params))
                    return nullptr;
                cxx_method_decl = clang::CXXMethodDecl::Create (*getASTContext(),
                                                                cxx_record_decl,
                                                                clang::SourceLocation(),
                                                                clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXOperatorName (op_kind), clang::SourceLocation()),
                                                                method_qual_type,
                                                                nullptr, // TypeSourceInfo *
                                                                SC,
                                                                is_inline,
                                                                false /*is_constexpr*/,
                                                                clang::SourceLocation());
            }
            else if (num_params == 0)
            {
                // Conversion operators don't take params...
                cxx_method_decl = clang::CXXConversionDecl::Create (*getASTContext(),
                                                                    cxx_record_decl,
                                                                    clang::SourceLocation(),
                                                                    clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXConversionFunctionName (getASTContext()->getCanonicalType (function_type->getReturnType())), clang::SourceLocation()),
                                                                    method_qual_type,
                                                                    nullptr, // TypeSourceInfo *
                                                                    is_inline,
                                                                    is_explicit,
                                                                    false /*is_constexpr*/,
                                                                    clang::SourceLocation());
            }
        }
        
        if (cxx_method_decl == nullptr)
        {
            cxx_method_decl = clang::CXXMethodDecl::Create (*getASTContext(),
                                                            cxx_record_decl,
                                                            clang::SourceLocation(),
                                                            clang::DeclarationNameInfo (decl_name, clang::SourceLocation()),
                                                            method_qual_type,
                                                            nullptr, // TypeSourceInfo *
                                                            SC,
                                                            is_inline,
                                                            false /*is_constexpr*/,
                                                            clang::SourceLocation());
        }
    }
    
    clang::AccessSpecifier access_specifier = ClangASTContext::ConvertAccessTypeToAccessSpecifier (access);
    
    cxx_method_decl->setAccess (access_specifier);
    cxx_method_decl->setVirtualAsWritten (is_virtual);
    
    if (is_attr_used)
        cxx_method_decl->addAttr(clang::UsedAttr::CreateImplicit(*getASTContext()));
    
    // Populate the method decl with parameter decls
    
    llvm::SmallVector<clang::ParmVarDecl *, 12> params;
    
    for (unsigned param_index = 0;
         param_index < num_params;
         ++param_index)
    {
        params.push_back (clang::ParmVarDecl::Create (*getASTContext(),
                                                      cxx_method_decl,
                                                      clang::SourceLocation(),
                                                      clang::SourceLocation(),
                                                      nullptr, // anonymous
                                                      method_function_prototype->getParamType(param_index),
                                                      nullptr,
                                                      clang::SC_None,
                                                      nullptr));
    }
    
    cxx_method_decl->setParams (llvm::ArrayRef<clang::ParmVarDecl*>(params));
    
    cxx_record_decl->addDecl (cxx_method_decl);
    
    // Sometimes the debug info will mention a constructor (default/copy/move),
    // destructor, or assignment operator (copy/move) but there won't be any
    // version of this in the code. So we check if the function was artificially
    // generated and if it is trivial and this lets the compiler/backend know
    // that it can inline the IR for these when it needs to and we can avoid a
    // "missing function" error when running expressions.
    
    if (is_artificial)
    {
        if (cxx_ctor_decl &&
            ((cxx_ctor_decl->isDefaultConstructor() && cxx_record_decl->hasTrivialDefaultConstructor ()) ||
             (cxx_ctor_decl->isCopyConstructor()    && cxx_record_decl->hasTrivialCopyConstructor    ()) ||
             (cxx_ctor_decl->isMoveConstructor()    && cxx_record_decl->hasTrivialMoveConstructor    ()) ))
        {
            cxx_ctor_decl->setDefaulted();
            cxx_ctor_decl->setTrivial(true);
        }
        else if (cxx_dtor_decl)
        {
            if (cxx_record_decl->hasTrivialDestructor())
            {
                cxx_dtor_decl->setDefaulted();
                cxx_dtor_decl->setTrivial(true);
            }
        }
        else if ((cxx_method_decl->isCopyAssignmentOperator() && cxx_record_decl->hasTrivialCopyAssignment()) ||
                 (cxx_method_decl->isMoveAssignmentOperator() && cxx_record_decl->hasTrivialMoveAssignment()))
        {
            cxx_method_decl->setDefaulted();
            cxx_method_decl->setTrivial(true);
        }
    }
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(cxx_method_decl);
#endif
    
    //    printf ("decl->isPolymorphic()             = %i\n", cxx_record_decl->isPolymorphic());
    //    printf ("decl->isAggregate()               = %i\n", cxx_record_decl->isAggregate());
    //    printf ("decl->isPOD()                     = %i\n", cxx_record_decl->isPOD());
    //    printf ("decl->isEmpty()                   = %i\n", cxx_record_decl->isEmpty());
    //    printf ("decl->isAbstract()                = %i\n", cxx_record_decl->isAbstract());
    //    printf ("decl->hasTrivialConstructor()     = %i\n", cxx_record_decl->hasTrivialConstructor());
    //    printf ("decl->hasTrivialCopyConstructor() = %i\n", cxx_record_decl->hasTrivialCopyConstructor());
    //    printf ("decl->hasTrivialCopyAssignment()  = %i\n", cxx_record_decl->hasTrivialCopyAssignment());
    //    printf ("decl->hasTrivialDestructor()      = %i\n", cxx_record_decl->hasTrivialDestructor());
    return cxx_method_decl;
}


#pragma mark C++ Base Classes

clang::CXXBaseSpecifier *
ClangASTContext::CreateBaseClassSpecifier (void* type, AccessType access, bool is_virtual, bool base_of_class)
{
    if (type)
        return new clang::CXXBaseSpecifier (clang::SourceRange(),
                                            is_virtual,
                                            base_of_class,
                                            ClangASTContext::ConvertAccessTypeToAccessSpecifier (access),
                                            getASTContext()->getTrivialTypeSourceInfo (GetQualType(type)),
                                            clang::SourceLocation());
    return nullptr;
}

void
ClangASTContext::DeleteBaseClassSpecifiers (clang::CXXBaseSpecifier **base_classes, unsigned num_base_classes)
{
    for (unsigned i=0; i<num_base_classes; ++i)
    {
        delete base_classes[i];
        base_classes[i] = nullptr;
    }
}

bool
ClangASTContext::SetBaseClassesForClassType (void* type, clang::CXXBaseSpecifier const * const *base_classes,
                                                unsigned num_base_classes)
{
    if (type)
    {
        clang::CXXRecordDecl *cxx_record_decl = GetAsCXXRecordDecl(type);
        if (cxx_record_decl)
        {
            cxx_record_decl->setBases(base_classes, num_base_classes);
            return true;
        }
    }
    return false;
}

bool
ClangASTContext::SetObjCSuperClass (const CompilerType& type, const CompilerType &superclass_clang_type)
{
    ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return false;
    clang::ASTContext* clang_ast = ast->getASTContext();

    if (type && superclass_clang_type.IsValid() && superclass_clang_type.GetTypeSystem() == type.GetTypeSystem())
    {
        clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl (type);
        clang::ObjCInterfaceDecl *super_interface_decl = GetAsObjCInterfaceDecl (superclass_clang_type);
        if (class_interface_decl && super_interface_decl)
        {
            class_interface_decl->setSuperClass(clang_ast->getTrivialTypeSourceInfo(clang_ast->getObjCInterfaceType(super_interface_decl)));
            return true;
        }
    }
    return false;
}

bool
ClangASTContext::AddObjCClassProperty (const CompilerType& type,
                                       const char *property_name,
                                       const CompilerType &property_clang_type,
                                       clang::ObjCIvarDecl *ivar_decl,
                                       const char *property_setter_name,
                                       const char *property_getter_name,
                                       uint32_t property_attributes,
                                       ClangASTMetadata *metadata)
{
    if (!type || !property_clang_type.IsValid() || property_name == nullptr || property_name[0] == '\0')
        return false;
    ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
    if (!ast)
        return false;
    clang::ASTContext* clang_ast = ast->getASTContext();
    
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl (type);
    
    if (class_interface_decl)
    {
        CompilerType property_clang_type_to_access;
        
        if (property_clang_type.IsValid())
            property_clang_type_to_access = property_clang_type;
        else if (ivar_decl)
            property_clang_type_to_access = CompilerType (clang_ast, ivar_decl->getType());
        
        if (class_interface_decl && property_clang_type_to_access.IsValid())
        {
            clang::TypeSourceInfo *prop_type_source;
            if (ivar_decl)
                prop_type_source = clang_ast->getTrivialTypeSourceInfo (ivar_decl->getType());
            else
                prop_type_source = clang_ast->getTrivialTypeSourceInfo (GetQualType(property_clang_type));
            
            clang::ObjCPropertyDecl *property_decl = clang::ObjCPropertyDecl::Create (*clang_ast,
                                                                                      class_interface_decl,
                                                                                      clang::SourceLocation(), // Source Location
                                                                                      &clang_ast->Idents.get(property_name),
                                                                                      clang::SourceLocation(), //Source Location for AT
                                                                                      clang::SourceLocation(), //Source location for (
                                                                                      ivar_decl ? ivar_decl->getType() : ClangASTContext::GetQualType(property_clang_type),
                                                                                      prop_type_source);
            
            if (property_decl)
            {
                if (metadata)
                    ClangASTContext::SetMetadata(clang_ast, property_decl, *metadata);
                
                class_interface_decl->addDecl (property_decl);
                
                clang::Selector setter_sel, getter_sel;
                
                if (property_setter_name != nullptr)
                {
                    std::string property_setter_no_colon(property_setter_name, strlen(property_setter_name) - 1);
                    clang::IdentifierInfo *setter_ident = &clang_ast->Idents.get(property_setter_no_colon.c_str());
                    setter_sel = clang_ast->Selectors.getSelector(1, &setter_ident);
                }
                else if (!(property_attributes & DW_APPLE_PROPERTY_readonly))
                {
                    std::string setter_sel_string("set");
                    setter_sel_string.push_back(::toupper(property_name[0]));
                    setter_sel_string.append(&property_name[1]);
                    clang::IdentifierInfo *setter_ident = &clang_ast->Idents.get(setter_sel_string.c_str());
                    setter_sel = clang_ast->Selectors.getSelector(1, &setter_ident);
                }
                property_decl->setSetterName(setter_sel);
                property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_setter);
                
                if (property_getter_name != nullptr)
                {
                    clang::IdentifierInfo *getter_ident = &clang_ast->Idents.get(property_getter_name);
                    getter_sel = clang_ast->Selectors.getSelector(0, &getter_ident);
                }
                else
                {
                    clang::IdentifierInfo *getter_ident = &clang_ast->Idents.get(property_name);
                    getter_sel = clang_ast->Selectors.getSelector(0, &getter_ident);
                }
                property_decl->setGetterName(getter_sel);
                property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_getter);
                
                if (ivar_decl)
                    property_decl->setPropertyIvarDecl (ivar_decl);
                
                if (property_attributes & DW_APPLE_PROPERTY_readonly)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_readonly);
                if (property_attributes & DW_APPLE_PROPERTY_readwrite)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_readwrite);
                if (property_attributes & DW_APPLE_PROPERTY_assign)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_assign);
                if (property_attributes & DW_APPLE_PROPERTY_retain)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_retain);
                if (property_attributes & DW_APPLE_PROPERTY_copy)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_copy);
                if (property_attributes & DW_APPLE_PROPERTY_nonatomic)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_nonatomic);
                
                if (!getter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(getter_sel))
                {
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const clang::ObjCMethodDecl::ImplementationControl impControl = clang::ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    clang::ObjCMethodDecl *getter = clang::ObjCMethodDecl::Create (*clang_ast,
                                                                                   clang::SourceLocation(),
                                                                                   clang::SourceLocation(),
                                                                                   getter_sel,
                                                                                   GetQualType(property_clang_type_to_access),
                                                                                   nullptr,
                                                                                   class_interface_decl,
                                                                                   isInstance,
                                                                                   isVariadic,
                                                                                   isSynthesized,
                                                                                   isImplicitlyDeclared,
                                                                                   isDefined,
                                                                                   impControl,
                                                                                   HasRelatedResultType);
                    
                    if (getter && metadata)
                        ClangASTContext::SetMetadata(clang_ast, getter, *metadata);
                    
                    if (getter)
                    {
                        getter->setMethodParams(*clang_ast, llvm::ArrayRef<clang::ParmVarDecl*>(), llvm::ArrayRef<clang::SourceLocation>());
                        
                        class_interface_decl->addDecl(getter);
                    }
                }
                
                if (!setter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(setter_sel))
                {
                    clang::QualType result_type = clang_ast->VoidTy;
                    
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const clang::ObjCMethodDecl::ImplementationControl impControl = clang::ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    clang::ObjCMethodDecl *setter = clang::ObjCMethodDecl::Create (*clang_ast,
                                                                                   clang::SourceLocation(),
                                                                                   clang::SourceLocation(),
                                                                                   setter_sel,
                                                                                   result_type,
                                                                                   nullptr,
                                                                                   class_interface_decl,
                                                                                   isInstance,
                                                                                   isVariadic,
                                                                                   isSynthesized,
                                                                                   isImplicitlyDeclared,
                                                                                   isDefined,
                                                                                   impControl,
                                                                                   HasRelatedResultType);
                    
                    if (setter && metadata)
                        ClangASTContext::SetMetadata(clang_ast, setter, *metadata);
                    
                    llvm::SmallVector<clang::ParmVarDecl *, 1> params;
                    
                    params.push_back (clang::ParmVarDecl::Create (*clang_ast,
                                                                  setter,
                                                                  clang::SourceLocation(),
                                                                  clang::SourceLocation(),
                                                                  nullptr, // anonymous
                                                                  GetQualType(property_clang_type_to_access),
                                                                  nullptr,
                                                                  clang::SC_Auto,
                                                                  nullptr));
                    
                    if (setter)
                    {
                        setter->setMethodParams(*clang_ast, llvm::ArrayRef<clang::ParmVarDecl*>(params), llvm::ArrayRef<clang::SourceLocation>());
                        
                        class_interface_decl->addDecl(setter);
                    }
                }
                
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTContext::IsObjCClassTypeAndHasIVars (const CompilerType& type, bool check_superclass)
{
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl (type);
    if (class_interface_decl)
        return ObjCDeclHasIVars (class_interface_decl, check_superclass);
    return false;
}


clang::ObjCMethodDecl *
ClangASTContext::AddMethodToObjCObjectType (const CompilerType& type,
                                            const char *name,  // the full symbol name as seen in the symbol table (void* type, "-[NString stringWithCString:]")
                                            const CompilerType &method_clang_type,
                                            lldb::AccessType access,
                                            bool is_artificial)
{
    if (!type || !method_clang_type.IsValid())
        return nullptr;
    
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl(type);
    
    if (class_interface_decl == nullptr)
        return nullptr;
    clang::ASTContext* ast = type.GetTypeSystem()->AsClangASTContext()->getASTContext();
    
    const char *selector_start = ::strchr (name, ' ');
    if (selector_start == nullptr)
        return nullptr;
    
    selector_start++;
    llvm::SmallVector<clang::IdentifierInfo *, 12> selector_idents;
    
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
        selector_idents.push_back (&ast->Idents.get (llvm::StringRef (start, len)));
        if (has_arg)
            len += 1;
    }
    
    
    if (selector_idents.size() == 0)
        return nullptr;
    
    clang::Selector method_selector = ast->Selectors.getSelector (num_selectors_with_args ? selector_idents.size() : 0,
                                                                    selector_idents.data());
    
    clang::QualType method_qual_type (GetQualType(method_clang_type));
    
    // Populate the method decl with parameter decls
    const clang::Type *method_type(method_qual_type.getTypePtr());
    
    if (method_type == nullptr)
        return nullptr;
    
    const clang::FunctionProtoType *method_function_prototype (llvm::dyn_cast<clang::FunctionProtoType>(method_type));
    
    if (!method_function_prototype)
        return nullptr;
    
    
    bool is_variadic = false;
    bool is_synthesized = false;
    bool is_defined = false;
    clang::ObjCMethodDecl::ImplementationControl imp_control = clang::ObjCMethodDecl::None;
    
    const unsigned num_args = method_function_prototype->getNumParams();
    
    if (num_args != num_selectors_with_args)
        return nullptr; // some debug information is corrupt.  We are not going to deal with it.
    
    clang::ObjCMethodDecl *objc_method_decl = clang::ObjCMethodDecl::Create (*ast,
                                                                             clang::SourceLocation(), // beginLoc,
                                                                             clang::SourceLocation(), // endLoc,
                                                                             method_selector,
                                                                             method_function_prototype->getReturnType(),
                                                                             nullptr, // TypeSourceInfo *ResultTInfo,
                                                                             ClangASTContext::GetASTContext(ast)->GetDeclContextForType(GetQualType(type)),
                                                                             name[0] == '-',
                                                                             is_variadic,
                                                                             is_synthesized,
                                                                             true, // is_implicitly_declared; we force this to true because we don't have source locations
                                                                             is_defined,
                                                                             imp_control,
                                                                             false /*has_related_result_type*/);
    
    
    if (objc_method_decl == nullptr)
        return nullptr;
    
    if (num_args > 0)
    {
        llvm::SmallVector<clang::ParmVarDecl *, 12> params;
        
        for (unsigned param_index = 0; param_index < num_args; ++param_index)
        {
            params.push_back (clang::ParmVarDecl::Create (*ast,
                                                          objc_method_decl,
                                                          clang::SourceLocation(),
                                                          clang::SourceLocation(),
                                                          nullptr, // anonymous
                                                          method_function_prototype->getParamType(param_index),
                                                          nullptr,
                                                          clang::SC_Auto,
                                                          nullptr));
        }
        
        objc_method_decl->setMethodParams(*ast, llvm::ArrayRef<clang::ParmVarDecl*>(params), llvm::ArrayRef<clang::SourceLocation>());
    }
    
    class_interface_decl->addDecl (objc_method_decl);
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(objc_method_decl);
#endif
    
    return objc_method_decl;
}

bool
ClangASTContext::SetHasExternalStorage (void* type, bool has_extern)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
        {
            clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
            if (cxx_record_decl)
            {
                cxx_record_decl->setHasExternalLexicalStorage (has_extern);
                cxx_record_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
            break;
            
        case clang::Type::Enum:
        {
            clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
            if (enum_decl)
            {
                enum_decl->setHasExternalLexicalStorage (has_extern);
                enum_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
        {
            const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
                if (class_interface_decl)
                {
                    class_interface_decl->setHasExternalLexicalStorage (has_extern);
                    class_interface_decl->setHasExternalVisibleStorage (has_extern);
                    return true;
                }
            }
        }
            break;
            
        case clang::Type::Typedef:
            return SetHasExternalStorage(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), has_extern);
            
        case clang::Type::Elaborated:
            return SetHasExternalStorage (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), has_extern);
            
        case clang::Type::Paren:
            return SetHasExternalStorage (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), has_extern);
            
        default:
            break;
    }
    return false;
}


#pragma mark TagDecl

bool
ClangASTContext::StartTagDeclarationDefinition (const CompilerType &type)
{
    if (type)
    {
        
        clang::QualType qual_type (GetQualType(type));
        const clang::Type *t = qual_type.getTypePtr();
        if (t)
        {
            const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(t);
            if (tag_type)
            {
                clang::TagDecl *tag_decl = tag_type->getDecl();
                if (tag_decl)
                {
                    tag_decl->startDefinition();
                    return true;
                }
            }
            
            const clang::ObjCObjectType *object_type = llvm::dyn_cast<clang::ObjCObjectType>(t);
            if (object_type)
            {
                clang::ObjCInterfaceDecl *interface_decl = object_type->getInterface();
                if (interface_decl)
                {
                    interface_decl->startDefinition();
                    return true;
                }
            }
        }
    }
    return false;
}

bool
ClangASTContext::CompleteTagDeclarationDefinition (const CompilerType& type)
{
    if (type)
    {
        clang::QualType qual_type (GetQualType(type));
        if (qual_type.isNull())
            return false;
        clang::ASTContext* ast = type.GetTypeSystem()->AsClangASTContext()->getASTContext();
        
        clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        
        if (cxx_record_decl)
        {
            cxx_record_decl->completeDefinition();
            
            return true;
        }
        
        const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(qual_type.getTypePtr());
        
        if (enutype)
        {
            clang::EnumDecl *enum_decl = enutype->getDecl();
            
            if (enum_decl)
            {
                /// TODO This really needs to be fixed.
                
                unsigned NumPositiveBits = 1;
                unsigned NumNegativeBits = 0;
                
                clang::QualType promotion_qual_type;
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

bool
ClangASTContext::AddEnumerationValueToEnumerationType (void* type,
                                                       const CompilerType &enumerator_clang_type,
                                                       const Declaration &decl,
                                                       const char *name,
                                                       int64_t enum_value,
                                                       uint32_t enum_value_bit_size)
{
    if (type && enumerator_clang_type.IsValid() && name && name[0])
    {
        clang::QualType enum_qual_type (GetCanonicalQualType(type));
        
        bool is_signed = false;
        enumerator_clang_type.IsIntegerType (is_signed);
        const clang::Type *clang_type = enum_qual_type.getTypePtr();
        if (clang_type)
        {
            const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(clang_type);
            
            if (enutype)
            {
                llvm::APSInt enum_llvm_apsint(enum_value_bit_size, is_signed);
                enum_llvm_apsint = enum_value;
                clang::EnumConstantDecl *enumerator_decl =
                clang::EnumConstantDecl::Create (*getASTContext(),
                                                 enutype->getDecl(),
                                                 clang::SourceLocation(),
                                                 name ? &getASTContext()->Idents.get(name) : nullptr,    // Identifier
                                                 GetQualType(enumerator_clang_type),
                                                 nullptr,
                                                 enum_llvm_apsint);
                
                if (enumerator_decl)
                {
                    enutype->getDecl()->addDecl(enumerator_decl);
                    
#ifdef LLDB_CONFIGURATION_DEBUG
                    VerifyDecl(enumerator_decl);
#endif
                    
                    return true;
                }
            }
        }
    }
    return false;
}

CompilerType
ClangASTContext::GetEnumerationIntegerType (void* type)
{
    clang::QualType enum_qual_type (GetCanonicalQualType(type));
    const clang::Type *clang_type = enum_qual_type.getTypePtr();
    if (clang_type)
    {
        const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(clang_type);
        if (enutype)
        {
            clang::EnumDecl *enum_decl = enutype->getDecl();
            if (enum_decl)
                return CompilerType (getASTContext(), enum_decl->getIntegerType());
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::CreateMemberPointerType (const CompilerType& type, const CompilerType &pointee_type)
{
    if (type && pointee_type.IsValid() && type.GetTypeSystem() == pointee_type.GetTypeSystem())
    {
        ClangASTContext* ast = type.GetTypeSystem()->AsClangASTContext();
        if (!ast)
            return CompilerType();
        return CompilerType (ast->getASTContext(),
                             ast->getASTContext()->getMemberPointerType (GetQualType(pointee_type),
                                                                         GetQualType(type).getTypePtr()));
    }
    return CompilerType();
}


size_t
ClangASTContext::ConvertStringToFloatValue (void* type, const char *s, uint8_t *dst, size_t dst_size)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        uint32_t count = 0;
        bool is_complex = false;
        if (IsFloatingPointType (type, count, is_complex))
        {
            // TODO: handle complex and vector types
            if (count != 1)
                return false;
            
            llvm::StringRef s_sref(s);
            llvm::APFloat ap_float(getASTContext()->getFloatTypeSemantics(qual_type), s_sref);
            
            const uint64_t bit_size = getASTContext()->getTypeSize (qual_type);
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



//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------
#define DEPTH_INCREMENT 2

void
ClangASTContext::DumpValue (void* type, ExecutionContext *exe_ctx,
                               Stream *s,
                               lldb::Format format,
                               const lldb_private::DataExtractor &data,
                               lldb::offset_t data_byte_offset,
                               size_t data_byte_size,
                               uint32_t bitfield_bit_size,
                               uint32_t bitfield_bit_offset,
                               bool show_types,
                               bool show_summary,
                               bool verbose,
                               uint32_t depth)
{
    if (!type)
        return;
    
    clang::QualType qual_type(GetQualType(type));
    switch (qual_type->getTypeClass())
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                uint32_t field_bit_offset = 0;
                uint32_t field_byte_offset = 0;
                const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;
                
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                        
                        // Skip empty base classes
                        if (verbose == false && ClangASTContext::RecordHasFields(base_class_decl) == false)
                            continue;
                        
                        if (base_class->isVirtual())
                            field_bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                        else
                            field_bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                        field_byte_offset = field_bit_offset / 8;
                        assert (field_bit_offset % 8 == 0);
                        if (child_idx == 0)
                            s->PutChar('{');
                        else
                            s->PutChar(',');
                        
                        clang::QualType base_class_qual_type = base_class->getType();
                        std::string base_class_type_name(base_class_qual_type.getAsString());
                        
                        // Indent and print the base class type name
                        s->Printf("\n%*s%s ", depth + DEPTH_INCREMENT, "", base_class_type_name.c_str());
                        
                        clang::TypeInfo base_class_type_info = getASTContext()->getTypeInfo(base_class_qual_type);
                        
                        // Dump the value of the member
                        CompilerType base_clang_type(getASTContext(), base_class_qual_type);
                        base_clang_type.DumpValue (exe_ctx,
                                                   s,                                   // Stream to dump to
                                                   base_clang_type.GetFormat(),         // The format with which to display the member
                                                   data,                                // Data buffer containing all bytes for this type
                                                   data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                                   base_class_type_info.Width / 8,      // Size of this type in bytes
                                                   0,                                   // Bitfield bit size
                                                   0,                                   // Bitfield bit offset
                                                   show_types,                          // Boolean indicating if we should show the variable types
                                                   show_summary,                        // Boolean indicating if we should show a summary for the current type
                                                   verbose,                             // Verbose output?
                                                   depth + DEPTH_INCREMENT);            // Scope depth for any types that have children
                        
                        ++child_idx;
                    }
                }
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
                {
                    // Print the starting squiggly bracket (if this is the
                    // first member) or comma (for member 2 and beyond) for
                    // the struct/union/class member.
                    if (child_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');
                    
                    // Indent
                    s->Printf("\n%*s", depth + DEPTH_INCREMENT, "");
                    
                    clang::QualType field_type = field->getType();
                    // Print the member type if requested
                    // Figure out the type byte size (field_type_info.first) and
                    // alignment (field_type_info.second) from the AST context.
                    clang::TypeInfo field_type_info = getASTContext()->getTypeInfo(field_type);
                    assert(field_idx < record_layout.getFieldCount());
                    // Figure out the field offset within the current struct/union/class type
                    field_bit_offset = record_layout.getFieldOffset (field_idx);
                    field_byte_offset = field_bit_offset / 8;
                    uint32_t field_bitfield_bit_size = 0;
                    uint32_t field_bitfield_bit_offset = 0;
                    if (ClangASTContext::FieldIsBitfield (getASTContext(), *field, field_bitfield_bit_size))
                        field_bitfield_bit_offset = field_bit_offset % 8;
                    
                    if (show_types)
                    {
                        std::string field_type_name(field_type.getAsString());
                        if (field_bitfield_bit_size > 0)
                            s->Printf("(%s:%u) ", field_type_name.c_str(), field_bitfield_bit_size);
                        else
                            s->Printf("(%s) ", field_type_name.c_str());
                    }
                    // Print the member name and equal sign
                    s->Printf("%s = ", field->getNameAsString().c_str());
                    
                    
                    // Dump the value of the member
                    CompilerType field_clang_type (getASTContext(), field_type);
                    field_clang_type.DumpValue (exe_ctx,
                                                s,                              // Stream to dump to
                                                field_clang_type.GetFormat(),   // The format with which to display the member
                                                data,                           // Data buffer containing all bytes for this type
                                                data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                                field_type_info.Width / 8,      // Size of this type in bytes
                                                field_bitfield_bit_size,        // Bitfield bit size
                                                field_bitfield_bit_offset,      // Bitfield bit offset
                                                show_types,                     // Boolean indicating if we should show the variable types
                                                show_summary,                   // Boolean indicating if we should show a summary for the current type
                                                verbose,                        // Verbose output?
                                                depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
                }
                
                // Indent the trailing squiggly bracket
                if (child_idx > 0)
                    s->Printf("\n%*s}", depth, "");
            }
            return;
            
        case clang::Type::Enum:
            if (GetCompleteType(type))
            {
                const clang::EnumType *enutype = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
                const clang::EnumDecl *enum_decl = enutype->getDecl();
                assert(enum_decl);
                clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                lldb::offset_t offset = data_byte_offset;
                const int64_t enum_value = data.GetMaxU64Bitfield(&offset, data_byte_size, bitfield_bit_size, bitfield_bit_offset);
                for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                {
                    if (enum_pos->getInitVal() == enum_value)
                    {
                        s->Printf("%s", enum_pos->getNameAsString().c_str());
                        return;
                    }
                }
                // If we have gotten here we didn't get find the enumerator in the
                // enum decl, so just print the integer.
                s->Printf("%" PRIi64, enum_value);
            }
            return;
            
        case clang::Type::ConstantArray:
        {
            const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr());
            bool is_array_of_characters = false;
            clang::QualType element_qual_type = array->getElementType();
            
            const clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
            if (canonical_type)
                is_array_of_characters = canonical_type->isCharType();
            
            const uint64_t element_count = array->getSize().getLimitedValue();
            
            clang::TypeInfo field_type_info = getASTContext()->getTypeInfo(element_qual_type);
            
            uint32_t element_idx = 0;
            uint32_t element_offset = 0;
            uint64_t element_byte_size = field_type_info.Width / 8;
            uint32_t element_stride = element_byte_size;
            
            if (is_array_of_characters)
            {
                s->PutChar('"');
                data.Dump(s, data_byte_offset, lldb::eFormatChar, element_byte_size, element_count, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                s->PutChar('"');
                return;
            }
            else
            {
                CompilerType element_clang_type(getASTContext(), element_qual_type);
                lldb::Format element_format = element_clang_type.GetFormat();
                
                for (element_idx = 0; element_idx < element_count; ++element_idx)
                {
                    // Print the starting squiggly bracket (if this is the
                    // first member) or comman (for member 2 and beyong) for
                    // the struct/union/class member.
                    if (element_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');
                    
                    // Indent and print the index
                    s->Printf("\n%*s[%u] ", depth + DEPTH_INCREMENT, "", element_idx);
                    
                    // Figure out the field offset within the current struct/union/class type
                    element_offset = element_idx * element_stride;
                    
                    // Dump the value of the member
                    element_clang_type.DumpValue (exe_ctx,
                                                  s,                              // Stream to dump to
                                                  element_format,                 // The format with which to display the element
                                                  data,                           // Data buffer containing all bytes for this type
                                                  data_byte_offset + element_offset,// Offset into "data" where to grab value from
                                                  element_byte_size,              // Size of this type in bytes
                                                  0,                              // Bitfield bit size
                                                  0,                              // Bitfield bit offset
                                                  show_types,                     // Boolean indicating if we should show the variable types
                                                  show_summary,                   // Boolean indicating if we should show a summary for the current type
                                                  verbose,                        // Verbose output?
                                                  depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
                }
                
                // Indent the trailing squiggly bracket
                if (element_idx > 0)
                    s->Printf("\n%*s}", depth, "");
            }
        }
            return;
            
        case clang::Type::Typedef:
        {
            clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
            
            CompilerType typedef_clang_type (getASTContext(), typedef_qual_type);
            lldb::Format typedef_format = typedef_clang_type.GetFormat();
            clang::TypeInfo typedef_type_info = getASTContext()->getTypeInfo(typedef_qual_type);
            uint64_t typedef_byte_size = typedef_type_info.Width / 8;
            
            return typedef_clang_type.DumpValue (exe_ctx,
                                                 s,                  // Stream to dump to
                                                 typedef_format,     // The format with which to display the element
                                                 data,               // Data buffer containing all bytes for this type
                                                 data_byte_offset,   // Offset into "data" where to grab value from
                                                 typedef_byte_size,  // Size of this type in bytes
                                                 bitfield_bit_size,  // Bitfield bit size
                                                 bitfield_bit_offset,// Bitfield bit offset
                                                 show_types,         // Boolean indicating if we should show the variable types
                                                 show_summary,       // Boolean indicating if we should show a summary for the current type
                                                 verbose,            // Verbose output?
                                                 depth);             // Scope depth for any types that have children
        }
            break;
            
        case clang::Type::Elaborated:
        {
            clang::QualType elaborated_qual_type = llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType();
            CompilerType elaborated_clang_type (getASTContext(), elaborated_qual_type);
            lldb::Format elaborated_format = elaborated_clang_type.GetFormat();
            clang::TypeInfo elaborated_type_info = getASTContext()->getTypeInfo(elaborated_qual_type);
            uint64_t elaborated_byte_size = elaborated_type_info.Width / 8;
            
            return elaborated_clang_type.DumpValue (exe_ctx,
                                                    s,                  // Stream to dump to
                                                    elaborated_format,  // The format with which to display the element
                                                    data,               // Data buffer containing all bytes for this type
                                                    data_byte_offset,   // Offset into "data" where to grab value from
                                                    elaborated_byte_size,  // Size of this type in bytes
                                                    bitfield_bit_size,  // Bitfield bit size
                                                    bitfield_bit_offset,// Bitfield bit offset
                                                    show_types,         // Boolean indicating if we should show the variable types
                                                    show_summary,       // Boolean indicating if we should show a summary for the current type
                                                    verbose,            // Verbose output?
                                                    depth);             // Scope depth for any types that have children
        }
            break;
            
        case clang::Type::Paren:
        {
            clang::QualType desugar_qual_type = llvm::cast<clang::ParenType>(qual_type)->desugar();
            CompilerType desugar_clang_type (getASTContext(), desugar_qual_type);
            
            lldb::Format desugar_format = desugar_clang_type.GetFormat();
            clang::TypeInfo desugar_type_info = getASTContext()->getTypeInfo(desugar_qual_type);
            uint64_t desugar_byte_size = desugar_type_info.Width / 8;
            
            return desugar_clang_type.DumpValue (exe_ctx,
                                                 s,                  // Stream to dump to
                                                 desugar_format,  // The format with which to display the element
                                                 data,               // Data buffer containing all bytes for this type
                                                 data_byte_offset,   // Offset into "data" where to grab value from
                                                 desugar_byte_size,  // Size of this type in bytes
                                                 bitfield_bit_size,  // Bitfield bit size
                                                 bitfield_bit_offset,// Bitfield bit offset
                                                 show_types,         // Boolean indicating if we should show the variable types
                                                 show_summary,       // Boolean indicating if we should show a summary for the current type
                                                 verbose,            // Verbose output?
                                                 depth);             // Scope depth for any types that have children
        }
            break;
            
        default:
            // We are down to a scalar type that we just need to display.
            data.Dump(s,
                      data_byte_offset,
                      format,
                      data_byte_size,
                      1,
                      UINT32_MAX,
                      LLDB_INVALID_ADDRESS,
                      bitfield_bit_size,
                      bitfield_bit_offset);
            
            if (show_summary)
                DumpSummary (type, exe_ctx, s, data, data_byte_offset, data_byte_size);
            break;
    }
}




bool
ClangASTContext::DumpTypeValue (void* type, Stream *s,
                                   lldb::Format format,
                                   const lldb_private::DataExtractor &data,
                                   lldb::offset_t byte_offset,
                                   size_t byte_size,
                                   uint32_t bitfield_bit_size,
                                   uint32_t bitfield_bit_offset,
                                   ExecutionContextScope *exe_scope)
{
    if (!type)
        return false;
    if (IsAggregateType(type))
    {
        return false;
    }
    else
    {
        clang::QualType qual_type(GetQualType(type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Typedef:
            {
                clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
                CompilerType typedef_clang_type (getASTContext(), typedef_qual_type);
                if (format == eFormatDefault)
                    format = typedef_clang_type.GetFormat();
                clang::TypeInfo typedef_type_info = getASTContext()->getTypeInfo(typedef_qual_type);
                uint64_t typedef_byte_size = typedef_type_info.Width / 8;
                
                return typedef_clang_type.DumpTypeValue (s,
                                                         format,                 // The format with which to display the element
                                                         data,                   // Data buffer containing all bytes for this type
                                                         byte_offset,            // Offset into "data" where to grab value from
                                                         typedef_byte_size,      // Size of this type in bytes
                                                         bitfield_bit_size,      // Size in bits of a bitfield value, if zero don't treat as a bitfield
                                                         bitfield_bit_offset,    // Offset in bits of a bitfield value if bitfield_bit_size != 0
                                                         exe_scope);
            }
                break;
                
            case clang::Type::Enum:
                // If our format is enum or default, show the enumeration value as
                // its enumeration string value, else just display it as requested.
                if ((format == eFormatEnum || format == eFormatDefault) && GetCompleteType(type))
                {
                    const clang::EnumType *enutype = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
                    const clang::EnumDecl *enum_decl = enutype->getDecl();
                    assert(enum_decl);
                    clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                    const bool is_signed = qual_type->isSignedIntegerOrEnumerationType();
                    lldb::offset_t offset = byte_offset;
                    if (is_signed)
                    {
                        const int64_t enum_svalue = data.GetMaxS64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                        for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                        {
                            if (enum_pos->getInitVal().getSExtValue() == enum_svalue)
                            {
                                s->PutCString (enum_pos->getNameAsString().c_str());
                                return true;
                            }
                        }
                        // If we have gotten here we didn't get find the enumerator in the
                        // enum decl, so just print the integer.
                        s->Printf("%" PRIi64, enum_svalue);
                    }
                    else
                    {
                        const uint64_t enum_uvalue = data.GetMaxU64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                        for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                        {
                            if (enum_pos->getInitVal().getZExtValue() == enum_uvalue)
                            {
                                s->PutCString (enum_pos->getNameAsString().c_str());
                                return true;
                            }
                        }
                        // If we have gotten here we didn't get find the enumerator in the
                        // enum decl, so just print the integer.
                        s->Printf("%" PRIu64, enum_uvalue);
                    }
                    return true;
                }
                // format was not enum, just fall through and dump the value as requested....
                
            default:
                // We are down to a scalar type that we just need to display.
            {
                uint32_t item_count = 1;
                // A few formats, we might need to modify our size and count for depending
                // on how we are trying to display the value...
                switch (format)
                {
                    default:
                    case eFormatBoolean:
                    case eFormatBinary:
                    case eFormatComplex:
                    case eFormatCString:         // NULL terminated C strings
                    case eFormatDecimal:
                    case eFormatEnum:
                    case eFormatHex:
                    case eFormatHexUppercase:
                    case eFormatFloat:
                    case eFormatOctal:
                    case eFormatOSType:
                    case eFormatUnsigned:
                    case eFormatPointer:
                    case eFormatVectorOfChar:
                    case eFormatVectorOfSInt8:
                    case eFormatVectorOfUInt8:
                    case eFormatVectorOfSInt16:
                    case eFormatVectorOfUInt16:
                    case eFormatVectorOfSInt32:
                    case eFormatVectorOfUInt32:
                    case eFormatVectorOfSInt64:
                    case eFormatVectorOfUInt64:
                    case eFormatVectorOfFloat32:
                    case eFormatVectorOfFloat64:
                    case eFormatVectorOfUInt128:
                        break;
                        
                    case eFormatChar:
                    case eFormatCharPrintable:
                    case eFormatCharArray:
                    case eFormatBytes:
                    case eFormatBytesWithASCII:
                        item_count = byte_size;
                        byte_size = 1;
                        break;
                        
                    case eFormatUnicode16:
                        item_count = byte_size / 2;
                        byte_size = 2;
                        break;
                        
                    case eFormatUnicode32:
                        item_count = byte_size / 4;
                        byte_size = 4;
                        break;
                }
                return data.Dump (s,
                                  byte_offset,
                                  format,
                                  byte_size,
                                  item_count,
                                  UINT32_MAX,
                                  LLDB_INVALID_ADDRESS,
                                  bitfield_bit_size,
                                  bitfield_bit_offset,
                                  exe_scope);
            }
                break;
        }
    }
    return 0;
}



void
ClangASTContext::DumpSummary (void* type, ExecutionContext *exe_ctx,
                                 Stream *s,
                                 const lldb_private::DataExtractor &data,
                                 lldb::offset_t data_byte_offset,
                                 size_t data_byte_size)
{
    uint32_t length = 0;
    if (IsCStringType (type, length))
    {
        if (exe_ctx)
        {
            Process *process = exe_ctx->GetProcessPtr();
            if (process)
            {
                lldb::offset_t offset = data_byte_offset;
                lldb::addr_t pointer_address = data.GetMaxU64(&offset, data_byte_size);
                std::vector<uint8_t> buf;
                if (length > 0)
                    buf.resize (length);
                else
                    buf.resize (256);
                
                lldb_private::DataExtractor cstr_data(&buf.front(), buf.size(), process->GetByteOrder(), 4);
                buf.back() = '\0';
                size_t bytes_read;
                size_t total_cstr_len = 0;
                Error error;
                while ((bytes_read = process->ReadMemory (pointer_address, &buf.front(), buf.size(), error)) > 0)
                {
                    const size_t len = strlen((const char *)&buf.front());
                    if (len == 0)
                        break;
                    if (total_cstr_len == 0)
                        s->PutCString (" \"");
                    cstr_data.Dump(s, 0, lldb::eFormatChar, 1, len, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                    total_cstr_len += len;
                    if (len < buf.size())
                        break;
                    pointer_address += total_cstr_len;
                }
                if (total_cstr_len > 0)
                    s->PutChar ('"');
            }
        }
    }
}

void
ClangASTContext::DumpTypeDescription (void* type)
{
    StreamFile s (stdout, false);
    DumpTypeDescription (&s);
    ClangASTMetadata *metadata = ClangASTContext::GetMetadata (getASTContext(), type);
    if (metadata)
    {
        metadata->Dump (&s);
    }
}

void
ClangASTContext::DumpTypeDescription (void* type, Stream *s)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        
        llvm::SmallVector<char, 1024> buf;
        llvm::raw_svector_ostream llvm_ostrm (buf);
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
            {
                GetCompleteType(type);
                
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    if (class_interface_decl)
                    {
                        clang::PrintingPolicy policy = getASTContext()->getPrintingPolicy();
                        class_interface_decl->print(llvm_ostrm, policy, s->GetIndentLevel());
                    }
                }
            }
                break;
                
            case clang::Type::Typedef:
            {
                const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
                if (typedef_type)
                {
                    const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
                    std::string clang_typedef_name (typedef_decl->getQualifiedNameAsString());
                    if (!clang_typedef_name.empty())
                    {
                        s->PutCString ("typedef ");
                        s->PutCString (clang_typedef_name.c_str());
                    }
                }
            }
                break;
                
            case clang::Type::Elaborated:
                CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).DumpTypeDescription(s);
                return;
                
            case clang::Type::Paren:
                CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).DumpTypeDescription(s);
                return;
                
            case clang::Type::Record:
            {
                GetCompleteType(type);
                
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                
                if (cxx_record_decl)
                    cxx_record_decl->print(llvm_ostrm, getASTContext()->getPrintingPolicy(), s->GetIndentLevel());
                else
                    record_decl->print(llvm_ostrm, getASTContext()->getPrintingPolicy(), s->GetIndentLevel());
            }
                break;
                
            default:
            {
                const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
                if (tag_type)
                {
                    clang::TagDecl *tag_decl = tag_type->getDecl();
                    if (tag_decl)
                        tag_decl->print(llvm_ostrm, 0);
                }
                else
                {
                    std::string clang_type_name(qual_type.getAsString());
                    if (!clang_type_name.empty())
                        s->PutCString (clang_type_name.c_str());
                }
            }
        }
        
        if (buf.size() > 0)
        {
            s->Write (buf.data(), buf.size());
        }
    }
}

// DWARF parsing functions
#pragma mark DWARF Parsing

#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/TypeList.h"

#include "Plugins/SymbolFile/DWARF/DWARFCompileUnit.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfoEntry.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "Plugins/SymbolFile/DWARF/DWARFDefines.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIECollection.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/DWARF/UniqueDWARFASTType.h"


class ClangASTContext::DelayedAddObjCClassProperty
{
public:
    DelayedAddObjCClassProperty(const CompilerType     &class_opaque_type,
                                const char             *property_name,
                                const CompilerType     &property_opaque_type,  // The property type is only required if you don't have an ivar decl
                                clang::ObjCIvarDecl    *ivar_decl,
                                const char             *property_setter_name,
                                const char             *property_getter_name,
                                uint32_t                property_attributes,
                                const ClangASTMetadata *metadata) :
        m_class_opaque_type     (class_opaque_type),
        m_property_name         (property_name),
        m_property_opaque_type  (property_opaque_type),
        m_ivar_decl             (ivar_decl),
        m_property_setter_name  (property_setter_name),
        m_property_getter_name  (property_getter_name),
        m_property_attributes   (property_attributes)
    {
        if (metadata != NULL)
        {
            m_metadata_ap.reset(new ClangASTMetadata());
            *m_metadata_ap = *metadata;
        }
    }

    DelayedAddObjCClassProperty (const DelayedAddObjCClassProperty &rhs)
    {
        *this = rhs;
    }

    DelayedAddObjCClassProperty& operator= (const DelayedAddObjCClassProperty &rhs)
    {
        m_class_opaque_type    = rhs.m_class_opaque_type;
        m_property_name        = rhs.m_property_name;
        m_property_opaque_type = rhs.m_property_opaque_type;
        m_ivar_decl            = rhs.m_ivar_decl;
        m_property_setter_name = rhs.m_property_setter_name;
        m_property_getter_name = rhs.m_property_getter_name;
        m_property_attributes  = rhs.m_property_attributes;

        if (rhs.m_metadata_ap.get())
        {
            m_metadata_ap.reset (new ClangASTMetadata());
            *m_metadata_ap = *rhs.m_metadata_ap;
        }
        return *this;
    }

    bool
    Finalize()
    {
        ClangASTContext* ast = m_class_opaque_type.GetTypeSystem()->AsClangASTContext();
        assert(ast);
        return ast->AddObjCClassProperty (m_class_opaque_type,
                                          m_property_name,
                                          m_property_opaque_type,
                                          m_ivar_decl,
                                          m_property_setter_name,
                                          m_property_getter_name,
                                          m_property_attributes,
                                          m_metadata_ap.get());
    }

private:
    CompilerType            m_class_opaque_type;
    const char             *m_property_name;
    CompilerType            m_property_opaque_type;
    clang::ObjCIvarDecl    *m_ivar_decl;
    const char             *m_property_setter_name;
    const char             *m_property_getter_name;
    uint32_t                m_property_attributes;
    std::unique_ptr<ClangASTMetadata> m_metadata_ap;
};

bool
ClangASTContext::ParseTemplateDIE (SymbolFileDWARF *dwarf,
                                   DWARFCompileUnit* dwarf_cu,
                                   const DWARFDebugInfoEntry *die,
                                   ClangASTContext::TemplateParameterInfos &template_param_infos)
{
    const dw_tag_t tag = die->Tag();

    switch (tag)
    {
        case DW_TAG_template_type_parameter:
        case DW_TAG_template_value_parameter:
        {
            const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize(), dwarf_cu->IsDWARF64());

            DWARFDebugInfoEntry::Attributes attributes;
            const size_t num_attributes = die->GetAttributes (dwarf,
                                                              dwarf_cu,
                                                              fixed_form_sizes,
                                                              attributes);
            const char *name = NULL;
            Type *lldb_type = NULL;
            CompilerType clang_type;
            uint64_t uval64 = 0;
            bool uval64_valid = false;
            if (num_attributes > 0)
            {
                DWARFFormValue form_value;
                for (size_t i=0; i<num_attributes; ++i)
                {
                    const dw_attr_t attr = attributes.AttributeAtIndex(i);

                    switch (attr)
                    {
                        case DW_AT_name:
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                                name = form_value.AsCString(&dwarf->get_debug_str_data());
                            break;

                        case DW_AT_type:
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                const dw_offset_t type_die_offset = form_value.Reference();
                                lldb_type = dwarf->ResolveTypeUID(type_die_offset);
                                if (lldb_type)
                                    clang_type = lldb_type->GetClangForwardType();
                            }
                            break;

                        case DW_AT_const_value:
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                uval64_valid = true;
                                uval64 = form_value.Unsigned();
                            }
                            break;
                        default:
                            break;
                    }
                }

                clang::ASTContext *ast = getASTContext();
                if (!clang_type)
                    clang_type = GetBasicType(eBasicTypeVoid);

                if (clang_type)
                {
                    bool is_signed = false;
                    if (name && name[0])
                        template_param_infos.names.push_back(name);
                    else
                        template_param_infos.names.push_back(NULL);

                    if (tag == DW_TAG_template_value_parameter &&
                        lldb_type != NULL &&
                        clang_type.IsIntegerType (is_signed) &&
                        uval64_valid)
                    {
                        llvm::APInt apint (lldb_type->GetByteSize() * 8, uval64, is_signed);
                        template_param_infos.args.push_back (clang::TemplateArgument (*ast,
                                                                                      llvm::APSInt(apint),
                                                                                      ClangASTContext::GetQualType(clang_type)));
                    }
                    else
                    {
                        template_param_infos.args.push_back (clang::TemplateArgument (ClangASTContext::GetQualType(clang_type)));
                    }
                }
                else
                {
                    return false;
                }

            }
        }
            return true;

        default:
            break;
    }
    return false;
}

bool
ClangASTContext::ParseTemplateParameterInfos (SymbolFileDWARF *dwarf,
                                              DWARFCompileUnit* dwarf_cu,
                                              const DWARFDebugInfoEntry *parent_die,
                                              ClangASTContext::TemplateParameterInfos &template_param_infos)
{

    if (parent_die == NULL)
        return false;

    Args template_parameter_names;
    for (const DWARFDebugInfoEntry *die = parent_die->GetFirstChild();
         die != NULL;
         die = die->GetSibling())
    {
        const dw_tag_t tag = die->Tag();

        switch (tag)
        {
            case DW_TAG_template_type_parameter:
            case DW_TAG_template_value_parameter:
                ParseTemplateDIE (dwarf, dwarf_cu, die, template_param_infos);
                break;

            default:
                break;
        }
    }
    if (template_param_infos.args.empty())
        return false;
    return template_param_infos.args.size() == template_param_infos.names.size();
}

clang::ClassTemplateDecl *
ClangASTContext::ParseClassTemplateDecl (SymbolFileDWARF *dwarf,
                                         clang::DeclContext *decl_ctx,
                                         lldb::AccessType access_type,
                                         const char *parent_name,
                                         int tag_decl_kind,
                                         const ClangASTContext::TemplateParameterInfos &template_param_infos)
{
    if (template_param_infos.IsValid())
    {
        std::string template_basename(parent_name);
        template_basename.erase (template_basename.find('<'));

        return CreateClassTemplateDecl (decl_ctx,
                                        access_type,
                                        template_basename.c_str(),
                                        tag_decl_kind,
                                        template_param_infos);
    }
    return NULL;
}

bool
ClangASTContext::ResolveClangOpaqueTypeDefinition (SymbolFileDWARF *dwarf,
                                                   DWARFCompileUnit *dwarf_cu,
                                                   const DWARFDebugInfoEntry* die,
                                                   lldb_private::Type *type,
                                                   CompilerType &clang_type)
{
    // Disable external storage for this type so we don't get anymore
    // clang::ExternalASTSource queries for this type.
    SetHasExternalStorage (clang_type.GetOpaqueQualType(), false);

    if (dwarf == nullptr || dwarf_cu == nullptr || die == nullptr)
        return false;

    const dw_tag_t tag = die->Tag();

    Log *log = nullptr; // (LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO|DWARF_LOG_TYPE_COMPLETION));
    if (log)
        dwarf->GetObjectFile()->GetModule()->LogMessageVerboseBacktrace (log,
                                                                         "0x%8.8" PRIx64 ": %s '%s' resolving forward declaration...",
                                                                         dwarf->MakeUserID(die->GetOffset()),
                                                                         DW_TAG_value_to_name(tag),
                                                                         type->GetName().AsCString());
    assert (clang_type);
    DWARFDebugInfoEntry::Attributes attributes;

    switch (tag)
    {
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_class_type:
        {
            LayoutInfo layout_info;

            {
                if (die->HasChildren())
                {
                    LanguageType class_language = eLanguageTypeUnknown;
                    if (ClangASTContext::IsObjCObjectOrInterfaceType(clang_type))
                    {
                        class_language = eLanguageTypeObjC;
                        // For objective C we don't start the definition when
                        // the class is created.
                        ClangASTContext::StartTagDeclarationDefinition (clang_type);
                    }

                    int tag_decl_kind = -1;
                    AccessType default_accessibility = eAccessNone;
                    if (tag == DW_TAG_structure_type)
                    {
                        tag_decl_kind = clang::TTK_Struct;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_union_type)
                    {
                        tag_decl_kind = clang::TTK_Union;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_class_type)
                    {
                        tag_decl_kind = clang::TTK_Class;
                        default_accessibility = eAccessPrivate;
                    }

                    SymbolContext sc(dwarf->GetCompUnitForDWARFCompUnit(dwarf_cu));
                    std::vector<clang::CXXBaseSpecifier *> base_classes;
                    std::vector<int> member_accessibilities;
                    bool is_a_class = false;
                    // Parse members and base classes first
                    DWARFDIECollection member_function_dies;

                    DelayedPropertyList delayed_properties;
                    ParseChildMembers (sc,
                                       dwarf,
                                       dwarf_cu,
                                       die,
                                       clang_type,
                                       class_language,
                                       base_classes,
                                       member_accessibilities,
                                       member_function_dies,
                                       delayed_properties,
                                       default_accessibility,
                                       is_a_class,
                                       layout_info);

                    // Now parse any methods if there were any...
                    size_t num_functions = member_function_dies.Size();
                    if (num_functions > 0)
                    {
                        for (size_t i=0; i<num_functions; ++i)
                        {
                            dwarf->ResolveType(dwarf_cu, member_function_dies.GetDIEPtrAtIndex(i));
                        }
                    }

                    if (class_language == eLanguageTypeObjC)
                    {
                        ConstString class_name (clang_type.GetTypeName());
                        if (class_name)
                        {
                            DIEArray method_die_offsets;
                            dwarf->GetObjCMethodDIEOffsets(class_name, method_die_offsets);

                            if (!method_die_offsets.empty())
                            {
                                DWARFDebugInfo* debug_info = dwarf->DebugInfo();

                                DWARFCompileUnit* method_cu = NULL;
                                const size_t num_matches = method_die_offsets.size();
                                for (size_t i=0; i<num_matches; ++i)
                                {
                                    const dw_offset_t die_offset = method_die_offsets[i];
                                    DWARFDebugInfoEntry *method_die = debug_info->GetDIEPtrWithCompileUnitHint (die_offset, &method_cu);

                                    if (method_die)
                                        dwarf->ResolveType (method_cu, method_die);
                                }
                            }

                            for (DelayedPropertyList::iterator pi = delayed_properties.begin(), pe = delayed_properties.end();
                                 pi != pe;
                                 ++pi)
                                pi->Finalize();
                        }
                    }

                    // If we have a DW_TAG_structure_type instead of a DW_TAG_class_type we
                    // need to tell the clang type it is actually a class.
                    if (class_language != eLanguageTypeObjC)
                    {
                        if (is_a_class && tag_decl_kind != clang::TTK_Class)
                            SetTagTypeKind (ClangASTContext::GetQualType(clang_type), clang::TTK_Class);
                    }

                    // Since DW_TAG_structure_type gets used for both classes
                    // and structures, we may need to set any DW_TAG_member
                    // fields to have a "private" access if none was specified.
                    // When we parsed the child members we tracked that actual
                    // accessibility value for each DW_TAG_member in the
                    // "member_accessibilities" array. If the value for the
                    // member is zero, then it was set to the "default_accessibility"
                    // which for structs was "public". Below we correct this
                    // by setting any fields to "private" that weren't correctly
                    // set.
                    if (is_a_class && !member_accessibilities.empty())
                    {
                        // This is a class and all members that didn't have
                        // their access specified are private.
                        SetDefaultAccessForRecordFields (GetAsRecordDecl(clang_type),
                                                         eAccessPrivate,
                                                         &member_accessibilities.front(),
                                                         member_accessibilities.size());
                    }

                    if (!base_classes.empty())
                    {
                        // Make sure all base classes refer to complete types and not
                        // forward declarations. If we don't do this, clang will crash
                        // with an assertion in the call to clang_type.SetBaseClassesForClassType()
                        bool base_class_error = false;
                        for (auto &base_class : base_classes)
                        {
                            clang::TypeSourceInfo *type_source_info = base_class->getTypeSourceInfo();
                            if (type_source_info)
                            {
                                CompilerType base_class_type (this, type_source_info->getType().getAsOpaquePtr());
                                if (base_class_type.GetCompleteType() == false)
                                {
                                    if (!base_class_error)
                                    {
                                        dwarf->GetObjectFile()->GetModule()->ReportError ("DWARF DIE at 0x%8.8x for class '%s' has a base class '%s' that is a forward declaration, not a complete definition.\nPlease file a bug against the compiler and include the preprocessed output for %s",
                                                                                          die->GetOffset(),
                                                                                          die->GetName(dwarf, dwarf_cu),
                                                                                          base_class_type.GetTypeName().GetCString(),
                                                                                          sc.comp_unit ? sc.comp_unit->GetPath().c_str() : "the source file");
                                    }
                                    // We have no choice other than to pretend that the base class
                                    // is complete. If we don't do this, clang will crash when we
                                    // call setBases() inside of "clang_type.SetBaseClassesForClassType()"
                                    // below. Since we provide layout assistance, all ivars in this
                                    // class and other classes will be fine, this is the best we can do
                                    // short of crashing.

                                    ClangASTContext::StartTagDeclarationDefinition (base_class_type);
                                    ClangASTContext::CompleteTagDeclarationDefinition (base_class_type);
                                }
                            }
                        }
                        SetBaseClassesForClassType (clang_type.GetOpaqueQualType(),
                                                    &base_classes.front(),
                                                    base_classes.size());

                        // Clang will copy each CXXBaseSpecifier in "base_classes"
                        // so we have to free them all.
                        ClangASTContext::DeleteBaseClassSpecifiers (&base_classes.front(),
                                                                    base_classes.size());
                    }
                }
            }

            ClangASTContext::BuildIndirectFields (clang_type);
            ClangASTContext::CompleteTagDeclarationDefinition (clang_type);

            if (!layout_info.field_offsets.empty() ||
                !layout_info.base_offsets.empty()  ||
                !layout_info.vbase_offsets.empty() )
            {
                if (type)
                    layout_info.bit_size = type->GetByteSize() * 8;
                if (layout_info.bit_size == 0)
                    layout_info.bit_size = die->GetAttributeValueAsUnsigned(dwarf, dwarf_cu, DW_AT_byte_size, 0) * 8;

                clang::CXXRecordDecl *record_decl = GetAsCXXRecordDecl(clang_type.GetOpaqueQualType());
                if (record_decl)
                {
                    if (log)
                    {
                        ModuleSP module_sp = dwarf->GetObjectFile()->GetModule();

                        if (module_sp)
                        {
                            module_sp->LogMessage (log,
                                                   "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) caching layout info for record_decl = %p, bit_size = %" PRIu64 ", alignment = %" PRIu64 ", field_offsets[%u], base_offsets[%u], vbase_offsets[%u])",
                                                   static_cast<void*>(clang_type.GetOpaqueQualType()),
                                                   static_cast<void*>(record_decl),
                                                   layout_info.bit_size,
                                                   layout_info.alignment,
                                                   static_cast<uint32_t>(layout_info.field_offsets.size()),
                                                   static_cast<uint32_t>(layout_info.base_offsets.size()),
                                                   static_cast<uint32_t>(layout_info.vbase_offsets.size()));

                            uint32_t idx;
                            {
                                llvm::DenseMap<const clang::FieldDecl *, uint64_t>::const_iterator pos,
                                end = layout_info.field_offsets.end();
                                for (idx = 0, pos = layout_info.field_offsets.begin(); pos != end; ++pos, ++idx)
                                {
                                    module_sp->LogMessage(log,
                                                          "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) field[%u] = { bit_offset=%u, name='%s' }",
                                                          static_cast<void *>(clang_type.GetOpaqueQualType()),
                                                          idx,
                                                          static_cast<uint32_t>(pos->second),
                                                          pos->first->getNameAsString().c_str());
                                }
                            }

                            {
                                llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>::const_iterator base_pos,
                                base_end = layout_info.base_offsets.end();
                                for (idx = 0, base_pos = layout_info.base_offsets.begin(); base_pos != base_end; ++base_pos, ++idx)
                                {
                                    module_sp->LogMessage(log,
                                                          "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) base[%u] = { byte_offset=%u, name='%s' }",
                                                          clang_type.GetOpaqueQualType(), idx, (uint32_t)base_pos->second.getQuantity(),
                                                          base_pos->first->getNameAsString().c_str());
                                }
                            }
                            {
                                llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>::const_iterator vbase_pos,
                                vbase_end = layout_info.vbase_offsets.end();
                                for (idx = 0, vbase_pos = layout_info.vbase_offsets.begin(); vbase_pos != vbase_end; ++vbase_pos, ++idx)
                                {
                                    module_sp->LogMessage(log,
                                                          "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition (clang_type = %p) vbase[%u] = { byte_offset=%u, name='%s' }",
                                                          static_cast<void *>(clang_type.GetOpaqueQualType()), idx,
                                                          static_cast<uint32_t>(vbase_pos->second.getQuantity()),
                                                          vbase_pos->first->getNameAsString().c_str());
                                }
                            }

                        }
                    }
                    m_record_decl_to_layout_map.insert(std::make_pair(record_decl, layout_info));
                }
            }
        }
            
            return (bool)clang_type;
            
        case DW_TAG_enumeration_type:
            ClangASTContext::StartTagDeclarationDefinition (clang_type);
            if (die->HasChildren())
            {
                SymbolContext sc(dwarf->GetCompUnitForDWARFCompUnit(dwarf_cu));
                bool is_signed = false;
                clang_type.IsIntegerType(is_signed);
                ParseChildEnumerators(sc, clang_type, is_signed, type->GetByteSize(), dwarf, dwarf_cu, die);
            }
            ClangASTContext::CompleteTagDeclarationDefinition (clang_type);
            return (bool)clang_type;
            
        default:
            assert(false && "not a forward clang type decl!");
            break;
    }

    return false;
}


bool
ClangASTContext::LayoutRecordType(SymbolFileDWARF *dwarf,
                                  const clang::RecordDecl *record_decl,
                                  uint64_t &bit_size,
                                  uint64_t &alignment,
                                  llvm::DenseMap<const clang::FieldDecl *, uint64_t> &field_offsets,
                                  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                                  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets)
{
    RecordDeclToLayoutMap::iterator pos = m_record_decl_to_layout_map.find (record_decl);
    bool success = false;
    base_offsets.clear();
    vbase_offsets.clear();
    if (pos != m_record_decl_to_layout_map.end())
    {
        bit_size = pos->second.bit_size;
        alignment = pos->second.alignment;
        field_offsets.swap(pos->second.field_offsets);
        base_offsets.swap (pos->second.base_offsets);
        vbase_offsets.swap (pos->second.vbase_offsets);
        m_record_decl_to_layout_map.erase(pos);
        success = true;
    }
    else
    {
        bit_size = 0;
        alignment = 0;
        field_offsets.clear();
    }
    return success;
}


size_t
ClangASTContext::ParseChildEnumerators (const SymbolContext& sc,
                                        lldb_private::CompilerType &clang_type,
                                        bool is_signed,
                                        uint32_t enumerator_byte_size,
                                        SymbolFileDWARF *dwarf,
                                        DWARFCompileUnit* dwarf_cu,
                                        const DWARFDebugInfoEntry *parent_die)
{
    if (parent_die == NULL)
        return 0;

    size_t enumerators_added = 0;
    const DWARFDebugInfoEntry *die;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize(), dwarf_cu->IsDWARF64());

    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        const dw_tag_t tag = die->Tag();
        if (tag == DW_TAG_enumerator)
        {
            DWARFDebugInfoEntry::Attributes attributes;
            const size_t num_child_attributes = die->GetAttributes(dwarf, dwarf_cu, fixed_form_sizes, attributes);
            if (num_child_attributes > 0)
            {
                const char *name = NULL;
                bool got_value = false;
                int64_t enum_value = 0;
                Declaration decl;

                uint32_t i;
                for (i=0; i<num_child_attributes; ++i)
                {
                    const dw_attr_t attr = attributes.AttributeAtIndex(i);
                    DWARFFormValue form_value;
                    if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                    {
                        switch (attr)
                        {
                            case DW_AT_const_value:
                                got_value = true;
                                if (is_signed)
                                    enum_value = form_value.Signed();
                                else
                                    enum_value = form_value.Unsigned();
                                break;

                            case DW_AT_name:
                                name = form_value.AsCString(&dwarf->get_debug_str_data());
                                break;

                            case DW_AT_description:
                            default:
                            case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                            case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                            case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                            case DW_AT_sibling:
                                break;
                        }
                    }
                }

                if (name && name[0] && got_value)
                {
                    AddEnumerationValueToEnumerationType (clang_type.GetOpaqueQualType(),
                                                          GetEnumerationIntegerType(clang_type.GetOpaqueQualType()),
                                                          decl,
                                                          name,
                                                          enum_value,
                                                          enumerator_byte_size * 8);
                    ++enumerators_added;
                }
            }
        }
    }
    return enumerators_added;
}

#if defined(LLDB_CONFIGURATION_DEBUG) || defined(LLDB_CONFIGURATION_RELEASE)

class DIEStack
{
public:

    void Push (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
    {
        m_dies.push_back (DIEInfo(cu, die));
    }


    void LogDIEs (Log *log, SymbolFileDWARF *dwarf)
    {
        StreamString log_strm;
        const size_t n = m_dies.size();
        log_strm.Printf("DIEStack[%" PRIu64 "]:\n", (uint64_t)n);
        for (size_t i=0; i<n; i++)
        {
            DWARFCompileUnit *cu = m_dies[i].cu;
            const DWARFDebugInfoEntry *die = m_dies[i].die;
            std::string qualified_name;
            die->GetQualifiedName(dwarf, cu, qualified_name);
            log_strm.Printf ("[%" PRIu64 "] 0x%8.8x: %s name='%s'\n",
                             (uint64_t)i,
                             die->GetOffset(),
                             DW_TAG_value_to_name(die->Tag()),
                             qualified_name.c_str());
        }
        log->PutCString(log_strm.GetData());
    }
    void Pop ()
    {
        m_dies.pop_back();
    }

    class ScopedPopper
    {
    public:
        ScopedPopper (DIEStack &die_stack) :
        m_die_stack (die_stack),
        m_valid (false)
        {
        }

        void
        Push (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
        {
            m_valid = true;
            m_die_stack.Push (cu, die);
        }

        ~ScopedPopper ()
        {
            if (m_valid)
                m_die_stack.Pop();
        }



    protected:
        DIEStack &m_die_stack;
        bool m_valid;
    };

protected:
    struct DIEInfo {
        DIEInfo (DWARFCompileUnit *c, const DWARFDebugInfoEntry *d) :
        cu(c),
        die(d)
        {
        }
        DWARFCompileUnit *cu;
        const DWARFDebugInfoEntry *die;
    };
    typedef std::vector<DIEInfo> Stack;
    Stack m_dies;
};
#endif



static AccessType
DW_ACCESS_to_AccessType (uint32_t dwarf_accessibility)
{
    switch (dwarf_accessibility)
    {
        case DW_ACCESS_public:      return eAccessPublic;
        case DW_ACCESS_private:     return eAccessPrivate;
        case DW_ACCESS_protected:   return eAccessProtected;
        default:                    break;
    }
    return eAccessNone;
}

static bool
DeclKindIsCXXClass (clang::Decl::Kind decl_kind)
{
    switch (decl_kind)
    {
        case clang::Decl::CXXRecord:
        case clang::Decl::ClassTemplateSpecialization:
            return true;
        default:
            break;
    }
    return false;
}

struct BitfieldInfo
{
    uint64_t bit_size;
    uint64_t bit_offset;

    BitfieldInfo () :
    bit_size (LLDB_INVALID_ADDRESS),
    bit_offset (LLDB_INVALID_ADDRESS)
    {
    }

    void
    Clear()
    {
        bit_size = LLDB_INVALID_ADDRESS;
        bit_offset = LLDB_INVALID_ADDRESS;
    }

    bool IsValid ()
    {
        return (bit_size != LLDB_INVALID_ADDRESS) &&
        (bit_offset != LLDB_INVALID_ADDRESS);
    }
};

Function *
ClangASTContext::ParseFunctionFromDWARF (const SymbolContext& sc,
                                         SymbolFileDWARF *dwarf,
                                         DWARFCompileUnit* dwarf_cu,
                                         const DWARFDebugInfoEntry *die)
{
    DWARFDebugRanges::RangeList func_ranges;
    const char *name = NULL;
    const char *mangled = NULL;
    int decl_file = 0;
    int decl_line = 0;
    int decl_column = 0;
    int call_file = 0;
    int call_line = 0;
    int call_column = 0;
    DWARFExpression frame_base;

    assert (die->Tag() == DW_TAG_subprogram);

    if (die->Tag() != DW_TAG_subprogram)
        return NULL;

    if (die->GetDIENamesAndRanges (dwarf,
                                   dwarf_cu,
                                   name,
                                   mangled,
                                   func_ranges,
                                   decl_file,
                                   decl_line,
                                   decl_column,
                                   call_file,
                                   call_line,
                                   call_column,
                                   &frame_base))
    {
        // Union of all ranges in the function DIE (if the function is discontiguous)
        AddressRange func_range;
        lldb::addr_t lowest_func_addr = func_ranges.GetMinRangeBase (0);
        lldb::addr_t highest_func_addr = func_ranges.GetMaxRangeEnd (0);
        if (lowest_func_addr != LLDB_INVALID_ADDRESS && lowest_func_addr <= highest_func_addr)
        {
            ModuleSP module_sp (dwarf->GetObjectFile()->GetModule());
            func_range.GetBaseAddress().ResolveAddressUsingFileSections (lowest_func_addr, module_sp->GetSectionList());
            if (func_range.GetBaseAddress().IsValid())
                func_range.SetByteSize(highest_func_addr - lowest_func_addr);
        }

        if (func_range.GetBaseAddress().IsValid())
        {
            Mangled func_name;
            if (mangled)
                func_name.SetValue(ConstString(mangled), true);
            else if (die->GetParent()->Tag() == DW_TAG_compile_unit &&
                     LanguageRuntime::LanguageIsCPlusPlus(dwarf_cu->GetLanguageType()) &&
                     name && strcmp(name, "main") != 0)
            {
                // If the mangled name is not present in the DWARF, generate the demangled name
                // using the decl context. We skip if the function is "main" as its name is
                // never mangled.
                bool is_static = false;
                bool is_variadic = false;
                unsigned type_quals = 0;
                std::vector<CompilerType> param_types;
                std::vector<clang::ParmVarDecl*> param_decls;
                const DWARFDebugInfoEntry *decl_ctx_die = NULL;
                DWARFDeclContext decl_ctx;
                StreamString sstr;

                die->GetDWARFDeclContext(dwarf, dwarf_cu, decl_ctx);
                sstr << decl_ctx.GetQualifiedName();

                clang::DeclContext *containing_decl_ctx = GetClangDeclContextContainingDIE(dwarf,
                                                                                           dwarf_cu,
                                                                                           die,
                                                                                           &decl_ctx_die);
                ParseChildParameters(sc,
                                     containing_decl_ctx,
                                     dwarf,
                                     dwarf_cu,
                                     die,
                                     true,
                                     is_static,
                                     is_variadic,
                                     param_types,
                                     param_decls,
                                     type_quals);
                sstr << "(";
                for (size_t i = 0; i < param_types.size(); i++)
                {
                    if (i > 0)
                        sstr << ", ";
                    sstr << param_types[i].GetTypeName();
                }
                if (is_variadic)
                    sstr << ", ...";
                sstr << ")";
                if (type_quals & clang::Qualifiers::Const)
                    sstr << " const";

                func_name.SetValue(ConstString(sstr.GetData()), false);
            }
            else
                func_name.SetValue(ConstString(name), false);

            FunctionSP func_sp;
            std::unique_ptr<Declaration> decl_ap;
            if (decl_file != 0 || decl_line != 0 || decl_column != 0)
                decl_ap.reset(new Declaration (sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file),
                                               decl_line,
                                               decl_column));

            // Supply the type _only_ if it has already been parsed
            Type *func_type = dwarf->m_die_to_type.lookup (die);

            assert(func_type == NULL || func_type != DIE_IS_BEING_PARSED);

            if (dwarf->FixupAddress (func_range.GetBaseAddress()))
            {
                const user_id_t func_user_id = dwarf->MakeUserID(die->GetOffset());
                func_sp.reset(new Function (sc.comp_unit,
                                            dwarf->MakeUserID(func_user_id),       // UserID is the DIE offset
                                            dwarf->MakeUserID(func_user_id),
                                            func_name,
                                            func_type,
                                            func_range));           // first address range

                if (func_sp.get() != NULL)
                {
                    if (frame_base.IsValid())
                        func_sp->GetFrameBaseExpression() = frame_base;
                    sc.comp_unit->AddFunction(func_sp);
                    return func_sp.get();
                }
            }
        }
    }
    return NULL;
}


size_t
ClangASTContext::ParseChildMembers (const SymbolContext& sc,
                                    SymbolFileDWARF *dwarf,
                                    DWARFCompileUnit* dwarf_cu,
                                    const DWARFDebugInfoEntry *parent_die,
                                    CompilerType &class_clang_type,
                                    const LanguageType class_language,
                                    std::vector<clang::CXXBaseSpecifier *>& base_classes,
                                    std::vector<int>& member_accessibilities,
                                    DWARFDIECollection& member_function_dies,
                                    DelayedPropertyList& delayed_properties,
                                    AccessType& default_accessibility,
                                    bool &is_a_class,
                                    LayoutInfo &layout_info)
{
    if (parent_die == NULL)
        return 0;

    size_t count = 0;
    const DWARFDebugInfoEntry *die;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize(), dwarf_cu->IsDWARF64());
    uint32_t member_idx = 0;
    BitfieldInfo last_field_info;
    ModuleSP module_sp = dwarf->GetObjectFile()->GetModule();
    ClangASTContext* ast = class_clang_type.GetTypeSystem()->AsClangASTContext();
    if (ast == nullptr)
        return 0;

    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        dw_tag_t tag = die->Tag();

        switch (tag)
        {
            case DW_TAG_member:
            case DW_TAG_APPLE_property:
            {
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_attributes = die->GetAttributes (dwarf,
                                                                  dwarf_cu,
                                                                  fixed_form_sizes,
                                                                  attributes);
                if (num_attributes > 0)
                {
                    Declaration decl;
                    //DWARFExpression location;
                    const char *name = NULL;
                    const char *prop_name = NULL;
                    const char *prop_getter_name = NULL;
                    const char *prop_setter_name = NULL;
                    uint32_t prop_attributes = 0;


                    bool is_artificial = false;
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;
                    AccessType accessibility = eAccessNone;
                    uint32_t member_byte_offset = UINT32_MAX;
                    size_t byte_size = 0;
                    size_t bit_offset = 0;
                    size_t bit_size = 0;
                    bool is_external = false; // On DW_TAG_members, this means the member is static
                    uint32_t i;
                    for (i=0; i<num_attributes && !is_artificial; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_name:        name = form_value.AsCString(&dwarf->get_debug_str_data()); break;
                                case DW_AT_type:        encoding_uid = form_value.Reference(); break;
                                case DW_AT_bit_offset:  bit_offset = form_value.Unsigned(); break;
                                case DW_AT_bit_size:    bit_size = form_value.Unsigned(); break;
                                case DW_AT_byte_size:   byte_size = form_value.Unsigned(); break;
                                case DW_AT_data_member_location:
                                    if (form_value.BlockData())
                                    {
                                        Value initialValue(0);
                                        Value memberOffset(0);
                                        const DWARFDataExtractor& debug_info_data = dwarf->get_debug_info_data();
                                        uint32_t block_length = form_value.Unsigned();
                                        uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                                        if (DWARFExpression::Evaluate(NULL, // ExecutionContext *
                                                                      NULL, // ClangExpressionVariableList *
                                                                      NULL, // ClangExpressionDeclMap *
                                                                      NULL, // RegisterContext *
                                                                      module_sp,
                                                                      debug_info_data,
                                                                      block_offset,
                                                                      block_length,
                                                                      eRegisterKindDWARF,
                                                                      &initialValue,
                                                                      memberOffset,
                                                                      NULL))
                                        {
                                            member_byte_offset = memberOffset.ResolveValue(NULL).UInt();
                                        }
                                    }
                                    else
                                    {
                                        // With DWARF 3 and later, if the value is an integer constant,
                                        // this form value is the offset in bytes from the beginning
                                        // of the containing entity.
                                        member_byte_offset = form_value.Unsigned();
                                    }
                                    break;

                                case DW_AT_accessibility: accessibility = DW_ACCESS_to_AccessType (form_value.Unsigned()); break;
                                case DW_AT_artificial: is_artificial = form_value.Boolean(); break;
                                case DW_AT_APPLE_property_name:      prop_name = form_value.AsCString(&dwarf->get_debug_str_data()); break;
                                case DW_AT_APPLE_property_getter:    prop_getter_name = form_value.AsCString(&dwarf->get_debug_str_data()); break;
                                case DW_AT_APPLE_property_setter:    prop_setter_name = form_value.AsCString(&dwarf->get_debug_str_data()); break;
                                case DW_AT_APPLE_property_attribute: prop_attributes = form_value.Unsigned(); break;
                                case DW_AT_external:                 is_external = form_value.Boolean(); break;

                                default:
                                case DW_AT_declaration:
                                case DW_AT_description:
                                case DW_AT_mutable:
                                case DW_AT_visibility:
                                case DW_AT_sibling:
                                    break;
                            }
                        }
                    }

                    if (prop_name)
                    {
                        ConstString fixed_getter;
                        ConstString fixed_setter;

                        // Check if the property getter/setter were provided as full
                        // names.  We want basenames, so we extract them.

                        if (prop_getter_name && prop_getter_name[0] == '-')
                        {
                            ObjCLanguageRuntime::MethodName prop_getter_method(prop_getter_name, true);
                            prop_getter_name = prop_getter_method.GetSelector().GetCString();
                        }

                        if (prop_setter_name && prop_setter_name[0] == '-')
                        {
                            ObjCLanguageRuntime::MethodName prop_setter_method(prop_setter_name, true);
                            prop_setter_name = prop_setter_method.GetSelector().GetCString();
                        }

                        // If the names haven't been provided, they need to be
                        // filled in.

                        if (!prop_getter_name)
                        {
                            prop_getter_name = prop_name;
                        }
                        if (!prop_setter_name && prop_name[0] && !(prop_attributes & DW_APPLE_PROPERTY_readonly))
                        {
                            StreamString ss;

                            ss.Printf("set%c%s:",
                                      toupper(prop_name[0]),
                                      &prop_name[1]);

                            fixed_setter.SetCString(ss.GetData());
                            prop_setter_name = fixed_setter.GetCString();
                        }
                    }

                    // Clang has a DWARF generation bug where sometimes it
                    // represents fields that are references with bad byte size
                    // and bit size/offset information such as:
                    //
                    //  DW_AT_byte_size( 0x00 )
                    //  DW_AT_bit_size( 0x40 )
                    //  DW_AT_bit_offset( 0xffffffffffffffc0 )
                    //
                    // So check the bit offset to make sure it is sane, and if
                    // the values are not sane, remove them. If we don't do this
                    // then we will end up with a crash if we try to use this
                    // type in an expression when clang becomes unhappy with its
                    // recycled debug info.

                    if (bit_offset > 128)
                    {
                        bit_size = 0;
                        bit_offset = 0;
                    }

                    // FIXME: Make Clang ignore Objective-C accessibility for expressions
                    if (class_language == eLanguageTypeObjC ||
                        class_language == eLanguageTypeObjC_plus_plus)
                        accessibility = eAccessNone;

                    if (member_idx == 0 && !is_artificial && name && (strstr (name, "_vptr$") == name))
                    {
                        // Not all compilers will mark the vtable pointer
                        // member as artificial (llvm-gcc). We can't have
                        // the virtual members in our classes otherwise it
                        // throws off all child offsets since we end up
                        // having and extra pointer sized member in our
                        // class layouts.
                        is_artificial = true;
                    }

                    // Handle static members
                    if (is_external && member_byte_offset == UINT32_MAX)
                    {
                        Type *var_type = dwarf->ResolveTypeUID(encoding_uid);

                        if (var_type)
                        {
                            if (accessibility == eAccessNone)
                                accessibility = eAccessPublic;
                            ClangASTContext::AddVariableToRecordType (class_clang_type,
                                                                      name,
                                                                      var_type->GetClangLayoutType(),
                                                                      accessibility);
                        }
                        break;
                    }

                    if (is_artificial == false)
                    {
                        Type *member_type = dwarf->ResolveTypeUID(encoding_uid);

                        clang::FieldDecl *field_decl = NULL;
                        if (tag == DW_TAG_member)
                        {
                            if (member_type)
                            {
                                if (accessibility == eAccessNone)
                                    accessibility = default_accessibility;
                                member_accessibilities.push_back(accessibility);

                                uint64_t field_bit_offset = (member_byte_offset == UINT32_MAX ? 0 : (member_byte_offset * 8));
                                if (bit_size > 0)
                                {

                                    BitfieldInfo this_field_info;
                                    this_field_info.bit_offset = field_bit_offset;
                                    this_field_info.bit_size = bit_size;

                                    /////////////////////////////////////////////////////////////
                                    // How to locate a field given the DWARF debug information
                                    //
                                    // AT_byte_size indicates the size of the word in which the
                                    // bit offset must be interpreted.
                                    //
                                    // AT_data_member_location indicates the byte offset of the
                                    // word from the base address of the structure.
                                    //
                                    // AT_bit_offset indicates how many bits into the word
                                    // (according to the host endianness) the low-order bit of
                                    // the field starts.  AT_bit_offset can be negative.
                                    //
                                    // AT_bit_size indicates the size of the field in bits.
                                    /////////////////////////////////////////////////////////////

                                    if (byte_size == 0)
                                        byte_size = member_type->GetByteSize();

                                    if (dwarf->GetObjectFile()->GetByteOrder() == eByteOrderLittle)
                                    {
                                        this_field_info.bit_offset += byte_size * 8;
                                        this_field_info.bit_offset -= (bit_offset + bit_size);
                                    }
                                    else
                                    {
                                        this_field_info.bit_offset += bit_offset;
                                    }

                                    // Update the field bit offset we will report for layout
                                    field_bit_offset = this_field_info.bit_offset;

                                    // If the member to be emitted did not start on a character boundary and there is
                                    // empty space between the last field and this one, then we need to emit an
                                    // anonymous member filling up the space up to its start.  There are three cases
                                    // here:
                                    //
                                    // 1 If the previous member ended on a character boundary, then we can emit an
                                    //   anonymous member starting at the most recent character boundary.
                                    //
                                    // 2 If the previous member did not end on a character boundary and the distance
                                    //   from the end of the previous member to the current member is less than a
                                    //   word width, then we can emit an anonymous member starting right after the
                                    //   previous member and right before this member.
                                    //
                                    // 3 If the previous member did not end on a character boundary and the distance
                                    //   from the end of the previous member to the current member is greater than
                                    //   or equal a word width, then we act as in Case 1.

                                    const uint64_t character_width = 8;
                                    const uint64_t word_width = 32;

                                    // Objective-C has invalid DW_AT_bit_offset values in older versions
                                    // of clang, so we have to be careful and only insert unnamed bitfields
                                    // if we have a new enough clang.
                                    bool detect_unnamed_bitfields = true;

                                    if (class_language == eLanguageTypeObjC || class_language == eLanguageTypeObjC_plus_plus)
                                        detect_unnamed_bitfields = dwarf_cu->Supports_unnamed_objc_bitfields ();

                                    if (detect_unnamed_bitfields)
                                    {
                                        BitfieldInfo anon_field_info;

                                        if ((this_field_info.bit_offset % character_width) != 0) // not char aligned
                                        {
                                            uint64_t last_field_end = 0;

                                            if (last_field_info.IsValid())
                                                last_field_end = last_field_info.bit_offset + last_field_info.bit_size;

                                            if (this_field_info.bit_offset != last_field_end)
                                            {
                                                if (((last_field_end % character_width) == 0) ||                    // case 1
                                                    (this_field_info.bit_offset - last_field_end >= word_width))    // case 3
                                                {
                                                    anon_field_info.bit_size = this_field_info.bit_offset % character_width;
                                                    anon_field_info.bit_offset = this_field_info.bit_offset - anon_field_info.bit_size;
                                                }
                                                else                                                                // case 2
                                                {
                                                    anon_field_info.bit_size = this_field_info.bit_offset - last_field_end;
                                                    anon_field_info.bit_offset = last_field_end;
                                                }
                                            }
                                        }

                                        if (anon_field_info.IsValid())
                                        {
                                            clang::FieldDecl *unnamed_bitfield_decl =
                                            ClangASTContext::AddFieldToRecordType (class_clang_type,
                                                                                   NULL,
                                                                                   GetBuiltinTypeForEncodingAndBitSize(eEncodingSint, word_width),
                                                                                   accessibility,
                                                                                   anon_field_info.bit_size);

                                            layout_info.field_offsets.insert(
                                                                             std::make_pair(unnamed_bitfield_decl, anon_field_info.bit_offset));
                                        }
                                    }
                                    last_field_info = this_field_info;
                                }
                                else
                                {
                                    last_field_info.Clear();
                                }

                                CompilerType member_clang_type = member_type->GetClangLayoutType();

                                {
                                    // Older versions of clang emit array[0] and array[1] in the same way (<rdar://problem/12566646>).
                                    // If the current field is at the end of the structure, then there is definitely no room for extra
                                    // elements and we override the type to array[0].

                                    CompilerType member_array_element_type;
                                    uint64_t member_array_size;
                                    bool member_array_is_incomplete;

                                    if (member_clang_type.IsArrayType(&member_array_element_type,
                                                                      &member_array_size,
                                                                      &member_array_is_incomplete) &&
                                        !member_array_is_incomplete)
                                    {
                                        uint64_t parent_byte_size = parent_die->GetAttributeValueAsUnsigned(dwarf, dwarf_cu, DW_AT_byte_size, UINT64_MAX);

                                        if (member_byte_offset >= parent_byte_size)
                                        {
                                            if (member_array_size != 1)
                                            {
                                                module_sp->ReportError ("0x%8.8" PRIx64 ": DW_TAG_member '%s' refers to type 0x%8.8" PRIx64 " which extends beyond the bounds of 0x%8.8" PRIx64,
                                                                                           dwarf->MakeUserID(die->GetOffset()),
                                                                                           name,
                                                                                           encoding_uid,
                                                                                           dwarf->MakeUserID(parent_die->GetOffset()));
                                            }

                                            member_clang_type = CreateArrayType(member_array_element_type, 0, false);
                                        }
                                    }
                                }

                                field_decl = ClangASTContext::AddFieldToRecordType (class_clang_type,
                                                                                    name,
                                                                                    member_clang_type,
                                                                                    accessibility,
                                                                                    bit_size);

                                SetMetadataAsUserID (field_decl, dwarf->MakeUserID(die->GetOffset()));

                                layout_info.field_offsets.insert(std::make_pair(field_decl, field_bit_offset));
                            }
                            else
                            {
                                if (name)
                                    module_sp->ReportError ("0x%8.8" PRIx64 ": DW_TAG_member '%s' refers to type 0x%8.8" PRIx64 " which was unable to be parsed",
                                                                               dwarf->MakeUserID(die->GetOffset()),
                                                                               name,
                                                                               encoding_uid);
                                else
                                    module_sp->ReportError ("0x%8.8" PRIx64 ": DW_TAG_member refers to type 0x%8.8" PRIx64 " which was unable to be parsed",
                                                                               dwarf->MakeUserID(die->GetOffset()),
                                                                               encoding_uid);
                            }
                        }

                        if (prop_name != NULL && member_type)
                        {
                            clang::ObjCIvarDecl *ivar_decl = NULL;

                            if (field_decl)
                            {
                                ivar_decl = clang::dyn_cast<clang::ObjCIvarDecl>(field_decl);
                                assert (ivar_decl != NULL);
                            }

                            ClangASTMetadata metadata;
                            metadata.SetUserID (dwarf->MakeUserID(die->GetOffset()));
                            delayed_properties.push_back(DelayedAddObjCClassProperty(class_clang_type,
                                                                                     prop_name,
                                                                                     member_type->GetClangLayoutType(),
                                                                                     ivar_decl,
                                                                                     prop_setter_name,
                                                                                     prop_getter_name,
                                                                                     prop_attributes,
                                                                                     &metadata));

                            if (ivar_decl)
                                SetMetadataAsUserID (ivar_decl, dwarf->MakeUserID(die->GetOffset()));
                        }
                    }
                }
                ++member_idx;
            }
                break;

            case DW_TAG_subprogram:
                // Let the type parsing code handle this one for us.
                member_function_dies.Append (die);
                break;

            case DW_TAG_inheritance:
            {
                is_a_class = true;
                if (default_accessibility == eAccessNone)
                    default_accessibility = eAccessPrivate;
                // TODO: implement DW_TAG_inheritance type parsing
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_attributes = die->GetAttributes (dwarf,
                                                                  dwarf_cu,
                                                                  fixed_form_sizes,
                                                                  attributes);
                if (num_attributes > 0)
                {
                    Declaration decl;
                    DWARFExpression location;
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;
                    AccessType accessibility = default_accessibility;
                    bool is_virtual = false;
                    bool is_base_of_class = true;
                    off_t member_byte_offset = 0;
                    uint32_t i;
                    for (i=0; i<num_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_type:        encoding_uid = form_value.Reference(); break;
                                case DW_AT_data_member_location:
                                    if (form_value.BlockData())
                                    {
                                        Value initialValue(0);
                                        Value memberOffset(0);
                                        const DWARFDataExtractor& debug_info_data = dwarf->get_debug_info_data();
                                        uint32_t block_length = form_value.Unsigned();
                                        uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                                        if (DWARFExpression::Evaluate (NULL,
                                                                       NULL,
                                                                       NULL,
                                                                       NULL,
                                                                       module_sp,
                                                                       debug_info_data,
                                                                       block_offset,
                                                                       block_length,
                                                                       eRegisterKindDWARF,
                                                                       &initialValue,
                                                                       memberOffset,
                                                                       NULL))
                                        {
                                            member_byte_offset = memberOffset.ResolveValue(NULL).UInt();
                                        }
                                    }
                                    else
                                    {
                                        // With DWARF 3 and later, if the value is an integer constant,
                                        // this form value is the offset in bytes from the beginning
                                        // of the containing entity.
                                        member_byte_offset = form_value.Unsigned();
                                    }
                                    break;

                                case DW_AT_accessibility:
                                    accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned());
                                    break;

                                case DW_AT_virtuality:
                                    is_virtual = form_value.Boolean();
                                    break;

                                case DW_AT_sibling:
                                    break;

                                default:
                                    break;
                            }
                        }
                    }

                    Type *base_class_type = dwarf->ResolveTypeUID(encoding_uid);
                    if (base_class_type == NULL)
                    {
                        module_sp->ReportError("0x%8.8x: DW_TAG_inheritance failed to resolve the base class at 0x%8.8" PRIx64 " from enclosing type 0x%8.8x. \nPlease file a bug and attach the file at the start of this error message",
                                                                  die->GetOffset(),
                                                                  encoding_uid,
                                                                  parent_die->GetOffset());
                        break;
                    }

                    CompilerType base_class_clang_type = base_class_type->GetClangFullType();
                    assert (base_class_clang_type);
                    if (class_language == eLanguageTypeObjC)
                    {
                        ast->SetObjCSuperClass(class_clang_type, base_class_clang_type);
                    }
                    else
                    {
                        base_classes.push_back (ast->CreateBaseClassSpecifier (base_class_clang_type.GetOpaqueQualType(),
                                                                               accessibility,
                                                                               is_virtual,
                                                                               is_base_of_class));

                        if (is_virtual)
                        {
                            // Do not specify any offset for virtual inheritance. The DWARF produced by clang doesn't
                            // give us a constant offset, but gives us a DWARF expressions that requires an actual object
                            // in memory. the DW_AT_data_member_location for a virtual base class looks like:
                            //      DW_AT_data_member_location( DW_OP_dup, DW_OP_deref, DW_OP_constu(0x00000018), DW_OP_minus, DW_OP_deref, DW_OP_plus )
                            // Given this, there is really no valid response we can give to clang for virtual base
                            // class offsets, and this should eventually be removed from LayoutRecordType() in the external
                            // AST source in clang.
                        }
                        else
                        {
                            layout_info.base_offsets.insert(
                                                            std::make_pair(ast->GetAsCXXRecordDecl(base_class_clang_type.GetOpaqueQualType()),
                                                                           clang::CharUnits::fromQuantity(member_byte_offset)));
                        }
                    }
                }
            }
                break;
                
            default:
                break;
        }
    }
    
    return count;
}


size_t
ClangASTContext::ParseChildParameters (const SymbolContext& sc,
                                       clang::DeclContext *containing_decl_ctx,
                                       SymbolFileDWARF *dwarf,
                                       DWARFCompileUnit* dwarf_cu,
                                       const DWARFDebugInfoEntry *parent_die,
                                       bool skip_artificial,
                                       bool &is_static,
                                       bool &is_variadic,
                                       std::vector<CompilerType>& function_param_types,
                                       std::vector<clang::ParmVarDecl*>& function_param_decls,
                                       unsigned &type_quals)
{
    if (parent_die == NULL)
        return 0;

    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize(), dwarf_cu->IsDWARF64());

    size_t arg_idx = 0;
    const DWARFDebugInfoEntry *die;
    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        dw_tag_t tag = die->Tag();
        switch (tag)
        {
            case DW_TAG_formal_parameter:
            {
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, fixed_form_sizes, attributes);
                if (num_attributes > 0)
                {
                    const char *name = NULL;
                    Declaration decl;
                    dw_offset_t param_type_die_offset = DW_INVALID_OFFSET;
                    bool is_artificial = false;
                    // one of None, Auto, Register, Extern, Static, PrivateExtern

                    clang::StorageClass storage = clang::SC_None;
                    uint32_t i;
                    for (i=0; i<num_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                case DW_AT_name:        name = form_value.AsCString(&dwarf->get_debug_str_data()); break;
                                case DW_AT_type:        param_type_die_offset = form_value.Reference(); break;
                                case DW_AT_artificial:  is_artificial = form_value.Boolean(); break;
                                case DW_AT_location:
                                    //                          if (form_value.BlockData())
                                    //                          {
                                    //                              const DWARFDataExtractor& debug_info_data = debug_info();
                                    //                              uint32_t block_length = form_value.Unsigned();
                                    //                              DWARFDataExtractor location(debug_info_data, form_value.BlockData() - debug_info_data.GetDataStart(), block_length);
                                    //                          }
                                    //                          else
                                    //                          {
                                    //                          }
                                    //                          break;
                                case DW_AT_const_value:
                                case DW_AT_default_value:
                                case DW_AT_description:
                                case DW_AT_endianity:
                                case DW_AT_is_optional:
                                case DW_AT_segment:
                                case DW_AT_variable_parameter:
                                default:
                                case DW_AT_abstract_origin:
                                case DW_AT_sibling:
                                    break;
                            }
                        }
                    }

                    bool skip = false;
                    if (skip_artificial)
                    {
                        if (is_artificial)
                        {
                            // In order to determine if a C++ member function is
                            // "const" we have to look at the const-ness of "this"...
                            // Ugly, but that
                            if (arg_idx == 0)
                            {
                                if (DeclKindIsCXXClass(containing_decl_ctx->getDeclKind()))
                                {
                                    // Often times compilers omit the "this" name for the
                                    // specification DIEs, so we can't rely upon the name
                                    // being in the formal parameter DIE...
                                    if (name == NULL || ::strcmp(name, "this")==0)
                                    {
                                        Type *this_type = dwarf->ResolveTypeUID (param_type_die_offset);
                                        if (this_type)
                                        {
                                            uint32_t encoding_mask = this_type->GetEncodingMask();
                                            if (encoding_mask & Type::eEncodingIsPointerUID)
                                            {
                                                is_static = false;

                                                if (encoding_mask & (1u << Type::eEncodingIsConstUID))
                                                    type_quals |= clang::Qualifiers::Const;
                                                if (encoding_mask & (1u << Type::eEncodingIsVolatileUID))
                                                    type_quals |= clang::Qualifiers::Volatile;
                                            }
                                        }
                                    }
                                }
                            }
                            skip = true;
                        }
                        else
                        {

                            // HACK: Objective C formal parameters "self" and "_cmd"
                            // are not marked as artificial in the DWARF...
                            CompileUnit *comp_unit = dwarf->GetCompUnitForDWARFCompUnit(dwarf_cu, UINT32_MAX);
                            if (comp_unit)
                            {
                                switch (comp_unit->GetLanguage())
                                {
                                    case eLanguageTypeObjC:
                                    case eLanguageTypeObjC_plus_plus:
                                        if (name && name[0] && (strcmp (name, "self") == 0 || strcmp (name, "_cmd") == 0))
                                            skip = true;
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                    }

                    if (!skip)
                    {
                        Type *type = dwarf->ResolveTypeUID(param_type_die_offset);
                        if (type)
                        {
                            function_param_types.push_back (type->GetClangForwardType());

                            clang::ParmVarDecl *param_var_decl = CreateParameterDeclaration (name,
                                                                                             type->GetClangForwardType(),
                                                                                             storage);
                            assert(param_var_decl);
                            function_param_decls.push_back(param_var_decl);

                            SetMetadataAsUserID (param_var_decl, dwarf->MakeUserID(die->GetOffset()));
                        }
                    }
                }
                arg_idx++;
            }
                break;

            case DW_TAG_unspecified_parameters:
                is_variadic = true;
                break;

            case DW_TAG_template_type_parameter:
            case DW_TAG_template_value_parameter:
                // The one caller of this was never using the template_param_infos,
                // and the local variable was taking up a large amount of stack space
                // in SymbolFileDWARF::ParseType() so this was removed. If we ever need
                // the template params back, we can add them back.
                // ParseTemplateDIE (dwarf_cu, die, template_param_infos);
                break;

            default:
                break;
        }
    }
    return arg_idx;
}

void
ClangASTContext::ParseChildArrayInfo (const SymbolContext& sc,
                                      SymbolFileDWARF *dwarf,
                                      DWARFCompileUnit* dwarf_cu,
                                      const DWARFDebugInfoEntry *parent_die,
                                      int64_t& first_index,
                                      std::vector<uint64_t>& element_orders,
                                      uint32_t& byte_stride,
                                      uint32_t& bit_stride)
{
    if (parent_die == NULL)
        return;

    const DWARFDebugInfoEntry *die;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (dwarf_cu->GetAddressByteSize(), dwarf_cu->IsDWARF64());
    for (die = parent_die->GetFirstChild(); die != NULL; die = die->GetSibling())
    {
        const dw_tag_t tag = die->Tag();
        switch (tag)
        {
            case DW_TAG_subrange_type:
            {
                DWARFDebugInfoEntry::Attributes attributes;
                const size_t num_child_attributes = die->GetAttributes(dwarf, dwarf_cu, fixed_form_sizes, attributes);
                if (num_child_attributes > 0)
                {
                    uint64_t num_elements = 0;
                    uint64_t lower_bound = 0;
                    uint64_t upper_bound = 0;
                    bool upper_bound_valid = false;
                    uint32_t i;
                    for (i=0; i<num_child_attributes; ++i)
                    {
                        const dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                        {
                            switch (attr)
                            {
                                case DW_AT_name:
                                    break;

                                case DW_AT_count:
                                    num_elements = form_value.Unsigned();
                                    break;

                                case DW_AT_bit_stride:
                                    bit_stride = form_value.Unsigned();
                                    break;

                                case DW_AT_byte_stride:
                                    byte_stride = form_value.Unsigned();
                                    break;

                                case DW_AT_lower_bound:
                                    lower_bound = form_value.Unsigned();
                                    break;

                                case DW_AT_upper_bound:
                                    upper_bound_valid = true;
                                    upper_bound = form_value.Unsigned();
                                    break;

                                default:
                                case DW_AT_abstract_origin:
                                case DW_AT_accessibility:
                                case DW_AT_allocated:
                                case DW_AT_associated:
                                case DW_AT_data_location:
                                case DW_AT_declaration:
                                case DW_AT_description:
                                case DW_AT_sibling:
                                case DW_AT_threads_scaled:
                                case DW_AT_type:
                                case DW_AT_visibility:
                                    break;
                            }
                        }
                    }
                    
                    if (num_elements == 0)
                    {
                        if (upper_bound_valid && upper_bound >= lower_bound)
                            num_elements = upper_bound - lower_bound + 1;
                    }
                    
                    element_orders.push_back (num_elements);
                }
            }
                break;
        }
    }
}

clang::DeclContext*
ClangASTContext::GetClangDeclContextContainingTypeUID (SymbolFileDWARF *dwarf, lldb::user_id_t type_uid)
{
    DWARFDebugInfo* debug_info = dwarf->DebugInfo();
    if (debug_info && dwarf->UserIDMatches(type_uid))
    {
        DWARFCompileUnitSP cu_sp;
        const DWARFDebugInfoEntry* die = debug_info->GetDIEPtr(type_uid, &cu_sp);
        if (die)
            return GetClangDeclContextContainingDIE (dwarf, cu_sp.get(), die, NULL);
    }
    return NULL;
}

clang::DeclContext*
ClangASTContext::GetClangDeclContextForTypeUID (SymbolFileDWARF *dwarf, const lldb_private::SymbolContext &sc, lldb::user_id_t type_uid)
{
    if (dwarf->UserIDMatches(type_uid))
        return GetClangDeclContextForDIEOffset (dwarf, sc, type_uid);
    return NULL;
}


clang::DeclContext *
ClangASTContext::GetClangDeclContextForDIE (SymbolFileDWARF *dwarf,
                                            const SymbolContext &sc,
                                            DWARFCompileUnit *cu,
                                            const DWARFDebugInfoEntry *die)
{
    clang::DeclContext *clang_decl_ctx = dwarf->GetCachedClangDeclContextForDIE (die);
    if (clang_decl_ctx)
        return clang_decl_ctx;
    // If this DIE has a specification, or an abstract origin, then trace to those.

    dw_offset_t die_offset = die->GetAttributeValueAsReference(dwarf, cu, DW_AT_specification, DW_INVALID_OFFSET);
    if (die_offset != DW_INVALID_OFFSET)
        return GetClangDeclContextForDIEOffset (dwarf, sc, die_offset);

    die_offset = die->GetAttributeValueAsReference(dwarf, cu, DW_AT_abstract_origin, DW_INVALID_OFFSET);
    if (die_offset != DW_INVALID_OFFSET)
        return GetClangDeclContextForDIEOffset (dwarf, sc, die_offset);

    Log *log = nullptr; //(LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
    if (log)
        dwarf->GetObjectFile()->GetModule()->LogMessage(log, "SymbolFileDWARF::GetClangDeclContextForDIE (die = 0x%8.8x) %s '%s'", die->GetOffset(), DW_TAG_value_to_name(die->Tag()), die->GetName(dwarf, cu));
    // This is the DIE we want.  Parse it, then query our map.
    bool assert_not_being_parsed = true;
    dwarf->ResolveTypeUID (cu, die, assert_not_being_parsed);

    clang_decl_ctx = dwarf->GetCachedClangDeclContextForDIE (die);

    return clang_decl_ctx;
}


clang::DeclContext *
ClangASTContext::GetClangDeclContextContainingDIEOffset (SymbolFileDWARF *dwarf,
                                                         dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        DWARFCompileUnitSP cu_sp;
        const DWARFDebugInfoEntry* die = dwarf->DebugInfo()->GetDIEPtr(die_offset, &cu_sp);
        return GetClangDeclContextContainingDIE (dwarf, cu_sp.get(), die, NULL);
    }
    return NULL;
}

clang::DeclContext *
ClangASTContext::GetClangDeclContextForDIEOffset (SymbolFileDWARF *dwarf,
                                                  const SymbolContext &sc,
                                                  dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        DWARFDebugInfo* debug_info = dwarf->DebugInfo();
        if (debug_info)
        {
            DWARFCompileUnitSP cu_sp;
            const DWARFDebugInfoEntry* die = debug_info->GetDIEPtr(die_offset, &cu_sp);
            if (die)
                return GetClangDeclContextForDIE (dwarf, sc, cu_sp.get(), die);
        }
    }
    return NULL;
}

clang::NamespaceDecl *
ClangASTContext::ResolveNamespaceDIE (SymbolFileDWARF *dwarf,
                                      DWARFCompileUnit *dwarf_cu,
                                      const DWARFDebugInfoEntry *die)
{
    if (die && die->Tag() == DW_TAG_namespace)
    {
        // See if we already parsed this namespace DIE and associated it with a
        // uniqued namespace declaration
        clang::NamespaceDecl *namespace_decl = static_cast<clang::NamespaceDecl *>(dwarf->m_die_to_decl_ctx[die]);
        if (namespace_decl)
            return namespace_decl;
        else
        {
            const char *namespace_name = die->GetAttributeValueAsString(dwarf, dwarf_cu, DW_AT_name, NULL);
            clang::DeclContext *containing_decl_ctx = GetClangDeclContextContainingDIE (dwarf, dwarf_cu, die, NULL);
            namespace_decl = GetUniqueNamespaceDeclaration (namespace_name, containing_decl_ctx);
            Log *log = nullptr;// (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
            if (log)
            {
                if (namespace_name)
                {
                    dwarf->GetObjectFile()->GetModule()->LogMessage (log,
                                                                     "ASTContext => %p: 0x%8.8" PRIx64 ": DW_TAG_namespace with DW_AT_name(\"%s\") => clang::NamespaceDecl *%p (original = %p)",
                                                                     static_cast<void*>(getASTContext()),
                                                                     dwarf->MakeUserID(die->GetOffset()),
                                                                     namespace_name,
                                                                     static_cast<void*>(namespace_decl),
                                                                     static_cast<void*>(namespace_decl->getOriginalNamespace()));
                }
                else
                {
                    dwarf->GetObjectFile()->GetModule()->LogMessage (log,
                                                                     "ASTContext => %p: 0x%8.8" PRIx64 ": DW_TAG_namespace (anonymous) => clang::NamespaceDecl *%p (original = %p)",
                                                                     static_cast<void*>(getASTContext()),
                                                                     dwarf->MakeUserID(die->GetOffset()),
                                                                     static_cast<void*>(namespace_decl),
                                                                     static_cast<void*>(namespace_decl->getOriginalNamespace()));
                }
            }

            if (namespace_decl)
                dwarf->LinkDeclContextToDIE((clang::DeclContext*)namespace_decl, die);
            return namespace_decl;
        }
    }
    return NULL;
}

clang::DeclContext *
ClangASTContext::GetClangDeclContextContainingDIE (SymbolFileDWARF *dwarf,
                                                   DWARFCompileUnit *cu,
                                                   const DWARFDebugInfoEntry *die,
                                                   const DWARFDebugInfoEntry **decl_ctx_die_copy)
{
    if (dwarf->m_clang_tu_decl == NULL)
        dwarf->m_clang_tu_decl = getASTContext()->getTranslationUnitDecl();

    const DWARFDebugInfoEntry *decl_ctx_die = dwarf->GetDeclContextDIEContainingDIE (cu, die);

    if (decl_ctx_die_copy)
        *decl_ctx_die_copy = decl_ctx_die;

    if (decl_ctx_die)
    {

        SymbolFileDWARF::DIEToDeclContextMap::iterator pos = dwarf->m_die_to_decl_ctx.find (decl_ctx_die);
        if (pos != dwarf->m_die_to_decl_ctx.end())
            return pos->second;

        switch (decl_ctx_die->Tag())
        {
            case DW_TAG_compile_unit:
                return dwarf->m_clang_tu_decl;

            case DW_TAG_namespace:
                return ResolveNamespaceDIE (dwarf, cu, decl_ctx_die);

            case DW_TAG_structure_type:
            case DW_TAG_union_type:
            case DW_TAG_class_type:
            {
                Type* type = dwarf->ResolveType (cu, decl_ctx_die);
                if (type)
                {
                    clang::DeclContext *decl_ctx = GetDeclContextForType(type->GetClangForwardType());
                    if (decl_ctx)
                    {
                        dwarf->LinkDeclContextToDIE (decl_ctx, decl_ctx_die);
                        if (decl_ctx)
                            return decl_ctx;
                    }
                }
            }
                break;
                
            default:
                break;
        }
    }
    return dwarf->m_clang_tu_decl;
}



TypeSP
ClangASTContext::ParseTypeFromDWARF (const SymbolContext& sc,
                                     SymbolFileDWARF *dwarf,
                                     DWARFCompileUnit* dwarf_cu,
                                     const DWARFDebugInfoEntry *die,
                                     Log *log,
                                     bool *type_is_new_ptr)
{
    TypeSP type_sp;

    if (type_is_new_ptr)
        *type_is_new_ptr = false;

#if defined(LLDB_CONFIGURATION_DEBUG) || defined(LLDB_CONFIGURATION_RELEASE)
    static DIEStack g_die_stack;
    DIEStack::ScopedPopper scoped_die_logger(g_die_stack);
#endif

    AccessType accessibility = eAccessNone;
    if (die != NULL)
    {
        if (log)
        {
            const DWARFDebugInfoEntry *context_die;
            clang::DeclContext *context = GetClangDeclContextContainingDIE (dwarf, dwarf_cu, die, &context_die);

            dwarf->GetObjectFile()->GetModule()->LogMessage (log, "SymbolFileDWARF::ParseType (die = 0x%8.8x, decl_ctx = %p (die 0x%8.8x)) %s name = '%s')",
                                                             die->GetOffset(),
                                                             static_cast<void*>(context),
                                                             context_die->GetOffset(),
                                                             DW_TAG_value_to_name(die->Tag()),
                                                             die->GetName(dwarf, dwarf_cu));

#if defined(LLDB_CONFIGURATION_DEBUG) || defined(LLDB_CONFIGURATION_RELEASE)
            scoped_die_logger.Push (dwarf_cu, die);
            g_die_stack.LogDIEs(log, dwarf);
#endif
        }
        //
        //        Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
        //        if (log && dwarf_cu)
        //        {
        //            StreamString s;
        //            die->DumpLocation (this, dwarf_cu, s);
        //            dwarf->GetObjectFile()->GetModule()->LogMessage (log, "SymbolFileDwarf::%s %s", __FUNCTION__, s.GetData());
        //
        //        }

        Type *type_ptr = dwarf->m_die_to_type.lookup (die);
        TypeList* type_list = dwarf->GetTypeList();
        if (type_ptr == NULL)
        {
            if (type_is_new_ptr)
                *type_is_new_ptr = true;

            const dw_tag_t tag = die->Tag();

            bool is_forward_declaration = false;
            DWARFDebugInfoEntry::Attributes attributes;
            const char *type_name_cstr = NULL;
            ConstString type_name_const_str;
            Type::ResolveState resolve_state = Type::eResolveStateUnresolved;
            uint64_t byte_size = 0;
            Declaration decl;

            Type::EncodingDataType encoding_data_type = Type::eEncodingIsUID;
            CompilerType clang_type;
            DWARFFormValue form_value;

            dw_attr_t attr;

            switch (tag)
            {
                case DW_TAG_base_type:
                case DW_TAG_pointer_type:
                case DW_TAG_reference_type:
                case DW_TAG_rvalue_reference_type:
                case DW_TAG_typedef:
                case DW_TAG_const_type:
                case DW_TAG_restrict_type:
                case DW_TAG_volatile_type:
                case DW_TAG_unspecified_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, NULL, attributes);
                    uint32_t encoding = 0;
                    lldb::user_id_t encoding_uid = LLDB_INVALID_UID;

                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                    case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                    case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                    case DW_AT_name:

                                        type_name_cstr = form_value.AsCString(&dwarf->get_debug_str_data());
                                        // Work around a bug in llvm-gcc where they give a name to a reference type which doesn't
                                        // include the "&"...
                                        if (tag == DW_TAG_reference_type)
                                        {
                                            if (strchr (type_name_cstr, '&') == NULL)
                                                type_name_cstr = NULL;
                                        }
                                        if (type_name_cstr)
                                            type_name_const_str.SetCString(type_name_cstr);
                                        break;
                                    case DW_AT_byte_size:   byte_size = form_value.Unsigned(); break;
                                    case DW_AT_encoding:    encoding = form_value.Unsigned(); break;
                                    case DW_AT_type:        encoding_uid = form_value.Reference(); break;
                                    default:
                                    case DW_AT_sibling:
                                        break;
                                }
                            }
                        }
                    }

                    DEBUG_PRINTF ("0x%8.8" PRIx64 ": %s (\"%s\") type => 0x%8.8lx\n", dwarf->MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr, encoding_uid);

                    switch (tag)
                    {
                        default:
                            break;

                        case DW_TAG_unspecified_type:
                            if (strcmp(type_name_cstr, "nullptr_t") == 0 ||
                                strcmp(type_name_cstr, "decltype(nullptr)") == 0 )
                            {
                                resolve_state = Type::eResolveStateFull;
                                clang_type = GetBasicType(eBasicTypeNullPtr);
                                break;
                            }
                            // Fall through to base type below in case we can handle the type there...

                        case DW_TAG_base_type:
                            resolve_state = Type::eResolveStateFull;
                            clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize (type_name_cstr,
                                                                                   encoding,
                                                                                   byte_size * 8);
                            break;

                        case DW_TAG_pointer_type:           encoding_data_type = Type::eEncodingIsPointerUID;           break;
                        case DW_TAG_reference_type:         encoding_data_type = Type::eEncodingIsLValueReferenceUID;   break;
                        case DW_TAG_rvalue_reference_type:  encoding_data_type = Type::eEncodingIsRValueReferenceUID;   break;
                        case DW_TAG_typedef:                encoding_data_type = Type::eEncodingIsTypedefUID;           break;
                        case DW_TAG_const_type:             encoding_data_type = Type::eEncodingIsConstUID;             break;
                        case DW_TAG_restrict_type:          encoding_data_type = Type::eEncodingIsRestrictUID;          break;
                        case DW_TAG_volatile_type:          encoding_data_type = Type::eEncodingIsVolatileUID;          break;
                    }

                    if (!clang_type && (encoding_data_type == Type::eEncodingIsPointerUID || encoding_data_type == Type::eEncodingIsTypedefUID) && sc.comp_unit != NULL)
                    {
                        bool translation_unit_is_objc = (sc.comp_unit->GetLanguage() == eLanguageTypeObjC || sc.comp_unit->GetLanguage() == eLanguageTypeObjC_plus_plus);

                        if (translation_unit_is_objc)
                        {
                            if (type_name_cstr != NULL)
                            {
                                static ConstString g_objc_type_name_id("id");
                                static ConstString g_objc_type_name_Class("Class");
                                static ConstString g_objc_type_name_selector("SEL");

                                if (type_name_const_str == g_objc_type_name_id)
                                {
                                    if (log)
                                        dwarf->GetObjectFile()->GetModule()->LogMessage (log, "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is Objective C 'id' built-in type.",
                                                                                  die->GetOffset(),
                                                                                  DW_TAG_value_to_name(die->Tag()),
                                                                                  die->GetName(dwarf, dwarf_cu));
                                    clang_type = GetBasicType(eBasicTypeObjCID);
                                    encoding_data_type = Type::eEncodingIsUID;
                                    encoding_uid = LLDB_INVALID_UID;
                                    resolve_state = Type::eResolveStateFull;

                                }
                                else if (type_name_const_str == g_objc_type_name_Class)
                                {
                                    if (log)
                                        dwarf->GetObjectFile()->GetModule()->LogMessage (log, "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is Objective C 'Class' built-in type.",
                                                                                  die->GetOffset(),
                                                                                  DW_TAG_value_to_name(die->Tag()),
                                                                                  die->GetName(dwarf, dwarf_cu));
                                    clang_type = GetBasicType(eBasicTypeObjCClass);
                                    encoding_data_type = Type::eEncodingIsUID;
                                    encoding_uid = LLDB_INVALID_UID;
                                    resolve_state = Type::eResolveStateFull;
                                }
                                else if (type_name_const_str == g_objc_type_name_selector)
                                {
                                    if (log)
                                        dwarf->GetObjectFile()->GetModule()->LogMessage (log, "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is Objective C 'selector' built-in type.",
                                                                                  die->GetOffset(),
                                                                                  DW_TAG_value_to_name(die->Tag()),
                                                                                  die->GetName(dwarf, dwarf_cu));
                                    clang_type = GetBasicType(eBasicTypeObjCSel);
                                    encoding_data_type = Type::eEncodingIsUID;
                                    encoding_uid = LLDB_INVALID_UID;
                                    resolve_state = Type::eResolveStateFull;
                                }
                            }
                            else if (encoding_data_type == Type::eEncodingIsPointerUID && encoding_uid != LLDB_INVALID_UID)
                            {
                                // Clang sometimes erroneously emits id as objc_object*.  In that case we fix up the type to "id".

                                DWARFDebugInfoEntry* encoding_die = dwarf_cu->GetDIEPtr(encoding_uid);

                                if (encoding_die && encoding_die->Tag() == DW_TAG_structure_type)
                                {
                                    if (const char *struct_name = encoding_die->GetAttributeValueAsString(dwarf, dwarf_cu, DW_AT_name, NULL))
                                    {
                                        if (!strcmp(struct_name, "objc_object"))
                                        {
                                            if (log)
                                                dwarf->GetObjectFile()->GetModule()->LogMessage (log, "SymbolFileDWARF::ParseType (die = 0x%8.8x) %s '%s' is 'objc_object*', which we overrode to 'id'.",
                                                                                          die->GetOffset(),
                                                                                          DW_TAG_value_to_name(die->Tag()),
                                                                                          die->GetName(dwarf, dwarf_cu));
                                            clang_type = GetBasicType(eBasicTypeObjCID);
                                            encoding_data_type = Type::eEncodingIsUID;
                                            encoding_uid = LLDB_INVALID_UID;
                                            resolve_state = Type::eResolveStateFull;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    type_sp.reset( new Type (dwarf->MakeUserID(die->GetOffset()),
                                             dwarf,
                                             type_name_const_str,
                                             byte_size,
                                             NULL,
                                             encoding_uid,
                                             encoding_data_type,
                                             &decl,
                                             clang_type,
                                             resolve_state));

                    dwarf->m_die_to_type[die] = type_sp.get();

                    //                  Type* encoding_type = GetUniquedTypeForDIEOffset(encoding_uid, type_sp, NULL, 0, 0, false);
                    //                  if (encoding_type != NULL)
                    //                  {
                    //                      if (encoding_type != DIE_IS_BEING_PARSED)
                    //                          type_sp->SetEncodingType(encoding_type);
                    //                      else
                    //                          m_indirect_fixups.push_back(type_sp.get());
                    //                  }
                }
                    break;

                case DW_TAG_structure_type:
                case DW_TAG_union_type:
                case DW_TAG_class_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die] = DIE_IS_BEING_PARSED;
                    bool byte_size_valid = false;

                    LanguageType class_language = eLanguageTypeUnknown;
                    bool is_complete_objc_class = false;
                    //bool struct_is_class = false;
                    const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, NULL, attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_decl_file:
                                        if (dwarf_cu->DW_AT_decl_file_attributes_are_invalid())
                                        {
                                            // llvm-gcc outputs invalid DW_AT_decl_file attributes that always
                                            // point to the compile unit file, so we clear this invalid value
                                            // so that we can still unique types efficiently.
                                            decl.SetFile(FileSpec ("<invalid>", false));
                                        }
                                        else
                                            decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned()));
                                        break;

                                    case DW_AT_decl_line:
                                        decl.SetLine(form_value.Unsigned());
                                        break;

                                    case DW_AT_decl_column:
                                        decl.SetColumn(form_value.Unsigned());
                                        break;

                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString(&dwarf->get_debug_str_data());
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;

                                    case DW_AT_byte_size:
                                        byte_size = form_value.Unsigned();
                                        byte_size_valid = true;
                                        break;

                                    case DW_AT_accessibility:
                                        accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned());
                                        break;

                                    case DW_AT_declaration:
                                        is_forward_declaration = form_value.Boolean();
                                        break;

                                    case DW_AT_APPLE_runtime_class:
                                        class_language = (LanguageType)form_value.Signed();
                                        break;

                                    case DW_AT_APPLE_objc_complete_type:
                                        is_complete_objc_class = form_value.Signed();
                                        break;

                                    case DW_AT_allocated:
                                    case DW_AT_associated:
                                    case DW_AT_data_location:
                                    case DW_AT_description:
                                    case DW_AT_start_scope:
                                    case DW_AT_visibility:
                                    default:
                                    case DW_AT_sibling:
                                        break;
                                }
                            }
                        }
                    }

                    // UniqueDWARFASTType is large, so don't create a local variables on the
                    // stack, put it on the heap. This function is often called recursively
                    // and clang isn't good and sharing the stack space for variables in different blocks.
                    std::unique_ptr<UniqueDWARFASTType> unique_ast_entry_ap(new UniqueDWARFASTType());

                    // Only try and unique the type if it has a name.
                    if (type_name_const_str &&
                        dwarf->GetUniqueDWARFASTTypeMap().Find (type_name_const_str,
                                                                dwarf,
                                                                dwarf_cu,
                                                                die,
                                                                decl,
                                                                byte_size_valid ? byte_size : -1,
                                                                *unique_ast_entry_ap))
                    {
                        // We have already parsed this type or from another
                        // compile unit. GCC loves to use the "one definition
                        // rule" which can result in multiple definitions
                        // of the same class over and over in each compile
                        // unit.
                        type_sp = unique_ast_entry_ap->m_type_sp;
                        if (type_sp)
                        {
                            dwarf->m_die_to_type[die] = type_sp.get();
                            return type_sp;
                        }
                    }

                    DEBUG_PRINTF ("0x%8.8" PRIx64 ": %s (\"%s\")\n", dwarf->MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                    int tag_decl_kind = -1;
                    AccessType default_accessibility = eAccessNone;
                    if (tag == DW_TAG_structure_type)
                    {
                        tag_decl_kind = clang::TTK_Struct;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_union_type)
                    {
                        tag_decl_kind = clang::TTK_Union;
                        default_accessibility = eAccessPublic;
                    }
                    else if (tag == DW_TAG_class_type)
                    {
                        tag_decl_kind = clang::TTK_Class;
                        default_accessibility = eAccessPrivate;
                    }

                    if (byte_size_valid && byte_size == 0 && type_name_cstr &&
                        die->HasChildren() == false &&
                        sc.comp_unit->GetLanguage() == eLanguageTypeObjC)
                    {
                        // Work around an issue with clang at the moment where
                        // forward declarations for objective C classes are emitted
                        // as:
                        //  DW_TAG_structure_type [2]
                        //  DW_AT_name( "ForwardObjcClass" )
                        //  DW_AT_byte_size( 0x00 )
                        //  DW_AT_decl_file( "..." )
                        //  DW_AT_decl_line( 1 )
                        //
                        // Note that there is no DW_AT_declaration and there are
                        // no children, and the byte size is zero.
                        is_forward_declaration = true;
                    }

                    if (class_language == eLanguageTypeObjC ||
                        class_language == eLanguageTypeObjC_plus_plus)
                    {
                        if (!is_complete_objc_class && dwarf->Supports_DW_AT_APPLE_objc_complete_type(dwarf_cu))
                        {
                            // We have a valid eSymbolTypeObjCClass class symbol whose
                            // name matches the current objective C class that we
                            // are trying to find and this DIE isn't the complete
                            // definition (we checked is_complete_objc_class above and
                            // know it is false), so the real definition is in here somewhere
                            type_sp = dwarf->FindCompleteObjCDefinitionTypeForDIE (die, type_name_const_str, true);

                            if (!type_sp)
                            {
                                SymbolFileDWARFDebugMap *debug_map_symfile = dwarf->GetDebugMapSymfile();
                                if (debug_map_symfile)
                                {
                                    // We weren't able to find a full declaration in
                                    // this DWARF, see if we have a declaration anywhere
                                    // else...
                                    type_sp = debug_map_symfile->FindCompleteObjCDefinitionTypeForDIE (die, type_name_const_str, true);
                                }
                            }

                            if (type_sp)
                            {
                                if (log)
                                {
                                    dwarf->GetObjectFile()->GetModule()->LogMessage (log,
                                                                              "SymbolFileDWARF(%p) - 0x%8.8x: %s type \"%s\" is an incomplete objc type, complete type is 0x%8.8" PRIx64,
                                                                              static_cast<void*>(this),
                                                                              die->GetOffset(),
                                                                              DW_TAG_value_to_name(tag),
                                                                              type_name_cstr,
                                                                              type_sp->GetID());
                                }

                                // We found a real definition for this type elsewhere
                                // so lets use it and cache the fact that we found
                                // a complete type for this die
                                dwarf->m_die_to_type[die] = type_sp.get();
                                return type_sp;
                            }
                        }
                    }


                    if (is_forward_declaration)
                    {
                        // We have a forward declaration to a type and we need
                        // to try and find a full declaration. We look in the
                        // current type index just in case we have a forward
                        // declaration followed by an actual declarations in the
                        // DWARF. If this fails, we need to look elsewhere...
                        if (log)
                        {
                            dwarf->GetObjectFile()->GetModule()->LogMessage (log,
                                                                      "SymbolFileDWARF(%p) - 0x%8.8x: %s type \"%s\" is a forward declaration, trying to find complete type",
                                                                      static_cast<void*>(this),
                                                                      die->GetOffset(),
                                                                      DW_TAG_value_to_name(tag),
                                                                      type_name_cstr);
                        }

                        DWARFDeclContext die_decl_ctx;
                        die->GetDWARFDeclContext(dwarf, dwarf_cu, die_decl_ctx);

                        //type_sp = FindDefinitionTypeForDIE (dwarf_cu, die, type_name_const_str);
                        type_sp = dwarf->FindDefinitionTypeForDWARFDeclContext (die_decl_ctx);

                        if (!type_sp)
                        {
                            SymbolFileDWARFDebugMap *debug_map_symfile = dwarf->GetDebugMapSymfile();
                            if (debug_map_symfile)
                            {
                                // We weren't able to find a full declaration in
                                // this DWARF, see if we have a declaration anywhere
                                // else...
                                type_sp = debug_map_symfile->FindDefinitionTypeForDWARFDeclContext (die_decl_ctx);
                            }
                        }

                        if (type_sp)
                        {
                            if (log)
                            {
                                dwarf->GetObjectFile()->GetModule()->LogMessage (log,
                                                                          "SymbolFileDWARF(%p) - 0x%8.8x: %s type \"%s\" is a forward declaration, complete type is 0x%8.8" PRIx64,
                                                                          static_cast<void*>(this),
                                                                          die->GetOffset(),
                                                                          DW_TAG_value_to_name(tag),
                                                                          type_name_cstr,
                                                                          type_sp->GetID());
                            }

                            // We found a real definition for this type elsewhere
                            // so lets use it and cache the fact that we found
                            // a complete type for this die
                            dwarf->m_die_to_type[die] = type_sp.get();
                            return type_sp;
                        }
                    }
                    assert (tag_decl_kind != -1);
                    bool clang_type_was_created = false;
                    clang_type.SetClangType(this, dwarf->m_forward_decl_die_to_clang_type.lookup (die));
                    if (!clang_type)
                    {
                        const DWARFDebugInfoEntry *decl_ctx_die;

                        clang::DeclContext *decl_ctx = GetClangDeclContextContainingDIE (dwarf, dwarf_cu, die, &decl_ctx_die);
                        if (accessibility == eAccessNone && decl_ctx)
                        {
                            // Check the decl context that contains this class/struct/union.
                            // If it is a class we must give it an accessibility.
                            const clang::Decl::Kind containing_decl_kind = decl_ctx->getDeclKind();
                            if (DeclKindIsCXXClass (containing_decl_kind))
                                accessibility = default_accessibility;
                        }

                        ClangASTMetadata metadata;
                        metadata.SetUserID(dwarf->MakeUserID(die->GetOffset()));
                        metadata.SetIsDynamicCXXType(dwarf->ClassOrStructIsVirtual (dwarf_cu, die));

                        if (type_name_cstr && strchr (type_name_cstr, '<'))
                        {
                            ClangASTContext::TemplateParameterInfos template_param_infos;
                            if (ParseTemplateParameterInfos (dwarf, dwarf_cu, die, template_param_infos))
                            {
                                clang::ClassTemplateDecl *class_template_decl = ParseClassTemplateDecl (dwarf,
                                                                                                        decl_ctx,
                                                                                                        accessibility,
                                                                                                        type_name_cstr,
                                                                                                        tag_decl_kind,
                                                                                                        template_param_infos);

                                clang::ClassTemplateSpecializationDecl *class_specialization_decl = CreateClassTemplateSpecializationDecl (decl_ctx,
                                                                                                                                           class_template_decl,
                                                                                                                                           tag_decl_kind,
                                                                                                                                           template_param_infos);
                                clang_type = CreateClassTemplateSpecializationType (class_specialization_decl);
                                clang_type_was_created = true;

                                SetMetadata (class_template_decl, metadata);
                                SetMetadata (class_specialization_decl, metadata);
                            }
                        }

                        if (!clang_type_was_created)
                        {
                            clang_type_was_created = true;
                            clang_type = CreateRecordType (decl_ctx,
                                                           accessibility,
                                                           type_name_cstr,
                                                           tag_decl_kind,
                                                           class_language,
                                                           &metadata);
                        }
                    }

                    // Store a forward declaration to this class type in case any
                    // parameters in any class methods need it for the clang
                    // types for function prototypes.
                    dwarf->LinkDeclContextToDIE(GetDeclContextForType(clang_type), die);
                    type_sp.reset (new Type (dwarf->MakeUserID(die->GetOffset()),
                                             dwarf,
                                             type_name_const_str,
                                             byte_size,
                                             NULL,
                                             LLDB_INVALID_UID,
                                             Type::eEncodingIsUID,
                                             &decl,
                                             clang_type,
                                             Type::eResolveStateForward));

                    type_sp->SetIsCompleteObjCClass(is_complete_objc_class);


                    // Add our type to the unique type map so we don't
                    // end up creating many copies of the same type over
                    // and over in the ASTContext for our module
                    unique_ast_entry_ap->m_type_sp = type_sp;
                    unique_ast_entry_ap->m_symfile = dwarf;
                    unique_ast_entry_ap->m_cu = dwarf_cu;
                    unique_ast_entry_ap->m_die = die;
                    unique_ast_entry_ap->m_declaration = decl;
                    unique_ast_entry_ap->m_byte_size = byte_size;
                    dwarf->GetUniqueDWARFASTTypeMap().Insert (type_name_const_str,
                                                              *unique_ast_entry_ap);

                    if (is_forward_declaration && die->HasChildren())
                    {
                        // Check to see if the DIE actually has a definition, some version of GCC will
                        // emit DIEs with DW_AT_declaration set to true, but yet still have subprogram,
                        // members, or inheritance, so we can't trust it
                        const DWARFDebugInfoEntry *child_die = die->GetFirstChild();
                        while (child_die)
                        {
                            switch (child_die->Tag())
                            {
                                case DW_TAG_inheritance:
                                case DW_TAG_subprogram:
                                case DW_TAG_member:
                                case DW_TAG_APPLE_property:
                                case DW_TAG_class_type:
                                case DW_TAG_structure_type:
                                case DW_TAG_enumeration_type:
                                case DW_TAG_typedef:
                                case DW_TAG_union_type:
                                    child_die = NULL;
                                    is_forward_declaration = false;
                                    break;
                                default:
                                    child_die = child_die->GetSibling();
                                    break;
                            }
                        }
                    }

                    if (!is_forward_declaration)
                    {
                        // Always start the definition for a class type so that
                        // if the class has child classes or types that require
                        // the class to be created for use as their decl contexts
                        // the class will be ready to accept these child definitions.
                        if (die->HasChildren() == false)
                        {
                            // No children for this struct/union/class, lets finish it
                            ClangASTContext::StartTagDeclarationDefinition (clang_type);
                            ClangASTContext::CompleteTagDeclarationDefinition (clang_type);

                            if (tag == DW_TAG_structure_type) // this only applies in C
                            {
                                clang::RecordDecl *record_decl = ClangASTContext::GetAsRecordDecl(clang_type);

                                if (record_decl)
                                    m_record_decl_to_layout_map.insert(std::make_pair(record_decl, LayoutInfo()));
                            }
                        }
                        else if (clang_type_was_created)
                        {
                            // Start the definition if the class is not objective C since
                            // the underlying decls respond to isCompleteDefinition(). Objective
                            // C decls don't respond to isCompleteDefinition() so we can't
                            // start the declaration definition right away. For C++ class/union/structs
                            // we want to start the definition in case the class is needed as the
                            // declaration context for a contained class or type without the need
                            // to complete that type..

                            if (class_language != eLanguageTypeObjC &&
                                class_language != eLanguageTypeObjC_plus_plus)
                                ClangASTContext::StartTagDeclarationDefinition (clang_type);

                            // Leave this as a forward declaration until we need
                            // to know the details of the type. lldb_private::Type
                            // will automatically call the SymbolFile virtual function
                            // "SymbolFileDWARF::ResolveClangOpaqueTypeDefinition(Type *)"
                            // When the definition needs to be defined.
                            dwarf->m_forward_decl_die_to_clang_type[die] = clang_type.GetOpaqueQualType();
                            dwarf->m_forward_decl_clang_type_to_die[ClangASTContext::RemoveFastQualifiers(clang_type).GetOpaqueQualType()] = die;
                            SetHasExternalStorage (clang_type.GetOpaqueQualType(), true);
                        }
                    }

                }
                    break;

                case DW_TAG_enumeration_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    lldb::user_id_t encoding_uid = DW_INVALID_OFFSET;

                    const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, NULL, attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;

                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_decl_file:       decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                    case DW_AT_decl_line:       decl.SetLine(form_value.Unsigned()); break;
                                    case DW_AT_decl_column:     decl.SetColumn(form_value.Unsigned()); break;
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString(&dwarf->get_debug_str_data());
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;
                                    case DW_AT_type:            encoding_uid = form_value.Reference(); break;
                                    case DW_AT_byte_size:       byte_size = form_value.Unsigned(); break;
                                    case DW_AT_accessibility:   break; //accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                                    case DW_AT_declaration:     break; //is_forward_declaration = form_value.Boolean(); break;
                                    case DW_AT_allocated:
                                    case DW_AT_associated:
                                    case DW_AT_bit_stride:
                                    case DW_AT_byte_stride:
                                    case DW_AT_data_location:
                                    case DW_AT_description:
                                    case DW_AT_start_scope:
                                    case DW_AT_visibility:
                                    case DW_AT_specification:
                                    case DW_AT_abstract_origin:
                                    case DW_AT_sibling:
                                        break;
                                }
                            }
                        }

                        DEBUG_PRINTF ("0x%8.8" PRIx64 ": %s (\"%s\")\n", dwarf->MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                        CompilerType enumerator_clang_type;
                        clang_type.SetClangType (this, dwarf->m_forward_decl_die_to_clang_type.lookup (die));
                        if (!clang_type)
                        {
                            if (encoding_uid != DW_INVALID_OFFSET)
                            {
                                Type *enumerator_type = dwarf->ResolveTypeUID(encoding_uid);
                                if (enumerator_type)
                                    enumerator_clang_type = enumerator_type->GetClangFullType();
                            }

                            if (!enumerator_clang_type)
                                enumerator_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize (NULL,
                                                                                                  DW_ATE_signed,
                                                                                                  byte_size * 8);

                            clang_type = CreateEnumerationType (type_name_cstr,
                                                                GetClangDeclContextContainingDIE (dwarf, dwarf_cu, die, NULL),
                                                                decl,
                                                                enumerator_clang_type);
                        }
                        else
                        {
                            enumerator_clang_type = GetEnumerationIntegerType (clang_type.GetOpaqueQualType());
                        }

                        dwarf->LinkDeclContextToDIE(ClangASTContext::GetDeclContextForType(clang_type), die);

                        type_sp.reset( new Type (dwarf->MakeUserID(die->GetOffset()),
                                                 dwarf,
                                                 type_name_const_str,
                                                 byte_size,
                                                 NULL,
                                                 encoding_uid,
                                                 Type::eEncodingIsUID,
                                                 &decl,
                                                 clang_type,
                                                 Type::eResolveStateForward));

                        ClangASTContext::StartTagDeclarationDefinition (clang_type);
                        if (die->HasChildren())
                        {
                            SymbolContext cu_sc(dwarf->GetCompUnitForDWARFCompUnit(dwarf_cu));
                            bool is_signed = false;
                            enumerator_clang_type.IsIntegerType(is_signed);
                            ParseChildEnumerators(cu_sc, clang_type, is_signed, type_sp->GetByteSize(), dwarf, dwarf_cu, die);
                        }
                        ClangASTContext::CompleteTagDeclarationDefinition (clang_type);
                    }
                }
                    break;

                case DW_TAG_inlined_subroutine:
                case DW_TAG_subprogram:
                case DW_TAG_subroutine_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    //const char *mangled = NULL;
                    dw_offset_t type_die_offset = DW_INVALID_OFFSET;
                    bool is_variadic = false;
                    bool is_inline = false;
                    bool is_static = false;
                    bool is_virtual = false;
                    bool is_explicit = false;
                    bool is_artificial = false;
                    dw_offset_t specification_die_offset = DW_INVALID_OFFSET;
                    dw_offset_t abstract_origin_die_offset = DW_INVALID_OFFSET;
                    dw_offset_t object_pointer_die_offset = DW_INVALID_OFFSET;

                    unsigned type_quals = 0;
                    clang::StorageClass storage = clang::SC_None;//, Extern, Static, PrivateExtern


                    const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, NULL, attributes);
                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                    case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                    case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString(&dwarf->get_debug_str_data());
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;

                                    case DW_AT_linkage_name:
                                    case DW_AT_MIPS_linkage_name:   break; // mangled = form_value.AsCString(&dwarf->get_debug_str_data()); break;
                                    case DW_AT_type:                type_die_offset = form_value.Reference(); break;
                                    case DW_AT_accessibility:       accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                                    case DW_AT_declaration:         break; // is_forward_declaration = form_value.Boolean(); break;
                                    case DW_AT_inline:              is_inline = form_value.Boolean(); break;
                                    case DW_AT_virtuality:          is_virtual = form_value.Boolean();  break;
                                    case DW_AT_explicit:            is_explicit = form_value.Boolean();  break;
                                    case DW_AT_artificial:          is_artificial = form_value.Boolean();  break;


                                    case DW_AT_external:
                                        if (form_value.Unsigned())
                                        {
                                            if (storage == clang::SC_None)
                                                storage = clang::SC_Extern;
                                            else
                                                storage = clang::SC_PrivateExtern;
                                        }
                                        break;

                                    case DW_AT_specification:
                                        specification_die_offset = form_value.Reference();
                                        break;

                                    case DW_AT_abstract_origin:
                                        abstract_origin_die_offset = form_value.Reference();
                                        break;

                                    case DW_AT_object_pointer:
                                        object_pointer_die_offset = form_value.Reference();
                                        break;

                                    case DW_AT_allocated:
                                    case DW_AT_associated:
                                    case DW_AT_address_class:
                                    case DW_AT_calling_convention:
                                    case DW_AT_data_location:
                                    case DW_AT_elemental:
                                    case DW_AT_entry_pc:
                                    case DW_AT_frame_base:
                                    case DW_AT_high_pc:
                                    case DW_AT_low_pc:
                                    case DW_AT_prototyped:
                                    case DW_AT_pure:
                                    case DW_AT_ranges:
                                    case DW_AT_recursive:
                                    case DW_AT_return_addr:
                                    case DW_AT_segment:
                                    case DW_AT_start_scope:
                                    case DW_AT_static_link:
                                    case DW_AT_trampoline:
                                    case DW_AT_visibility:
                                    case DW_AT_vtable_elem_location:
                                    case DW_AT_description:
                                    case DW_AT_sibling:
                                        break;
                                }
                            }
                        }
                    }

                    std::string object_pointer_name;
                    if (object_pointer_die_offset != DW_INVALID_OFFSET)
                    {
                        // Get the name from the object pointer die
                        StreamString s;
                        if (DWARFDebugInfoEntry::GetName (dwarf, dwarf_cu, object_pointer_die_offset, s))
                        {
                            object_pointer_name.assign(s.GetData());
                        }
                    }

                    DEBUG_PRINTF ("0x%8.8" PRIx64 ": %s (\"%s\")\n", dwarf->MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                    CompilerType return_clang_type;
                    Type *func_type = NULL;

                    if (type_die_offset != DW_INVALID_OFFSET)
                        func_type = dwarf->ResolveTypeUID(type_die_offset);

                    if (func_type)
                        return_clang_type = func_type->GetClangForwardType();
                    else
                        return_clang_type = GetBasicType(eBasicTypeVoid);


                    std::vector<CompilerType> function_param_types;
                    std::vector<clang::ParmVarDecl*> function_param_decls;

                    // Parse the function children for the parameters

                    const DWARFDebugInfoEntry *decl_ctx_die = NULL;
                    clang::DeclContext *containing_decl_ctx = GetClangDeclContextContainingDIE (dwarf, dwarf_cu, die, &decl_ctx_die);
                    const clang::Decl::Kind containing_decl_kind = containing_decl_ctx->getDeclKind();

                    const bool is_cxx_method = DeclKindIsCXXClass (containing_decl_kind);
                    // Start off static. This will be set to false in ParseChildParameters(...)
                    // if we find a "this" parameters as the first parameter
                    if (is_cxx_method)
                        is_static = true;

                    if (die->HasChildren())
                    {
                        bool skip_artificial = true;
                        ParseChildParameters (sc,
                                              containing_decl_ctx,
                                              dwarf,
                                              dwarf_cu,
                                              die,
                                              skip_artificial,
                                              is_static,
                                              is_variadic,
                                              function_param_types,
                                              function_param_decls,
                                              type_quals);
                    }

                    // clang_type will get the function prototype clang type after this call
                    clang_type = CreateFunctionType (return_clang_type,
                                                     function_param_types.data(),
                                                     function_param_types.size(),
                                                     is_variadic,
                                                     type_quals);

                    bool ignore_containing_context = false;

                    if (type_name_cstr)
                    {
                        bool type_handled = false;
                        if (tag == DW_TAG_subprogram)
                        {
                            ObjCLanguageRuntime::MethodName objc_method (type_name_cstr, true);
                            if (objc_method.IsValid(true))
                            {
                                CompilerType class_opaque_type;
                                ConstString class_name(objc_method.GetClassName());
                                if (class_name)
                                {
                                    TypeSP complete_objc_class_type_sp (dwarf->FindCompleteObjCDefinitionTypeForDIE (NULL, class_name, false));

                                    if (complete_objc_class_type_sp)
                                    {
                                        CompilerType type_clang_forward_type = complete_objc_class_type_sp->GetClangForwardType();
                                        if (ClangASTContext::IsObjCObjectOrInterfaceType(type_clang_forward_type))
                                            class_opaque_type = type_clang_forward_type;
                                    }
                                }

                                if (class_opaque_type)
                                {
                                    // If accessibility isn't set to anything valid, assume public for
                                    // now...
                                    if (accessibility == eAccessNone)
                                        accessibility = eAccessPublic;

                                    clang::ObjCMethodDecl *objc_method_decl = AddMethodToObjCObjectType (class_opaque_type,
                                                                                                         type_name_cstr,
                                                                                                         clang_type,
                                                                                                         accessibility,
                                                                                                         is_artificial);
                                    type_handled = objc_method_decl != NULL;
                                    if (type_handled)
                                    {
                                        dwarf->LinkDeclContextToDIE(ClangASTContext::GetAsDeclContext(objc_method_decl), die);
                                        SetMetadataAsUserID (objc_method_decl, dwarf->MakeUserID(die->GetOffset()));
                                    }
                                    else
                                    {
                                        dwarf->GetObjectFile()->GetModule()->ReportError ("{0x%8.8x}: invalid Objective-C method 0x%4.4x (%s), please file a bug and attach the file at the start of this error message",
                                                                                   die->GetOffset(),
                                                                                   tag,
                                                                                   DW_TAG_value_to_name(tag));
                                    }
                                }
                            }
                            else if (is_cxx_method)
                            {
                                // Look at the parent of this DIE and see if is is
                                // a class or struct and see if this is actually a
                                // C++ method
                                Type *class_type = dwarf->ResolveType (dwarf_cu, decl_ctx_die);
                                if (class_type)
                                {
                                    if (class_type->GetID() != dwarf->MakeUserID(decl_ctx_die->GetOffset()))
                                    {
                                        // We uniqued the parent class of this function to another class
                                        // so we now need to associate all dies under "decl_ctx_die" to
                                        // DIEs in the DIE for "class_type"...
                                        SymbolFileDWARF *class_symfile = NULL;
                                        DWARFCompileUnitSP class_type_cu_sp;
                                        const DWARFDebugInfoEntry *class_type_die = NULL;

                                        SymbolFileDWARFDebugMap *debug_map_symfile = dwarf->GetDebugMapSymfile();
                                        if (debug_map_symfile)
                                        {
                                            class_symfile = debug_map_symfile->GetSymbolFileByOSOIndex(SymbolFileDWARFDebugMap::GetOSOIndexFromUserID(class_type->GetID()));
                                            class_type_die = class_symfile->DebugInfo()->GetDIEPtr(class_type->GetID(), &class_type_cu_sp);
                                        }
                                        else
                                        {
                                            class_symfile = dwarf;
                                            class_type_die = dwarf->DebugInfo()->GetDIEPtr(class_type->GetID(), &class_type_cu_sp);
                                        }
                                        if (class_type_die)
                                        {
                                            DWARFDIECollection failures;

                                            CopyUniqueClassMethodTypes (dwarf,
                                                                        class_symfile,
                                                                        class_type,
                                                                        class_type_cu_sp.get(),
                                                                        class_type_die,
                                                                        dwarf_cu,
                                                                        decl_ctx_die,
                                                                        failures);

                                            // FIXME do something with these failures that's smarter than
                                            // just dropping them on the ground.  Unfortunately classes don't
                                            // like having stuff added to them after their definitions are
                                            // complete...

                                            type_ptr = dwarf->m_die_to_type[die];
                                            if (type_ptr && type_ptr != DIE_IS_BEING_PARSED)
                                            {
                                                type_sp = type_ptr->shared_from_this();
                                                break;
                                            }
                                        }
                                    }

                                    if (specification_die_offset != DW_INVALID_OFFSET)
                                    {
                                        // We have a specification which we are going to base our function
                                        // prototype off of, so we need this type to be completed so that the
                                        // m_die_to_decl_ctx for the method in the specification has a valid
                                        // clang decl context.
                                        class_type->GetClangForwardType();
                                        // If we have a specification, then the function type should have been
                                        // made with the specification and not with this die.
                                        DWARFCompileUnitSP spec_cu_sp;
                                        const DWARFDebugInfoEntry* spec_die = dwarf->DebugInfo()->GetDIEPtr(specification_die_offset, &spec_cu_sp);
                                        clang::DeclContext *spec_clang_decl_ctx = GetClangDeclContextForDIE (dwarf, sc, dwarf_cu, spec_die);
                                        if (spec_clang_decl_ctx)
                                        {
                                            dwarf->LinkDeclContextToDIE(spec_clang_decl_ctx, die);
                                        }
                                        else
                                        {
                                            dwarf->GetObjectFile()->GetModule()->ReportWarning ("0x%8.8" PRIx64 ": DW_AT_specification(0x%8.8x) has no decl\n",
                                                                                         dwarf->MakeUserID(die->GetOffset()),
                                                                                         specification_die_offset);
                                        }
                                        type_handled = true;
                                    }
                                    else if (abstract_origin_die_offset != DW_INVALID_OFFSET)
                                    {
                                        // We have a specification which we are going to base our function
                                        // prototype off of, so we need this type to be completed so that the
                                        // m_die_to_decl_ctx for the method in the abstract origin has a valid
                                        // clang decl context.
                                        class_type->GetClangForwardType();

                                        DWARFCompileUnitSP abs_cu_sp;
                                        const DWARFDebugInfoEntry* abs_die = dwarf->DebugInfo()->GetDIEPtr(abstract_origin_die_offset, &abs_cu_sp);
                                        clang::DeclContext *abs_clang_decl_ctx = GetClangDeclContextForDIE (dwarf, sc, dwarf_cu, abs_die);
                                        if (abs_clang_decl_ctx)
                                        {
                                            dwarf->LinkDeclContextToDIE (abs_clang_decl_ctx, die);
                                        }
                                        else
                                        {
                                            dwarf->GetObjectFile()->GetModule()->ReportWarning ("0x%8.8" PRIx64 ": DW_AT_abstract_origin(0x%8.8x) has no decl\n",
                                                                                         dwarf->MakeUserID(die->GetOffset()),
                                                                                         abstract_origin_die_offset);
                                        }
                                        type_handled = true;
                                    }
                                    else
                                    {
                                        CompilerType class_opaque_type = class_type->GetClangForwardType();
                                        if (ClangASTContext::IsCXXClassType(class_opaque_type))
                                        {
                                            if (class_opaque_type.IsBeingDefined ())
                                            {
                                                // Neither GCC 4.2 nor clang++ currently set a valid accessibility
                                                // in the DWARF for C++ methods... Default to public for now...
                                                if (accessibility == eAccessNone)
                                                    accessibility = eAccessPublic;

                                                if (!is_static && !die->HasChildren())
                                                {
                                                    // We have a C++ member function with no children (this pointer!)
                                                    // and clang will get mad if we try and make a function that isn't
                                                    // well formed in the DWARF, so we will just skip it...
                                                    type_handled = true;
                                                }
                                                else
                                                {
                                                    clang::CXXMethodDecl *cxx_method_decl;
                                                    // REMOVE THE CRASH DESCRIPTION BELOW
                                                    Host::SetCrashDescriptionWithFormat ("SymbolFileDWARF::ParseType() is adding a method %s to class %s in DIE 0x%8.8" PRIx64 " from %s",
                                                                                         type_name_cstr,
                                                                                         class_type->GetName().GetCString(),
                                                                                         dwarf->MakeUserID(die->GetOffset()),
                                                                                         dwarf->GetObjectFile()->GetFileSpec().GetPath().c_str());

                                                    const bool is_attr_used = false;

                                                    cxx_method_decl = AddMethodToCXXRecordType (class_opaque_type.GetOpaqueQualType(),
                                                                                                type_name_cstr,
                                                                                                clang_type,
                                                                                                accessibility,
                                                                                                is_virtual,
                                                                                                is_static,
                                                                                                is_inline,
                                                                                                is_explicit,
                                                                                                is_attr_used,
                                                                                                is_artificial);

                                                    type_handled = cxx_method_decl != NULL;

                                                    if (type_handled)
                                                    {
                                                        dwarf->LinkDeclContextToDIE(ClangASTContext::GetAsDeclContext(cxx_method_decl), die);

                                                        Host::SetCrashDescription (NULL);


                                                        ClangASTMetadata metadata;
                                                        metadata.SetUserID(dwarf->MakeUserID(die->GetOffset()));

                                                        if (!object_pointer_name.empty())
                                                        {
                                                            metadata.SetObjectPtrName(object_pointer_name.c_str());
                                                            if (log)
                                                                log->Printf ("Setting object pointer name: %s on method object %p.\n",
                                                                             object_pointer_name.c_str(),
                                                                             static_cast<void*>(cxx_method_decl));
                                                        }
                                                        SetMetadata (cxx_method_decl, metadata);
                                                    }
                                                    else
                                                    {
                                                        ignore_containing_context = true;
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                // We were asked to parse the type for a method in a class, yet the
                                                // class hasn't been asked to complete itself through the
                                                // clang::ExternalASTSource protocol, so we need to just have the
                                                // class complete itself and do things the right way, then our
                                                // DIE should then have an entry in the dwarf->m_die_to_type map. First
                                                // we need to modify the dwarf->m_die_to_type so it doesn't think we are
                                                // trying to parse this DIE anymore...
                                                dwarf->m_die_to_type[die] = NULL;

                                                // Now we get the full type to force our class type to complete itself
                                                // using the clang::ExternalASTSource protocol which will parse all
                                                // base classes and all methods (including the method for this DIE).
                                                class_type->GetClangFullType();

                                                // The type for this DIE should have been filled in the function call above
                                                type_ptr = dwarf->m_die_to_type[die];
                                                if (type_ptr && type_ptr != DIE_IS_BEING_PARSED)
                                                {
                                                    type_sp = type_ptr->shared_from_this();
                                                    break;
                                                }

                                                // FIXME This is fixing some even uglier behavior but we really need to
                                                // uniq the methods of each class as well as the class itself.
                                                // <rdar://problem/11240464>
                                                type_handled = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (!type_handled)
                        {
                            // We just have a function that isn't part of a class
                            clang::FunctionDecl *function_decl = CreateFunctionDeclaration (ignore_containing_context ? GetTranslationUnitDecl() : containing_decl_ctx,
                                                                                            type_name_cstr,
                                                                                            clang_type,
                                                                                            storage,
                                                                                            is_inline);

                            //                            if (template_param_infos.GetSize() > 0)
                            //                            {
                            //                                clang::FunctionTemplateDecl *func_template_decl = CreateFunctionTemplateDecl (containing_decl_ctx,
                            //                                                                                                              function_decl,
                            //                                                                                                              type_name_cstr,
                            //                                                                                                              template_param_infos);
                            //
                            //                                CreateFunctionTemplateSpecializationInfo (function_decl,
                            //                                                                          func_template_decl,
                            //                                                                          template_param_infos);
                            //                            }
                            // Add the decl to our DIE to decl context map
                            assert (function_decl);
                            dwarf->LinkDeclContextToDIE(function_decl, die);
                            if (!function_param_decls.empty())
                                SetFunctionParameters (function_decl,
                                                       &function_param_decls.front(),
                                                       function_param_decls.size());

                            ClangASTMetadata metadata;
                            metadata.SetUserID(dwarf->MakeUserID(die->GetOffset()));

                            if (!object_pointer_name.empty())
                            {
                                metadata.SetObjectPtrName(object_pointer_name.c_str());
                                if (log)
                                    log->Printf ("Setting object pointer name: %s on function object %p.",
                                                 object_pointer_name.c_str(),
                                                 static_cast<void*>(function_decl));
                            }
                            SetMetadata (function_decl, metadata);
                        }
                    }
                    type_sp.reset( new Type (dwarf->MakeUserID(die->GetOffset()),
                                             dwarf,
                                             type_name_const_str,
                                             0,
                                             NULL,
                                             LLDB_INVALID_UID,
                                             Type::eEncodingIsUID,
                                             &decl,
                                             clang_type,
                                             Type::eResolveStateFull));
                    assert(type_sp.get());
                }
                    break;

                case DW_TAG_array_type:
                {
                    // Set a bit that lets us know that we are currently parsing this
                    dwarf->m_die_to_type[die] = DIE_IS_BEING_PARSED;

                    lldb::user_id_t type_die_offset = DW_INVALID_OFFSET;
                    int64_t first_index = 0;
                    uint32_t byte_stride = 0;
                    uint32_t bit_stride = 0;
                    bool is_vector = false;
                    const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, NULL, attributes);

                    if (num_attributes > 0)
                    {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_decl_file:   decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(form_value.Unsigned())); break;
                                    case DW_AT_decl_line:   decl.SetLine(form_value.Unsigned()); break;
                                    case DW_AT_decl_column: decl.SetColumn(form_value.Unsigned()); break;
                                    case DW_AT_name:
                                        type_name_cstr = form_value.AsCString(&dwarf->get_debug_str_data());
                                        type_name_const_str.SetCString(type_name_cstr);
                                        break;

                                    case DW_AT_type:            type_die_offset = form_value.Reference(); break;
                                    case DW_AT_byte_size:       break; // byte_size = form_value.Unsigned(); break;
                                    case DW_AT_byte_stride:     byte_stride = form_value.Unsigned(); break;
                                    case DW_AT_bit_stride:      bit_stride = form_value.Unsigned(); break;
                                    case DW_AT_GNU_vector:      is_vector = form_value.Boolean(); break;
                                    case DW_AT_accessibility:   break; // accessibility = DW_ACCESS_to_AccessType(form_value.Unsigned()); break;
                                    case DW_AT_declaration:     break; // is_forward_declaration = form_value.Boolean(); break;
                                    case DW_AT_allocated:
                                    case DW_AT_associated:
                                    case DW_AT_data_location:
                                    case DW_AT_description:
                                    case DW_AT_ordering:
                                    case DW_AT_start_scope:
                                    case DW_AT_visibility:
                                    case DW_AT_specification:
                                    case DW_AT_abstract_origin:
                                    case DW_AT_sibling:
                                        break;
                                }
                            }
                        }

                        DEBUG_PRINTF ("0x%8.8" PRIx64 ": %s (\"%s\")\n", dwarf->MakeUserID(die->GetOffset()), DW_TAG_value_to_name(tag), type_name_cstr);

                        Type *element_type = dwarf->ResolveTypeUID(type_die_offset);

                        if (element_type)
                        {
                            std::vector<uint64_t> element_orders;
                            ParseChildArrayInfo(sc, dwarf, dwarf_cu, die, first_index, element_orders, byte_stride, bit_stride);
                            if (byte_stride == 0 && bit_stride == 0)
                                byte_stride = element_type->GetByteSize();
                            CompilerType array_element_type = element_type->GetClangForwardType();
                            uint64_t array_element_bit_stride = byte_stride * 8 + bit_stride;
                            if (element_orders.size() > 0)
                            {
                                uint64_t num_elements = 0;
                                std::vector<uint64_t>::const_reverse_iterator pos;
                                std::vector<uint64_t>::const_reverse_iterator end = element_orders.rend();
                                for (pos = element_orders.rbegin(); pos != end; ++pos)
                                {
                                    num_elements = *pos;
                                    clang_type = CreateArrayType (array_element_type,
                                                                  num_elements,
                                                                  is_vector);
                                    array_element_type = clang_type;
                                    array_element_bit_stride = num_elements ?
                                    array_element_bit_stride * num_elements :
                                    array_element_bit_stride;
                                }
                            }
                            else
                            {
                                clang_type = CreateArrayType (array_element_type, 0, is_vector);
                            }
                            ConstString empty_name;
                            type_sp.reset( new Type (dwarf->MakeUserID(die->GetOffset()),
                                                     dwarf,
                                                     empty_name, 
                                                     array_element_bit_stride / 8, 
                                                     NULL, 
                                                     type_die_offset, 
                                                     Type::eEncodingIsUID, 
                                                     &decl, 
                                                     clang_type, 
                                                     Type::eResolveStateFull));
                            type_sp->SetEncodingType (element_type);
                        }
                    }
                }
                    break;
                    
                case DW_TAG_ptr_to_member_type:
                {
                    dw_offset_t type_die_offset = DW_INVALID_OFFSET;
                    dw_offset_t containing_type_die_offset = DW_INVALID_OFFSET;
                    
                    const size_t num_attributes = die->GetAttributes(dwarf, dwarf_cu, NULL, attributes);
                    
                    if (num_attributes > 0) {
                        uint32_t i;
                        for (i=0; i<num_attributes; ++i)
                        {
                            attr = attributes.AttributeAtIndex(i);
                            if (attributes.ExtractFormValueAtIndex(dwarf, i, form_value))
                            {
                                switch (attr)
                                {
                                    case DW_AT_type:
                                        type_die_offset = form_value.Reference(); break;
                                    case DW_AT_containing_type:
                                        containing_type_die_offset = form_value.Reference(); break;
                                }
                            }
                        }
                        
                        Type *pointee_type = dwarf->ResolveTypeUID(type_die_offset);
                        Type *class_type = dwarf->ResolveTypeUID(containing_type_die_offset);
                        
                        CompilerType pointee_clang_type = pointee_type->GetClangForwardType();
                        CompilerType class_clang_type = class_type->GetClangLayoutType();
                        
                        clang_type = ClangASTContext::CreateMemberPointerType(pointee_clang_type, class_clang_type);
                        
                        byte_size = clang_type.GetByteSize(nullptr);
                        
                        type_sp.reset( new Type (dwarf->MakeUserID(die->GetOffset()),
                                                 dwarf,
                                                 type_name_const_str, 
                                                 byte_size, 
                                                 NULL, 
                                                 LLDB_INVALID_UID, 
                                                 Type::eEncodingIsUID, 
                                                 NULL, 
                                                 clang_type, 
                                                 Type::eResolveStateForward));
                    }
                    
                    break;
                }
                default:
                    dwarf->GetObjectFile()->GetModule()->ReportError ("{0x%8.8x}: unhandled type tag 0x%4.4x (%s), please file a bug and attach the file at the start of this error message",
                                                               die->GetOffset(),
                                                               tag,
                                                               DW_TAG_value_to_name(tag));
                    break;
            }
            
            if (type_sp.get())
            {
                const DWARFDebugInfoEntry *sc_parent_die = SymbolFileDWARF::GetParentSymbolContextDIE(die);
                dw_tag_t sc_parent_tag = sc_parent_die ? sc_parent_die->Tag() : 0;
                
                SymbolContextScope * symbol_context_scope = NULL;
                if (sc_parent_tag == DW_TAG_compile_unit)
                {
                    symbol_context_scope = sc.comp_unit;
                }
                else if (sc.function != NULL && sc_parent_die)
                {
                    symbol_context_scope = sc.function->GetBlock(true).FindBlockByID(dwarf->MakeUserID(sc_parent_die->GetOffset()));
                    if (symbol_context_scope == NULL)
                        symbol_context_scope = sc.function;
                }
                
                if (symbol_context_scope != NULL)
                {
                    type_sp->SetSymbolContextScope(symbol_context_scope);
                }
                
                // We are ready to put this type into the uniqued list up at the module level
                type_list->Insert (type_sp);
                
                dwarf->m_die_to_type[die] = type_sp.get();
            }
        }
        else if (type_ptr != DIE_IS_BEING_PARSED)
        {
            type_sp = type_ptr->shared_from_this();
        }
    }
    return type_sp;
}


bool
ClangASTContext::CopyUniqueClassMethodTypes (SymbolFileDWARF *dst_symfile,
                                             SymbolFileDWARF *src_symfile,
                                             lldb_private::Type *class_type,
                                             DWARFCompileUnit* src_cu,
                                             const DWARFDebugInfoEntry *src_class_die,
                                             DWARFCompileUnit* dst_cu,
                                             const DWARFDebugInfoEntry *dst_class_die,
                                             DWARFDIECollection &failures)
{
    if (!class_type || !src_cu || !src_class_die || !dst_cu || !dst_class_die)
        return false;
    if (src_class_die->Tag() != dst_class_die->Tag())
        return false;

    // We need to complete the class type so we can get all of the method types
    // parsed so we can then unique those types to their equivalent counterparts
    // in "dst_cu" and "dst_class_die"
    class_type->GetClangFullType();

    const DWARFDebugInfoEntry *src_die;
    const DWARFDebugInfoEntry *dst_die;
    UniqueCStringMap<const DWARFDebugInfoEntry *> src_name_to_die;
    UniqueCStringMap<const DWARFDebugInfoEntry *> dst_name_to_die;
    UniqueCStringMap<const DWARFDebugInfoEntry *> src_name_to_die_artificial;
    UniqueCStringMap<const DWARFDebugInfoEntry *> dst_name_to_die_artificial;
    for (src_die = src_class_die->GetFirstChild(); src_die != NULL; src_die = src_die->GetSibling())
    {
        if (src_die->Tag() == DW_TAG_subprogram)
        {
            // Make sure this is a declaration and not a concrete instance by looking
            // for DW_AT_declaration set to 1. Sometimes concrete function instances
            // are placed inside the class definitions and shouldn't be included in
            // the list of things are are tracking here.
            if (src_die->GetAttributeValueAsUnsigned(src_symfile, src_cu, DW_AT_declaration, 0) == 1)
            {
                const char *src_name = src_die->GetMangledName (src_symfile, src_cu);
                if (src_name)
                {
                    ConstString src_const_name(src_name);
                    if (src_die->GetAttributeValueAsUnsigned(src_symfile, src_cu, DW_AT_artificial, 0))
                        src_name_to_die_artificial.Append(src_const_name.GetCString(), src_die);
                    else
                        src_name_to_die.Append(src_const_name.GetCString(), src_die);
                }
            }
        }
    }
    for (dst_die = dst_class_die->GetFirstChild(); dst_die != NULL; dst_die = dst_die->GetSibling())
    {
        if (dst_die->Tag() == DW_TAG_subprogram)
        {
            // Make sure this is a declaration and not a concrete instance by looking
            // for DW_AT_declaration set to 1. Sometimes concrete function instances
            // are placed inside the class definitions and shouldn't be included in
            // the list of things are are tracking here.
            if (dst_die->GetAttributeValueAsUnsigned(dst_symfile, dst_cu, DW_AT_declaration, 0) == 1)
            {
                const char *dst_name = dst_die->GetMangledName (dst_symfile, dst_cu);
                if (dst_name)
                {
                    ConstString dst_const_name(dst_name);
                    if (dst_die->GetAttributeValueAsUnsigned(dst_symfile, dst_cu, DW_AT_artificial, 0))
                        dst_name_to_die_artificial.Append(dst_const_name.GetCString(), dst_die);
                    else
                        dst_name_to_die.Append(dst_const_name.GetCString(), dst_die);
                }
            }
        }
    }
    const uint32_t src_size = src_name_to_die.GetSize ();
    const uint32_t dst_size = dst_name_to_die.GetSize ();
    Log *log = nullptr; // (LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO | DWARF_LOG_TYPE_COMPLETION));

    // Is everything kosher so we can go through the members at top speed?
    bool fast_path = true;

    if (src_size != dst_size)
    {
        if (src_size != 0 && dst_size != 0)
        {
            if (log)
                log->Printf("warning: trying to unique class DIE 0x%8.8x to 0x%8.8x, but they didn't have the same size (src=%d, dst=%d)",
                            src_class_die->GetOffset(),
                            dst_class_die->GetOffset(),
                            src_size,
                            dst_size);
        }

        fast_path = false;
    }

    uint32_t idx;

    if (fast_path)
    {
        for (idx = 0; idx < src_size; ++idx)
        {
            src_die = src_name_to_die.GetValueAtIndexUnchecked (idx);
            dst_die = dst_name_to_die.GetValueAtIndexUnchecked (idx);

            if (src_die->Tag() != dst_die->Tag())
            {
                if (log)
                    log->Printf("warning: tried to unique class DIE 0x%8.8x to 0x%8.8x, but 0x%8.8x (%s) tags didn't match 0x%8.8x (%s)",
                                src_class_die->GetOffset(),
                                dst_class_die->GetOffset(),
                                src_die->GetOffset(),
                                DW_TAG_value_to_name(src_die->Tag()),
                                dst_die->GetOffset(),
                                DW_TAG_value_to_name(src_die->Tag()));
                fast_path = false;
            }

            const char *src_name = src_die->GetMangledName (src_symfile, src_cu);
            const char *dst_name = dst_die->GetMangledName (dst_symfile, dst_cu);

            // Make sure the names match
            if (src_name == dst_name || (strcmp (src_name, dst_name) == 0))
                continue;

            if (log)
                log->Printf("warning: tried to unique class DIE 0x%8.8x to 0x%8.8x, but 0x%8.8x (%s) names didn't match 0x%8.8x (%s)",
                            src_class_die->GetOffset(),
                            dst_class_die->GetOffset(),
                            src_die->GetOffset(),
                            src_name,
                            dst_die->GetOffset(),
                            dst_name);

            fast_path = false;
        }
    }

    // Now do the work of linking the DeclContexts and Types.
    if (fast_path)
    {
        // We can do this quickly.  Just run across the tables index-for-index since
        // we know each node has matching names and tags.
        for (idx = 0; idx < src_size; ++idx)
        {
            src_die = src_name_to_die.GetValueAtIndexUnchecked (idx);
            dst_die = dst_name_to_die.GetValueAtIndexUnchecked (idx);

            clang::DeclContext *src_decl_ctx = src_symfile->m_die_to_decl_ctx[src_die];
            if (src_decl_ctx)
            {
                if (log)
                    log->Printf ("uniquing decl context %p from 0x%8.8x for 0x%8.8x",
                                 static_cast<void*>(src_decl_ctx),
                                 src_die->GetOffset(), dst_die->GetOffset());
                dst_symfile->LinkDeclContextToDIE (src_decl_ctx, dst_die);
            }
            else
            {
                if (log)
                    log->Printf ("warning: tried to unique decl context from 0x%8.8x for 0x%8.8x, but none was found",
                                 src_die->GetOffset(), dst_die->GetOffset());
            }

            Type *src_child_type = dst_symfile->m_die_to_type[src_die];
            if (src_child_type)
            {
                if (log)
                    log->Printf ("uniquing type %p (uid=0x%" PRIx64 ") from 0x%8.8x for 0x%8.8x",
                                 static_cast<void*>(src_child_type),
                                 src_child_type->GetID(),
                                 src_die->GetOffset(), dst_die->GetOffset());
                dst_symfile->m_die_to_type[dst_die] = src_child_type;
            }
            else
            {
                if (log)
                    log->Printf ("warning: tried to unique lldb_private::Type from 0x%8.8x for 0x%8.8x, but none was found", src_die->GetOffset(), dst_die->GetOffset());
            }
        }
    }
    else
    {
        // We must do this slowly.  For each member of the destination, look
        // up a member in the source with the same name, check its tag, and
        // unique them if everything matches up.  Report failures.

        if (!src_name_to_die.IsEmpty() && !dst_name_to_die.IsEmpty())
        {
            src_name_to_die.Sort();

            for (idx = 0; idx < dst_size; ++idx)
            {
                const char *dst_name = dst_name_to_die.GetCStringAtIndex(idx);
                dst_die = dst_name_to_die.GetValueAtIndexUnchecked(idx);
                src_die = src_name_to_die.Find(dst_name, NULL);

                if (src_die && (src_die->Tag() == dst_die->Tag()))
                {
                    clang::DeclContext *src_decl_ctx = src_symfile->m_die_to_decl_ctx[src_die];
                    if (src_decl_ctx)
                    {
                        if (log)
                            log->Printf ("uniquing decl context %p from 0x%8.8x for 0x%8.8x",
                                         static_cast<void*>(src_decl_ctx),
                                         src_die->GetOffset(),
                                         dst_die->GetOffset());
                        dst_symfile->LinkDeclContextToDIE (src_decl_ctx, dst_die);
                    }
                    else
                    {
                        if (log)
                            log->Printf ("warning: tried to unique decl context from 0x%8.8x for 0x%8.8x, but none was found", src_die->GetOffset(), dst_die->GetOffset());
                    }

                    Type *src_child_type = dst_symfile->m_die_to_type[src_die];
                    if (src_child_type)
                    {
                        if (log)
                            log->Printf ("uniquing type %p (uid=0x%" PRIx64 ") from 0x%8.8x for 0x%8.8x",
                                         static_cast<void*>(src_child_type),
                                         src_child_type->GetID(),
                                         src_die->GetOffset(),
                                         dst_die->GetOffset());
                        dst_symfile->m_die_to_type[dst_die] = src_child_type;
                    }
                    else
                    {
                        if (log)
                            log->Printf ("warning: tried to unique lldb_private::Type from 0x%8.8x for 0x%8.8x, but none was found", src_die->GetOffset(), dst_die->GetOffset());
                    }
                }
                else
                {
                    if (log)
                        log->Printf ("warning: couldn't find a match for 0x%8.8x", dst_die->GetOffset());

                    failures.Append(dst_die);
                }
            }
        }
    }

    const uint32_t src_size_artificial = src_name_to_die_artificial.GetSize ();
    const uint32_t dst_size_artificial = dst_name_to_die_artificial.GetSize ();

    UniqueCStringMap<const DWARFDebugInfoEntry *> name_to_die_artificial_not_in_src;

    if (src_size_artificial && dst_size_artificial)
    {
        dst_name_to_die_artificial.Sort();

        for (idx = 0; idx < src_size_artificial; ++idx)
        {
            const char *src_name_artificial = src_name_to_die_artificial.GetCStringAtIndex(idx);
            src_die = src_name_to_die_artificial.GetValueAtIndexUnchecked (idx);
            dst_die = dst_name_to_die_artificial.Find(src_name_artificial, NULL);

            if (dst_die)
            {
                // Both classes have the artificial types, link them
                clang::DeclContext *src_decl_ctx = dst_symfile->m_die_to_decl_ctx[src_die];
                if (src_decl_ctx)
                {
                    if (log)
                        log->Printf ("uniquing decl context %p from 0x%8.8x for 0x%8.8x",
                                     static_cast<void*>(src_decl_ctx),
                                     src_die->GetOffset(), dst_die->GetOffset());
                    dst_symfile->LinkDeclContextToDIE (src_decl_ctx, dst_die);
                }
                else
                {
                    if (log)
                        log->Printf ("warning: tried to unique decl context from 0x%8.8x for 0x%8.8x, but none was found", src_die->GetOffset(), dst_die->GetOffset());
                }

                Type *src_child_type = dst_symfile->m_die_to_type[src_die];
                if (src_child_type)
                {
                    if (log)
                        log->Printf ("uniquing type %p (uid=0x%" PRIx64 ") from 0x%8.8x for 0x%8.8x",
                                     static_cast<void*>(src_child_type),
                                     src_child_type->GetID(),
                                     src_die->GetOffset(), dst_die->GetOffset());
                    dst_symfile->m_die_to_type[dst_die] = src_child_type;
                }
                else
                {
                    if (log)
                        log->Printf ("warning: tried to unique lldb_private::Type from 0x%8.8x for 0x%8.8x, but none was found", src_die->GetOffset(), dst_die->GetOffset());
                }
            }
        }
    }

    if (dst_size_artificial)
    {
        for (idx = 0; idx < dst_size_artificial; ++idx)
        {
            const char *dst_name_artificial = dst_name_to_die_artificial.GetCStringAtIndex(idx);
            dst_die = dst_name_to_die_artificial.GetValueAtIndexUnchecked (idx);
            if (log)
                log->Printf ("warning: need to create artificial method for 0x%8.8x for method '%s'", dst_die->GetOffset(), dst_name_artificial);
            
            failures.Append(dst_die);
        }
    }
    
    return (failures.Size() != 0);
}


bool
ClangASTContext::DIEIsInNamespace (const ClangNamespaceDecl *namespace_decl,
                                   SymbolFileDWARF *dwarf,
                                   DWARFCompileUnit *cu,
                                   const DWARFDebugInfoEntry *die)
{
    // No namespace specified, so the answer is
    if (namespace_decl == NULL)
        return true;

    Log *log = nullptr; //(LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS));

    const DWARFDebugInfoEntry *decl_ctx_die = NULL;
    clang::DeclContext *die_clang_decl_ctx = GetClangDeclContextContainingDIE (dwarf, cu, die, &decl_ctx_die);
    if (decl_ctx_die)
    {
        clang::NamespaceDecl *clang_namespace_decl = namespace_decl->GetNamespaceDecl();

        if (clang_namespace_decl)
        {
            if (decl_ctx_die->Tag() != DW_TAG_namespace)
            {
                if (log)
                    dwarf->GetObjectFile()->GetModule()->LogMessage(log, "Found a match, but its parent is not a namespace");
                return false;
            }

            if (clang_namespace_decl == die_clang_decl_ctx)
                return true;
            else
                return false;
        }
        else
        {
            // We have a namespace_decl that was not NULL but it contained
            // a NULL "clang::NamespaceDecl", so this means the global namespace
            // So as long the contained decl context DIE isn't a namespace
            // we should be ok.
            if (decl_ctx_die->Tag() != DW_TAG_namespace)
                return true;
        }
    }

    if (log)
        dwarf->GetObjectFile()->GetModule()->LogMessage(log, "Found a match, but its parent doesn't exist");

    return false;
}

