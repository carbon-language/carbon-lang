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
#define NDEBUG
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/LangStandard.h"
#undef NDEBUG

#include "lldb/Core/dwarf.h"

#include <stdio.h>

using namespace lldb_private;
using namespace llvm;
using namespace clang;

static AccessSpecifier
ConvertAccessTypeToAccessSpecifier (ClangASTContext::AccessType access)
{
    switch (access)
    {
    default:                                break;
    case ClangASTContext::eAccessNone:      return AS_none;
    case ClangASTContext::eAccessPublic:    return AS_public;
    case ClangASTContext::eAccessPrivate:   return AS_private;
    case ClangASTContext::eAccessProtected: return AS_protected;
    }
    return AS_none;
}

static ObjCIvarDecl::AccessControl
ConvertAccessTypeToObjCIvarAccessControl (ClangASTContext::AccessType access)
{
    switch (access)
    {
    default:                                break;
    case ClangASTContext::eAccessNone:      return ObjCIvarDecl::None;
    case ClangASTContext::eAccessPublic:    return ObjCIvarDecl::Public;
    case ClangASTContext::eAccessPrivate:   return ObjCIvarDecl::Private;
    case ClangASTContext::eAccessProtected: return ObjCIvarDecl::Protected;
    case ClangASTContext::eAccessPackage:   return ObjCIvarDecl::Package;
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
                assert(0 && "Invalid input kind!");
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
        Opts.setVisibilityMode(LangOptions::Default);
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


ClangASTContext::ClangASTContext(const char *target_triple) :
    m_target_triple(),
    m_ast_context_ap(),
    m_language_options_ap(),
    m_source_manager_ap(),
    m_diagnostic_ap(),
    m_target_options_ap(),
    m_target_info_ap(),
    m_identifier_table_ap(),
    m_selector_table_ap(),
    m_builtins_ap()
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
    m_ast_context_ap.reset();
}


void
ClangASTContext::Clear()
{
    m_ast_context_ap.reset();
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


ASTContext *
ClangASTContext::getASTContext()
{
    if (m_ast_context_ap.get() == NULL)
    {
        m_ast_context_ap.reset(
            new ASTContext(
                *getLanguageOptions(),
                *getSourceManager(),
                *getTargetInfo(),
                *getIdentifierTable(),
                *getSelectorTable(),
                *getBuiltinContext()));
    }
    return m_ast_context_ap.get();
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

clang::SourceManager *
ClangASTContext::getSourceManager()
{
    if (m_source_manager_ap.get() == NULL)
        m_source_manager_ap.reset(new clang::SourceManager(*getDiagnostic()));
    return m_source_manager_ap.get();
}

Diagnostic *
ClangASTContext::getDiagnostic()
{
    if (m_diagnostic_ap.get() == NULL)
        m_diagnostic_ap.reset(new Diagnostic());
    return m_diagnostic_ap.get();
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
QualTypeMatchesBitSize(const uint64_t bit_size, ASTContext *ast_context, QualType qual_type)
{
    uint64_t qual_type_bit_size = ast_context->getTypeSize(qual_type);
    if (qual_type_bit_size == bit_size)
        return true;
    return false;
}

void *
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (lldb::Encoding encoding, uint32_t bit_size)
{
    ASTContext *ast_context = getASTContext();

    assert (ast_context != NULL);

    return GetBuiltinTypeForEncodingAndBitSize (ast_context, encoding, bit_size);
}

void *
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (clang::ASTContext *ast_context, lldb::Encoding encoding, uint32_t bit_size)
{
    if (!ast_context)
        return NULL;
    
    switch (encoding)
    {
    case lldb::eEncodingInvalid:
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->VoidPtrTy))
            return ast_context->VoidPtrTy.getAsOpaquePtr();
        break;
        
    case lldb::eEncodingUint:
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedCharTy))
            return ast_context->UnsignedCharTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedShortTy))
            return ast_context->UnsignedShortTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedIntTy))
            return ast_context->UnsignedIntTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedLongTy))
            return ast_context->UnsignedLongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedLongLongTy))
            return ast_context->UnsignedLongLongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedInt128Ty))
            return ast_context->UnsignedInt128Ty.getAsOpaquePtr();
        break;
        
    case lldb::eEncodingSint:
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->CharTy))
            return ast_context->CharTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->ShortTy))
            return ast_context->ShortTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->IntTy))
            return ast_context->IntTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongTy))
            return ast_context->LongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongLongTy))
            return ast_context->LongLongTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->Int128Ty))
            return ast_context->Int128Ty.getAsOpaquePtr();
        break;
        
    case lldb::eEncodingIEEE754:
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->FloatTy))
            return ast_context->FloatTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->DoubleTy))
            return ast_context->DoubleTy.getAsOpaquePtr();
        if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongDoubleTy))
            return ast_context->LongDoubleTy.getAsOpaquePtr();
        break;
        
    case lldb::eEncodingVector:
    default:
        break;
    }
    
    return NULL;
}

void *
ClangASTContext::GetBuiltinTypeForDWARFEncodingAndBitSize (const char *type_name, uint32_t dw_ate, uint32_t bit_size)
{
    ASTContext *ast_context = getASTContext();

    #define streq(a,b) strcmp(a,b) == 0
    assert (ast_context != NULL);
    if (ast_context)
    {
        switch (dw_ate)
        {
        default:
            break;

        case DW_ATE_address:
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->VoidPtrTy))
                return ast_context->VoidPtrTy.getAsOpaquePtr();
            break;

        case DW_ATE_boolean:
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->BoolTy))
                return ast_context->BoolTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedCharTy))
                return ast_context->UnsignedCharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedShortTy))
                return ast_context->UnsignedShortTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedIntTy))
                return ast_context->UnsignedIntTy.getAsOpaquePtr();
            break;

        case DW_ATE_complex_float:
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->FloatComplexTy))
                return ast_context->FloatComplexTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->DoubleComplexTy))
                return ast_context->DoubleComplexTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongDoubleComplexTy))
                return ast_context->LongDoubleComplexTy.getAsOpaquePtr();
            break;

        case DW_ATE_float:
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->FloatTy))
                return ast_context->FloatTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->DoubleTy))
                return ast_context->DoubleTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongDoubleTy))
                return ast_context->LongDoubleTy.getAsOpaquePtr();
            break;

        case DW_ATE_signed:
            if (type_name)
            {
                if (streq(type_name, "int") ||
                    streq(type_name, "signed int"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->IntTy))
                        return ast_context->IntTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->Int128Ty))
                        return ast_context->Int128Ty.getAsOpaquePtr();
                }

                if (streq(type_name, "long int") ||
                    streq(type_name, "long long int") ||
                    streq(type_name, "signed long long"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongTy))
                        return ast_context->LongTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongLongTy))
                        return ast_context->LongLongTy.getAsOpaquePtr();
                }

                if (streq(type_name, "short") ||
                    streq(type_name, "short int") ||
                    streq(type_name, "signed short") ||
                    streq(type_name, "short signed int"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->ShortTy))
                        return ast_context->ShortTy.getAsOpaquePtr();
                }

                if (streq(type_name, "char") ||
                    streq(type_name, "signed char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->CharTy))
                        return ast_context->CharTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->SignedCharTy))
                        return ast_context->SignedCharTy.getAsOpaquePtr();
                }

                if (streq(type_name, "wchar_t"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->WCharTy))
                        return ast_context->WCharTy.getAsOpaquePtr();
                }

            }
            // We weren't able to match up a type name, just search by size
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->CharTy))
                return ast_context->CharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->ShortTy))
                return ast_context->ShortTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->IntTy))
                return ast_context->IntTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongTy))
                return ast_context->LongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->LongLongTy))
                return ast_context->LongLongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->Int128Ty))
                return ast_context->Int128Ty.getAsOpaquePtr();
            break;

        case DW_ATE_signed_char:
            if (type_name)
            {
                if (streq(type_name, "signed char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->SignedCharTy))
                        return ast_context->SignedCharTy.getAsOpaquePtr();
                }
            }
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->CharTy))
                return ast_context->CharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->SignedCharTy))
                return ast_context->SignedCharTy.getAsOpaquePtr();
            break;

        case DW_ATE_unsigned:
            if (type_name)
            {
                if (streq(type_name, "unsigned int"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedIntTy))
                        return ast_context->UnsignedIntTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedInt128Ty))
                        return ast_context->UnsignedInt128Ty.getAsOpaquePtr();
                }

                if (streq(type_name, "unsigned int") ||
                    streq(type_name, "long unsigned int") ||
                    streq(type_name, "unsigned long long"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedLongTy))
                        return ast_context->UnsignedLongTy.getAsOpaquePtr();
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedLongLongTy))
                        return ast_context->UnsignedLongLongTy.getAsOpaquePtr();
                }

                if (streq(type_name, "unsigned short") ||
                    streq(type_name, "short unsigned int"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedShortTy))
                        return ast_context->UnsignedShortTy.getAsOpaquePtr();
                }
                if (streq(type_name, "unsigned char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedCharTy))
                        return ast_context->UnsignedCharTy.getAsOpaquePtr();
                }

            }
            // We weren't able to match up a type name, just search by size
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedCharTy))
                return ast_context->UnsignedCharTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedShortTy))
                return ast_context->UnsignedShortTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedIntTy))
                return ast_context->UnsignedIntTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedLongTy))
                return ast_context->UnsignedLongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedLongLongTy))
                return ast_context->UnsignedLongLongTy.getAsOpaquePtr();
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedInt128Ty))
                return ast_context->UnsignedInt128Ty.getAsOpaquePtr();
            break;

        case DW_ATE_unsigned_char:
            if (QualTypeMatchesBitSize (bit_size, ast_context, ast_context->UnsignedCharTy))
                return ast_context->UnsignedCharTy.getAsOpaquePtr();
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

void *
ClangASTContext::GetVoidBuiltInType()
{
    return getASTContext()->VoidTy.getAsOpaquePtr();
}

void *
ClangASTContext::GetCStringType (bool is_const)
{
    QualType char_type(getASTContext()->CharTy);
    
    if (is_const)
        char_type.addConst();
    
    return getASTContext()->getPointerType(char_type).getAsOpaquePtr();
}

void *
ClangASTContext::GetVoidPtrType (bool is_const)
{
    return GetVoidPtrType(getASTContext(), is_const);
}

void *
ClangASTContext::GetVoidPtrType (clang::ASTContext *ast_context, bool is_const)
{
    QualType void_ptr_type(ast_context->VoidPtrTy);
    
    if (is_const)
        void_ptr_type.addConst();
    
    return void_ptr_type.getAsOpaquePtr();
}

void *
ClangASTContext::CopyType(clang::ASTContext *dest_context, 
                          clang::ASTContext *source_context,
                          void *clang_type)
{
    Diagnostic diagnostics;
    FileManager file_manager;
    ASTImporter importer(diagnostics,
                         *dest_context, file_manager,
                         *source_context, file_manager);
    QualType ret = importer.Import(QualType::getFromOpaquePtr(clang_type));
    return ret.getAsOpaquePtr();
}

bool
ClangASTContext::AreTypesSame(clang::ASTContext *ast_context,
             void *type1,
             void *type2)
{
    return ast_context->hasSameType(QualType::getFromOpaquePtr(type1),
                                    QualType::getFromOpaquePtr(type2));
}

#pragma mark CVR modifiers

void *
ClangASTContext::AddConstModifier (void *clang_type)
{
    if (clang_type)
    {
        QualType result(QualType::getFromOpaquePtr(clang_type));
        result.addConst();
        return result.getAsOpaquePtr();
    }
    return NULL;
}

void *
ClangASTContext::AddRestrictModifier (void *clang_type)
{
    if (clang_type)
    {
        QualType result(QualType::getFromOpaquePtr(clang_type));
        result.getQualifiers().setRestrict (true);
        return result.getAsOpaquePtr();
    }
    return NULL;
}

void *
ClangASTContext::AddVolatileModifier (void *clang_type)
{
    if (clang_type)
    {
        QualType result(QualType::getFromOpaquePtr(clang_type));
        result.getQualifiers().setVolatile (true);
        return result.getAsOpaquePtr();
    }
    return NULL;
}

#pragma mark Structure, Unions, Classes

void *
ClangASTContext::CreateRecordType (const char *name, int kind, DeclContext *decl_ctx, lldb::LanguageType language)
{
    ASTContext *ast_context = getASTContext();
    assert (ast_context != NULL);

    if (decl_ctx == NULL)
        decl_ctx = ast_context->getTranslationUnitDecl();


    if (language == lldb::eLanguageTypeObjC)
    {
        bool isForwardDecl = false;
        bool isInternal = false;
        return CreateObjCClass (name, decl_ctx, isForwardDecl, isInternal);
    }

    // NOTE: Eventually CXXRecordDecl will be merged back into RecordDecl and
    // we will need to update this code. I was told to currently always use
    // the CXXRecordDecl class since we often don't know from debug information
    // if something is struct or a class, so we default to always use the more
    // complete definition just in case.
    CXXRecordDecl *decl = CXXRecordDecl::Create(*ast_context,
                                                (TagDecl::TagKind)kind,
                                                decl_ctx,
                                                SourceLocation(),
                                                name && name[0] ? &ast_context->Idents.get(name) : NULL);

    return ast_context->getTagDeclType(decl).getAsOpaquePtr();
}

bool
ClangASTContext::AddFieldToRecordType 
(
    void *record_clang_type, 
    const char *name, 
    void *field_type, 
    AccessType access, 
    uint32_t bitfield_bit_size
)
{
    if (record_clang_type == NULL || field_type == NULL)
        return false;

    ASTContext *ast_context = getASTContext();
    IdentifierTable *identifier_table = getIdentifierTable();

    assert (ast_context != NULL);
    assert (identifier_table != NULL);

    QualType record_qual_type(QualType::getFromOpaquePtr(record_clang_type));

    clang::Type *clang_type = record_qual_type.getTypePtr();
    if (clang_type)
    {
        const RecordType *record_type = dyn_cast<RecordType>(clang_type);

        if (record_type)
        {
            RecordDecl *record_decl = record_type->getDecl();

            CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
            if (cxx_record_decl)
                cxx_record_decl->setEmpty (false);

            clang::Expr *bit_width = NULL;
            if (bitfield_bit_size != 0)
            {
                APInt bitfield_bit_size_apint(ast_context->getTypeSize(ast_context->IntTy), bitfield_bit_size);
                bit_width = new (*ast_context)IntegerLiteral (bitfield_bit_size_apint, ast_context->IntTy, SourceLocation());
            }
            FieldDecl *field = FieldDecl::Create (*ast_context,
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
                return true;
            }
        }
        else
        {
            ObjCObjectType *objc_class_type = dyn_cast<ObjCObjectType>(clang_type);
            if (objc_class_type)
            {
                bool isSynthesized = false;
                ClangASTContext::AddObjCClassIVar (record_clang_type,
                                                   name,
                                                   field_type,
                                                   access,
                                                   bitfield_bit_size,
                                                   isSynthesized);
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
    ASTContext *ast_context,
    FieldDecl* field,
    uint32_t& bitfield_bit_size
)
{
    if (ast_context == NULL || field == NULL)
        return false;

    if (field->isBitField())
    {
        Expr* bit_width_expr = field->getBitWidth();
        if (bit_width_expr)
        {
            llvm::APSInt bit_width_apsint;
            if (bit_width_expr->isIntegerConstantExpr(bit_width_apsint, *ast_context))
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
ClangASTContext::SetDefaultAccessForRecordFields (void *clang_qual_type, int default_accessibility, int *assigned_accessibilities, size_t num_assigned_accessibilities)
{
    if (clang_qual_type)
    {
        QualType qual_type(QualType::getFromOpaquePtr(clang_qual_type));
        clang::Type *clang_type = qual_type.getTypePtr();
        if (clang_type)
        {
            RecordType *record_type = dyn_cast<RecordType>(clang_type);
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
}

#pragma mark C++ Base Classes

CXXBaseSpecifier *
ClangASTContext::CreateBaseClassSpecifier (void *base_class_type, AccessType access, bool is_virtual, bool base_of_class)
{
    if (base_class_type)
        return new CXXBaseSpecifier(SourceRange(), is_virtual, base_of_class, (AccessSpecifier)access, QualType::getFromOpaquePtr(base_class_type));
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
ClangASTContext::SetBaseClassesForClassType (void *class_clang_type, CXXBaseSpecifier const * const *base_classes, unsigned num_base_classes)
{
    if (class_clang_type)
    {
        clang::Type *clang_type = QualType::getFromOpaquePtr(class_clang_type).getTypePtr();
        if (clang_type)
        {
            RecordType *record_type = dyn_cast<RecordType>(clang_type);
            if (record_type)
            {
                CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_type->getDecl());
                if (cxx_record_decl)
                {
                    //cxx_record_decl->setEmpty (false);
                    cxx_record_decl->setBases(base_classes, num_base_classes);
                    return true;
                }
            }
        }
    }
    return false;
}
#pragma mark Objective C Classes

void *
ClangASTContext::CreateObjCClass 
(
    const char *name, 
    DeclContext *decl_ctx, 
    bool isForwardDecl, 
    bool isInternal
)
{
    ASTContext *ast_context = getASTContext();
    assert (ast_context != NULL);
    assert (name && name[0]);
    if (decl_ctx == NULL)
        decl_ctx = ast_context->getTranslationUnitDecl();

    // NOTE: Eventually CXXRecordDecl will be merged back into RecordDecl and
    // we will need to update this code. I was told to currently always use
    // the CXXRecordDecl class since we often don't know from debug information
    // if something is struct or a class, so we default to always use the more
    // complete definition just in case.
    ObjCInterfaceDecl *decl = ObjCInterfaceDecl::Create (*ast_context,
                                                         decl_ctx,
                                                         SourceLocation(),
                                                         &ast_context->Idents.get(name),
                                                         SourceLocation(),
                                                         isForwardDecl,
                                                         isInternal);
    
    return ast_context->getObjCInterfaceType(decl).getAsOpaquePtr();
}

bool
ClangASTContext::SetObjCSuperClass (void *class_opaque_type, void *super_opaque_type)
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
    void *class_opaque_type, 
    const char *name, 
    void *ivar_opaque_type, 
    AccessType access, 
    uint32_t bitfield_bit_size, 
    bool isSynthesized
)
{
    if (class_opaque_type == NULL || ivar_opaque_type == NULL)
        return false;

    ASTContext *ast_context = getASTContext();
    IdentifierTable *identifier_table = getIdentifierTable();

    assert (ast_context != NULL);
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
                    APInt bitfield_bit_size_apint(ast_context->getTypeSize(ast_context->IntTy), bitfield_bit_size);
                    bit_width = new (*ast_context)IntegerLiteral (bitfield_bit_size_apint, ast_context->IntTy, SourceLocation());
                }
                
                ObjCIvarDecl *field = ObjCIvarDecl::Create (*ast_context,
                                                            class_interface_decl,
                                                            SourceLocation(),
                                                            &identifier_table->get(name), // Identifier
                                                            QualType::getFromOpaquePtr(ivar_opaque_type), // Field type
                                                            NULL, // TypeSourceInfo *
                                                            ConvertAccessTypeToObjCIvarAccessControl (access),
                                                            bit_width,
                                                            isSynthesized);
                
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
ClangASTContext::ObjCTypeHasIVars (void *class_opaque_type, bool check_superclass)
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
    

#pragma mark Aggregate Types

bool
ClangASTContext::IsAggregateType (void *clang_type)
{
    if (clang_type == NULL)
        return false;

    QualType qual_type (QualType::getFromOpaquePtr(clang_type));

    if (qual_type->isAggregateType ())
        return true;

    switch (qual_type->getTypeClass())
    {
    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:
    case clang::Type::ConstantArray:
    case clang::Type::ExtVector:
    case clang::Type::Vector:
    case clang::Type::Record:
    case clang::Type::ObjCObject:
    case clang::Type::ObjCInterface:
    case clang::Type::ObjCObjectPointer:
        return true;

    case clang::Type::Typedef:
        return ClangASTContext::IsAggregateType (cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr());

    default:
        break;
    }
    // The clang type does have a value
    return false;
}

uint32_t
ClangASTContext::GetNumChildren (void *clang_qual_type, bool omit_empty_base_classes)
{
    if (clang_qual_type == NULL)
        return 0;

    uint32_t num_children = 0;
    QualType qual_type(QualType::getFromOpaquePtr(clang_qual_type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
    case clang::Type::Record:
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
        return ClangASTContext::GetNumChildren (cast<ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType().getAsOpaquePtr(), 
                                                omit_empty_base_classes);

    case clang::Type::ConstantArray:
        num_children = cast<ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
        break;

    case clang::Type::Pointer:
        {
            PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
            QualType pointee_type = pointer_type->getPointeeType();
            uint32_t num_pointee_children = ClangASTContext::GetNumChildren (pointee_type.getAsOpaquePtr(), 
                                                                             omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
        break;

    case clang::Type::Typedef:
        num_children = ClangASTContext::GetNumChildren (cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr(), omit_empty_base_classes);
        break;

    default:
        break;
    }
    return num_children;
}


void *
ClangASTContext::GetChildClangTypeAtIndex
(
    const char *parent_name,
    void *parent_clang_type,
    uint32_t idx,
    bool transparent_pointers,
    bool omit_empty_base_classes,
    std::string& child_name,
    uint32_t &child_byte_size,
    int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset
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
                                         child_bitfield_bit_offset);
    return NULL;
}

void *
ClangASTContext::GetChildClangTypeAtIndex
(
    ASTContext *ast_context,
    const char *parent_name,
    void *parent_clang_type,
    uint32_t idx,
    bool transparent_pointers,
    bool omit_empty_base_classes,
    std::string& child_name,
    uint32_t &child_byte_size,
    int32_t &child_byte_offset,
    uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset
)
{
    if (parent_clang_type == NULL)
        return NULL;

    if (idx < ClangASTContext::GetNumChildren (parent_clang_type, omit_empty_base_classes))
    {
        uint32_t bit_offset;
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;
        QualType parent_qual_type(QualType::getFromOpaquePtr(parent_clang_type));
        switch (parent_qual_type->getTypeClass())
        {
        case clang::Type::Record:
            {
                const RecordType *record_type = cast<RecordType>(parent_qual_type.getTypePtr());
                const RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const ASTRecordLayout &record_layout = ast_context->getASTRecordLayout(record_decl);
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
                                bit_offset = record_layout.getVBaseClassOffset(base_class_decl);
                            else
                                bit_offset = record_layout.getBaseClassOffset(base_class_decl);

                            // Base classes should be a multiple of 8 bits in size
                            assert (bit_offset % 8 == 0);
                            child_byte_offset = bit_offset/8;
                            std::string base_class_type_name(base_class->getType().getAsString());

                            child_name.assign(base_class_type_name.c_str());

                            uint64_t clang_type_info_bit_size = ast_context->getTypeSize(base_class->getType());

                            // Base classes biut sizes should be a multiple of 8 bits in size
                            assert (clang_type_info_bit_size % 8 == 0);
                            child_byte_size = clang_type_info_bit_size / 8;
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
                        std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(field->getType());
                        assert(field_idx < record_layout.getFieldCount());

                        child_byte_size = field_type_info.first / 8;

                        // Figure out the field offset within the current struct/union/class type
                        bit_offset = record_layout.getFieldOffset (field_idx);
                        child_byte_offset = bit_offset / 8;
                        if (ClangASTContext::FieldIsBitfield (ast_context, *field, child_bitfield_bit_size))
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
                
                        const ASTRecordLayout &interface_layout = ast_context->getASTObjCInterfaceLayout(class_interface_decl);
                        ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                if (ClangASTContext::GetNumChildren(superclass_interface_decl, omit_empty_base_classes) > 0)
                                {
                                    if (idx == 0)
                                    {
                                        QualType ivar_qual_type(ast_context->getObjCInterfaceType(superclass_interface_decl));
                                        

                                        child_name.assign(superclass_interface_decl->getNameAsString().c_str());

                                        std::pair<uint64_t, unsigned> ivar_type_info = ast_context->getTypeInfo(ivar_qual_type.getTypePtr());

                                        child_byte_size = ivar_type_info.first / 8;

                                        // Figure out the field offset within the current struct/union/class type
                                        bit_offset = interface_layout.getFieldOffset (child_idx);
                                        child_byte_offset = bit_offset / 8;

                                        return ivar_qual_type.getAsOpaquePtr();
                                    }

                                    ++child_idx;
                                }
                            }
                            else
                                ++child_idx;
                        }

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

                                    std::pair<uint64_t, unsigned> ivar_type_info = ast_context->getTypeInfo(ivar_qual_type.getTypePtr());

                                    child_byte_size = ivar_type_info.first / 8;

                                    // Figure out the field offset within the current struct/union/class type
                                    bit_offset = interface_layout.getFieldOffset (child_idx);
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
                return GetChildClangTypeAtIndex (ast_context,
                                                 parent_name,
                                                 cast<ObjCObjectPointerType>(parent_qual_type.getTypePtr())->getPointeeType().getAsOpaquePtr(),
                                                 idx,
                                                 transparent_pointers,
                                                 omit_empty_base_classes,
                                                 child_name,
                                                 child_byte_size,
                                                 child_byte_offset,
                                                 child_bitfield_bit_size,
                                                 child_bitfield_bit_offset);
            }
            break;

        case clang::Type::ConstantArray:
            {
                const ConstantArrayType *array = cast<ConstantArrayType>(parent_qual_type.getTypePtr());
                const uint64_t element_count = array->getSize().getLimitedValue();

                if (idx < element_count)
                {
                    std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(array->getElementType());

                    char element_name[32];
                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);

                    child_name.assign(element_name);
                    assert(field_type_info.first % 8 == 0);
                    child_byte_size = field_type_info.first / 8;
                    child_byte_offset = idx * child_byte_size;
                    return array->getElementType().getAsOpaquePtr();
                }
            }
            break;

        case clang::Type::Pointer:
            {
                PointerType *pointer_type = cast<PointerType>(parent_qual_type.getTypePtr());
                QualType pointee_type = pointer_type->getPointeeType();

                if (transparent_pointers && ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                {
                    return GetChildClangTypeAtIndex (ast_context,
                                                     parent_name,
                                                     pointer_type->getPointeeType().getAsOpaquePtr(),
                                                     idx,
                                                     transparent_pointers,
                                                     omit_empty_base_classes,
                                                     child_name,
                                                     child_byte_size,
                                                     child_byte_offset,
                                                     child_bitfield_bit_size,
                                                     child_bitfield_bit_offset);
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
                        std::pair<uint64_t, unsigned> clang_type_info = ast_context->getTypeInfo(pointee_type);
                        assert(clang_type_info.first % 8 == 0);
                        child_byte_size = clang_type_info.first / 8;
                        child_byte_offset = 0;
                        return pointee_type.getAsOpaquePtr();
                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetChildClangTypeAtIndex (ast_context,
                                             parent_name,
                                             cast<TypedefType>(parent_qual_type)->LookThroughTypedefs().getAsOpaquePtr(),
                                             idx,
                                             transparent_pointers,
                                             omit_empty_base_classes,
                                             child_name,
                                             child_byte_size,
                                             child_byte_offset,
                                             child_bitfield_bit_size,
                                             child_bitfield_bit_offset);
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
    return ClangASTContext::RecordHasFields(cast<CXXRecordDecl>(b->getType()->getAs<RecordType>()->getDecl())) == false;
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
    ASTContext *ast_context,
    void *clang_type,
    const char *name,
    bool omit_empty_base_classes,
    std::vector<uint32_t>& child_indexes
)
{
    if (clang_type && name && name[0])
    {
        QualType qual_type(QualType::getFromOpaquePtr(clang_type));
        switch (qual_type->getTypeClass())
        {
        case clang::Type::Record:
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
                    IdentifierInfo & ident_ref = ast_context->Idents.get(name, name + strlen (name));
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
                        
                        for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
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
                            
                            if (GetIndexOfChildMemberWithName (ast_context,
                                                               ast_context->getObjCInterfaceType(superclass_interface_decl).getAsOpaquePtr(),
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
                return GetIndexOfChildMemberWithName (ast_context,
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
//                    std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(array->getElementType());
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
//                    return GetIndexOfChildWithName (ast_context,
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
                    return GetIndexOfChildMemberWithName (ast_context,
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
                    return GetIndexOfChildMemberWithName (ast_context,
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
//                        std::pair<uint64_t, unsigned> clang_type_info = ast_context->getTypeInfo(pointee_type);
//                        assert(clang_type_info.first % 8 == 0);
//                        child_byte_size = clang_type_info.first / 8;
//                        child_byte_offset = 0;
//                        return pointee_type.getAsOpaquePtr();
//                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetIndexOfChildMemberWithName (ast_context,
                                                  cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr(),
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
    ASTContext *ast_context,
    void *clang_type,
    const char *name,
    bool omit_empty_base_classes
)
{
    if (clang_type && name && name[0])
    {
        QualType qual_type(QualType::getFromOpaquePtr(clang_type));
        
        clang::Type::TypeClass qual_type_class = qual_type->getTypeClass();

        switch (qual_type_class)
        {
        case clang::Type::Record:
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
                return GetIndexOfChildWithName (ast_context,
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
//                    std::pair<uint64_t, unsigned> field_type_info = ast_context->getTypeInfo(array->getElementType());
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
//                    return GetIndexOfChildWithName (ast_context,
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
                    return GetIndexOfChildWithName (ast_context,
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
                    return GetIndexOfChildWithName (ast_context,
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
//                        std::pair<uint64_t, unsigned> clang_type_info = ast_context->getTypeInfo(pointee_type);
//                        assert(clang_type_info.first % 8 == 0);
//                        child_byte_size = clang_type_info.first / 8;
//                        child_byte_offset = 0;
//                        return pointee_type.getAsOpaquePtr();
//                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetIndexOfChildWithName (ast_context,
                                            cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr(),
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
ClangASTContext::SetTagTypeKind (void *tag_clang_type, int kind)
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
ClangASTContext::GetDeclContextForType (void *clang_type)
{
    if (clang_type == NULL)
        return NULL;

    QualType qual_type(QualType::getFromOpaquePtr(clang_type));
    switch (qual_type->getTypeClass())
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
    case clang::Type::Typedef:                  return ClangASTContext::GetDeclContextForType (cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr());

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
        ASTContext *ast_context = getASTContext();
        if (decl_ctx == NULL)
            decl_ctx = ast_context->getTranslationUnitDecl();
        return NamespaceDecl::Create(*ast_context, decl_ctx, SourceLocation(), &ast_context->Idents.get(name));
    }
    return NULL;
}


#pragma mark Function Types

FunctionDecl *
ClangASTContext::CreateFunctionDeclaration (const char *name, void *function_clang_type, int storage, bool is_inline)
{
    if (name)
    {
        ASTContext *ast_context = getASTContext();
        assert (ast_context != NULL);

        if (name && name[0])
        {
            return FunctionDecl::Create(*ast_context,
                                        ast_context->getTranslationUnitDecl(),
                                        SourceLocation(),
                                        DeclarationName (&ast_context->Idents.get(name)),
                                        QualType::getFromOpaquePtr(function_clang_type),
                                        NULL,
                                        (FunctionDecl::StorageClass)storage,
                                        (FunctionDecl::StorageClass)storage,
                                        is_inline);
        }
        else
        {
            return FunctionDecl::Create(*ast_context,
                                        ast_context->getTranslationUnitDecl(),
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

void *
ClangASTContext::CreateFunctionType (void *result_type, void **args, unsigned num_args, bool isVariadic, unsigned TypeQuals)
{
    ASTContext *ast_context = getASTContext();
    assert (ast_context != NULL);
    std::vector<QualType> qual_type_args;
    for (unsigned i=0; i<num_args; ++i)
        qual_type_args.push_back (QualType::getFromOpaquePtr(args[i]));

    // TODO: Detect calling convention in DWARF?
    return ast_context->getFunctionType(QualType::getFromOpaquePtr(result_type),
                                        qual_type_args.empty() ? NULL : &qual_type_args.front(),
                                        qual_type_args.size(),
                                        isVariadic,
                                        TypeQuals,
                                        false,  // hasExceptionSpec
                                        false,  // hasAnyExceptionSpec,
                                        0,      // NumExs
                                        0,      // const QualType *ExArray
                                        FunctionType::ExtInfo ()).getAsOpaquePtr();    // NoReturn);
}

ParmVarDecl *
ClangASTContext::CreateParmeterDeclaration (const char *name, void *return_type, int storage)
{
    ASTContext *ast_context = getASTContext();
    assert (ast_context != NULL);
    return ParmVarDecl::Create(*ast_context,
                                ast_context->getTranslationUnitDecl(),
                                SourceLocation(),
                                name && name[0] ? &ast_context->Idents.get(name) : NULL,
                                QualType::getFromOpaquePtr(return_type),
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

void *
ClangASTContext::CreateArrayType (void *element_type, size_t element_count, uint32_t bit_stride)
{
    if (element_type)
    {
        ASTContext *ast_context = getASTContext();
        assert (ast_context != NULL);
        llvm::APInt ap_element_count (64, element_count);
        return ast_context->getConstantArrayType(QualType::getFromOpaquePtr(element_type),
                                                 ap_element_count,
                                                 ArrayType::Normal,
                                                 0).getAsOpaquePtr(); // ElemQuals
    }
    return NULL;
}


#pragma mark TagDecl

bool
ClangASTContext::StartTagDeclarationDefinition (void *clang_type)
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
ClangASTContext::CompleteTagDeclarationDefinition (void *clang_type)
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
                    tag_decl->completeDefinition();
                    return true;
                }
            }
        }
    }
    return false;
}


#pragma mark Enumeration Types

void *
ClangASTContext::CreateEnumerationType (const Declaration &decl, const char *name)
{
    // TODO: Do something intelligent with the Declaration object passed in
    // like maybe filling in the SourceLocation with it...
    ASTContext *ast_context = getASTContext();
    assert (ast_context != NULL);
    EnumDecl *enum_decl = EnumDecl::Create(*ast_context,
                                           ast_context->getTranslationUnitDecl(),
                                           SourceLocation(),
                                           name && name[0] ? &ast_context->Idents.get(name) : NULL,
                                           SourceLocation(),
                                           NULL);
    if (enum_decl)
        return ast_context->getTagDeclType(enum_decl).getAsOpaquePtr();
    return NULL;
}

bool
ClangASTContext::AddEnumerationValueToEnumerationType
(
    void *enum_clang_type,
    void *enumerator_clang_type,
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
        ASTContext *ast_context = getASTContext();
        IdentifierTable *identifier_table = getIdentifierTable();

        assert (ast_context != NULL);
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
                    EnumConstantDecl::Create(*ast_context,
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

void *
ClangASTContext::CreatePointerType (void *clang_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));

        switch (qual_type->getTypeClass())
        {
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
        // TODO: find out if I need to make a pointer or objc pointer for "clang::Type::ObjCObjectPointer" types
        //case clang::Type::ObjCObjectPointer: 
            return getASTContext()->getObjCObjectPointerType(qual_type).getAsOpaquePtr();

        // TODO: can we detect if this type is a block type?
//      case clang::Type::BlockType:
//          return getASTContext()->getBlockPointerType(qual_type).getAsOpaquePtr();
        
        default:
            return getASTContext()->getPointerType(qual_type).getAsOpaquePtr();
        }
    }
    return NULL;
}

void *
ClangASTContext::CreateLValueReferenceType (void *clang_type)
{
    if (clang_type)
        return getASTContext()->getLValueReferenceType (QualType::getFromOpaquePtr(clang_type)).getAsOpaquePtr();
    return NULL;
}

void *
ClangASTContext::CreateRValueReferenceType (void *clang_type)
{
    if (clang_type)
        return getASTContext()->getRValueReferenceType (QualType::getFromOpaquePtr(clang_type)).getAsOpaquePtr();
    return NULL;
}

void *
ClangASTContext::CreateMemberPointerType (void *clang_pointee_type, void *clang_class_type)
{
    if (clang_pointee_type && clang_pointee_type)
        return getASTContext()->getMemberPointerType(QualType::getFromOpaquePtr(clang_pointee_type),
                                                     QualType::getFromOpaquePtr(clang_class_type).getTypePtr()).getAsOpaquePtr();
    return NULL;
}

size_t
ClangASTContext::GetPointerBitSize ()
{
    ASTContext *ast_context = getASTContext();
    return ast_context->getTypeSize(ast_context->VoidPtrTy);
}

bool
ClangASTContext::IsPointerOrReferenceType (void *clang_type, void **target_type)
{
    if (clang_type == NULL)
        return false;

    QualType qual_type (QualType::getFromOpaquePtr(clang_type));
    switch (qual_type->getTypeClass())
    {
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
        return ClangASTContext::IsPointerOrReferenceType (cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr());
    default:
        break;
    }
    return false;
}

size_t
ClangASTContext::GetTypeBitSize (clang::ASTContext *ast_context, void *clang_type)
{
    if (clang_type)
        return ast_context->getTypeSize(QualType::getFromOpaquePtr(clang_type));
    return 0;
}

size_t
ClangASTContext::GetTypeBitAlign (clang::ASTContext *ast_context, void *clang_type)
{
    if (clang_type)
        return ast_context->getTypeAlign(QualType::getFromOpaquePtr(clang_type));
    return 0;
}

bool
ClangASTContext::IsIntegerType (void *clang_type, bool &is_signed)
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
ClangASTContext::IsPointerType (void *clang_type, void **target_type)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        switch (qual_type->getTypeClass())
        {
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
            return ClangASTContext::IsPointerOrReferenceType (cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr(), target_type);
        default:
            break;
        }
    }
    return false;
}

bool
ClangASTContext::IsFloatingPointType (void *clang_type, uint32_t &count, bool &is_complex)
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
ClangASTContext::IsCStringType (void *clang_type, uint32_t &length)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        switch (qual_type->getTypeClass())
        {
        case clang::Type::ConstantArray:
            {
                ConstantArrayType *array = cast<ConstantArrayType>(qual_type.getTypePtr());
                QualType element_qual_type = array->getElementType();
                clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
                if (canonical_type && canonical_type->isCharType())
                {
                    // We know the size of the array and it could be a C string
                    // since it is an array of characters
                    length = array->getSize().getLimitedValue();
                    return true;
                }
            }
            break;

        case clang::Type::Pointer:
            {
                PointerType *pointer_type = cast<PointerType>(qual_type.getTypePtr());
                clang::Type *pointee_type_ptr = pointer_type->getPointeeType().getTypePtr();
                if (pointee_type_ptr)
                {
                    clang::Type *canonical_type_ptr = pointee_type_ptr->getCanonicalTypeInternal().getTypePtr();
                    length = 0; // No length info, read until a NULL terminator is received
                    if (canonical_type_ptr)
                        return canonical_type_ptr->isCharType();
                    else
                        return pointee_type_ptr->isCharType();
                }
            }
            break;

        case clang::Type::Typedef:
            return ClangASTContext::IsCStringType (cast<TypedefType>(qual_type)->LookThroughTypedefs().getAsOpaquePtr(), length);

        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            {
                ReferenceType *reference_type = cast<ReferenceType>(qual_type.getTypePtr());
                clang::Type *pointee_type_ptr = reference_type->getPointeeType().getTypePtr();
                if (pointee_type_ptr)
                {
                    clang::Type *canonical_type_ptr = pointee_type_ptr->getCanonicalTypeInternal().getTypePtr();
                    length = 0; // No length info, read until a NULL terminator is received
                    if (canonical_type_ptr)
                        return canonical_type_ptr->isCharType();
                    else
                        return pointee_type_ptr->isCharType();
                }
            }
            break;
        }
    }
    return false;
}

bool
ClangASTContext::IsArrayType (void *clang_type, void **member_type, uint64_t *size)
{
    if (!clang_type)
        return false;
    
    QualType qual_type (QualType::getFromOpaquePtr(clang_type));
    
    switch (qual_type->getTypeClass())
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

void *
ClangASTContext::CreateTypedefType (const char *name, void *clang_type, DeclContext *decl_ctx)
{
    if (clang_type)
    {
        QualType qual_type (QualType::getFromOpaquePtr(clang_type));
        ASTContext *ast_context = getASTContext();
        IdentifierTable *identifier_table = getIdentifierTable();
        assert (ast_context != NULL);
        assert (identifier_table != NULL);
        if (decl_ctx == NULL)
            decl_ctx = ast_context->getTranslationUnitDecl();
        TypedefDecl *decl = TypedefDecl::Create(*ast_context,
                                                decl_ctx,
                                                SourceLocation(),
                                                name ? &identifier_table->get(name) : NULL, // Identifier
                                                ast_context->CreateTypeSourceInfo(qual_type));

        // Get a uniqued QualType for the typedef decl type
        return ast_context->getTypedefType (decl).getAsOpaquePtr();
    }
    return NULL;
}


std::string
ClangASTContext::GetTypeName (void *opaque_qual_type)
{
    std::string return_name;
    
    clang::QualType qual_type(clang::QualType::getFromOpaquePtr(opaque_qual_type));

    const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
    if (typedef_type)
    {
        const clang::TypedefDecl *typedef_decl = typedef_type->getDecl();
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
//ClangASTContext::ConvertFloatValueToString (ASTContext *ast_context, void *clang_type, const uint8_t* bytes, size_t byte_size, int apint_byte_order, std::string &float_str)
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
ClangASTContext::ConvertStringToFloatValue (ASTContext *ast_context, void *clang_type, const char *s, uint8_t *dst, size_t dst_size)
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
            APFloat ap_float(ast_context->getFloatTypeSemantics(qual_type), s_sref);

            const uint64_t bit_size = ast_context->getTypeSize (qual_type);
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
