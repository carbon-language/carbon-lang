//===-- ClangModulesDeclVendor.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangModulesDeclVendor.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"

#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Lookup.h"
#include "clang/Serialization/ASTReader.h"

#include <mutex>

using namespace lldb_private;

namespace {
    // Any Clang compiler requires a consumer for diagnostics.  This one stores them as strings
    // so we can provide them to the user in case a module failed to load.
    class StoringDiagnosticConsumer : public clang::DiagnosticConsumer
    {
    public:
        StoringDiagnosticConsumer ();
        void
        HandleDiagnostic (clang::DiagnosticsEngine::Level DiagLevel, const clang::Diagnostic &info);
        
        void
        ClearDiagnostics ();
        
        void
        DumpDiagnostics (Stream &error_stream);
    private:
        typedef std::pair<clang::DiagnosticsEngine::Level, std::string> IDAndDiagnostic;
        std::vector<IDAndDiagnostic> m_diagnostics;
        Log * m_log;
    };
    
    // The private implementation of our ClangModulesDeclVendor.  Contains all the Clang state required
    // to load modules.
    class ClangModulesDeclVendorImpl : public ClangModulesDeclVendor
    {
    public:
        ClangModulesDeclVendorImpl(llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> &diagnostics_engine,
                                   llvm::IntrusiveRefCntPtr<clang::CompilerInvocation> &compiler_invocation,
                                   std::unique_ptr<clang::CompilerInstance> &&compiler_instance,
                                   std::unique_ptr<clang::Parser> &&parser);
        
        virtual bool
        AddModule(std::vector<llvm::StringRef> &path,
                  Stream &error_stream);
        
        virtual uint32_t
        FindDecls (const ConstString &name,
                   bool append,
                   uint32_t max_matches,
                   std::vector <clang::NamedDecl*> &decls);
        
        ~ClangModulesDeclVendorImpl();
        
    private:
        clang::ModuleLoadResult
        DoGetModule(clang::ModuleIdPath path, bool make_visible);
        
        llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine>  m_diagnostics_engine;
        llvm::IntrusiveRefCntPtr<clang::CompilerInvocation> m_compiler_invocation;
        std::unique_ptr<clang::CompilerInstance>            m_compiler_instance;
        std::unique_ptr<clang::Parser>                      m_parser;
    };
}

StoringDiagnosticConsumer::StoringDiagnosticConsumer ()
{
    m_log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
}

void
StoringDiagnosticConsumer::HandleDiagnostic (clang::DiagnosticsEngine::Level DiagLevel, const clang::Diagnostic &info)
{
    llvm::SmallVector<char, 256> diagnostic_string;
    
    info.FormatDiagnostic(diagnostic_string);
    
    m_diagnostics.push_back(IDAndDiagnostic(DiagLevel, std::string(diagnostic_string.data(), diagnostic_string.size())));
}

void
StoringDiagnosticConsumer::ClearDiagnostics ()
{
    m_diagnostics.clear();
}

void
StoringDiagnosticConsumer::DumpDiagnostics (Stream &error_stream)
{
    for (IDAndDiagnostic &diag : m_diagnostics)
    {
        switch (diag.first)
        {
            default:
                error_stream.PutCString(diag.second.c_str());
                error_stream.PutChar('\n');
                break;
            case clang::DiagnosticsEngine::Level::Ignored:
                break;
        }
    }
}

static FileSpec
GetResourceDir ()
{
    static FileSpec g_cached_resource_dir;
    
    static std::once_flag g_once_flag;
    
    std::call_once(g_once_flag, [](){
        HostInfo::GetLLDBPath (lldb::ePathTypeClangDir, g_cached_resource_dir);
    });
    
    return g_cached_resource_dir;
}


ClangModulesDeclVendor::ClangModulesDeclVendor()
{
}

ClangModulesDeclVendor::~ClangModulesDeclVendor()
{
}

ClangModulesDeclVendorImpl::ClangModulesDeclVendorImpl(llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> &diagnostics_engine,
                                                       llvm::IntrusiveRefCntPtr<clang::CompilerInvocation> &compiler_invocation,
                                                       std::unique_ptr<clang::CompilerInstance> &&compiler_instance,
                                                       std::unique_ptr<clang::Parser> &&parser) :
    ClangModulesDeclVendor(),
    m_diagnostics_engine(diagnostics_engine),
    m_compiler_invocation(compiler_invocation),
    m_compiler_instance(std::move(compiler_instance)),
    m_parser(std::move(parser))
{
}

bool
ClangModulesDeclVendorImpl::AddModule(std::vector<llvm::StringRef> &path,
                                      Stream &error_stream)
{
    // Fail early.
    
    if (m_compiler_instance->hadModuleLoaderFatalFailure())
    {
        error_stream.PutCString("error: Couldn't load a module because the module loader is in a fatal state.\n");
        return false;
    }
    
    if (!m_compiler_instance->getPreprocessor().getHeaderSearchInfo().lookupModule(path[0]))
    {
        error_stream.Printf("error: Header search couldn't locate module %s\n", path[0].str().c_str());
        return false;
    }
    
    llvm::SmallVector<std::pair<clang::IdentifierInfo *, clang::SourceLocation>, 4> clang_path;
    
    {
        size_t source_loc_counter = 0;
        clang::SourceManager &source_manager = m_compiler_instance->getASTContext().getSourceManager();
        
        for (llvm::StringRef &component : path)
        {
            clang_path.push_back(std::make_pair(&m_compiler_instance->getASTContext().Idents.get(component),
                                                source_manager.getLocForStartOfFile(source_manager.getMainFileID()).getLocWithOffset(source_loc_counter++)));
        }
    }
    
    StoringDiagnosticConsumer *diagnostic_consumer = static_cast<StoringDiagnosticConsumer *>(m_compiler_instance->getDiagnostics().getClient());
    
    diagnostic_consumer->ClearDiagnostics();
    
    clang::Module *top_level_module = DoGetModule(clang_path.front(), false);
    
    if (!top_level_module)
    {
        diagnostic_consumer->DumpDiagnostics(error_stream);
        error_stream.Printf("error: Couldn't load top-level module %s\n", path[0].str().c_str());
        return false;
    }
    
    clang::Module *submodule = top_level_module;
    
    for (size_t ci = 1; ci < path.size(); ++ci)
    {
        llvm::StringRef &component = path[ci];
        submodule = submodule->findSubmodule(component.str());
        if (!submodule)
        {
            diagnostic_consumer->DumpDiagnostics(error_stream);
            error_stream.Printf("error: Couldn't load submodule %s\n", component.str().c_str());
            return false;
        }
    }
    
    clang::Module *requested_module = DoGetModule(clang_path, true);
    
    return (requested_module != nullptr);
}

// ClangImporter::lookupValue

uint32_t
ClangModulesDeclVendorImpl::FindDecls (const ConstString &name,
                                       bool append,
                                       uint32_t max_matches,
                                       std::vector <clang::NamedDecl*> &decls)
{
    if (!append)
        decls.clear();
    
    clang::IdentifierInfo &ident = m_compiler_instance->getASTContext().Idents.get(name.GetStringRef());
    
    clang::LookupResult lookup_result(m_compiler_instance->getSema(),
                                      clang::DeclarationName(&ident),
                                      clang::SourceLocation(),
                                      clang::Sema::LookupOrdinaryName);
    
    m_compiler_instance->getSema().LookupName(lookup_result, m_compiler_instance->getSema().getScopeForContext(m_compiler_instance->getASTContext().getTranslationUnitDecl()));
    
    uint32_t num_matches = 0;
    
    for (clang::NamedDecl *named_decl : lookup_result)
    {
        if (num_matches >= max_matches)
            return num_matches;
        
        decls.push_back(named_decl);
        ++num_matches;
    }
    
    return num_matches;
}

ClangModulesDeclVendorImpl::~ClangModulesDeclVendorImpl()
{
}

clang::ModuleLoadResult
ClangModulesDeclVendorImpl::DoGetModule(clang::ModuleIdPath path,
                                        bool make_visible)
{
    clang::Module::NameVisibilityKind visibility = make_visible ? clang::Module::AllVisible : clang::Module::Hidden;
    
    const bool is_inclusion_directive = false;
    
    return m_compiler_instance->loadModule(path.front().second, path, visibility, is_inclusion_directive);
}

static const char *ModuleImportBufferName = "LLDBModulesMemoryBuffer";

lldb_private::ClangModulesDeclVendor *
ClangModulesDeclVendor::Create(Target &target)
{
    // FIXME we should insure programmatically that the expression parser's compiler and the modules runtime's
    // compiler are both initialized in the same way – preferably by the same code.
    
    if (!target.GetPlatform()->SupportsModules())
        return nullptr;
    
    const ArchSpec &arch = target.GetArchitecture();
    
    std::vector<std::string> compiler_invocation_arguments =
    {
        "-fmodules",
        "-fcxx-modules",
        "-fsyntax-only",
        "-femit-all-decls",
        "-target", arch.GetTriple().str(),
        "-fmodules-validate-system-headers",
        "-Werror=non-modular-include-in-framework-module"
    };
    
    target.GetPlatform()->AddClangModuleCompilationOptions(&target, compiler_invocation_arguments);

    compiler_invocation_arguments.push_back(ModuleImportBufferName);

    // Add additional search paths with { "-I", path } or { "-F", path } here.
   
    {
        llvm::SmallString<128> DefaultModuleCache;
        const bool erased_on_reboot = false;
        llvm::sys::path::system_temp_directory(erased_on_reboot, DefaultModuleCache);
        llvm::sys::path::append(DefaultModuleCache, "org.llvm.clang");
        llvm::sys::path::append(DefaultModuleCache, "ModuleCache");
        std::string module_cache_argument("-fmodules-cache-path=");
        module_cache_argument.append(DefaultModuleCache.str().str());
        compiler_invocation_arguments.push_back(module_cache_argument);
    }
    
    {
        FileSpec clang_resource_dir = GetResourceDir();
        
        if (clang_resource_dir.IsDirectory())
        {
            compiler_invocation_arguments.push_back("-resource-dir");
            compiler_invocation_arguments.push_back(clang_resource_dir.GetPath());
        }
    }
    
    llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics_engine = clang::CompilerInstance::createDiagnostics(new clang::DiagnosticOptions,
                                                                                                                       new StoringDiagnosticConsumer);
    
    std::vector<const char *> compiler_invocation_argument_cstrs;
    
    for (const std::string &arg : compiler_invocation_arguments) {
        compiler_invocation_argument_cstrs.push_back(arg.c_str());
    }
    
    llvm::IntrusiveRefCntPtr<clang::CompilerInvocation> invocation(clang::createInvocationFromCommandLine(compiler_invocation_argument_cstrs, diagnostics_engine));
    
    if (!invocation)
        return nullptr;
    
    std::unique_ptr<llvm::MemoryBuffer> source_buffer = llvm::MemoryBuffer::getMemBuffer("extern int __lldb __attribute__((unavailable));",
                                                                                         ModuleImportBufferName);
    
    invocation->getPreprocessorOpts().addRemappedFile(ModuleImportBufferName, source_buffer.release());
    
    std::unique_ptr<clang::CompilerInstance> instance(new clang::CompilerInstance);
    
    instance->setDiagnostics(diagnostics_engine.get());
    instance->setInvocation(invocation.get());
    
    std::unique_ptr<clang::FrontendAction> action(new clang::SyntaxOnlyAction);
    
    instance->setTarget(clang::TargetInfo::CreateTargetInfo(*diagnostics_engine, instance->getInvocation().TargetOpts));
    
    if (!instance->hasTarget())
        return nullptr;
    
    instance->getTarget().adjust(instance->getLangOpts());
    
    if (!action->BeginSourceFile(*instance, instance->getFrontendOpts().Inputs[0]))
        return nullptr;
    
    instance->getPreprocessor().enableIncrementalProcessing();
    
    instance->createModuleManager();
    
    instance->createSema(action->getTranslationUnitKind(), nullptr);
    
    const bool skipFunctionBodies = false;
    std::unique_ptr<clang::Parser> parser(new clang::Parser(instance->getPreprocessor(), instance->getSema(), skipFunctionBodies));
    
    instance->getPreprocessor().EnterMainSourceFile();
    parser->Initialize();
    
    clang::Parser::DeclGroupPtrTy parsed;
    
    while (!parser->ParseTopLevelDecl(parsed));
    
    return new ClangModulesDeclVendorImpl (diagnostics_engine, invocation, std::move(instance), std::move(parser));
}
