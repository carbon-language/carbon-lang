//===-- ClangExpression.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

// C++ Includes
#include <cstdlib>
#include <string>
#include <map>

// Other libraries and framework includes
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/ParseAST.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/System/Host.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetSelect.h"

// Project includes
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangStmtVisitor.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Expression/RecordingMemoryManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"

#define NO_RTTI
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Core/dwarf.h"


using namespace lldb_private;
using namespace clang;
using namespace llvm;

namespace clang {

class AnalyzerOptions;
class CodeGenOptions;
class DependencyOutputOptions;
class DiagnosticOptions;
class FrontendOptions;
class HeaderSearchOptions;
class LangOptions;
class PreprocessorOptions;
class PreprocessorOutputOptions;
class TargetInfo;
class TargetOptions;

} // end namespace clang



//===----------------------------------------------------------------------===//
// Utility Methods
//===----------------------------------------------------------------------===//

std::string GetBuiltinIncludePath(const char *Argv0) {
  llvm::sys::Path P =
    llvm::sys::Path::GetMainExecutable(Argv0,
                                       (void*)(intptr_t) GetBuiltinIncludePath);

  if (!P.isEmpty()) {
    P.eraseComponent();  // Remove /clang from foo/bin/clang
    P.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/<version>/include
    P.appendComponent("lib");
    P.appendComponent("clang");
    P.appendComponent(CLANG_VERSION_STRING);
    P.appendComponent("include");
  }

  return P.str();
}


//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

void LLVMErrorHandler(void *UserData, const std::string &Message) {
    Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

    Diags.Report(diag::err_fe_error_backend) << Message;

    // We cannot recover from llvm errors.
    exit(1);
}

static FrontendAction *CreateFrontendBaseAction(CompilerInstance &CI) {
    using namespace clang::frontend;

    switch (CI.getFrontendOpts().ProgramAction) {
        default:
            llvm_unreachable("Invalid program action!");

        case ASTDump:                return new ASTDumpAction();
        case ASTPrint:               return new ASTPrintAction();
        case ASTPrintXML:            return new ASTPrintXMLAction();
        case ASTView:                return new ASTViewAction();
        case DumpRawTokens:          return new DumpRawTokensAction();
        case DumpTokens:             return new DumpTokensAction();
        case EmitAssembly:           return new EmitAssemblyAction();
        case EmitBC:                 return new EmitBCAction();
        case EmitHTML:               return new HTMLPrintAction();
        case EmitLLVM:               return new EmitLLVMAction();
        case EmitLLVMOnly:           return new EmitLLVMOnlyAction();
        case EmitObj:                return new EmitObjAction();
        case FixIt:                  return new FixItAction();
        case GeneratePCH:            return new GeneratePCHAction();
        case GeneratePTH:            return new GeneratePTHAction();
        case InheritanceView:        return new InheritanceViewAction();
        case InitOnly:               return new InitOnlyAction();
        case ParseNoop:              return new ParseOnlyAction();
        case ParsePrintCallbacks:    return new PrintParseAction();
        case ParseSyntaxOnly:        return new SyntaxOnlyAction();

        case PluginAction: {
            if (CI.getFrontendOpts().ActionName == "help") {
                llvm::errs() << "clang -cc1 plugins:\n";
                for (FrontendPluginRegistry::iterator it =
                     FrontendPluginRegistry::begin(),
                     ie = FrontendPluginRegistry::end();
                     it != ie; ++it)
                    llvm::errs() << "  " << it->getName() << " - " << it->getDesc() << "\n";
                return 0;
            }

            for (FrontendPluginRegistry::iterator it =
                 FrontendPluginRegistry::begin(), ie = FrontendPluginRegistry::end();
                 it != ie; ++it) {
                if (it->getName() == CI.getFrontendOpts().ActionName)
                    return it->instantiate();
            }

            CI.getDiagnostics().Report(diag::err_fe_invalid_plugin_name)
            << CI.getFrontendOpts().ActionName;
            return 0;
        }

        case PrintDeclContext:       return new DeclContextPrintAction();
        case PrintPreprocessedInput: return new PrintPreprocessedAction();
        case RewriteMacros:          return new RewriteMacrosAction();
        case RewriteObjC:            return new RewriteObjCAction();
        case RewriteTest:            return new RewriteTestAction();
        case RunAnalysis:            return new AnalysisAction();
        case RunPreprocessorOnly:    return new PreprocessOnlyAction();
    }
}

//----------------------------------------------------------------------
// ClangExpression constructor
//----------------------------------------------------------------------
ClangExpression::ClangExpression(const char *target_triple,
                                 ClangExpressionDeclMap *decl_map) :
    m_target_triple (),
    m_jit_mm_ptr (NULL),
    m_code_generator_ptr (NULL),
    m_decl_map (decl_map)
{
    if (target_triple && target_triple[0])
        m_target_triple = target_triple;
    else
        m_target_triple = llvm::sys::getHostTriple();
}


//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ClangExpression::~ClangExpression()
{
    if (m_code_generator_ptr && !m_execution_engine.get())
        delete m_code_generator_ptr;
}

bool
ClangExpression::CreateCompilerInstance (bool &IsAST)
{
    // Initialize targets first, so that --version shows registered targets.
    static struct InitializeLLVM {
        InitializeLLVM() {
            llvm::InitializeAllTargets();
            llvm::InitializeAllAsmPrinters();
        }
    } InitializeLLVM;

    // 1. Create a new compiler instance.
    m_clang_ap.reset(new CompilerInstance());    
    m_clang_ap->setLLVMContext(new LLVMContext());
    
    // 2. Set options.

    // Parse expressions as Objective C++ regardless of context.
    // Our hook into Clang's lookup mechanism only works in C++.
    m_clang_ap->getLangOpts().CPlusPlus = true;
    m_clang_ap->getLangOpts().ObjC1 = true;

    // Disable some warnings.
    m_clang_ap->getDiagnosticOpts().Warnings.push_back("no-unused-value");
    
    // Set the target triple.
    m_clang_ap->getTargetOpts().Triple = m_target_triple;
    
    // 3. Set up various important bits of infrastructure.
    
    m_clang_ap->createDiagnostics(0, 0);
    m_clang_ap->getLangOpts().CPlusPlus = true;

    // Create the target instance.
    m_clang_ap->setTarget(TargetInfo::CreateTargetInfo(m_clang_ap->getDiagnostics(),
                                                       m_clang_ap->getTargetOpts()));
    if (!m_clang_ap->hasTarget())
    {
        m_clang_ap.reset();
        return false;
    }

    // Inform the target of the language options
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    m_clang_ap->getTarget().setForcedLangOptions(m_clang_ap->getLangOpts());

    return m_clang_ap.get();
}

Mutex &
ClangExpression::GetClangMutex ()
{
    static Mutex g_clang_mutex(Mutex::eMutexTypeRecursive);  // Control access to the clang compiler
    return g_clang_mutex;
}


clang::ASTContext *
ClangExpression::GetASTContext ()
{
    CompilerInstance *compiler_instance = GetCompilerInstance();
    if (compiler_instance)
        return &compiler_instance->getASTContext();
    return NULL;
}

unsigned
ClangExpression::ParseExpression (const char *expr_text, Stream &stream)
{
    // HACK: for now we have to make a function body around our expression
    // since there is no way to parse a single expression line in LLVM/Clang.
    std::string func_expr("void ___clang_expr()\n{\n\t");
    func_expr.append(expr_text);
    func_expr.append(";\n}");
    return ParseBareExpression (func_expr, stream);

}

unsigned
ClangExpression::ParseBareExpression (llvm::StringRef expr_text, Stream &stream)
{
    Mutex::Locker locker(GetClangMutex ());

    TextDiagnosticBuffer text_diagnostic_buffer;

    bool IsAST = false;
    if (!CreateCompilerInstance (IsAST))
    {
        stream.Printf("error: couldn't create compiler instance\n");
        return 1;
    }
    
    // This code is matched below by a setClient to NULL.
    // We cannot return out of this code without doing that.
    m_clang_ap->getDiagnostics().setClient(&text_diagnostic_buffer);
    text_diagnostic_buffer.FlushDiagnostics (m_clang_ap->getDiagnostics());
    
    MemoryBuffer *memory_buffer = MemoryBuffer::getMemBufferCopy(expr_text, __FUNCTION__);

    if (!m_clang_ap->hasSourceManager())
        m_clang_ap->createSourceManager();

    m_clang_ap->createFileManager();
    m_clang_ap->createPreprocessor();
    
    // Build the ASTContext.  Most of this we inherit from the
    // CompilerInstance, but we also want to give the context
    // an ExternalASTSource.
    SelectorTable selector_table;
    std::auto_ptr<Builtin::Context> builtin_ap(new Builtin::Context(m_clang_ap->getTarget()));
    ASTContext *Context = new ASTContext(m_clang_ap->getLangOpts(),
                                         m_clang_ap->getSourceManager(),
                                         m_clang_ap->getTarget(),
                                         m_clang_ap->getPreprocessor().getIdentifierTable(),
                                         selector_table,
                                         *builtin_ap.get());
    
    llvm::OwningPtr<ExternalASTSource> ASTSource(new ClangASTSource(*Context, *m_decl_map));

    if (m_decl_map)
    {
        Context->setExternalSource(ASTSource);
    }
    
    m_clang_ap->setASTContext(Context);

    FileID memory_buffer_file_id = m_clang_ap->getSourceManager().createMainFileIDForMemBuffer (memory_buffer);
    std::string module_name("test_func");
    text_diagnostic_buffer.BeginSourceFile(m_clang_ap->getLangOpts(), &m_clang_ap->getPreprocessor());

    if (m_code_generator_ptr)
        delete m_code_generator_ptr;
    
    m_code_generator_ptr = CreateLLVMCodeGen(m_clang_ap->getDiagnostics(),
                                             module_name,
                                             m_clang_ap->getCodeGenOpts(),
                                             m_clang_ap->getLLVMContext());


    // - CodeGeneration ASTConsumer (include/clang/ModuleBuilder.h), which will be passed in when you call...
    // - Call clang::ParseAST (in lib/Sema/ParseAST.cpp) to parse the buffer. The CodeGenerator will generate code for __dbg_expr.
    // - Once ParseAST completes, you can grab the llvm::Module from the CodeGenerator, which will have an llvm::Function you can hand off to the JIT.
    ParseAST(m_clang_ap->getPreprocessor(), m_code_generator_ptr, m_clang_ap->getASTContext());

    text_diagnostic_buffer.EndSourceFile();

    //compiler_instance->getASTContext().getTranslationUnitDecl()->dump();

    //if (compiler_instance->getFrontendOpts().ShowStats) {
    //    compiler_instance->getFileManager().PrintStats();
    //    fprintf(stderr, "\n");
    //}
    
    // This code resolves the setClient above.
    m_clang_ap->getDiagnostics().setClient(0);
    
    TextDiagnosticBuffer::const_iterator diag_iterator;
    
    int num_errors = 0;

#ifdef COUNT_WARNINGS_AND_ERRORS
    int num_warnings = 0;
    
    for (diag_iterator = text_diagnostic_buffer.warn_begin();
         diag_iterator != text_diagnostic_buffer.warn_end();
         ++diag_iterator)
        num_warnings++;
    
    for (diag_iterator = text_diagnostic_buffer.err_begin();
         diag_iterator != text_diagnostic_buffer.err_end();
         ++diag_iterator)
        num_errors++;    
    
    if (num_warnings || num_errors)
    {
        if (num_warnings)
            stream.Printf("%u warning%s%s", num_warnings, (num_warnings == 1 ? "" : "s"), (num_errors ? " and " : ""));
        if (num_errors)
            stream.Printf("%u error%s", num_errors, (num_errors == 1 ? "" : "s"));
        stream.Printf("\n");
    }
#endif
    
    for (diag_iterator = text_diagnostic_buffer.warn_begin();
         diag_iterator != text_diagnostic_buffer.warn_end();
         ++diag_iterator)
        stream.Printf("warning: %s\n", (*diag_iterator).second.c_str());
    
    num_errors = 0;
    
    for (diag_iterator = text_diagnostic_buffer.err_begin();
         diag_iterator != text_diagnostic_buffer.err_end();
         ++diag_iterator)
    {
        num_errors++;
        stream.Printf("error: %s\n", (*diag_iterator).second.c_str());
    }
    
    return num_errors;
}


static FrontendAction *
CreateFrontendAction(CompilerInstance &CI)
{
    // Create the underlying action.
    FrontendAction *Act = CreateFrontendBaseAction(CI);
    if (!Act)
        return 0;

    // If there are any AST files to merge, create a frontend action
    // adaptor to perform the merge.
    if (!CI.getFrontendOpts().ASTMergeFiles.empty())
        Act = new ASTMergeAction(Act, &CI.getFrontendOpts().ASTMergeFiles[0],
                                 CI.getFrontendOpts().ASTMergeFiles.size());

    return Act;
}


unsigned
ClangExpression::ConvertExpressionToDWARF (ClangExpressionVariableList& expr_local_variable_list, 
                                           StreamString &dwarf_opcode_strm)
{
    CompilerInstance *compiler_instance = GetCompilerInstance();

    DeclarationName hack_func_name(&compiler_instance->getASTContext().Idents.get("___clang_expr"));
    DeclContext::lookup_result result = compiler_instance->getASTContext().getTranslationUnitDecl()->lookup(hack_func_name);

    if (result.first != result.second)
    {
        Decl *decl = *result.first;
        Stmt *decl_stmt = decl->getBody();
        if (decl_stmt)
        {
            ClangStmtVisitor visitor(compiler_instance->getASTContext(), expr_local_variable_list, m_decl_map, dwarf_opcode_strm);

            visitor.Visit (decl_stmt);
        }
    }
    return 0;
}

bool
ClangExpression::JITFunction (const ExecutionContext &exc_context, const char *name)
{

    llvm::Module *module = m_code_generator_ptr->GetModule();

    if (module)
    {
        std::string error;

        if (m_jit_mm_ptr == NULL)
            m_jit_mm_ptr = new RecordingMemoryManager();

        //llvm::InitializeNativeTarget();
        if (m_execution_engine.get() == 0)
            m_execution_engine.reset(llvm::ExecutionEngine::createJIT (module, &error, m_jit_mm_ptr));
        m_execution_engine->DisableLazyCompilation();
        llvm::Function *function = module->getFunction (llvm::StringRef (name));

        // We don't actually need the function pointer here, this just forces it to get resolved.
        void *fun_ptr = m_execution_engine->getPointerToFunction(function);
        // Note, you probably won't get here on error, since the LLVM JIT tends to just
        // exit on error at present...  So be careful.
        if (fun_ptr == 0)
            return false;
        m_jitted_functions.push_back(ClangExpression::JittedFunction(name, (lldb::addr_t) fun_ptr));

    }
    return true;
}

bool
ClangExpression::WriteJITCode (const ExecutionContext &exc_context)
{
    if (m_jit_mm_ptr == NULL)
        return false;

    if (exc_context.process == NULL)
        return false;

    // Look over the regions allocated for the function compiled.  The JIT
    // tries to allocate the functions & stubs close together, so we should try to
    // write them that way too...
    // For now I only write functions with no stubs, globals, exception tables,
    // etc.  So I only need to write the functions.

    size_t size = 0;
    std::map<uint8_t *, uint8_t *>::iterator fun_pos, fun_end = m_jit_mm_ptr->m_functions.end();
    for (fun_pos = m_jit_mm_ptr->m_functions.begin(); fun_pos != fun_end; fun_pos++)
    {
        size += (*fun_pos).second - (*fun_pos).first;
    }

    Error error;
    lldb::addr_t target_addr = exc_context.process->AllocateMemory (size, lldb::ePermissionsReadable|lldb::ePermissionsExecutable, error);

    if (target_addr == LLDB_INVALID_ADDRESS)
        return false;

    lldb::addr_t cursor = target_addr;
    for (fun_pos = m_jit_mm_ptr->m_functions.begin(); fun_pos != fun_end; fun_pos++)
    {
        lldb::addr_t lstart = (lldb::addr_t) (*fun_pos).first;
        lldb::addr_t lend = (lldb::addr_t) (*fun_pos).second;
        size_t size = lend - lstart;
        exc_context.process->WriteMemory(cursor, (void *) lstart, size, error);
        m_jit_mm_ptr->AddToLocalToRemoteMap (lstart, size, cursor);
        cursor += size;
    }

    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();

    for (pos = m_jitted_functions.begin(); pos != end; pos++)
    {
        (*pos).m_remote_addr = m_jit_mm_ptr->GetRemoteAddressForLocal ((*pos).m_local_addr);
    }
    return true;
}

lldb::addr_t
ClangExpression::GetFunctionAddress (const char *name)
{
    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();

    for (pos = m_jitted_functions.begin(); pos < end; pos++)
    {
        if (strcmp ((*pos).m_name.c_str(), name) == 0)
            return (*pos).m_remote_addr;
    }
    return LLDB_INVALID_ADDRESS;
}

unsigned
ClangExpression::Compile()
{
    Mutex::Locker locker(GetClangMutex ());
    bool IsAST = false;
    
    if (CreateCompilerInstance(IsAST))
    {
        // Validate/process some options
        if (m_clang_ap->getHeaderSearchOpts().Verbose)
            llvm::errs() << "clang-cc version " CLANG_VERSION_STRING
            << " based upon " << PACKAGE_STRING
            << " hosted on " << llvm::sys::getHostTriple() << "\n";

        // Enforce certain implications.
        if (!m_clang_ap->getFrontendOpts().ViewClassInheritance.empty())
            m_clang_ap->getFrontendOpts().ProgramAction = frontend::InheritanceView;
//        if (!compiler_instance->getFrontendOpts().FixItSuffix.empty())
//            compiler_instance->getFrontendOpts().ProgramAction = frontend::FixIt;

        for (unsigned i = 0, e = m_clang_ap->getFrontendOpts().Inputs.size(); i != e; ++i) {
            const std::string &InFile = m_clang_ap->getFrontendOpts().Inputs[i].second;

            // If we aren't using an AST file, setup the file and source managers and
            // the preprocessor.
            if (!IsAST) {
                if (!i) {
                    // Create a file manager object to provide access to and cache the
                    // filesystem.
                    m_clang_ap->createFileManager();

                    // Create the source manager.
                    m_clang_ap->createSourceManager();
                } else {
                    // Reset the ID tables if we are reusing the SourceManager.
                    m_clang_ap->getSourceManager().clearIDTables();
                }

                // Create the preprocessor.
                m_clang_ap->createPreprocessor();
            }

            llvm::OwningPtr<FrontendAction> Act(CreateFrontendAction(*m_clang_ap.get()));
            if (!Act)
                break;

            if (Act->BeginSourceFile(*m_clang_ap, InFile, IsAST)) {
                Act->Execute();
                Act->EndSourceFile();
            }
        }

        if (m_clang_ap->getDiagnosticOpts().ShowCarets)
        {
            unsigned NumWarnings = m_clang_ap->getDiagnostics().getNumWarnings();
            unsigned NumErrors = m_clang_ap->getDiagnostics().getNumErrors() -
            m_clang_ap->getDiagnostics().getNumErrorsSuppressed();

            if (NumWarnings || NumErrors)
            {
                if (NumWarnings)
                    fprintf (stderr, "%u warning%s%s", NumWarnings, (NumWarnings == 1 ? "" : "s"), (NumErrors ? " and " : ""));
                if (NumErrors)
                    fprintf (stderr, "%u error%s", NumErrors, (NumErrors == 1 ? "" : "s"));
                fprintf (stderr, " generated.\n");
            }
        }

        if (m_clang_ap->getFrontendOpts().ShowStats) {
            m_clang_ap->getFileManager().PrintStats();
            fprintf(stderr, "\n");
        }

        // Return the appropriate status when verifying diagnostics.
        //
        // FIXME: If we could make getNumErrors() do the right thing, we wouldn't need
        // this.
        if (m_clang_ap->getDiagnosticOpts().VerifyDiagnostics)
            return static_cast<VerifyDiagnosticsClient&>(m_clang_ap->getDiagnosticClient()).HadErrors();

        return m_clang_ap->getDiagnostics().getNumErrors();
    }
    return 1;
}
