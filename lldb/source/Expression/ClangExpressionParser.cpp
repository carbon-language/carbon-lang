//===-- ClangExpressionParser.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangExpressionParser.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/IRDynamicChecks.h"
#include "lldb/Expression/IRForTarget.h"
#include "lldb/Expression/IRToDWARF.h"
#include "lldb/Expression/RecordingMemoryManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Checker/FrontendActions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Rewrite/FrontendActions.h"
#include "clang/Sema/SemaConsumer.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Signals.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"

using namespace clang;
using namespace llvm;
using namespace lldb_private;

//===----------------------------------------------------------------------===//
// Utility Methods for Clang
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
// Main driver for Clang
//===----------------------------------------------------------------------===//

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
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
        case BoostCon:               return new BoostConAction();
        case DumpRawTokens:          return new DumpRawTokensAction();
        case DumpTokens:             return new DumpTokensAction();
        case EmitAssembly:           return new EmitAssemblyAction();
        case EmitBC:                 return new EmitBCAction();
        case EmitHTML:               return new HTMLPrintAction();
        case EmitLLVM:               return new EmitLLVMAction();
        case EmitLLVMOnly:           return new EmitLLVMOnlyAction();
        case EmitCodeGenOnly:        return new EmitCodeGenOnlyAction();
        case EmitObj:                return new EmitObjAction();
        case FixIt:                  return new FixItAction();
        case GeneratePCH:            return new GeneratePCHAction();
        case GeneratePTH:            return new GeneratePTHAction();
        case InheritanceView:        return new InheritanceViewAction();
        case InitOnly:               return new InitOnlyAction();
        case ParseSyntaxOnly:        return new SyntaxOnlyAction();
            
        case PluginAction: {
            for (FrontendPluginRegistry::iterator it =
                 FrontendPluginRegistry::begin(), ie = FrontendPluginRegistry::end();
                 it != ie; ++it) {
                if (it->getName() == CI.getFrontendOpts().ActionName) {
                    llvm::OwningPtr<PluginASTAction> P(it->instantiate());
                    if (!P->ParseArgs(CI, CI.getFrontendOpts().PluginArgs))
                        return 0;
                    return P.take();
                }
            }
            
            CI.getDiagnostics().Report(diag::err_fe_invalid_plugin_name)
            << CI.getFrontendOpts().ActionName;
            return 0;
        }
            
        case PrintDeclContext:       return new DeclContextPrintAction();
        case PrintPreamble:          return new PrintPreambleAction();
        case PrintPreprocessedInput: return new PrintPreprocessedAction();
        case RewriteMacros:          return new RewriteMacrosAction();
        case RewriteObjC:            return new RewriteObjCAction();
        case RewriteTest:            return new RewriteTestAction();
        case RunAnalysis:            return new AnalysisAction();
        case RunPreprocessorOnly:    return new PreprocessOnlyAction();
    }
}

static FrontendAction *CreateFrontendAction(CompilerInstance &CI) {
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

//===----------------------------------------------------------------------===//
// Implementation of ClangExpressionParser
//===----------------------------------------------------------------------===//

ClangExpressionParser::ClangExpressionParser(const char *target_triple,
                                             ClangExpression &expr) :
    m_expr(expr),
    m_target_triple (),
    m_compiler (),
    m_code_generator (NULL),
    m_execution_engine (),
    m_jitted_functions ()
{
    // Initialize targets first, so that --version shows registered targets.
    static struct InitializeLLVM {
        InitializeLLVM() {
            llvm::InitializeAllTargets();
            llvm::InitializeAllAsmPrinters();
        }
    } InitializeLLVM;
    
    if (target_triple && target_triple[0])
        m_target_triple = target_triple;
    else
        m_target_triple = llvm::sys::getHostTriple();
    
    // 1. Create a new compiler instance.
    m_compiler.reset(new CompilerInstance());    
    m_compiler->setLLVMContext(new LLVMContext());
    
    // 2. Set options.
    
    // Parse expressions as Objective C++ regardless of context.
    // Our hook into Clang's lookup mechanism only works in C++.
    m_compiler->getLangOpts().CPlusPlus = true;
    m_compiler->getLangOpts().ObjC1 = true;
    m_compiler->getLangOpts().ThreadsafeStatics = false;
    m_compiler->getLangOpts().AccessControl = false; // Debuggers get universal access
    m_compiler->getLangOpts().DollarIdents = true; // $ indicates a persistent variable name
    
    // Set CodeGen options
    m_compiler->getCodeGenOpts().EmitDeclMetadata = true;
    m_compiler->getCodeGenOpts().InstrumentFunctions = false;
    
    // Disable some warnings.
    m_compiler->getDiagnosticOpts().Warnings.push_back("no-unused-value");
    
    // Set the target triple.
    m_compiler->getTargetOpts().Triple = m_target_triple;
    
    // 3. Set up various important bits of infrastructure.
    m_compiler->createDiagnostics(0, 0);
    
    // Create the target instance.
    m_compiler->setTarget(TargetInfo::CreateTargetInfo(m_compiler->getDiagnostics(),
                                                       m_compiler->getTargetOpts()));
    
    assert (m_compiler->hasTarget());
    
    // Inform the target of the language options
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    m_compiler->getTarget().setForcedLangOptions(m_compiler->getLangOpts());
    
    // 4. Set up the diagnostic buffer for reporting errors
    
    m_compiler->getDiagnostics().setClient(new clang::TextDiagnosticBuffer);
    
    // 5. Set up the source management objects inside the compiler
    
    clang::FileSystemOptions file_system_options;
    m_file_manager.reset(new clang::FileManager(file_system_options));
    
    if (!m_compiler->hasSourceManager())
        m_compiler->createSourceManager(*m_file_manager.get());
    
    m_compiler->createFileManager();
    m_compiler->createPreprocessor();
    
    // 6. Most of this we get from the CompilerInstance, but we 
    // also want to give the context an ExternalASTSource.
    m_selector_table.reset(new SelectorTable());
    m_builtin_context.reset(new Builtin::Context(m_compiler->getTarget()));
    
    std::auto_ptr<clang::ASTContext> ast_context(new ASTContext(m_compiler->getLangOpts(),
                                                                m_compiler->getSourceManager(),
                                                                m_compiler->getTarget(),
                                                                m_compiler->getPreprocessor().getIdentifierTable(),
                                                                *m_selector_table.get(),
                                                                *m_builtin_context.get(),
                                                                0));
    
    ClangExpressionDeclMap *decl_map = m_expr.DeclMap();
    
    if (decl_map)
    {
        OwningPtr<clang::ExternalASTSource> ast_source(new ClangASTSource(*ast_context, *decl_map));
        ast_context->setExternalSource(ast_source);
    }
    
    m_compiler->setASTContext(ast_context.release());
    
    std::string module_name("$__lldb_module");

    m_code_generator.reset(CreateLLVMCodeGen(m_compiler->getDiagnostics(),
                                             module_name,
                                             m_compiler->getCodeGenOpts(),
                                             m_compiler->getLLVMContext()));
}

ClangExpressionParser::~ClangExpressionParser()
{
}

unsigned
ClangExpressionParser::Parse (Stream &stream)
{
    TextDiagnosticBuffer *diag_buf = static_cast<TextDiagnosticBuffer*>(m_compiler->getDiagnostics().getClient());
        
    diag_buf->FlushDiagnostics (m_compiler->getDiagnostics());
    
    MemoryBuffer *memory_buffer = MemoryBuffer::getMemBufferCopy(m_expr.Text(), __FUNCTION__);
    FileID memory_buffer_file_id = m_compiler->getSourceManager().createMainFileIDForMemBuffer (memory_buffer);
    
    diag_buf->BeginSourceFile(m_compiler->getLangOpts(), &m_compiler->getPreprocessor());
    
    ASTConsumer *ast_transformer = m_expr.ASTTransformer(m_code_generator.get());
    
    if (ast_transformer)
        ParseAST(m_compiler->getPreprocessor(), ast_transformer, m_compiler->getASTContext());
    else 
        ParseAST(m_compiler->getPreprocessor(), m_code_generator.get(), m_compiler->getASTContext());    
    
    diag_buf->EndSourceFile();
    
    TextDiagnosticBuffer::const_iterator diag_iterator;
    
    int num_errors = 0;
    
    for (diag_iterator = diag_buf->warn_begin();
         diag_iterator != diag_buf->warn_end();
         ++diag_iterator)
        stream.Printf("warning: %s\n", (*diag_iterator).second.c_str());
    
    num_errors = 0;
    
    for (diag_iterator = diag_buf->err_begin();
         diag_iterator != diag_buf->err_end();
         ++diag_iterator)
    {
        num_errors++;
        stream.Printf("error: %s\n", (*diag_iterator).second.c_str());
    }
    
    for (diag_iterator = diag_buf->note_begin();
         diag_iterator != diag_buf->note_end();
         ++diag_iterator)
        stream.Printf("note: %s\n", (*diag_iterator).second.c_str());
    
    return num_errors;
}

static bool FindFunctionInModule (std::string &mangled_name,
                                  llvm::Module *module,
                                  const char *orig_name)
{
    for (llvm::Module::iterator fi = module->getFunctionList().begin(), fe = module->getFunctionList().end();
         fi != fe;
         ++fi)
    {        
        if (fi->getName().str().find(orig_name) != std::string::npos)
        {
            mangled_name = fi->getName().str();
            return true;
        }
    }
    
    return false;
}

Error
ClangExpressionParser::MakeDWARF ()
{
    Error err;
    
    llvm::Module *module = m_code_generator->GetModule();
    
    if (!module)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("IR doesn't contain a module");
        return err;
    }
    
    ClangExpressionVariableStore *local_variables = m_expr.LocalVariables();
    ClangExpressionDeclMap *decl_map = m_expr.DeclMap();
    
    if (!local_variables)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Can't convert an expression without a VariableList to DWARF");
        return err;
    }
    
    if (!decl_map)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Can't convert an expression without a DeclMap to DWARF");
        return err;
    }
    
    std::string function_name;
    
    if (!FindFunctionInModule(function_name, module, m_expr.FunctionName()))
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't find %s() in the module", m_expr.FunctionName());
        return err;
    }
    
    IRToDWARF ir_to_dwarf(*local_variables, decl_map, m_expr.DwarfOpcodeStream(), function_name.c_str());
    
    if (!ir_to_dwarf.runOnModule(*module))
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Couldn't convert the expression to DWARF");
        return err;
    }
    
    err.Clear();
    return err;
}

Error
ClangExpressionParser::MakeJIT (lldb::addr_t &func_addr, 
                                lldb::addr_t &func_end, 
                                ExecutionContext &exe_ctx)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    Error err;
    
    llvm::Module *module = m_code_generator->ReleaseModule();

    if (!module)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("IR doesn't contain a module");
        return err;
    }
    
    // Find the actual name of the function (it's often mangled somehow)
    
    std::string function_name;
    
    if (!FindFunctionInModule(function_name, module, m_expr.FunctionName()))
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't find %s() in the module", m_expr.FunctionName());
        return err;
    }
    else
    {
        if(log)
            log->Printf("Found function %s for %s", function_name.c_str(), m_expr.FunctionName());
    }
    
    ClangExpressionDeclMap *decl_map = m_expr.DeclMap(); // result can be NULL
    
    if (decl_map)
    {
        IRForTarget ir_for_target(decl_map, 
                                  m_expr.NeedsVariableResolution(),
                                  function_name.c_str());
        
        if (!ir_for_target.runOnModule(*module))
        {
            err.SetErrorToGenericError();
            err.SetErrorString("Couldn't convert the expression to DWARF");
            return err;
        }
        
        if (m_expr.NeedsValidation() && exe_ctx.process->GetDynamicCheckers())
        {
            IRDynamicChecks ir_dynamic_checks(*exe_ctx.process->GetDynamicCheckers(), function_name.c_str());
        
            if (!ir_dynamic_checks.runOnModule(*module))
            {
                err.SetErrorToGenericError();
                err.SetErrorString("Couldn't add dynamic checks to the expression");
                return err;
            }
        }
    }
    
    m_jit_mm = new RecordingMemoryManager();
    
    std::string error_string;
        
    llvm::TargetMachine::setRelocationModel(llvm::Reloc::PIC_);
    
    m_execution_engine.reset(llvm::ExecutionEngine::createJIT (module, 
                                                               &error_string, 
                                                               m_jit_mm,
                                                               CodeGenOpt::Less,
                                                               true,
                                                               CodeModel::Small));
        
    if (!m_execution_engine.get())
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't JIT the function: %s", error_string.c_str());
        return err;
    }
    
    m_execution_engine->DisableLazyCompilation();
    
    llvm::Function *function = module->getFunction (function_name.c_str());
    
    // We don't actually need the function pointer here, this just forces it to get resolved.
    
    void *fun_ptr = m_execution_engine->getPointerToFunction(function);
    
    // Errors usually cause failures in the JIT, but if we're lucky we get here.
    
    if (!fun_ptr)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Couldn't JIT the function");
        return err;
    }
    
    m_jitted_functions.push_back (ClangExpressionParser::JittedFunction(function_name.c_str(), (lldb::addr_t)fun_ptr));
    
    ExecutionContext &exc_context(exe_ctx);
    
    if (exc_context.process == NULL)
    {
        err.SetErrorToGenericError();
        err.SetErrorString("Couldn't write the JIT compiled code into the target because there is no target");
        return err;
    }
    
    // Look over the regions allocated for the function compiled.  The JIT
    // tries to allocate the functions & stubs close together, so we should try to
    // write them that way too...
    // For now I only write functions with no stubs, globals, exception tables,
    // etc.  So I only need to write the functions.
    
    size_t alloc_size = 0;
    
    std::map<uint8_t *, uint8_t *>::iterator fun_pos = m_jit_mm->m_functions.begin();
    std::map<uint8_t *, uint8_t *>::iterator fun_end = m_jit_mm->m_functions.end();
    
    for (; fun_pos != fun_end; ++fun_pos)
        alloc_size += (*fun_pos).second - (*fun_pos).first;
    
    Error alloc_error;
    lldb::addr_t target_addr = exc_context.process->AllocateMemory (alloc_size, lldb::ePermissionsReadable|lldb::ePermissionsExecutable, alloc_error);
    
    if (target_addr == LLDB_INVALID_ADDRESS)
    {
        err.SetErrorToGenericError();
        err.SetErrorStringWithFormat("Couldn't allocate memory for the JITted function: %s", alloc_error.AsCString("unknown error"));
        return err;
    }
    
    lldb::addr_t cursor = target_addr;
        
    for (fun_pos = m_jit_mm->m_functions.begin(); fun_pos != fun_end; fun_pos++)
    {
        lldb::addr_t lstart = (lldb::addr_t) (*fun_pos).first;
        lldb::addr_t lend = (lldb::addr_t) (*fun_pos).second;
        size_t size = lend - lstart;
        
        Error write_error;
        
        if (exc_context.process->WriteMemory(cursor, (void *) lstart, size, write_error) != size)
        {
            err.SetErrorToGenericError();
            err.SetErrorStringWithFormat("Couldn't copy JITted function into the target: %s", write_error.AsCString("unknown error"));
            return err;
        }
            
        m_jit_mm->AddToLocalToRemoteMap (lstart, size, cursor);
        cursor += size;
    }
    
    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();
    
    for (pos = m_jitted_functions.begin(); pos != end; pos++)
    {
        (*pos).m_remote_addr = m_jit_mm->GetRemoteAddressForLocal ((*pos).m_local_addr);
    
        if (!(*pos).m_name.compare(function_name.c_str()))
        {
            func_end = m_jit_mm->GetRemoteRangeForLocal ((*pos).m_local_addr).second;
            func_addr = (*pos).m_remote_addr;
        }
    }
    
    if (log)
    {
        log->Printf("Code can be run in the target.");
        
        StreamString disassembly_stream;
        
        Error err = DisassembleFunction(disassembly_stream, exe_ctx);
        
        if (!err.Success())
        {
            log->Printf("Couldn't disassemble function : %s", err.AsCString("unknown error"));
        }
        else
        {
            log->Printf("Function disassembly:\n%s", disassembly_stream.GetData());
        }
    }
    
    err.Clear();
    return err;
}

Error
ClangExpressionParser::DisassembleFunction (Stream &stream, ExecutionContext &exe_ctx)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    const char *name = m_expr.FunctionName();
    
    Error ret;
    
    ret.Clear();
    
    lldb::addr_t func_local_addr = LLDB_INVALID_ADDRESS;
    lldb::addr_t func_remote_addr = LLDB_INVALID_ADDRESS;
    
    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();
    
    for (pos = m_jitted_functions.begin(); pos < end; pos++)
    {
        if (strstr(pos->m_name.c_str(), name))
        {
            func_local_addr = pos->m_local_addr;
            func_remote_addr = pos->m_remote_addr;
        }
    }
    
    if (func_local_addr == LLDB_INVALID_ADDRESS)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't find function %s for disassembly", name);
        return ret;
    }
    
    if(log)
        log->Printf("Found function, has local address 0x%llx and remote address 0x%llx", (uint64_t)func_local_addr, (uint64_t)func_remote_addr);
    
    std::pair <lldb::addr_t, lldb::addr_t> func_range;
    
    func_range = m_jit_mm->GetRemoteRangeForLocal(func_local_addr);
    
    if (func_range.first == 0 && func_range.second == 0)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't find code range for function %s", name);
        return ret;
    }
    
    if(log)
        log->Printf("Function's code range is [0x%llx-0x%llx]", func_range.first, func_range.second);
    
    if (!exe_ctx.target)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorString("Couldn't find the target");
    }
    
    lldb::DataBufferSP buffer_sp(new DataBufferHeap(func_range.second - func_remote_addr, 0));
    
    Error err;
    exe_ctx.process->ReadMemory(func_remote_addr, buffer_sp->GetBytes(), buffer_sp->GetByteSize(), err);
    
    if (!err.Success())
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Couldn't read from process: %s", err.AsCString("unknown error"));
        return ret;
    }
    
    ArchSpec arch(exe_ctx.target->GetArchitecture());
    
    Disassembler *disassembler = Disassembler::FindPlugin(arch);
    
    if (disassembler == NULL)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorStringWithFormat("Unable to find disassembler plug-in for %s architecture.", arch.AsCString());
        return ret;
    }
    
    if (!exe_ctx.process)
    {
        ret.SetErrorToGenericError();
        ret.SetErrorString("Couldn't find the process");
        return ret;
    }
    
    DataExtractor extractor(buffer_sp, 
                            exe_ctx.process->GetByteOrder(),
                            exe_ctx.target->GetArchitecture().GetAddressByteSize());
    
    if (log)
    {
        log->Printf("Function data has contents:");
        extractor.PutToLog (log.get(),
                            0,
                            extractor.GetByteSize(),
                            func_remote_addr,
                            16,
                            DataExtractor::TypeUInt8);
    }
    
    disassembler->DecodeInstructions (Address (NULL, func_remote_addr), extractor, 0, UINT32_MAX);
    
    InstructionList &instruction_list = disassembler->GetInstructionList();
    
    uint32_t bytes_offset = 0;
    
    for (uint32_t instruction_index = 0, num_instructions = instruction_list.GetSize(); 
         instruction_index < num_instructions; 
         ++instruction_index)
    {
        Instruction *instruction = instruction_list.GetInstructionAtIndex(instruction_index).get();
        instruction->Dump (&stream,
                           true,
                           &extractor, 
                           bytes_offset, 
                           &exe_ctx, 
                           true);
        stream.PutChar('\n');
        bytes_offset += instruction->GetByteSize();
    }
    
    return ret;
}
