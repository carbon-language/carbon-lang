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
#include "clang/Rewrite/FrontendActions.h"
#include "clang/Sema/ParseAST.h"
#include "clang/Sema/SemaConsumer.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/System/Host.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetSelect.h"

// Project includes
#include "lldb/Core/Log.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Expression/ClangExpression.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangResultSynthesizer.h"
#include "lldb/Expression/IRForTarget.h"
#include "lldb/Expression/IRToDWARF.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Expression/RecordingMemoryManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "lldb/Core/StreamString.h"
#include "lldb/Host/Mutex.h"


using namespace lldb_private;
using namespace clang;
using namespace llvm;


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

//----------------------------------------------------------------------
// ClangExpression constructor
//----------------------------------------------------------------------
ClangExpression::ClangExpression(const char *target_triple,
                                 ClangExpressionDeclMap *decl_map) :
    m_target_triple (),
    m_decl_map (decl_map),
    m_clang_ap (),
    m_code_generator_ptr (NULL),
    m_jit_mm_ptr (NULL),
    m_execution_engine (),
    m_jitted_functions ()
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
ClangExpression::CreateCompilerInstance ()
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
    m_clang_ap->getLangOpts().ThreadsafeStatics = false;
    m_clang_ap->getLangOpts().AccessControl = false; // Debuggers get universal access
    m_clang_ap->getLangOpts().DollarIdents = true; // $ indicates a persistent variable name
    
    // Set CodeGen options
    m_clang_ap->getCodeGenOpts().EmitDeclMetadata = true;
    m_clang_ap->getCodeGenOpts().InstrumentFunctions = false;

    // Disable some warnings.
    m_clang_ap->getDiagnosticOpts().Warnings.push_back("no-unused-value");
    
    // Set the target triple.
    m_clang_ap->getTargetOpts().Triple = m_target_triple;
    
    // 3. Set up various important bits of infrastructure.
    
    m_clang_ap->createDiagnostics(0, 0);

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
ClangExpression::ParseExpression (const char *expr_text,
                                  Stream &stream,
                                  bool add_result_var)
{
    // HACK: for now we have to make a function body around our expression
    // since there is no way to parse a single expression line in LLVM/Clang.
    std::string func_expr("extern \"C\" void ___clang_expr(void *___clang_arg)\n{\n\t");
    func_expr.append(expr_text);
    func_expr.append(";\n}");
    return ParseBareExpression (func_expr, stream, add_result_var);

}

unsigned
ClangExpression::ParseBareExpression (llvm::StringRef expr_text, 
                                      Stream &stream,
                                      bool add_result_var)
{
    Mutex::Locker locker(GetClangMutex ());

    TextDiagnosticBuffer text_diagnostic_buffer;

    if (!CreateCompilerInstance ())
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
                                         *builtin_ap.get(),
                                         0);
    
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
    
    if (add_result_var)
    {
        ClangResultSynthesizer result_synthesizer(m_code_generator_ptr);
        ParseAST(m_clang_ap->getPreprocessor(), &result_synthesizer, m_clang_ap->getASTContext());
    }
    else 
    {
        ParseAST(m_clang_ap->getPreprocessor(), m_code_generator_ptr, m_clang_ap->getASTContext());
    }

    
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

bool
ClangExpression::ConvertIRToDWARF (ClangExpressionVariableList &expr_local_variable_list,
                                   StreamString &dwarf_opcode_strm)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    llvm::Module *module = m_code_generator_ptr->GetModule();
        
    if (!module)
    {
        if (log)
            log->Printf("IR doesn't contain a module");
        
        return 1;
    }
    
    IRToDWARF ir_to_dwarf(expr_local_variable_list, m_decl_map, dwarf_opcode_strm);
    
    return ir_to_dwarf.runOnModule(*module);
}

bool
ClangExpression::PrepareIRForTarget ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    llvm::Module *module = m_code_generator_ptr->GetModule();
    
    if (!module)
    {
        if (log)
            log->Printf("IR doesn't contain a module");
        
        return 1;
    }
    
    llvm::Triple target_triple = m_clang_ap->getTarget().getTriple();
    
    std::string err;
    
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(m_target_triple, err);
    
    if (!target)
    {
        if (log)
            log->Printf("Couldn't find a target for %s", m_target_triple.c_str());
        
        return 1;
    }
    
    std::auto_ptr<llvm::TargetMachine> target_machine(target->createTargetMachine(m_target_triple, ""));
    
    IRForTarget ir_for_target(m_decl_map, target_machine->getTargetData());
    
    return ir_for_target.runOnModule(*module);
}

bool
ClangExpression::JITFunction (const char *name)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    llvm::Module *module = m_code_generator_ptr->GetModule();

    if (module)
    {
        std::string error;

        if (m_jit_mm_ptr == NULL)
            m_jit_mm_ptr = new RecordingMemoryManager();

        //llvm::InitializeNativeTarget();
        
        if (log)
        {
            const char *relocation_model_string;
            
            switch (llvm::TargetMachine::getRelocationModel())
            {
                case llvm::Reloc::Default:
                    relocation_model_string = "Default";
                    break;
                case llvm::Reloc::Static:
                    relocation_model_string = "Static";
                    break;
                case llvm::Reloc::PIC_:
                    relocation_model_string = "PIC_";
                    break;
                case llvm::Reloc::DynamicNoPIC:
                    relocation_model_string = "DynamicNoPIC";
                    break;
            }
            
            log->Printf("Target machine's relocation model: %s", relocation_model_string);
        }
                        
        if (m_execution_engine.get() == 0)
            m_execution_engine.reset(llvm::ExecutionEngine::createJIT (module, 
                                                                       &error, 
                                                                       m_jit_mm_ptr,
                                                                       CodeGenOpt::Default,
                                                                       true,
                                                                       CodeModel::Small)); // set to small so RIP-relative relocations work in PIC
        
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    if (m_jit_mm_ptr == NULL)
        return false;

    if (exc_context.process == NULL)
        return false;

    // Look over the regions allocated for the function compiled.  The JIT
    // tries to allocate the functions & stubs close together, so we should try to
    // write them that way too...
    // For now I only write functions with no stubs, globals, exception tables,
    // etc.  So I only need to write the functions.

    size_t alloc_size = 0;
    std::map<uint8_t *, uint8_t *>::iterator fun_pos, fun_end = m_jit_mm_ptr->m_functions.end();
    for (fun_pos = m_jit_mm_ptr->m_functions.begin(); fun_pos != fun_end; fun_pos++)
    {
        alloc_size += (*fun_pos).second - (*fun_pos).first;
    }

    Error error;
    lldb::addr_t target_addr = exc_context.process->AllocateMemory (alloc_size, lldb::ePermissionsReadable|lldb::ePermissionsExecutable, error);

    if (target_addr == LLDB_INVALID_ADDRESS)
        return false;

    lldb::addr_t cursor = target_addr;
    for (fun_pos = m_jit_mm_ptr->m_functions.begin(); fun_pos != fun_end; fun_pos++)
    {
        if (log)
            log->Printf("Reading [%p-%p] from m_functions", fun_pos->first, fun_pos->second);
        
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

Error
ClangExpression::DisassembleFunction (Stream &stream, ExecutionContext &exe_ctx, const char *name)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    Error ret;
    
    ret.Clear();
    
    lldb::addr_t func_local_addr = LLDB_INVALID_ADDRESS;
    lldb::addr_t func_remote_addr = LLDB_INVALID_ADDRESS;
    
    std::vector<JittedFunction>::iterator pos, end = m_jitted_functions.end();
    
    for (pos = m_jitted_functions.begin(); pos < end; pos++)
    {
        if (strcmp(pos->m_name.c_str(), name) == 0)
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
    
    func_range = m_jit_mm_ptr->GetRemoteRangeForLocal(func_local_addr);
    
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
    
    if(log)
    {
        log->Printf("Function data has contents:");
        extractor.PutToLog (log,
                            0,
                            extractor.GetByteSize(),
                            func_remote_addr,
                            16,
                            DataExtractor::TypeUInt8);
    }
            
    disassembler->DecodeInstructions(extractor, 0, UINT32_MAX);
    
    Disassembler::InstructionList &instruction_list = disassembler->GetInstructionList();
    
    uint32_t bytes_offset = 0;
    
    for (uint32_t instruction_index = 0, num_instructions = instruction_list.GetSize(); 
         instruction_index < num_instructions; 
         ++instruction_index)
    {
        Disassembler::Instruction *instruction = instruction_list.GetInstructionAtIndex(instruction_index);
        Address addr(NULL, func_remote_addr + bytes_offset);
        instruction->Dump (&stream,
                           &addr,
                           &extractor, 
                           bytes_offset, 
                           exe_ctx, 
                           true);
        stream.PutChar('\n');
        bytes_offset += instruction->GetByteSize();
    }
    
    return ret;
}
