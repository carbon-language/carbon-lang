//===-- IRDynamicChecks.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/IRDynamicChecks.h"
#include "lldb/Expression/ClangUtilityFunction.h"

#include "lldb/Core/Log.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Value.h"

using namespace llvm;
using namespace lldb_private;

static char ID;

static const char valid_pointer_check_text[] = 
    "extern \"C\" void "
    "___clang_valid_pointer_check (unsigned char *ptr)"
    "{"
        "unsigned char val = *ptr;"
    "}";

static const char valid_pointer_check_name[] = 
    "___clang_valid_pointer_check";

DynamicCheckerFunctions::DynamicCheckerFunctions ()
{
    m_valid_pointer_check.reset(new ClangUtilityFunction(valid_pointer_check_text,
                                                         valid_pointer_check_name));
}

DynamicCheckerFunctions::~DynamicCheckerFunctions ()
{
}

bool
DynamicCheckerFunctions::Install(Stream &error_stream,
                                 ExecutionContext &exe_ctx)
{
    if (!m_valid_pointer_check->Install(error_stream, exe_ctx))
        return false;
        
    return true;
}

IRDynamicChecks::IRDynamicChecks(DynamicCheckerFunctions &checker_functions,
                                 const char *func_name) :
    ModulePass(&ID),
    m_checker_functions(checker_functions),
    m_func_name(func_name)
{
}

/* A handy utility function used at several places in the code */

static std::string 
PrintValue(llvm::Value *V, bool truncate = false)
{
    std::string s;
    raw_string_ostream rso(s);
    V->print(rso);
    rso.flush();
    if (truncate)
        s.resize(s.length() - 1);
    return s;
}

IRDynamicChecks::~IRDynamicChecks()
{
}

bool
IRDynamicChecks::runOnModule(llvm::Module &M)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    llvm::Function* function = M.getFunction(StringRef(m_func_name.c_str()));
    
    if (!function)
    {
        if (log)
            log->Printf("Couldn't find %s() in the module", m_func_name.c_str());
        
        return false;
    }
    
    llvm::Function::iterator bbi;
    
    for (bbi = function->begin();
         bbi != function->end();
         ++bbi)
    {
    }
    
    return true;    
}

void
IRDynamicChecks::assignPassManager(PMStack &PMS,
                                   PassManagerType T)
{
}

PassManagerType
IRDynamicChecks::getPotentialPassManagerType() const
{
    return PMT_ModulePassManager;
}
