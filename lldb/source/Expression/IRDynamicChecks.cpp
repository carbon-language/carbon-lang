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
#include "llvm/Instructions.h"
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

//----------------------------------------------------------------------
/// @class Instrumenter IRDynamicChecks.cpp
/// @brief Finds and instruments individual LLVM IR instructions
///
/// When instrumenting LLVM IR, it is frequently desirable to first search
/// for instructions, and then later modify them.  This way iterators
/// remain intact, and multiple passes can look at the same code base without
/// treading on each other's toes.
///
/// The Instrumenter class implements this functionality.  A client first
/// calls Inspect on a function, which populates a list of instructions to
/// be instrumented.  Then, later, when all passes' Inspect functions have
/// been called, the client calls Instrument, which adds the desired
/// instrumentation.
///
/// A subclass of Instrumenter must override InstrumentInstruction, which
/// is responsible for adding whatever instrumentation is necessary.
///
/// A subclass of Instrumenter may override:
///
/// - InspectInstruction [default: does nothing]
///
/// - InspectBasicBlock [default: iterates through the instructions in a 
///   basic block calling InspectInstruction]
///
/// - InspectFunction [default: iterates through the basic blocks in a 
///   function calling InspectBasicBlock]
//----------------------------------------------------------------------
class Instrumenter {
public:
    //------------------------------------------------------------------
    /// Constructor
    ///
    /// @param[in] module
    ///     The module being instrumented.
    //------------------------------------------------------------------
    Instrumenter (llvm::Module &module,
                  DynamicCheckerFunctions &checker_functions) :
        m_module(module),
        m_checker_functions(checker_functions)
    {
    }
    
    //------------------------------------------------------------------
    /// Inspect a function to find instructions to instrument
    ///
    /// @param[in] function
    ///     The function to inspect.
    ///
    /// @return
    ///     True on success; false on error.
    //------------------------------------------------------------------
    bool Inspect (llvm::Function &function)
    {
        return InspectFunction(function);
    }
    
    //------------------------------------------------------------------
    /// Instrument all the instructions found by Inspect()
    ///
    /// @return
    ///     True on success; false on error.
    //------------------------------------------------------------------
    bool Instrument ()
    {
        for (InstIterator ii = m_to_instrument.begin(), last_ii = m_to_instrument.end();
             ii != last_ii;
             ++ii)
        {
            if (!InstrumentInstruction(*ii))
                return false;
        }
        
        return true;
    }
protected:
    //------------------------------------------------------------------
    /// Add instrumentation to a single instruction
    ///
    /// @param[in] inst
    ///     The instruction to be instrumented. 
    ///
    /// @return
    ///     True on success; false otherwise.
    //------------------------------------------------------------------
    virtual bool InstrumentInstruction(llvm::Instruction *inst) = 0;
    
    //------------------------------------------------------------------
    /// Register a single instruction to be instrumented
    ///
    /// @param[in] inst
    ///     The instruction to be instrumented.
    //------------------------------------------------------------------
    void RegisterInstruction(llvm::Instruction &i)
    {
        m_to_instrument.push_back(&i);
    }
    
    //------------------------------------------------------------------
    /// Determine whether a single instruction is interesting to
    /// instrument, and, if so, call RegisterInstruction
    ///
    /// @param[in] i
    ///     The instruction to be inspected.
    ///
    /// @return
    ///     False if there was an error scanning; true otherwise.
    //------------------------------------------------------------------
    virtual bool InspectInstruction(llvm::Instruction &i)
    {
        return true;
    }
    
    //------------------------------------------------------------------
    /// Scan a basic block to see if any instructions are interesting
    ///
    /// @param[in] bb
    ///     The basic block to be inspected.
    ///
    /// @return
    ///     False if there was an error scanning; true otherwise.
    //------------------------------------------------------------------
    virtual bool InspectBasicBlock(llvm::BasicBlock &bb)
    {
        for (llvm::BasicBlock::iterator ii = bb.begin(), last_ii = bb.end();
             ii != last_ii;
             ++ii)
        {
            if (!InspectInstruction(*ii))
                return false;
        }
        
        return true;
    }
    
    //------------------------------------------------------------------
    /// Scan a function to see if any instructions are interesting
    ///
    /// @param[in] f
    ///     The function to be inspected. 
    ///
    /// @return
    ///     False if there was an error scanning; true otherwise.
    //------------------------------------------------------------------
    virtual bool InspectFunction(llvm::Function &f)
    {
        for (llvm::Function::iterator bbi = f.begin(), last_bbi = f.end();
             bbi != last_bbi;
             ++bbi)
        {
            if (!InspectBasicBlock(*bbi))
                return false;
        }
        
        return true;
    }
    
    typedef std::vector <llvm::Instruction *>   InstVector;
    typedef InstVector::iterator                InstIterator;
    
    InstVector                  m_to_instrument;        ///< List of instructions the inspector found
    llvm::Module               &m_module;               ///< The module which is being instrumented
    DynamicCheckerFunctions    &m_checker_functions;    ///< The dynamic checker functions for the process
};

class ValidPointerChecker : public Instrumenter
{
public:
    ValidPointerChecker(llvm::Module &module,
                        DynamicCheckerFunctions &checker_functions) :
        Instrumenter(module, checker_functions),
        m_valid_pointer_check_func(NULL)
    {
    }
private:
    bool InstrumentInstruction(llvm::Instruction *inst)
    {
        lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

        if(log)
            log->Printf("Instrumenting load/store instruction: %s\n", 
                        PrintValue(inst).c_str());
        
        if (!m_valid_pointer_check_func)
        {
            std::vector<const llvm::Type*> params;
            
            const IntegerType *intptr_ty = llvm::Type::getIntNTy(m_module.getContext(),
                                                                 (m_module.getPointerSize() == llvm::Module::Pointer64) ? 64 : 32);
            
            m_i8ptr_ty = llvm::Type::getInt8PtrTy(m_module.getContext());
            
            params.push_back(m_i8ptr_ty);
            
            FunctionType *fun_ty = FunctionType::get(llvm::Type::getVoidTy(m_module.getContext()), params, true);
            PointerType *fun_ptr_ty = PointerType::getUnqual(fun_ty);
            Constant *fun_addr_int = ConstantInt::get(intptr_ty, m_checker_functions.m_valid_pointer_check->StartAddress(), false);
            m_valid_pointer_check_func = ConstantExpr::getIntToPtr(fun_addr_int, fun_ptr_ty);
        }
        
        llvm::Value *dereferenced_ptr;
        
        if (llvm::LoadInst *li = dyn_cast<llvm::LoadInst> (inst))
            dereferenced_ptr = li->getPointerOperand();
        else if (llvm::StoreInst *si = dyn_cast<llvm::StoreInst> (inst))
            dereferenced_ptr = si->getPointerOperand();
        else
            return false;
        
        // Insert an instruction to cast the loaded value to int8_t*
        
        BitCastInst *bit_cast = new BitCastInst(dereferenced_ptr,
                                                m_i8ptr_ty,
                                                "",
                                                inst);
        
        // Insert an instruction to call the helper with the result
        
        SmallVector <llvm::Value*, 1> args;
        args.push_back(bit_cast);
        
        CallInst::Create(m_valid_pointer_check_func, 
                         args.begin(),
                         args.end(),
                         "",
                         inst);
            
        return true;
    }
    
    bool InspectInstruction(llvm::Instruction &i)
    {
        if (dyn_cast<llvm::LoadInst> (&i) ||
            dyn_cast<llvm::StoreInst> (&i))
            RegisterInstruction(i);
        
        return true;
    }
    
    llvm::Value         *m_valid_pointer_check_func;
    const PointerType   *m_i8ptr_ty;
};

IRDynamicChecks::IRDynamicChecks(DynamicCheckerFunctions &checker_functions,
                                 const char *func_name) :
    ModulePass(&ID),
    m_checker_functions(checker_functions),
    m_func_name(func_name)
{
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

    ValidPointerChecker vpc(M, m_checker_functions);
    
    if (!vpc.Inspect(*function))
        return false;
    
    if (!vpc.Instrument())
        return false;
    
    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        M.print(oss, NULL);
        
        oss.flush();
        
        log->Printf("Module after dynamic checks: \n%s", s.c_str());
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
