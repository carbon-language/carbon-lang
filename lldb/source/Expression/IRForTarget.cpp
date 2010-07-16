//===-- IRForTarget.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/IRForTarget.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"

#include "clang/AST/ASTContext.h"

#include "lldb/Core/dwarf.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"

#include <map>

using namespace llvm;

IRForTarget::IRForTarget(const void *pid,
                         lldb_private::ClangExpressionDeclMap *decl_map,
                         const llvm::TargetData *target_data) :
    ModulePass(pid),
    m_decl_map(decl_map),
    m_target_data(target_data)
{
}

IRForTarget::~IRForTarget()
{
}

static clang::NamedDecl *
DeclForGlobalValue(llvm::Module &module,
                   llvm::GlobalValue *global_value)
{
    NamedMDNode *named_metadata = module.getNamedMetadata("clang.global.decl.ptrs");
    
    if (!named_metadata)
        return NULL;
    
    unsigned num_nodes = named_metadata->getNumOperands();
    unsigned node_index;
    
    for (node_index = 0;
         node_index < num_nodes;
         ++node_index)
    {
        MDNode *metadata_node = named_metadata->getOperand(node_index);
        
        if (!metadata_node)
            return NULL;
        
        if (metadata_node->getNumOperands() != 2)
            return NULL;
        
        if (metadata_node->getOperand(0) != global_value)
            continue;
        
        ConstantInt *constant_int = dyn_cast<ConstantInt>(metadata_node->getOperand(1));
        
        if (!constant_int)
            return NULL;
        
        uintptr_t ptr = constant_int->getZExtValue();
        
        return reinterpret_cast<clang::NamedDecl *>(ptr);
    }
    
    return NULL;
}

bool 
IRForTarget::MaybeHandleVariable(Module &M, 
                                 lldb_private::ClangExpressionDeclMap *DM,
                                 llvm::Value *V,
                                 bool Store)
{
    if (GlobalVariable *global_variable = dyn_cast<GlobalVariable>(V))
    {        
        clang::NamedDecl *named_decl = DeclForGlobalValue(M, global_variable);
        
        std::string name = named_decl->getName().str();
        
        void *qual_type = NULL;
        
        if (clang::ValueDecl *value_decl = dyn_cast<clang::ValueDecl>(named_decl))
            qual_type = value_decl->getType().getAsOpaquePtr();
        else
            return false;
        
        const llvm::Type *value_type = global_variable->getType();
        
        size_t value_size = m_target_data->getTypeStoreSize(value_type);
        off_t value_alignment = m_target_data->getPrefTypeAlignment(value_type);
        
        if (named_decl && !DM->AddValueToStruct(V, 
                                                named_decl,
                                                name,
                                                qual_type,
                                                value_size, 
                                                value_alignment))
            return false;
    }
    
    return true;
}

bool
IRForTarget::runOnBasicBlock(Module &M, BasicBlock &BB)
{        
    /////////////////////////////////////////////////////////////////////////
    // Prepare the current basic block for execution in the remote process
    //
    
    llvm::BasicBlock::iterator ii;

    for (ii = BB.begin();
         ii != BB.end();
         ++ii)
    {
        Instruction &inst = *ii;
        
        if (LoadInst *load = dyn_cast<LoadInst>(&inst))
            if (!MaybeHandleVariable(M, m_decl_map, load->getPointerOperand(), false))
                return false;
        
        if (StoreInst *store = dyn_cast<StoreInst>(&inst))
            if (!MaybeHandleVariable(M, m_decl_map, store->getPointerOperand(), false))
                return false;
    }
    
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

// UnfoldConstant operates on a constant [C] which has just been replaced with a value
// [new_value].  We assume that new_value has been properly placed early in the function,
// most likely somewhere in front of the first instruction in the entry basic block 
// [first_entry_instruction].  
//
// UnfoldConstant reads through the uses of C and replaces C in those uses with new_value.
// Where those uses are constants, the function generates new instructions to compute the
// result of the new, non-constant expression and places them before first_entry_instruction.  
// These instructions replace the constant uses, so UnfoldConstant calls itself recursively
// for those.

static bool
UnfoldConstant(llvm::Constant *C, llvm::Value *new_value, llvm::Instruction *first_entry_instruction)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    Value::use_iterator ui;
    
    for (ui = C->use_begin();
         ui != C->use_end();
         ++ui)
    {
        User *user = *ui;
        
        if (Constant *constant = dyn_cast<Constant>(user))
        {
            // synthesize a new non-constant equivalent of the constant
            
            if (ConstantExpr *constant_expr = dyn_cast<ConstantExpr>(constant))
            {
                switch (constant_expr->getOpcode())
                {
                default:
                    if (log)
                        log->Printf("Unhandled constant expression type: %s", PrintValue(constant_expr).c_str());
                    return false;
                case Instruction::BitCast:
                    {
                        // UnaryExpr
                        //   OperandList[0] is value
                        
                        Value *s = constant_expr->getOperand(0);
                        
                        if (s == C)
                            s = new_value;
                        
                        BitCastInst *bit_cast(new BitCastInst(s, C->getType(), "", first_entry_instruction));
                        
                        UnfoldConstant(constant_expr, bit_cast, first_entry_instruction);
                    }
                    break;
                case Instruction::GetElementPtr:
                    {
                        // GetElementPtrConstantExpr
                        //   OperandList[0] is base
                        //   OperandList[1]... are indices
                        
                        Value *ptr = constant_expr->getOperand(0);
                        
                        if (ptr == C)
                            ptr = new_value;
                        
                        SmallVector<Value*, 16> indices;
                        
                        unsigned operand_index;
                        unsigned num_operands = constant_expr->getNumOperands();
                        
                        for (operand_index = 1;
                             operand_index < num_operands;
                             ++operand_index)
                        {
                            Value *operand = constant_expr->getOperand(operand_index);
                            
                            if (operand == C)
                                operand = new_value;
                            
                            indices.push_back(operand);
                        }
                        
                        GetElementPtrInst *get_element_ptr(GetElementPtrInst::Create(ptr, indices.begin(), indices.end(), "", first_entry_instruction));
                        
                        UnfoldConstant(constant_expr, get_element_ptr, first_entry_instruction);
                    }
                    break;
                }
            }
            else
            {
                if (log)
                    log->Printf("Unhandled constant type: %s", PrintValue(constant).c_str());
                return false;
            }
        }
        else
        {
            // simple fall-through case for non-constants
            user->replaceUsesOfWith(C, new_value);
        }
    }
    
    return true;
}

bool 
IRForTarget::replaceVariables(Module &M, Function *F)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);

    m_decl_map->DoStructLayout();
    
    if (log)
        log->Printf("Element arrangement:");
    
    uint32_t num_elements;
    uint32_t element_index;
    
    size_t size;
    off_t alignment;
    
    if (!m_decl_map->GetStructInfo (num_elements, size, alignment))
        return false;
    
    Function::arg_iterator iter(F->getArgumentList().begin());
    
    if (iter == F->getArgumentList().end())
        return false;
    
    llvm::Argument *argument = iter;
    
    if (!argument->getName().equals("___clang_arg"))
        return false;
    
    if (log)
        log->Printf("Arg: %s", PrintValue(argument).c_str());
    
    llvm::BasicBlock &entry_block(F->getEntryBlock());
    llvm::Instruction *first_entry_instruction(entry_block.getFirstNonPHIOrDbg());
    
    if (!first_entry_instruction)
        return false;
    
    LLVMContext &context(M.getContext());
    const IntegerType *offset_type(Type::getInt32Ty(context));
    
    if (!offset_type)
        return false;
        
    for (element_index = 0; element_index < num_elements; ++element_index)
    {
        const clang::NamedDecl *decl;
        llvm::Value *value;
        off_t offset;
        
        if (!m_decl_map->GetStructElement (decl, value, offset, element_index))
            return false;
        
        if (log)
            log->Printf("  %s (%s) placed at %d",
                        decl->getIdentifier()->getNameStart(),
                        PrintValue(value, true).c_str(),
                        offset);
        
        ConstantInt *offset_int(ConstantInt::getSigned(offset_type, offset));
        GetElementPtrInst *get_element_ptr = GetElementPtrInst::Create(argument, offset_int, "", first_entry_instruction);
        BitCastInst *bit_cast = new BitCastInst(get_element_ptr, value->getType(), "", first_entry_instruction);
        
        if (Constant *constant = dyn_cast<Constant>(value))
            UnfoldConstant(constant, bit_cast, first_entry_instruction);
        else
            value->replaceAllUsesWith(bit_cast);
    }
    
    if (log)
        log->Printf("Total structure [align %d, size %d]", alignment, size);
    
    return true;
}

bool
IRForTarget::runOnModule(Module &M)
{
    lldb_private::Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    
    llvm::Function* function = M.getFunction(StringRef("___clang_expr"));
    
    if (!function)
    {
        if (log)
            log->Printf("Couldn't find ___clang_expr() in the module");
        
        return false;
    }
        
    llvm::Function::iterator bbi;
    
    for (bbi = function->begin();
         bbi != function->end();
         ++bbi)
    {
        if (!runOnBasicBlock(M, *bbi))
            return false;
    }
    
    if (!replaceVariables(M, function))
        return false;
    
    if (log)
    {
        for (bbi = function->begin();
             bbi != function->end();
             ++bbi)
        {
            log->Printf("Rewrote basic block %s for running: \n%s", 
                        bbi->hasName() ? bbi->getNameStr().c_str() : "[anonymous]",
                        PrintValue(bbi).c_str());
        }
        
    }
    
    return true;    
}

void
IRForTarget::assignPassManager(PMStack &PMS,
                               PassManagerType T)
{
}

PassManagerType
IRForTarget::getPotentialPassManagerType() const
{
    return PMT_ModulePassManager;
}
