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
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ValueSymbolTable.h"

#include "clang/AST/ASTContext.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"

#include <map>

using namespace llvm;

static char ID;

IRForTarget::IRForTarget (lldb_private::ClangExpressionDeclMap *decl_map,
                          bool resolve_vars,
                          const char *func_name) :
    ModulePass(ID),
    m_decl_map(decl_map),
    m_CFStringCreateWithBytes(NULL),
    m_sel_registerName(NULL),
    m_func_name(func_name),
    m_resolve_vars(resolve_vars)
{
}

/* Handy utility functions used at several places in the code */

static std::string 
PrintValue(const Value *value, bool truncate = false)
{
    std::string s;
    raw_string_ostream rso(s);
    value->print(rso);
    rso.flush();
    if (truncate)
        s.resize(s.length() - 1);
    return s;
}

static std::string
PrintType(const Type *type, bool truncate = false)
{
    std::string s;
    raw_string_ostream rso(s);
    type->print(rso);
    rso.flush();
    if (truncate)
        s.resize(s.length() - 1);
    return s;
}

IRForTarget::~IRForTarget()
{
}

bool 
IRForTarget::CreateResultVariable (llvm::Module &llvm_module, llvm::Function &llvm_function)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (!m_resolve_vars)
        return true;
    
    // Find the result variable.  If it doesn't exist, we can give up right here.
    
    ValueSymbolTable& value_symbol_table = llvm_module.getValueSymbolTable();
    
    const char *result_name = NULL;
    
    for (ValueSymbolTable::iterator vi = value_symbol_table.begin(), ve = value_symbol_table.end();
         vi != ve;
         ++vi)
    {
        if (strstr(vi->first(), "$__lldb_expr_result") &&
            !strstr(vi->first(), "GV")) 
        {
            result_name = vi->first();
            break;
        }
    }
    
    if (!result_name)
    {
        if (log)
            log->PutCString("Couldn't find result variable");
        
        return true;
    }
    
    if (log)
        log->Printf("Result name: \"%s\"", result_name);
    
    Value *result_value = llvm_module.getNamedValue(result_name);
    
    if (!result_value)
    {
        if (log)
            log->PutCString("Result variable had no data");
                
        return false;
    }
        
    if (log)
        log->Printf("Found result in the IR: \"%s\"", PrintValue(result_value, false).c_str());
    
    GlobalVariable *result_global = dyn_cast<GlobalVariable>(result_value);
    
    if (!result_global)
    {
        if (log)
            log->PutCString("Result variable isn't a GlobalVariable");
        return false;
    }
    
    // Find the metadata and follow it to the VarDecl
    
    NamedMDNode *named_metadata = llvm_module.getNamedMetadata("clang.global.decl.ptrs");
    
    if (!named_metadata)
    {
        if (log)
            log->PutCString("No global metadata");
        
        return false;
    }
        
    unsigned num_nodes = named_metadata->getNumOperands();
    unsigned node_index;
    
    MDNode *metadata_node = NULL;
    
    for (node_index = 0;
         node_index < num_nodes;
         ++node_index)
    {
        metadata_node = named_metadata->getOperand(node_index);
        
        if (metadata_node->getNumOperands() != 2)
            continue;
        
        if (metadata_node->getOperand(0) == result_global)
            break;
    }
    
    if (!metadata_node)
    {
        if (log)
            log->PutCString("Couldn't find result metadata");
        return false;
    }
        
    ConstantInt *constant_int = dyn_cast<ConstantInt>(metadata_node->getOperand(1));
        
    lldb::addr_t result_decl_intptr = constant_int->getZExtValue();
    
    clang::VarDecl *result_decl = reinterpret_cast<clang::VarDecl *>(result_decl_intptr);
        
    // Get the next available result name from m_decl_map and create the persistent
    // variable for it
    
    lldb_private::TypeFromParser result_decl_type (result_decl->getType().getAsOpaquePtr(),
                                                   &result_decl->getASTContext());

    lldb_private::ConstString new_result_name (m_decl_map->GetPersistentResultName());
    m_decl_map->AddPersistentVariable(result_decl, new_result_name, result_decl_type);
    
    if (log)
        log->Printf("Creating a new result global: \"%s\"", new_result_name.GetCString());
        
    // Construct a new result global and set up its metadata
    
    GlobalVariable *new_result_global = new GlobalVariable(llvm_module, 
                                                           result_global->getType()->getElementType(),
                                                           false, /* not constant */
                                                           GlobalValue::ExternalLinkage,
                                                           NULL, /* no initializer */
                                                           new_result_name.GetCString ());
    
    // It's too late in compilation to create a new VarDecl for this, but we don't
    // need to.  We point the metadata at the old VarDecl.  This creates an odd
    // anomaly: a variable with a Value whose name is something like $0 and a
    // Decl whose name is $__lldb_expr_result.  This condition is handled in
    // ClangExpressionDeclMap::DoMaterialize, and the name of the variable is
    // fixed up.
    
    ConstantInt *new_constant_int = ConstantInt::get(constant_int->getType(), 
                                                     result_decl_intptr,
                                                     false);
    
    llvm::Value* values[2];
    values[0] = new_result_global;
    values[1] = new_constant_int;
    
    MDNode *persistent_global_md = MDNode::get(llvm_module.getContext(), values, 2);
    named_metadata->addOperand(persistent_global_md);
    
    if (log)
        log->Printf("Replacing \"%s\" with \"%s\"",
                    PrintValue(result_global).c_str(),
                    PrintValue(new_result_global).c_str());
    
    if (result_global->hasNUses(0))
    {
        // We need to synthesize a store for this variable, because otherwise
        // there's nothing to put into its equivalent persistent variable.
        
        BasicBlock &entry_block(llvm_function.getEntryBlock());
        Instruction *first_entry_instruction(entry_block.getFirstNonPHIOrDbg());
        
        if (!first_entry_instruction)
            return false;
        
        if (!result_global->hasInitializer())
        {
            if (log)
                log->Printf("Couldn't find initializer for unused variable");
            return false;
        }
        
        Constant *initializer = result_global->getInitializer();
                
        StoreInst *synthesized_store = new StoreInst::StoreInst(initializer,
                                                                new_result_global,
                                                                first_entry_instruction);
        
        if (log)
            log->Printf("Synthesized result store \"%s\"\n", PrintValue(synthesized_store).c_str());
    }
    else
    {
        result_global->replaceAllUsesWith(new_result_global);
    }
        
    result_global->eraseFromParent();
    
    return true;
}

static void DebugUsers(lldb::LogSP &log, Value *value, uint8_t depth)
{    
    if (!depth)
        return;
    
    depth--;
    
    log->Printf("  <Begin %d users>", value->getNumUses());
    
    for (Value::use_iterator ui = value->use_begin(), ue = value->use_end();
         ui != ue;
         ++ui)
    {
        log->Printf("  <Use %p> %s", *ui, PrintValue(*ui).c_str());
        DebugUsers(log, *ui, depth);
    }
    
    log->Printf("  <End uses>");
}

bool 
IRForTarget::RewriteObjCConstString (llvm::Module &llvm_module,
                                     llvm::GlobalVariable *ns_str,
                                     llvm::GlobalVariable *cstr,
                                     Instruction *FirstEntryInstruction)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    const Type *i8_ptr_ty = Type::getInt8PtrTy(llvm_module.getContext());
    const IntegerType *intptr_ty = Type::getIntNTy(llvm_module.getContext(),
                                                   (llvm_module.getPointerSize() == Module::Pointer64) ? 64 : 32);
    const Type *i32_ty = Type::getInt32Ty(llvm_module.getContext());
    const Type *i8_ty = Type::getInt8Ty(llvm_module.getContext());
    
    if (!m_CFStringCreateWithBytes)
    {
        lldb::addr_t CFStringCreateWithBytes_addr;
        
        static lldb_private::ConstString g_CFStringCreateWithBytes_str ("CFStringCreateWithBytes");
        
        if (!m_decl_map->GetFunctionAddress (g_CFStringCreateWithBytes_str, CFStringCreateWithBytes_addr))
        {
            if (log)
                log->PutCString("Couldn't find CFStringCreateWithBytes in the target");
            
            return false;
        }
            
        if (log)
            log->Printf("Found CFStringCreateWithBytes at 0x%llx", CFStringCreateWithBytes_addr);
        
        // Build the function type:
        //
        // CFStringRef CFStringCreateWithBytes (
        //   CFAllocatorRef alloc,
        //   const UInt8 *bytes,
        //   CFIndex numBytes,
        //   CFStringEncoding encoding,
        //   Boolean isExternalRepresentation
        // );
        //
        // We make the following substitutions:
        //
        // CFStringRef -> i8*
        // CFAllocatorRef -> i8*
        // UInt8 * -> i8*
        // CFIndex -> long (i32 or i64, as appropriate; we ask the module for its pointer size for now)
        // CFStringEncoding -> i32
        // Boolean -> i8
        
        std::vector <const Type *> CFSCWB_arg_types;
        CFSCWB_arg_types.push_back(i8_ptr_ty);
        CFSCWB_arg_types.push_back(i8_ptr_ty);
        CFSCWB_arg_types.push_back(intptr_ty);
        CFSCWB_arg_types.push_back(i32_ty);
        CFSCWB_arg_types.push_back(i8_ty);
        llvm::Type *CFSCWB_ty = FunctionType::get(i8_ptr_ty, CFSCWB_arg_types, false);
        
        // Build the constant containing the pointer to the function
        PointerType *CFSCWB_ptr_ty = PointerType::getUnqual(CFSCWB_ty);
        Constant *CFSCWB_addr_int = ConstantInt::get(intptr_ty, CFStringCreateWithBytes_addr, false);
        m_CFStringCreateWithBytes = ConstantExpr::getIntToPtr(CFSCWB_addr_int, CFSCWB_ptr_ty);
    }
    
    ConstantArray *string_array = dyn_cast<ConstantArray>(cstr->getInitializer());
                        
    SmallVector <Value*, 5> CFSCWB_arguments;
    
    Constant *alloc_arg         = Constant::getNullValue(i8_ptr_ty);
    Constant *bytes_arg         = ConstantExpr::getBitCast(cstr, i8_ptr_ty);
    Constant *numBytes_arg      = ConstantInt::get(intptr_ty, string_array->getType()->getNumElements(), false);
    Constant *encoding_arg      = ConstantInt::get(i32_ty, 0x0600, false); /* 0x0600 is kCFStringEncodingASCII */
    Constant *isExternal_arg    = ConstantInt::get(i8_ty, 0x0, false); /* 0x0 is false */
    
    CFSCWB_arguments.push_back(alloc_arg);
    CFSCWB_arguments.push_back(bytes_arg);
    CFSCWB_arguments.push_back(numBytes_arg);
    CFSCWB_arguments.push_back(encoding_arg);
    CFSCWB_arguments.push_back(isExternal_arg);
    
    CallInst *CFSCWB_call = CallInst::Create(m_CFStringCreateWithBytes, 
                                             CFSCWB_arguments.begin(),
                                             CFSCWB_arguments.end(),
                                             "CFStringCreateWithBytes",
                                             FirstEntryInstruction);
            
    if (!UnfoldConstant(ns_str, CFSCWB_call, FirstEntryInstruction))
    {
        if (log)
            log->PutCString("Couldn't replace the NSString with the result of the call");
        
        return false;
    }
    
    ns_str->eraseFromParent();
    
    return true;
}

bool
IRForTarget::RewriteObjCConstStrings(Module &llvm_module, Function &llvm_function)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    ValueSymbolTable& value_symbol_table = llvm_module.getValueSymbolTable();
    
    BasicBlock &entry_block(llvm_function.getEntryBlock());
    Instruction *FirstEntryInstruction(entry_block.getFirstNonPHIOrDbg());
    
    if (!FirstEntryInstruction)
    {
        if (log)
            log->PutCString("Couldn't find first instruction for rewritten Objective-C strings");
        
        return false;
    }
    
    for (ValueSymbolTable::iterator vi = value_symbol_table.begin(), ve = value_symbol_table.end();
         vi != ve;
         ++vi)
    {
        if (strstr(vi->first(), "_unnamed_cfstring_"))
        {
            Value *nsstring_value = vi->second;
            
            GlobalVariable *nsstring_global = dyn_cast<GlobalVariable>(nsstring_value);
            
            if (!nsstring_global)
            {
                if (log)
                    log->PutCString("NSString variable is not a GlobalVariable");
                return false;
            }
            
            if (!nsstring_global->hasInitializer())
            {
                if (log)
                    log->PutCString("NSString variable does not have an initializer");
                return false;
            }
            
            ConstantStruct *nsstring_struct = dyn_cast<ConstantStruct>(nsstring_global->getInitializer());
            
            if (!nsstring_struct)
            {
                if (log)
                    log->PutCString("NSString variable's initializer is not a ConstantStruct");
                return false;
            }
            
            // We expect the following structure:
            //
            // struct {
            //   int *isa;
            //   int flags;
            //   char *str;
            //   long length;
            // };
            
            if (nsstring_struct->getNumOperands() != 4)
            {
                if (log)
                    log->Printf("NSString variable's initializer structure has an unexpected number of members.  Should be 4, is %d", nsstring_struct->getNumOperands());
                return false;
            }
            
            Constant *nsstring_member = nsstring_struct->getOperand(2);
            
            if (!nsstring_member)
            {
                if (log)
                    log->PutCString("NSString initializer's str element was empty");
                return false;
            }
            
            ConstantExpr *nsstring_expr = dyn_cast<ConstantExpr>(nsstring_member);
            
            if (!nsstring_expr)
            {
                if (log)
                    log->PutCString("NSString initializer's str element is not a ConstantExpr");
                return false;
            }
            
            if (nsstring_expr->getOpcode() != Instruction::GetElementPtr)
            {
                if (log)
                    log->Printf("NSString initializer's str element is not a GetElementPtr expression, it's a %s", nsstring_expr->getOpcodeName());
                return false;
            }
            
            Constant *nsstring_cstr = nsstring_expr->getOperand(0);
            
            GlobalVariable *cstr_global = dyn_cast<GlobalVariable>(nsstring_cstr);
            
            if (!cstr_global)
            {
                if (log)
                    log->PutCString("NSString initializer's str element is not a GlobalVariable");
                    
                return false;
            }
            
            if (!cstr_global->hasInitializer())
            {
                if (log)
                    log->PutCString("NSString initializer's str element does not have an initializer");
                return false;
            }
            
            ConstantArray *cstr_array = dyn_cast<ConstantArray>(cstr_global->getInitializer());
            
            if (!cstr_array)
            {
                if (log)
                    log->PutCString("NSString initializer's str element is not a ConstantArray");
                return false;
            }
            
            if (!cstr_array->isCString())
            {
                if (log)
                    log->PutCString("NSString initializer's str element is not a C string array");
                return false;
            }
            
            if (log)
                log->Printf("Found NSString constant %s, which contains \"%s\"", vi->first(), cstr_array->getAsString().c_str());
            
            if (!RewriteObjCConstString(llvm_module, nsstring_global, cstr_global, FirstEntryInstruction))
            {
                if (log)
                    log->PutCString("Error rewriting the constant string");
                return false;
            }
            
            
        }
    }
    
    for (ValueSymbolTable::iterator vi = value_symbol_table.begin(), ve = value_symbol_table.end();
         vi != ve;
         ++vi)
    {
        if (!strcmp(vi->first(), "__CFConstantStringClassReference"))
        {
            GlobalVariable *gv = dyn_cast<GlobalVariable>(vi->second);
            
            if (!gv)
            {
                if (log)
                    log->PutCString("__CFConstantStringClassReference is not a global variable");
                return false;
            }
                
            gv->eraseFromParent();
                
            break;
        }
    }
    
    return true;
}

static bool IsObjCSelectorRef (Value *value)
{
    GlobalVariable *global_variable = dyn_cast<GlobalVariable>(value);
    
    if (!global_variable || !global_variable->hasName() || !global_variable->getName().startswith("\01L_OBJC_SELECTOR_REFERENCES_"))
        return false;
    
    return true;
}

bool 
IRForTarget::RewriteObjCSelector (Instruction* selector_load, Module &llvm_module)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    LoadInst *load = dyn_cast<LoadInst>(selector_load);
    
    if (!load)
        return false;
    
    // Unpack the message name from the selector.  In LLVM IR, an objc_msgSend gets represented as
    //
    // %tmp     = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_" ; <i8*>
    // %call    = call i8* (i8*, i8*, ...)* @objc_msgSend(i8* %obj, i8* %tmp, ...) ; <i8*>
    //
    // where %obj is the object pointer and %tmp is the selector.
    // 
    // @"\01L_OBJC_SELECTOR_REFERENCES_" is a pointer to a character array called @"\01L_OBJC_llvm_moduleETH_VAR_NAllvm_moduleE_".
    // @"\01L_OBJC_llvm_moduleETH_VAR_NAllvm_moduleE_" contains the string.
    
    // Find the pointer's initializer (a ConstantExpr with opcode GetElementPtr) and get the string from its target
    
    GlobalVariable *_objc_selector_references_ = dyn_cast<GlobalVariable>(load->getPointerOperand());
    
    if (!_objc_selector_references_ || !_objc_selector_references_->hasInitializer())
        return false;
    
    Constant *osr_initializer = _objc_selector_references_->getInitializer();
    
    ConstantExpr *osr_initializer_expr = dyn_cast<ConstantExpr>(osr_initializer);
    
    if (!osr_initializer_expr || osr_initializer_expr->getOpcode() != Instruction::GetElementPtr)
        return false;
    
    Value *osr_initializer_base = osr_initializer_expr->getOperand(0);

    if (!osr_initializer_base)
        return false;
    
    // Find the string's initializer (a ConstantArray) and get the string from it
    
    GlobalVariable *_objc_meth_var_name_ = dyn_cast<GlobalVariable>(osr_initializer_base);
    
    if (!_objc_meth_var_name_ || !_objc_meth_var_name_->hasInitializer())
        return false;
    
    Constant *omvn_initializer = _objc_meth_var_name_->getInitializer();

    ConstantArray *omvn_initializer_array = dyn_cast<ConstantArray>(omvn_initializer);
    
    if (!omvn_initializer_array->isString())
        return false;
    
    std::string omvn_initializer_string = omvn_initializer_array->getAsString();
    
    if (log)
        log->Printf("Found Objective-C selector reference \"%s\"", omvn_initializer_string.c_str());
    
    // Construct a call to sel_registerName
    
    if (!m_sel_registerName)
    {
        lldb::addr_t sel_registerName_addr;
        
        static lldb_private::ConstString g_sel_registerName_str ("sel_registerName");
        if (!m_decl_map->GetFunctionAddress (g_sel_registerName_str, sel_registerName_addr))
            return false;
        
        if (log)
            log->Printf("Found sel_registerName at 0x%llx", sel_registerName_addr);
        
        // Build the function type: struct objc_selector *sel_registerName(uint8_t*)
        
        // The below code would be "more correct," but in actuality what's required is uint8_t*
        //Type *sel_type = StructType::get(llvm_module.getContext());
        //Type *sel_ptr_type = PointerType::getUnqual(sel_type);
        const Type *sel_ptr_type = Type::getInt8PtrTy(llvm_module.getContext());
        
        std::vector <const Type *> srN_arg_types;
        srN_arg_types.push_back(Type::getInt8PtrTy(llvm_module.getContext()));
        llvm::Type *srN_type = FunctionType::get(sel_ptr_type, srN_arg_types, false);
        
        // Build the constant containing the pointer to the function
        const IntegerType *intptr_ty = Type::getIntNTy(llvm_module.getContext(),
                                                       (llvm_module.getPointerSize() == Module::Pointer64) ? 64 : 32);
        PointerType *srN_ptr_ty = PointerType::getUnqual(srN_type);
        Constant *srN_addr_int = ConstantInt::get(intptr_ty, sel_registerName_addr, false);
        m_sel_registerName = ConstantExpr::getIntToPtr(srN_addr_int, srN_ptr_ty);
    }
    
    SmallVector <Value*, 1> srN_arguments;
    
    Constant *omvn_pointer = ConstantExpr::getBitCast(_objc_meth_var_name_, Type::getInt8PtrTy(llvm_module.getContext()));
    
    srN_arguments.push_back(omvn_pointer);
    
    CallInst *srN_call = CallInst::Create(m_sel_registerName, 
                                          srN_arguments.begin(),
                                          srN_arguments.end(),
                                          "sel_registerName",
                                          selector_load);
    
    // Replace the load with the call in all users
    
    selector_load->replaceAllUsesWith(srN_call);
    
    selector_load->eraseFromParent();
    
    return true;
}

bool
IRForTarget::RewriteObjCSelectors (Module &llvm_module, BasicBlock &basic_block)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    BasicBlock::iterator ii;
    
    typedef SmallVector <Instruction*, 2> InstrList;
    typedef InstrList::iterator InstrIterator;
    
    InstrList selector_loads;
    
    for (ii = basic_block.begin();
         ii != basic_block.end();
         ++ii)
    {
        Instruction &inst = *ii;
        
        if (LoadInst *load = dyn_cast<LoadInst>(&inst))
            if (IsObjCSelectorRef(load->getPointerOperand()))
                selector_loads.push_back(&inst);
    }
    
    InstrIterator iter;
    
    for (iter = selector_loads.begin();
         iter != selector_loads.end();
         ++iter)
    {
        if (!RewriteObjCSelector(*iter, llvm_module))
        {
            if(log)
                log->PutCString("Couldn't rewrite a reference to an Objective-C selector");
            return false;
        }
    }
        
    return true;
}

bool 
IRForTarget::RewritePersistentAlloc (llvm::Instruction *persistent_alloc,
                                     llvm::Module &llvm_module)
{
    AllocaInst *alloc = dyn_cast<AllocaInst>(persistent_alloc);
    
    MDNode *alloc_md = alloc->getMetadata("clang.decl.ptr");

    if (!alloc_md || !alloc_md->getNumOperands())
        return false;
    
    ConstantInt *constant_int = dyn_cast<ConstantInt>(alloc_md->getOperand(0));
    
    if (!constant_int)
        return false;
    
    // We attempt to register this as a new persistent variable with the DeclMap.
    
    uintptr_t ptr = constant_int->getZExtValue();
    
    clang::VarDecl *decl = reinterpret_cast<clang::VarDecl *>(ptr);
    
    lldb_private::TypeFromParser result_decl_type (decl->getType().getAsOpaquePtr(),
                                                   &decl->getASTContext());
    
    StringRef decl_name (decl->getName());
    lldb_private::ConstString persistent_variable_name (decl_name.data(), decl_name.size());
    if (!m_decl_map->AddPersistentVariable(decl, persistent_variable_name, result_decl_type))
        return false;
    
    GlobalVariable *persistent_global = new GlobalVariable(llvm_module, 
                                                           alloc->getType()->getElementType(),
                                                           false, /* not constant */
                                                           GlobalValue::ExternalLinkage,
                                                           NULL, /* no initializer */
                                                           alloc->getName().str().c_str());
    
    // What we're going to do here is make believe this was a regular old external
    // variable.  That means we need to make the metadata valid.
    
    NamedMDNode *named_metadata = llvm_module.getNamedMetadata("clang.global.decl.ptrs");
    
    llvm::Value* values[2];
    values[0] = persistent_global;
    values[1] = constant_int;

    MDNode *persistent_global_md = MDNode::get(llvm_module.getContext(), values, 2);
    named_metadata->addOperand(persistent_global_md);
    
    alloc->replaceAllUsesWith(persistent_global);
    alloc->eraseFromParent();
    
    return true;
}

bool 
IRForTarget::RewritePersistentAllocs(llvm::Module &llvm_module, llvm::BasicBlock &basic_block)
{
    if (!m_resolve_vars)
        return true;
    
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    BasicBlock::iterator ii;
    
    typedef SmallVector <Instruction*, 2> InstrList;
    typedef InstrList::iterator InstrIterator;
    
    InstrList pvar_allocs;
    
    for (ii = basic_block.begin();
         ii != basic_block.end();
         ++ii)
    {
        Instruction &inst = *ii;
        
        if (AllocaInst *alloc = dyn_cast<AllocaInst>(&inst))
            if (alloc->getName().startswith("$") &&
                !alloc->getName().startswith("$__lldb"))
                pvar_allocs.push_back(alloc);
    }
    
    InstrIterator iter;
    
    for (iter = pvar_allocs.begin();
         iter != pvar_allocs.end();
         ++iter)
    {
        if (!RewritePersistentAlloc(*iter, llvm_module))
        {
            if(log)
                log->PutCString("Couldn't rewrite the creation of a persistent variable");
            return false;
        }
    }
    
    return true;
}

static clang::NamedDecl *
DeclForGlobalValue(Module &module, GlobalValue *global_value)
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
            continue;
        
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
IRForTarget::MaybeHandleVariable (Module &llvm_module, Value *llvm_value_ptr)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("MaybeHandleVariable (%s)\n", PrintValue(llvm_value_ptr).c_str());

    if (ConstantExpr *constant_expr = dyn_cast<ConstantExpr>(llvm_value_ptr))
    {
        switch (constant_expr->getOpcode())
        {
        default:
            break;
        case Instruction::GetElementPtr:
        case Instruction::BitCast:
            Value *s = constant_expr->getOperand(0);
            if (!MaybeHandleVariable(llvm_module, s))
                return false;
        }
    }
    if (GlobalVariable *global_variable = dyn_cast<GlobalVariable>(llvm_value_ptr))
    {
        clang::NamedDecl *named_decl = DeclForGlobalValue(llvm_module, global_variable);
        
        if (!named_decl)
        {
            if (IsObjCSelectorRef(llvm_value_ptr))
                return true;
            
            if (log)
                log->Printf("Found global variable \"%s\" without metadata", global_variable->getName().str().c_str());
            return false;
        }
        
        std::string name (named_decl->getName().str());
        
        void *opaque_type = NULL;
        clang::ASTContext *ast_context = NULL;
        
        if (clang::ValueDecl *value_decl = dyn_cast<clang::ValueDecl>(named_decl))
        {
            opaque_type = value_decl->getType().getAsOpaquePtr();
            ast_context = &value_decl->getASTContext();
        }
        else
        {
            return false;
        }
        
        clang::QualType qual_type(clang::QualType::getFromOpaquePtr(opaque_type));
            
        const Type *value_type = global_variable->getType();
                
        size_t value_size = (ast_context->getTypeSize(qual_type) + 7) / 8;
        off_t value_alignment = (ast_context->getTypeAlign(qual_type) + 7) / 8;
                
        if (log)
            log->Printf("Type of \"%s\" is [clang \"%s\", lldb \"%s\"] [size %d, align %d]", 
                        name.c_str(), 
                        qual_type.getAsString().c_str(), 
                        PrintType(value_type).c_str(), 
                        value_size, 
                        value_alignment);

        
        if (named_decl && !m_decl_map->AddValueToStruct(named_decl,
                                                        lldb_private::ConstString (name.c_str()),
                                                        llvm_value_ptr,
                                                        value_size, 
                                                        value_alignment))
            return false;
    }
    else if (dyn_cast<llvm::Function>(llvm_value_ptr))
    {
        if (log)
            log->Printf("Function pointers aren't handled right now");
        
        return false;
    }
    
    return true;
}

bool
IRForTarget::MaybeHandleCallArguments (Module &llvm_module, CallInst *Old)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
        log->Printf("MaybeHandleCallArguments(%s)", PrintValue(Old).c_str());
    
    for (unsigned op_index = 0, num_ops = Old->getNumArgOperands();
         op_index < num_ops;
         ++op_index)
        if (!MaybeHandleVariable(llvm_module, Old->getArgOperand(op_index))) // conservatively believe that this is a store
            return false;
    
    return true;
}

bool
IRForTarget::MaybeHandleCall (Module &llvm_module, CallInst *llvm_call_inst)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    Function *fun = llvm_call_inst->getCalledFunction();
    
    if (fun == NULL)
    {
        Value *val = llvm_call_inst->getCalledValue();
        
        ConstantExpr *const_expr = dyn_cast<ConstantExpr>(val);
        
        if (const_expr && const_expr->getOpcode() == Instruction::BitCast)
        {
            fun = dyn_cast<Function>(const_expr->getOperand(0));
            
            if (!fun)
                return false;
        }
        else
        {
            return false;
        }
    }
    
    lldb_private::ConstString str;
    
    if (fun->isIntrinsic())
    {
        Intrinsic::ID intrinsic_id = (Intrinsic::ID)fun->getIntrinsicID();
        
        switch (intrinsic_id)
        {
        default:
            if (log)
                log->Printf("Unresolved intrinsic \"%s\"", Intrinsic::getName(intrinsic_id).c_str());
            return false;
        case Intrinsic::memcpy:
            {
                static lldb_private::ConstString g_memcpy_str ("memcpy");
                str = g_memcpy_str;
            }
            break;
        }
        
        if (log && str)
            log->Printf("Resolved intrinsic name \"%s\"", str.GetCString());
    }
    else
    {
        str.SetCStringWithLength (fun->getName().data(), fun->getName().size());
    }
    
    clang::NamedDecl *fun_decl = DeclForGlobalValue (llvm_module, fun);
    lldb::addr_t fun_addr = LLDB_INVALID_ADDRESS;
    Value **fun_value_ptr = NULL;
    
    if (fun_decl)
    {
        if (!m_decl_map->GetFunctionInfo (fun_decl, fun_value_ptr, fun_addr)) 
        {
            fun_value_ptr = NULL;
            
            if (!m_decl_map->GetFunctionAddress (str, fun_addr))
            {
                if (log)
                    log->Printf("Function \"%s\" had no address", str.GetCString());
                
                return false;
            }
        }
    }
    else 
    {
        if (!m_decl_map->GetFunctionAddress (str, fun_addr))
        {
            if (log)
                log->Printf ("Metadataless function \"%s\" had no address", str.GetCString());
        }
    }
        
    if (log)
        log->Printf("Found \"%s\" at 0x%llx", str.GetCString(), fun_addr);
    
    Value *fun_addr_ptr;
            
    if (!fun_value_ptr || !*fun_value_ptr)
    {
        const IntegerType *intptr_ty = Type::getIntNTy(llvm_module.getContext(),
                                                       (llvm_module.getPointerSize() == Module::Pointer64) ? 64 : 32);
        const FunctionType *fun_ty = fun->getFunctionType();
        PointerType *fun_ptr_ty = PointerType::getUnqual(fun_ty);
        Constant *fun_addr_int = ConstantInt::get(intptr_ty, fun_addr, false);
        fun_addr_ptr = ConstantExpr::getIntToPtr(fun_addr_int, fun_ptr_ty);
            
        if (fun_value_ptr)
            *fun_value_ptr = fun_addr_ptr;
    }
            
    if (fun_value_ptr)
        fun_addr_ptr = *fun_value_ptr;
    
    llvm_call_inst->setCalledFunction(fun_addr_ptr);
    
    ConstantArray *func_name = (ConstantArray*)ConstantArray::get(llvm_module.getContext(), str.GetCString());
    
    Value *values[1];
    values[0] = func_name;
    MDNode *func_metadata = MDNode::get(llvm_module.getContext(), values, 1);
    
    llvm_call_inst->setMetadata("lldb.call.realName", func_metadata);
    
    if (log)
        log->Printf("Set metadata for %p [%d, \"%s\"]", llvm_call_inst, func_name->isString(), func_name->getAsString().c_str());
    
    return true;
}

bool
IRForTarget::ResolveCalls(Module &llvm_module, BasicBlock &basic_block)
{        
    /////////////////////////////////////////////////////////////////////////
    // Prepare the current basic block for execution in the remote process
    //
    
    BasicBlock::iterator ii;

    for (ii = basic_block.begin();
         ii != basic_block.end();
         ++ii)
    {
        Instruction &inst = *ii;
        
        CallInst *call = dyn_cast<CallInst>(&inst);
        
        if (call && !MaybeHandleCall(llvm_module, call))
            return false;
        
        if (call && !MaybeHandleCallArguments(llvm_module, call))
            return false;
    }
    
    return true;
}

bool
IRForTarget::ResolveExternals (Module &llvm_module, Function &llvm_function)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    for (Module::global_iterator global = llvm_module.global_begin(), end = llvm_module.global_end();
         global != end;
         ++global)
    {
        if (log)
            log->Printf("Examining %s, DeclForGlobalValue returns %p", 
                        (*global).getName().str().c_str(),
                        DeclForGlobalValue(llvm_module, global));
    
        if (DeclForGlobalValue(llvm_module, global) &&
            !MaybeHandleVariable (llvm_module, global))
            return false;
    }
        
    return true;
}

static bool isGuardVariableRef(Value *V)
{
    Constant *Old;
    
    if (!(Old = dyn_cast<Constant>(V)))
        return false;
    
    ConstantExpr *CE;
    
    if ((CE = dyn_cast<ConstantExpr>(V)))
    {
        if (CE->getOpcode() != Instruction::BitCast)
            return false;
        
        Old = CE->getOperand(0);
    }
    
    GlobalVariable *GV = dyn_cast<GlobalVariable>(Old);
    
    if (!GV || !GV->hasName() || !GV->getName().startswith("_ZGV"))
        return false;
    
    return true;
}

static void TurnGuardLoadIntoZero(Instruction* guard_load, Module &llvm_module)
{
    Constant* zero(ConstantInt::get(Type::getInt8Ty(llvm_module.getContext()), 0, true));

    Value::use_iterator ui;
    
    for (ui = guard_load->use_begin();
         ui != guard_load->use_end();
         ++ui)
    {
        if (isa<Constant>(*ui))
        {
            // do nothing for the moment
        }
        else
        {
            ui->replaceUsesOfWith(guard_load, zero);
        }
    }
    
    guard_load->eraseFromParent();
}

static void ExciseGuardStore(Instruction* guard_store)
{
    guard_store->eraseFromParent();
}

bool
IRForTarget::RemoveGuards(Module &llvm_module, BasicBlock &basic_block)
{        
    ///////////////////////////////////////////////////////
    // Eliminate any reference to guard variables found.
    //
    
    BasicBlock::iterator ii;
    
    typedef SmallVector <Instruction*, 2> InstrList;
    typedef InstrList::iterator InstrIterator;
    
    InstrList guard_loads;
    InstrList guard_stores;
    
    for (ii = basic_block.begin();
         ii != basic_block.end();
         ++ii)
    {
        Instruction &inst = *ii;
        
        if (LoadInst *load = dyn_cast<LoadInst>(&inst))
            if (isGuardVariableRef(load->getPointerOperand()))
                guard_loads.push_back(&inst);                
        
        if (StoreInst *store = dyn_cast<StoreInst>(&inst))            
            if (isGuardVariableRef(store->getPointerOperand()))
                guard_stores.push_back(&inst);
    }
    
    InstrIterator iter;
    
    for (iter = guard_loads.begin();
         iter != guard_loads.end();
         ++iter)
        TurnGuardLoadIntoZero(*iter, llvm_module);
    
    for (iter = guard_stores.begin();
         iter != guard_stores.end();
         ++iter)
        ExciseGuardStore(*iter);
    
    return true;
}

bool
IRForTarget::UnfoldConstant(Constant *old_constant, Value *new_constant, Instruction *first_entry_inst)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    Value::use_iterator ui;
    
    SmallVector<User*, 16> users;
    
    // We do this because the use list might change, invalidating our iterator.
    // Much better to keep a work list ourselves.
    for (ui = old_constant->use_begin();
         ui != old_constant->use_end();
         ++ui)
        users.push_back(*ui);
        
    for (int i = 0;
         i < users.size();
         ++i)
    {
        User *user = users[i];
                
        if (Constant *constant = dyn_cast<Constant>(user))
        {
            // synthesize a new non-constant equivalent of the constant
            
            if (ConstantExpr *constant_expr = dyn_cast<ConstantExpr>(constant))
            {
                switch (constant_expr->getOpcode())
                {
                default:
                    if (log)
                        log->Printf("Unhandled constant expression type: \"%s\"", PrintValue(constant_expr).c_str());
                    return false;
                case Instruction::BitCast:
                    {
                        // UnaryExpr
                        //   OperandList[0] is value
                        
                        Value *s = constant_expr->getOperand(0);
                        
                        if (s == old_constant)
                            s = new_constant;
                        
                        BitCastInst *bit_cast(new BitCastInst(s, old_constant->getType(), "", first_entry_inst));
                        
                        UnfoldConstant(constant_expr, bit_cast, first_entry_inst);
                    }
                    break;
                case Instruction::GetElementPtr:
                    {
                        // GetElementPtrConstantExpr
                        //   OperandList[0] is base
                        //   OperandList[1]... are indices
                        
                        Value *ptr = constant_expr->getOperand(0);
                        
                        if (ptr == old_constant)
                            ptr = new_constant;
                        
                        SmallVector<Value*, 16> indices;
                        
                        unsigned operand_index;
                        unsigned num_operands = constant_expr->getNumOperands();
                        
                        for (operand_index = 1;
                             operand_index < num_operands;
                             ++operand_index)
                        {
                            Value *operand = constant_expr->getOperand(operand_index);
                            
                            if (operand == old_constant)
                                operand = new_constant;
                            
                            indices.push_back(operand);
                        }
                        
                        GetElementPtrInst *get_element_ptr(GetElementPtrInst::Create(ptr, indices.begin(), indices.end(), "", first_entry_inst));
                        
                        UnfoldConstant(constant_expr, get_element_ptr, first_entry_inst);
                    }
                    break;
                }
            }
            else
            {
                if (log)
                    log->Printf("Unhandled constant type: \"%s\"", PrintValue(constant).c_str());
                return false;
            }
        }
        else
        {
            // simple fall-through case for non-constants
            user->replaceUsesOfWith(old_constant, new_constant);
        }
    }
    
    return true;
}

bool 
IRForTarget::ReplaceVariables (Module &llvm_module, Function &llvm_function)
{
    if (!m_resolve_vars)
        return true;
    
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    m_decl_map->DoStructLayout();
    
    if (log)
        log->Printf("Element arrangement:");
    
    uint32_t num_elements;
    uint32_t element_index;
    
    size_t size;
    off_t alignment;
    
    if (!m_decl_map->GetStructInfo (num_elements, size, alignment))
        return false;
    
    Function::arg_iterator iter(llvm_function.getArgumentList().begin());
    
    if (iter == llvm_function.getArgumentList().end())
        return false;
    
    Argument *argument = iter;
    
    if (argument->getName().equals("this"))
    {
        ++iter;
        
        if (iter == llvm_function.getArgumentList().end())
            return false;
        
        argument = iter;
    }
    
    if (!argument->getName().equals("$__lldb_arg"))
        return false;
    
    if (log)
        log->Printf("Arg: \"%s\"", PrintValue(argument).c_str());
    
    BasicBlock &entry_block(llvm_function.getEntryBlock());
    Instruction *FirstEntryInstruction(entry_block.getFirstNonPHIOrDbg());
    
    if (!FirstEntryInstruction)
        return false;
    
    LLVMContext &context(llvm_module.getContext());
    const IntegerType *offset_type(Type::getInt32Ty(context));
    
    if (!offset_type)
        return false;
        
    for (element_index = 0; element_index < num_elements; ++element_index)
    {
        const clang::NamedDecl *decl;
        Value *value;
        off_t offset;
        lldb_private::ConstString name;
        
        if (!m_decl_map->GetStructElement (decl, value, offset, name, element_index))
            return false;
        
        if (log)
            log->Printf("  \"%s\" [\"%s\"] (\"%s\") placed at %d",
                        value->getName().str().c_str(),
                        name.GetCString(),
                        PrintValue(value, true).c_str(),
                        offset);
        
        ConstantInt *offset_int(ConstantInt::getSigned(offset_type, offset));
        GetElementPtrInst *get_element_ptr = GetElementPtrInst::Create(argument, offset_int, "", FirstEntryInstruction);
        BitCastInst *bit_cast = new BitCastInst(get_element_ptr, value->getType(), "", FirstEntryInstruction);
        
        if (Constant *constant = dyn_cast<Constant>(value))
            UnfoldConstant(constant, bit_cast, FirstEntryInstruction);
        else
            value->replaceAllUsesWith(bit_cast);
        
        if (GlobalVariable *var = dyn_cast<GlobalVariable>(value))
            var->eraseFromParent();
    }
    
    if (log)
        log->Printf("Total structure [align %d, size %d]", alignment, size);
    
    return true;
}

bool
IRForTarget::runOnModule (Module &llvm_module)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    Function* function = llvm_module.getFunction(StringRef(m_func_name.c_str()));
    
    if (!function)
    {
        if (log)
            log->Printf("Couldn't find \"%s()\" in the module", m_func_name.c_str());
        
        return false;
    }
        
    Function::iterator bbi;
    
    ////////////////////////////////////////////////////////////
    // Replace $__lldb_expr_result with a persistent variable
    //
    
    if (!CreateResultVariable(llvm_module, *function))
        return false;
    
    ///////////////////////////////////////////////////////////////////////////////
    // Fix all Objective-C constant strings to use NSStringWithCString:encoding:
    //
    
    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        llvm_module.print(oss, NULL);
        
        oss.flush();
        
        log->Printf("Module after creating the result variable: \n\"%s\"", s.c_str());
    }
    
    if (!RewriteObjCConstStrings(llvm_module, *function))
        return false;
    
    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        llvm_module.print(oss, NULL);
        
        oss.flush();
        
        log->Printf("Module after rewriting Objective-C const strings: \n\"%s\"", s.c_str());
    }
    
    //////////////////////////////////
    // Run basic-block level passes
    //
    
    for (bbi = function->begin();
         bbi != function->end();
         ++bbi)
    {
        if (!RemoveGuards(llvm_module, *bbi))
            return false;
        
        if (!RewritePersistentAllocs(llvm_module, *bbi))
            return false;
        
        if (!RewriteObjCSelectors(llvm_module, *bbi))
            return false;

        if (!ResolveCalls(llvm_module, *bbi))
            return false;
    }
    
    ///////////////////////////////
    // Run function-level passes
    //
    
    if (!ResolveExternals(llvm_module, *function))
        return false;
    
    if (!ReplaceVariables(llvm_module, *function))
        return false;
    
    if (log)
    {
        std::string s;
        raw_string_ostream oss(s);
        
        llvm_module.print(oss, NULL);
        
        oss.flush();
        
        log->Printf("Module after preparing for execution: \n\"%s\"", s.c_str());
    }
    
    return true;    
}

void
IRForTarget::assignPassManager (PMStack &pass_mgr_stack, PassManagerType pass_mgr_type)
{
}

PassManagerType
IRForTarget::getPotentialPassManagerType() const
{
    return PMT_ModulePassManager;
}
