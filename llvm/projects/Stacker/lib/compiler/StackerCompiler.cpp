//===-- StackerCompiler.cpp - Parser for llvm assembly files ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and donated to the LLVM research 
// group and is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file implements the compiler for the "Stacker" language.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//            Globasl - Global variables we use 
//===----------------------------------------------------------------------===//

#include <llvm/Analysis/Verifier.h>
#include <llvm/iMemory.h>
#include <llvm/iOperators.h>
#include <llvm/iOther.h>
#include <llvm/iTerminators.h>
#include <Support/Statistic.h>
#include "StackerCompiler.h"
#include "StackerParser.h"
#include <string>

// Lexer/Parser defined variables and functions
extern std::FILE *Stackerin;
extern int Stackerlineno;
extern char* Stackertext;
extern int Stackerleng;
extern int Stackerparse();

StackerCompiler* StackerCompiler::TheInstance = 0;

static Statistic<> NumDefinitions(
	"numdefs","The # of definitions encoutered while compiling Stacker");

StackerCompiler::StackerCompiler()
    : CurFilename("")
    , TheModule(0)
    , TheFunction(0)
    , DefinitionType(0)
    , TheStack(0)
    , TheIndex(0)
    , TheScanf(0)
    , ThePrintf(0)
    , TheExit(0)
    , StrFormat(0)
    , NumFormat(0)
    , ChrFormat(0)
    , InStrFormat(0)
    , InNumFormat(0)
    , InChrFormat(0)
    , Zero(0)
    , One(0)
    , Two(0)
    , Three(0)
    , Four(0)
    , Five(0)
    , IZero(0)
    , IOne(0)
    , ITwo(0)
    , no_arguments()
    , echo(false)
    , stack_size(256)
    , stack_type(0)
{
}

StackerCompiler::~StackerCompiler()
{
    // delete TheModule; << don't do this! 
    // TheModule is passed to caller of the compile() method .. its their 
    // problem.  Likewise for the other allocated objects (which become part 
    // of TheModule.
    TheModule = 0;
    DefinitionType = 0;
    TheStack = 0;
    TheIndex = 0;
}

Module*
StackerCompiler::compile(
    const std::string& filename,
    bool should_echo,
    size_t the_stack_size
)
{
    // TODO: Provide a global lock to protect the singled-threaded compiler
    // and its global variables. Should be in guard object on the stack so
    // that its destructor causes lock to be released (multiple exits from
    // this function).

    // Assign parameters
    CurFilename = filename;
    echo = should_echo;
    stack_size = the_stack_size;

    /// Default the file to read
    FILE *F = stdin;

    ///
    if (filename != "-") 
    {
	F = fopen(filename.c_str(), "r");

	if (F == 0)
	{
	    throw ParseException(filename, 
		"Could not open file '" + filename + "'");
	}
    }

    Module *Result;
    try 
    {
	// Create the module we'll return
	TheModule = new Module( CurFilename );

	// Create a type to represent the stack. This is the same as the LLVM 
	// Assembly type [ 256 x int ]
	stack_type = ArrayType::get( Type::IntTy, stack_size );

	// Create a global variable for the stack. Note the use of appending 
	// linkage linkage so that multiple modules will make the stack larger. 
	// Also note that the last argument causes the global to be inserted 
	// automatically into the module.
	TheStack = new GlobalVariable( 
	    /*type=*/ stack_type, 
	    /*isConstant=*/ false, 
	    /*Linkage=*/ GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/ Constant::getNullValue(stack_type),
	    /*name=*/ "_stack_",
	    /*parent=*/ TheModule 
	);

	// Create a global variable for indexing into the stack. Note the use 
	// of LinkOnce linkage. Only one copy of _index_ will be retained 
	// after linking
	TheIndex = new GlobalVariable( 
	    /*type=*/Type::LongTy, 
	    /*isConstant=*/false,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/ Constant::getNullValue(Type::LongTy),
	    /*name=*/"_index_",
	    /*parent=*/TheModule
	);

	// Create a function prototype for definitions. No parameters, no 
	// result.  This is used below any time a function is created.
	std::vector<const Type*> params; // No parameters
	DefinitionType = FunctionType::get( Type::VoidTy, params, false );

	// Create a function for printf(3)
	params.push_back( PointerType::get( Type::SByteTy ) );
	FunctionType* printf_type = 
	    FunctionType::get( Type::IntTy, params, true );
	ThePrintf = new Function( 
	    printf_type, GlobalValue::ExternalLinkage, "printf", TheModule);

	// Create a function for scanf(3)
	TheScanf = new Function( 
	    printf_type, GlobalValue::ExternalLinkage, "scanf", TheModule);

	// Create a function for exit(3)
	params.clear();
	params.push_back( Type::IntTy );
	FunctionType* exit_type = 
	    FunctionType::get( Type::VoidTy, params, false );
	TheExit = new Function( 
	    exit_type, GlobalValue::ExternalLinkage, "exit", TheModule);

	Constant* str_format = ConstantArray::get("%s");
	StrFormat = new GlobalVariable( 
	    /*type=*/ArrayType::get( Type::SByteTy,  3 ),
	    /*isConstant=*/true,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/str_format, 
	    /*name=*/"_str_format_",
	    /*parent=*/TheModule
	);

	Constant* in_str_format = ConstantArray::get(" %as");
	InStrFormat = new GlobalVariable( 
	    /*type=*/ArrayType::get( Type::SByteTy,  5 ),
	    /*isConstant=*/true,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/in_str_format, 
	    /*name=*/"_in_str_format_",
	    /*parent=*/TheModule
	);

	Constant* num_format = ConstantArray::get("%d");
	NumFormat = new GlobalVariable( 
	    /*type=*/ArrayType::get( Type::SByteTy,  3 ),
	    /*isConstant=*/true,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/num_format, 
	    /*name=*/"_num_format_",
	    /*parent=*/TheModule
	);

	Constant* in_num_format = ConstantArray::get(" %d");
	InNumFormat = new GlobalVariable( 
	    /*type=*/ArrayType::get( Type::SByteTy,  4 ),
	    /*isConstant=*/true,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/in_num_format, 
	    /*name=*/"_in_num_format_",
	    /*parent=*/TheModule
	);

	Constant* chr_format = ConstantArray::get("%c");
	ChrFormat = new GlobalVariable( 
	    /*type=*/ArrayType::get( Type::SByteTy,  3 ),
	    /*isConstant=*/true,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/chr_format, 
	    /*name=*/"_chr_format_",
	    /*parent=*/TheModule
	);

	Constant* in_chr_format = ConstantArray::get(" %c");
	InChrFormat = new GlobalVariable( 
	    /*type=*/ArrayType::get( Type::SByteTy,  4 ),
	    /*isConstant=*/true,
	    /*Linkage=*/GlobalValue::LinkOnceLinkage, 
	    /*initializer=*/in_chr_format, 
	    /*name=*/"_in_chr_format_",
	    /*parent=*/TheModule
	);

	// Get some constants so we aren't always creating them
	Zero = ConstantInt::get( Type::LongTy, 0 );
	One = ConstantInt::get( Type::LongTy, 1 );
	Two = ConstantInt::get( Type::LongTy, 2 );
	Three = ConstantInt::get( Type::LongTy, 3 );
	Four = ConstantInt::get( Type::LongTy, 4 );
	Five = ConstantInt::get( Type::LongTy, 5 );
	IZero = ConstantInt::get( Type::IntTy, 0 );
	IOne = ConstantInt::get( Type::IntTy, 1 );
	ITwo = ConstantInt::get( Type::IntTy, 2 );

	// Reset the current line number
	Stackerlineno = 1;    

	// Reset the parser's input to F
	Stackerin = F;		// Set the input file.

	// Let the parse know about this instance
	TheInstance = this;

	// Parse the file. The parser (see StackParser.y) will call back to 
	// the StackCompiler via the "handle*" methods 
	Stackerparse(); 

	// Avoid potential illegal use (TheInstance might be on the stack)
	TheInstance = 0;

    } catch (...) {
	if (F != stdin) fclose(F);      // Make sure to close file descriptor 
	throw;                          // if an exception is thrown
    }

    // Close the file
    if (F != stdin) fclose(F);
    
    // Return the compiled module to the caller
    return TheModule;
}

//===----------------------------------------------------------------------===//
//            Internal Functions, used by handleXXX below.
//            These represent the basic stack operations.
//===----------------------------------------------------------------------===//

Instruction*
StackerCompiler::incr_stack_index( BasicBlock* bb, Value* ival = 0 )
{
    // Load the value from the TheIndex
    LoadInst* loadop = new LoadInst( TheIndex );
    bb->getInstList().push_back( loadop );

    // Increment the loaded index value
    if ( ival == 0 ) ival = One;
    CastInst* caster = new CastInst( ival, Type::LongTy );
    bb->getInstList().push_back( caster );
    BinaryOperator* addop = BinaryOperator::create( Instruction::Add, 
	    loadop, caster);
    bb->getInstList().push_back( addop );

    // Store the incremented value
    StoreInst* storeop = new StoreInst( addop, TheIndex );
    bb->getInstList().push_back( storeop );
    return storeop;
}

Instruction*
StackerCompiler::decr_stack_index( BasicBlock* bb, Value* ival = 0 )
{
    // Load the value from the TheIndex
    LoadInst* loadop = new LoadInst( TheIndex );
    bb->getInstList().push_back( loadop );

    // Decrement the loaded index value
    if ( ival == 0 ) ival = One;
    CastInst* caster = new CastInst( ival, Type::LongTy );
    bb->getInstList().push_back( caster );
    BinaryOperator* subop = BinaryOperator::create( Instruction::Sub, 
	    loadop, caster);
    bb->getInstList().push_back( subop );

    // Store the incremented value
    StoreInst* storeop = new StoreInst( subop, TheIndex );
    bb->getInstList().push_back( storeop );

    return storeop;
}

Instruction*
StackerCompiler::get_stack_pointer( BasicBlock* bb, Value* index = 0 )
{
    // Load the value of the Stack Index 
    LoadInst* loadop = new LoadInst( TheIndex );
    bb->getInstList().push_back( loadop );

    // Index into the stack to get its address. NOTE the use of two
    // elements in this vector. The first de-references the pointer that
    // "TheStack" represents. The second indexes into the pointed to array.
    // Think of the first index as getting the address of the 0th element
    // of the array.
    std::vector<Value*> indexVec;
    indexVec.push_back( Zero );

    if ( index == 0 )
    {
	indexVec.push_back(loadop);	
    }
    else
    {
	CastInst* caster = new CastInst( index, Type::LongTy );
	bb->getInstList().push_back( caster );
	BinaryOperator* subop = BinaryOperator::create( 
	    Instruction::Sub, loadop, caster );
	bb->getInstList().push_back( subop );
	indexVec.push_back(subop);
    }

    // Get the address of the indexed stack element
    GetElementPtrInst* gep = new GetElementPtrInst( TheStack, indexVec );
    bb->getInstList().push_back( gep );		// Put GEP in Block

    return gep;
}

Instruction*
StackerCompiler::push_value( BasicBlock* bb, Value* val )
{
    // Get location of 
    incr_stack_index(bb);

    // Get the stack pointer
    GetElementPtrInst* gep = cast<GetElementPtrInst>( 
	    get_stack_pointer( bb ) );

    // Cast the value to an integer .. hopefully it works
    CastInst* cast_inst = new CastInst( val, Type::IntTy );
    bb->getInstList().push_back( cast_inst );

    // Store the value
    StoreInst* storeop = new StoreInst( cast_inst, gep );
    bb->getInstList().push_back( storeop );

    return storeop;
}

Instruction*
StackerCompiler::push_integer(BasicBlock* bb, int32_t value )
{
    // Just push a constant integer value
    return push_value( bb, ConstantSInt::get( Type::IntTy, value ) );
}

Instruction*
StackerCompiler::pop_integer( BasicBlock*bb )
{
    // Get the stack pointer
    GetElementPtrInst* gep = cast<GetElementPtrInst>(
	get_stack_pointer( bb ));

    // Load the value
    LoadInst* load_inst = new LoadInst( gep );
    bb->getInstList().push_back( load_inst );

    // Decrement the stack index
    decr_stack_index( bb );

    // Return the value
    return load_inst;
}

Instruction*
StackerCompiler::push_string( BasicBlock* bb, const char* value )
{
    // Get length of the string
    size_t len = strlen( value );

    // Create a type for the string constant. Length is +1 for 
    // the terminating 0.
    ArrayType* char_array = ArrayType::get( Type::SByteTy, len + 1 );

    // Create an initializer for the value
    Constant* initVal = ConstantArray::get( value );

    // Create an internal linkage global variable to hold the constant.
    GlobalVariable* strconst = new GlobalVariable( 
	char_array, 
	/*isConstant=*/true, 
	GlobalValue::InternalLinkage, 
	/*initializer=*/initVal,
	"",
	TheModule
    );

    // Push the casted value
    return push_value( bb, strconst );
}

Instruction*
StackerCompiler::pop_string( BasicBlock* bb )
{
    // Get location of stack pointer
    GetElementPtrInst* gep = cast<GetElementPtrInst>(
	get_stack_pointer( bb ));

    // Load the value from the stack
    LoadInst* loader = new LoadInst( gep );
    bb->getInstList().push_back( loader );

    // Cast the integer to a sbyte*
    CastInst* caster = new CastInst( loader, PointerType::get(Type::SByteTy) );
    bb->getInstList().push_back( caster );

    // Decrement stack index
    decr_stack_index( bb );

    // Return the value
    return caster;
}

Instruction*
StackerCompiler::replace_top( BasicBlock* bb, Value* new_top, Value* index = 0 )
{
    // Get the stack pointer
    GetElementPtrInst* gep = cast<GetElementPtrInst>(
	    get_stack_pointer( bb, index ));
    
    // Store the value there
    StoreInst* store_inst = new StoreInst( new_top, gep );
    bb->getInstList().push_back( store_inst );

    // Return the value
    return store_inst;
}

Instruction*
StackerCompiler::stack_top( BasicBlock* bb, Value* index = 0 )
{
    // Get the stack pointer
    GetElementPtrInst* gep = cast<GetElementPtrInst>(
	get_stack_pointer( bb, index ));

    // Load the value
    LoadInst* load_inst = new LoadInst( gep );
    bb->getInstList().push_back( load_inst );

    // Return the value
    return load_inst;
}

Instruction*
StackerCompiler::stack_top_string( BasicBlock* bb, Value* index = 0 )
{
    // Get location of stack pointer
    GetElementPtrInst* gep = cast<GetElementPtrInst>(
	get_stack_pointer( bb, index ));

    // Load the value from the stack
    LoadInst* loader = new LoadInst( gep );
    bb->getInstList().push_back( loader );

    // Cast the integer to a sbyte*
    CastInst* caster = new CastInst( loader, PointerType::get(Type::SByteTy) );
    bb->getInstList().push_back( caster );

    // Return the value
    return caster;
}

static void
add_block( Function*f, BasicBlock* bb )
{
    if ( ! f->empty() && f->back().getTerminator() == 0 )
    {
	BranchInst* branch = new BranchInst(bb);
	f->back().getInstList().push_back( branch );
    }
    f->getBasicBlockList().push_back( bb );
}


//===----------------------------------------------------------------------===//
//            handleXXX - Handle semantics of parser productions
//===----------------------------------------------------------------------===//

Module*
StackerCompiler::handle_module_start( )
{
    // Return the newly created module
    return TheModule;
}

Module* 
StackerCompiler::handle_module_end( Module* mod )
{
    // Return the module.
    return mod;
}

Module*
StackerCompiler::handle_definition_list_start()
{
    return TheModule;
}

Module* 
StackerCompiler::handle_definition_list_end( Module* mod, Function* definition )
{
    if ( ! definition->empty() )
    {
	BasicBlock& last_block = definition->back();
	if ( last_block.getTerminator() == 0 )
	{
	    last_block.getInstList().push_back( new ReturnInst() );
	}
    }
    // Insert the definition into the module
    mod->getFunctionList().push_back( definition );

    // Bump our (sample) statistic.
    ++NumDefinitions;
    return mod;
}

Function*
StackerCompiler::handle_main_definition( Function* func )
{
    // Set the name of the function defined as the Stacker main
    // This will get called by the "main" that is defined in 
    // the runtime library.
    func->setName( "_MAIN_");

    // Turn "_stack_" into an initialized variable since this is the main
    // module. This causes it to not be "external" but defined in this module.
    TheStack->setInitializer( Constant::getNullValue(stack_type) );
    TheStack->setLinkage( GlobalValue::LinkOnceLinkage );

    // Turn "_index_" into an intialized variable for the same reason.
    TheIndex->setInitializer( Constant::getNullValue(Type::LongTy) );
    TheIndex->setLinkage( GlobalValue::LinkOnceLinkage );

    return func;
}

Function* 
StackerCompiler::handle_forward( char * name )
{
    // Just create a placeholder function
    Function* the_function = new Function ( 
	DefinitionType, 
	GlobalValue::ExternalLinkage, 
	name ); 
    assert( the_function->isExternal() );

    free( name );
    return the_function;
}

Function* 
StackerCompiler::handle_definition( char * name, Function* f )
{
    // Look up the function name in the module to see if it was forward
    // declared.
    Function* existing_function = TheModule->getNamedFunction( name );

#if 0
    // If the function already exists...
    if ( existing_function )
    {
	// Just get rid of the placeholder
	existing_function->dropAllReferences();
	delete existing_function;
    }
#endif

    // Just set the name of the function now that we know what it is.
    f->setName( name );

    free( name );

    return f;
}

Function*
StackerCompiler::handle_word_list_start()
{
    TheFunction = new Function(DefinitionType, GlobalValue::ExternalLinkage);
    return TheFunction;
}

Function*
StackerCompiler::handle_word_list_end( Function* f, BasicBlock* bb )
{
    add_block( f, bb );
    return f;
}

BasicBlock* 
StackerCompiler::handle_if( char* ifTrue, char* ifFalse )
{
    // Create a basic block for the preamble
    BasicBlock* bb = new BasicBlock((echo?"if":""));

    // Get the condition value
    LoadInst* cond = cast<LoadInst>( pop_integer(bb) );

    // Compare the condition against 0 
    SetCondInst* cond_inst = new SetCondInst( Instruction::SetNE, cond, 
	ConstantSInt::get( Type::IntTy, 0) );
    bb->getInstList().push_back( cond_inst );

    // Create an exit block
    BasicBlock* exit_bb = new BasicBlock((echo?"endif":""));

    // Create the true_block
    BasicBlock* true_bb = new BasicBlock((echo?"then":""));

    // Create the false_block
    BasicBlock* false_bb = 0;
    if ( ifFalse ) false_bb = new BasicBlock((echo?"else":""));

    // Create a branch on the SetCond
    BranchInst* br_inst = new BranchInst( true_bb, 
	( ifFalse ? false_bb : exit_bb ), cond_inst );
    bb->getInstList().push_back( br_inst );

    // Fill the true block 
    std::vector<Value*> args;
    if ( Function* true_func = TheModule->getNamedFunction(ifTrue) )
    {
	true_bb->getInstList().push_back( 
		new CallInst( true_func, args ) );
	true_bb->getInstList().push_back( 
		new BranchInst( exit_bb ) );
    }
    else
    {
	ThrowException(std::string("Function '") + ifTrue + 
	    "' must be declared first.'");
    }

    free( ifTrue );

    // Fill the false block
    if ( false_bb )
    {
	if ( Function* false_func = TheModule->getNamedFunction(ifFalse) )
	{
	    false_bb->getInstList().push_back( 
		    new CallInst( false_func, args ) );
	    false_bb->getInstList().push_back( 
		    new BranchInst( exit_bb ) );
	}
	else
	{
	    ThrowException(std::string("Function '") + ifFalse + 
		    "' must be declared first.'");
	}
	free( ifFalse );
    }

    // Add the blocks to the function
    add_block( TheFunction, bb );
    add_block( TheFunction, true_bb );
    if ( false_bb ) add_block( TheFunction, false_bb );

    return exit_bb;
}

BasicBlock* 
StackerCompiler::handle_while( char* todo )
{

    // Create a basic block for the loop test
    BasicBlock* test = new BasicBlock((echo?"while":""));

    // Create an exit block
    BasicBlock* exit = new BasicBlock((echo?"end":""));

    // Create a loop body block
    BasicBlock* body = new BasicBlock((echo?"do":""));

    // Create a root node
    BasicBlock* bb = new BasicBlock((echo?"root":""));
    BranchInst* root_br_inst = new BranchInst( test );
    bb->getInstList().push_back( root_br_inst );

    // Pop the condition value
    LoadInst* cond = cast<LoadInst>( stack_top(test) );

    // Compare the condition against 0 
    SetCondInst* cond_inst = new SetCondInst( 
	Instruction::SetNE, cond, ConstantSInt::get( Type::IntTy, 0) );
    test->getInstList().push_back( cond_inst );

    // Add the branch instruction
    BranchInst* br_inst = new BranchInst( body, exit, cond_inst );
    test->getInstList().push_back( br_inst );

    // Fill in the body
    std::vector<Value*> args;
    if ( Function* body_func = TheModule->getNamedFunction(todo) )
    {
	body->getInstList().push_back( new CallInst( body_func, args ) );
	body->getInstList().push_back( new BranchInst( test ) );
    }
    else
    {
	ThrowException(std::string("Function '") + todo + 
	    "' must be declared first.'");
    }

    free( todo );

    // Add the blocks
    add_block( TheFunction, bb );
    add_block( TheFunction, test );
    add_block( TheFunction, body );

    return exit;
}

BasicBlock* 
StackerCompiler::handle_identifier( char * name )
{
    Function* func = TheModule->getNamedFunction( name );
    BasicBlock* bb = new BasicBlock((echo?"call":""));
    if ( func )
    {
	CallInst* call_def = new CallInst( func , no_arguments );
	bb->getInstList().push_back( call_def );
    }
    else
    {
	ThrowException(std::string("Definition '") + name + 
	    "' must be defined before it can be used.");
    }

    free( name );
    return bb;
}

BasicBlock* 
StackerCompiler::handle_string( char * value )
{
    // Create a new basic block for the push operation
    BasicBlock* bb = new BasicBlock((echo?"string":""));

    // Push the string onto the stack
    push_string(bb, value);

    // Free the strdup'd string
    free( value );

    return bb;
}

BasicBlock* 
StackerCompiler::handle_integer( const int32_t value )
{
    // Create a new basic block for the push operation
    BasicBlock* bb = new BasicBlock((echo?"int":""));

    // Push the integer onto the stack
    push_integer(bb, value );

    return bb;
}

BasicBlock* 
StackerCompiler::handle_word( int tkn )
{
    // Create a new basic block to hold the instruction(s)
    BasicBlock* bb = new BasicBlock();

    /* Fill the basic block with the appropriate instructions */
    switch ( tkn )
    {
    case DUMP :  // Dump the stack (debugging aid)
    {
	if (echo) bb->setName("DUMP");
	Function* f = TheModule->getOrInsertFunction(
	    "_stacker_dump_stack_", DefinitionType);
	std::vector<Value*> args;
	bb->getInstList().push_back( new CallInst( f, args ) );
	break;
    }

    // Logical Operations
    case TRUETOK :  // -- -1
    {
	if (echo) bb->setName("TRUE");
	push_integer(bb,-1); 
	break;
    }
    case FALSETOK : // -- 0
    {
	if (echo) bb->setName("FALSE");
	push_integer(bb,0); 
	break;
    }
    case LESS : // w1 w2 -- w2<w1
    {
	if (echo) bb->setName("LESS");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetLT, op1, op2 );
	bb->getInstList().push_back( cond_inst );
	push_value( bb, cond_inst );
	break;
    }
    case MORE : // w1 w2 -- w2>w1
    {
	if (echo) bb->setName("MORE");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetGT, op1, op2 );
	bb->getInstList().push_back( cond_inst );
	push_value( bb, cond_inst );
	break;
    }
    case LESS_EQUAL : // w1 w2 -- w2<=w1
    {
	if (echo) bb->setName("LE");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetLE, op1, op2 );
	bb->getInstList().push_back( cond_inst );
	push_value( bb, cond_inst );
	break;
    }
    case MORE_EQUAL : // w1 w2 -- w2>=w1
    {
	if (echo) bb->setName("GE");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetGE, op1, op2 );
	bb->getInstList().push_back( cond_inst );
	push_value( bb, cond_inst );
	break;
    }
    case NOT_EQUAL : // w1 w2 -- w2!=w1
    {
	if (echo) bb->setName("NE");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetNE, op1, op2 );
	bb->getInstList().push_back( cond_inst );
	push_value( bb, cond_inst );
	break;
    }
    case EQUAL : // w1 w2 -- w1==w2
    {
	if (echo) bb->setName("EQ");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetEQ, op1, op2 );
	bb->getInstList().push_back( cond_inst );
	push_value( bb, cond_inst );
	break;
    }

    // Arithmetic Operations
    case PLUS : // w1 w2 -- w2+w1
    {
	if (echo) bb->setName("ADD");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* addop = 
	    BinaryOperator::create( Instruction::Add, op1, op2);
	bb->getInstList().push_back( addop );
	push_value( bb, addop );
	break;
    }
    case MINUS : // w1 w2 -- w2-w1
    {
	if (echo) bb->setName("SUB");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* subop = 
	    BinaryOperator::create( Instruction::Sub, op1, op2);
	bb->getInstList().push_back( subop );
	push_value( bb, subop );
	break;
    }
    case INCR :  // w1 -- w1+1
    {
	if (echo) bb->setName("INCR");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* addop = 
	    BinaryOperator::create( Instruction::Add, op1, IOne );
	bb->getInstList().push_back( addop );
	push_value( bb, addop );
	break;
    }
    case DECR : // w1 -- w1-1
    {
	if (echo) bb->setName("DECR");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* subop = BinaryOperator::create( Instruction::Sub, op1,
	    ConstantSInt::get( Type::IntTy, 1 ) );
	bb->getInstList().push_back( subop );
	push_value( bb, subop );
	break;
    }
    case MULT : // w1 w2 -- w2*w1
    {
	if (echo) bb->setName("MUL");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* multop = 
	    BinaryOperator::create( Instruction::Mul, op1, op2);
	bb->getInstList().push_back( multop );
	push_value( bb, multop );
	break;
    }
    case DIV :// w1 w2 -- w2/w1
    {
	if (echo) bb->setName("DIV");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* divop = 
	    BinaryOperator::create( Instruction::Div, op1, op2);
	bb->getInstList().push_back( divop );
	push_value( bb, divop );
	break;
    }
    case MODULUS : // w1 w2 -- w2%w1
    {
	if (echo) bb->setName("MOD");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* divop = 
	    BinaryOperator::create( Instruction::Rem, op1, op2);
	bb->getInstList().push_back( divop );
	push_value( bb, divop );
	break;
    }
    case STAR_SLASH : // w1 w2 w3 -- (w3*w2)/w1
    {
	if (echo) bb->setName("STAR_SLASH");
	// Get the operands
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op3 = cast<LoadInst>(pop_integer(bb));

	// Multiply the first two
	BinaryOperator* multop = 
	    BinaryOperator::create( Instruction::Mul, op1, op2);
	bb->getInstList().push_back( multop );

	// Divide by the third operand
	BinaryOperator* divop = 
	    BinaryOperator::create( Instruction::Div, multop, op3);
	bb->getInstList().push_back( divop );

	// Push the result
	push_value( bb, divop );

	break;
    }
    case NEGATE : // w1 -- -w1
    {
	if (echo) bb->setName("NEG");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	// APPARENTLY, the following doesn't work:
	// BinaryOperator* negop = BinaryOperator::createNeg( op1 );
	// bb->getInstList().push_back( negop );
	// So we'll multiply by -1 (ugh)
	BinaryOperator* multop = BinaryOperator::create( Instruction::Mul, op1,
	    ConstantSInt::get( Type::IntTy, -1 ) );
	bb->getInstList().push_back( multop );
	push_value( bb, multop );
	break;
    }
    case ABS : // w1 -- |w1|
    {
	if (echo) bb->setName("ABS");
	// Get the top of stack value
	LoadInst* op1 = cast<LoadInst>(stack_top(bb));

	// Determine if its negative
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetLT, op1, IZero );
	bb->getInstList().push_back( cond_inst );

	// Create a block for storing the result
	BasicBlock* exit_bb = new BasicBlock((echo?"exit":""));

	// Create a block for making it a positive value
	BasicBlock* pos_bb = new BasicBlock((echo?"neg":""));

	// Create the branch on the SetCond
	BranchInst* br_inst = new BranchInst( pos_bb, exit_bb, cond_inst );
	bb->getInstList().push_back( br_inst );

	// Fill out the negation block
	LoadInst* pop_op = cast<LoadInst>( pop_integer(pos_bb) );
	BinaryOperator* neg_op = BinaryOperator::createNeg( pop_op );
	pos_bb->getInstList().push_back( neg_op );
	push_value( pos_bb, neg_op );
	pos_bb->getInstList().push_back( new BranchInst( exit_bb ) );

	// Add the new blocks in the correct order
	add_block( TheFunction, bb );
	add_block( TheFunction, pos_bb );
	bb = exit_bb;
	break;
    }
    case MIN : // w1 w2 -- (w2<w1?w2:w1)
    {
	if (echo) bb->setName("MIN");

	// Create the three blocks
	BasicBlock* exit_bb  = new BasicBlock((echo?"exit":""));
	BasicBlock* op1_block = new BasicBlock((echo?"less":""));
	BasicBlock* op2_block = new BasicBlock((echo?"more":""));

	// Get the two operands
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));

	// Compare them 
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetLT, op1, op2);
	bb->getInstList().push_back( cond_inst );

	// Create a branch on the SetCond
	BranchInst* br_inst = 
	    new BranchInst( op1_block, op2_block, cond_inst );
	bb->getInstList().push_back( br_inst );

	// Create a block for pushing the first one
	push_value(op1_block, op1);
	op1_block->getInstList().push_back( new BranchInst( exit_bb ) );

	// Create a block for pushing the second one
	push_value(op2_block, op2);
	op2_block->getInstList().push_back( new BranchInst( exit_bb ) );

	// Add the blocks
	add_block( TheFunction, bb );
	add_block( TheFunction, op1_block );
	add_block( TheFunction, op2_block );
	bb = exit_bb;
	break;
    }
    case MAX : // w1 w2 -- (w2>w1?w2:w1)
    {
	if (echo) bb->setName("MAX");
	// Get the two operands
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));

	// Compare them 
	SetCondInst* cond_inst = 
	    new SetCondInst( Instruction::SetGT, op1, op2);
	bb->getInstList().push_back( cond_inst );

	// Create an exit block
	BasicBlock* exit_bb = new BasicBlock((echo?"exit":""));

	// Create a block for pushing the larger one
	BasicBlock* op1_block = new BasicBlock((echo?"more":""));
	push_value(op1_block, op1);
	op1_block->getInstList().push_back( new BranchInst( exit_bb ) );

	// Create a block for pushing the smaller or equal one
	BasicBlock* op2_block = new BasicBlock((echo?"less":""));
	push_value(op2_block, op2);
	op2_block->getInstList().push_back( new BranchInst( exit_bb ) );

	// Create a banch on the SetCond
	BranchInst* br_inst = 
	    new BranchInst( op1_block, op2_block, cond_inst );
	bb->getInstList().push_back( br_inst );

	// Add the blocks
	add_block( TheFunction, bb );
	add_block( TheFunction, op1_block );
	add_block( TheFunction, op2_block );

	bb = exit_bb;
	break;
    }

    // Bitwise Operators
    case AND : // w1 w2 -- w2&w1
    {
	if (echo) bb->setName("AND");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* andop = 
	    BinaryOperator::create( Instruction::And, op1, op2);
	bb->getInstList().push_back( andop );
	push_value( bb, andop );
	break;
    }
    case OR : // w1 w2 -- w2|w1
    {
	if (echo) bb->setName("OR");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* orop = 
	    BinaryOperator::create( Instruction::Or, op1, op2);
	bb->getInstList().push_back( orop );
	push_value( bb, orop );
	break;
    }
    case XOR : // w1 w2 -- w2^w1
    {
	if (echo) bb->setName("XOR");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	BinaryOperator* xorop = 
	    BinaryOperator::create( Instruction::Xor, op1, op2);
	bb->getInstList().push_back( xorop );
	push_value( bb, xorop );
	break;
    }
    case LSHIFT : // w1 w2 -- w1<<w2
    {
	if (echo) bb->setName("SHL");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	CastInst* castop = new CastInst( op1, Type::UByteTy );
	bb->getInstList().push_back( castop );
	ShiftInst* shlop = new ShiftInst( Instruction::Shl, op2, castop );
	bb->getInstList().push_back( shlop );
	push_value( bb, shlop );
	break;
    }
    case RSHIFT :  // w1 w2 -- w1>>w2
    {
	if (echo) bb->setName("SHR");
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));
	LoadInst* op2 = cast<LoadInst>(pop_integer(bb));
	CastInst* castop = new CastInst( op1, Type::UByteTy );
	bb->getInstList().push_back( castop );
	ShiftInst* shrop = new ShiftInst( Instruction::Shr, op2, castop );
	bb->getInstList().push_back( shrop );
	push_value( bb, shrop );
	break;
    }

    // Stack Manipulation Operations
    case DROP:   	// w -- 
    {
	if (echo) bb->setName("DROP");
	decr_stack_index(bb, One);
	break;
    }
    case DROP2:	// w1 w2 -- 
    {
	if (echo) bb->setName("DROP2");
	decr_stack_index( bb, Two );
	break;
    }
    case NIP:	// w1 w2 -- w2
    {
	if (echo) bb->setName("NIP");
	LoadInst* w2 = cast<LoadInst>( stack_top( bb ) );
	decr_stack_index( bb  );
	replace_top( bb, w2 );
	break;
    }
    case NIP2:	// w1 w2 w3 w4 -- w3 w4
    {
	if (echo) bb->setName("NIP2");
	LoadInst* w4 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w3 = cast<LoadInst>( stack_top( bb, One ) );
	decr_stack_index( bb, Two );
	replace_top( bb, w4 );
	replace_top( bb, w3, One );
	break;
    }
    case DUP:	// w -- w w
    {
	if (echo) bb->setName("DUP");
	LoadInst* w = cast<LoadInst>( stack_top( bb ) );
	push_value( bb, w );
	break;
    }
    case DUP2:	// w1 w2 -- w1 w2 w1 w2
    {
	if (echo) bb->setName("DUP2");
	LoadInst* w2 = cast<LoadInst>( stack_top(bb) );
	LoadInst* w1 = cast<LoadInst>( stack_top(bb, One ) );
	incr_stack_index( bb, Two );
	replace_top( bb, w1, One );
	replace_top( bb, w2 );
	break;
    }
    case SWAP:	// w1 w2 -- w2 w1
    {
	if (echo) bb->setName("SWAP");
	LoadInst* w2 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, One ) );
	replace_top( bb, w1 );
	replace_top( bb, w2, One );
	break;
    }
    case SWAP2:	// w1 w2 w3 w4 -- w3 w4 w1 w2
    {
	if (echo) bb->setName("SWAP2");
	LoadInst* w4 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w3 = cast<LoadInst>( stack_top( bb, One ) );
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, Two ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Three ) );
	replace_top( bb, w2 );
	replace_top( bb, w1, One );
	replace_top( bb, w4, Two );
	replace_top( bb, w3, Three );
	break;
    }
    case OVER:	// w1 w2 -- w1 w2 w1
    {
	if (echo) bb->setName("OVER");
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, One ) );
	push_value( bb, w1 );
	break;
    }
    case OVER2:	// w1 w2 w3 w4 -- w1 w2 w3 w4 w1 w2
    {
	if (echo) bb->setName("OVER2");
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, Two ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Three ) );
	incr_stack_index( bb, Two );
	replace_top( bb, w2 );
	replace_top( bb, w1, One );
	break;
    }
    case ROT:	// w1 w2 w3 -- w2 w3 w1
    {
	if (echo) bb->setName("ROT");
	LoadInst* w3 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, One ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Two ) );
	replace_top( bb, w1 );
	replace_top( bb, w3, One );
	replace_top( bb, w2, Two );
	break;
    }
    case ROT2:	// w1 w2 w3 w4 w5 w6 -- w3 w4 w5 w6 w1 w2
    {
	if (echo) bb->setName("ROT2");
	LoadInst* w6 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w5 = cast<LoadInst>( stack_top( bb, One ) );
	LoadInst* w4 = cast<LoadInst>( stack_top( bb, Two ) );
	LoadInst* w3 = cast<LoadInst>( stack_top( bb, Three) );
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, Four ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Five ) );
	replace_top( bb, w2 );
	replace_top( bb, w1, One );
	replace_top( bb, w6, Two );
	replace_top( bb, w5, Three );
	replace_top( bb, w4, Four );
	replace_top( bb, w3, Five );
	break;
    }
    case RROT:	// w1 w2 w3 -- w3 w1 w2
    {
	if (echo) bb->setName("RROT2");
	LoadInst* w3 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, One ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Two ) );
	replace_top( bb, w2 );
	replace_top( bb, w1, One );
	replace_top( bb, w3, Two );
	break;
    }
    case RROT2:	// w1 w2 w3 w4 w5 w6 -- w5 w6 w1 w2 w3 w4
    {
	if (echo) bb->setName("RROT2");
	LoadInst* w6 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w5 = cast<LoadInst>( stack_top( bb, One ) );
	LoadInst* w4 = cast<LoadInst>( stack_top( bb, Two ) );
	LoadInst* w3 = cast<LoadInst>( stack_top( bb, Three) );
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, Four ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Five ) );
	replace_top( bb, w4 );
	replace_top( bb, w3, One );
	replace_top( bb, w2, Two );
	replace_top( bb, w1, Three );
	replace_top( bb, w6, Four );
	replace_top( bb, w5, Five );
	break;
    }
    case TUCK:	// w1 w2 -- w2 w1 w2
    {
	if (echo) bb->setName("TUCK");
	LoadInst* w2 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, One ) );
	incr_stack_index( bb );
	replace_top( bb, w2 );
	replace_top( bb, w1, One );
	replace_top( bb, w2, Two );
	break;
    }
    case TUCK2:	// w1 w2 w3 w4 -- w3 w4 w1 w2 w3 w4 
    {
	if (echo) bb->setName("TUCK2");
	LoadInst* w4 = cast<LoadInst>( stack_top( bb ) );
	LoadInst* w3 = cast<LoadInst>( stack_top( bb, One ) );
	LoadInst* w2 = cast<LoadInst>( stack_top( bb, Two ) );
	LoadInst* w1 = cast<LoadInst>( stack_top( bb, Three) );
	incr_stack_index( bb, Two );
	replace_top( bb, w4 );
	replace_top( bb, w3, One );
	replace_top( bb, w2, Two );
	replace_top( bb, w1, Three );
	replace_top( bb, w4, Four );
	replace_top( bb, w3, Five );
	break;
    }
    case ROLL:	// x0 x1 .. xn n -- x1 .. xn x0
    {
	/// THIS OEPRATOR IS OMITTED PURPOSEFULLY AND IS LEFT TO THE 
	/// READER AS AN EXERCISE. THIS IS ONE OF THE MORE COMPLICATED
	/// OPERATORS. IF YOU CAN GET THIS ONE RIGHT, YOU COMPLETELY
	/// UNDERSTAND HOW BOTH LLVM AND STACKER WOR.  
	/// HINT: LOOK AT PICK AND SELECT. ROLL IS SIMILAR.
	if (echo) bb->setName("ROLL");
	break;
    }
    case PICK:	// x0 ... Xn n -- x0 ... Xn x0
    {
	if (echo) bb->setName("PICK");
	LoadInst* n = cast<LoadInst>( stack_top( bb ) );
	BinaryOperator* addop = 
	    BinaryOperator::create( Instruction::Add, n, IOne );
	bb->getInstList().push_back( addop );
	LoadInst* x0 = cast<LoadInst>( stack_top( bb, addop ) );
	replace_top( bb, x0 );
	break;
    }
    case SELECT:	// m n X0..Xm Xm+1 .. Xn -- Xm
    {
	if (echo) bb->setName("SELECT");
	LoadInst* m = cast<LoadInst>( stack_top(bb) );
	LoadInst* n = cast<LoadInst>( stack_top(bb, One) );
	BinaryOperator* index = 
	    BinaryOperator::create( Instruction::Add, m, IOne );
	bb->getInstList().push_back( index );
	LoadInst* Xm = cast<LoadInst>( stack_top(bb, index ) );
	BinaryOperator* n_plus_1 = 
	    BinaryOperator::create( Instruction::Add, n, IOne );
	bb->getInstList().push_back( n_plus_1 );
	decr_stack_index( bb, n_plus_1 );
	replace_top( bb, Xm );
	break;
    }
    case MALLOC : // n -- p
    {
	if (echo) bb->setName("MALLOC");
	// Get the number of bytes to mallocate
	LoadInst* op1 = cast<LoadInst>( pop_integer(bb) );

	// Make sure its a UIntTy
	CastInst* caster = new CastInst( op1, Type::UIntTy );
	bb->getInstList().push_back( caster );

	// Allocate the bytes
	MallocInst* mi = new MallocInst( Type::SByteTy, caster );
	bb->getInstList().push_back( mi );

	// Push the pointer
	push_value( bb, mi );
	break;
    }
    case FREE :  // p --
    {
	if (echo) bb->setName("FREE");
	// Pop the value off the stack
	CastInst* ptr = cast<CastInst>( pop_string(bb) );

	// Free the memory
	FreeInst* fi = new FreeInst( ptr );
	bb->getInstList().push_back( fi );

	break;
    }
    case GET : // p w1 -- p w2
    {
	if (echo) bb->setName("GET");
	// Get the character index
	LoadInst* op1 = cast<LoadInst>( stack_top(bb) );
	CastInst* chr_idx = new CastInst( op1, Type::LongTy );
	bb->getInstList().push_back( chr_idx );

	// Get the String pointer
	CastInst* ptr = cast<CastInst>( stack_top_string(bb,One) );

	// Get address of op1'th element of the string
	std::vector<Value*> indexVec;
	indexVec.push_back( chr_idx );
	GetElementPtrInst* gep = new GetElementPtrInst( ptr, indexVec );
	bb->getInstList().push_back( gep );

	// Get the value and push it
	LoadInst* loader = new LoadInst( gep );
	bb->getInstList().push_back( loader );
	CastInst* caster = new CastInst( loader, Type::IntTy );
	bb->getInstList().push_back( caster );

	// Push the result back on stack
	replace_top( bb, caster );

	break;
    }
    case PUT : // p w2 w1  -- p
    {
	if (echo) bb->setName("PUT");

	// Get the value to put
	LoadInst* w1 = cast<LoadInst>( pop_integer(bb) );

	// Get the character index
	LoadInst* w2 = cast<LoadInst>( pop_integer(bb) );
	CastInst* chr_idx = new CastInst( w2, Type::LongTy );
	bb->getInstList().push_back( chr_idx );

	// Get the String pointer
	CastInst* ptr = cast<CastInst>( stack_top_string(bb) );

	// Get address of op2'th element of the string
	std::vector<Value*> indexVec;
	indexVec.push_back( chr_idx );
	GetElementPtrInst* gep = new GetElementPtrInst( ptr, indexVec );
	bb->getInstList().push_back( gep );

	// Cast the value and put it
	CastInst* caster = new CastInst( w1, Type::SByteTy );
	bb->getInstList().push_back( caster );
	StoreInst* storer = new StoreInst( caster, gep );
	bb->getInstList().push_back( storer );

	break;
    }
    case RECURSE : 
    {
	if (echo) bb->setName("RECURSE");
	std::vector<Value*> params;
	CallInst* call_inst = new CallInst( TheFunction, params );
	bb->getInstList().push_back( call_inst );
	break;
    }
    case RETURN : 
    {
	if (echo) bb->setName("RETURN");
	bb->getInstList().push_back( new ReturnInst() );
	break;
    }
    case EXIT : 
    {
	if (echo) bb->setName("EXIT");
	// Get the result value
	LoadInst* op1 = cast<LoadInst>(pop_integer(bb));

	// Call exit(3)
	std::vector<Value*> params;
	params.push_back(op1);
	CallInst* call_inst = new CallInst( TheExit, params );
	bb->getInstList().push_back( call_inst );
	break;
    }
    case TAB :
    {
	if (echo) bb->setName("TAB");
	// Get the format string for a character
	std::vector<Value*> indexVec;
	indexVec.push_back( Zero );
	indexVec.push_back( Zero );
	GetElementPtrInst* format_gep = 
	    new GetElementPtrInst( ChrFormat, indexVec );
	bb->getInstList().push_back( format_gep );

	// Get the character to print (a newline)
	ConstantSInt* newline = ConstantSInt::get(Type::IntTy, 
	    static_cast<int>('\t'));

	// Call printf
	std::vector<Value*> args;
	args.push_back( format_gep );
	args.push_back( newline );
	bb->getInstList().push_back( new CallInst( ThePrintf, args ) );
	break;
    }
    case SPACE : 
    {
	if (echo) bb->setName("SPACE");
	// Get the format string for a character
	std::vector<Value*> indexVec;
	indexVec.push_back( Zero );
	indexVec.push_back( Zero );
	GetElementPtrInst* format_gep = 
	    new GetElementPtrInst( ChrFormat, indexVec );
	bb->getInstList().push_back( format_gep );

	// Get the character to print (a newline)
	ConstantSInt* newline = ConstantSInt::get(Type::IntTy, 
	    static_cast<int>(' '));

	// Call printf
	std::vector<Value*> args;
	args.push_back( format_gep );
	args.push_back( newline );
	bb->getInstList().push_back( new CallInst( ThePrintf, args ) );
	break;
    }
    case CR : 
    {
	if (echo) bb->setName("CR");
	// Get the format string for a character
	std::vector<Value*> indexVec;
	indexVec.push_back( Zero );
	indexVec.push_back( Zero );
	GetElementPtrInst* format_gep = 
	    new GetElementPtrInst( ChrFormat, indexVec );
	bb->getInstList().push_back( format_gep );

	// Get the character to print (a newline)
	ConstantSInt* newline = ConstantSInt::get(Type::IntTy, 
	    static_cast<int>('\n'));

	// Call printf
	std::vector<Value*> args;
	args.push_back( format_gep );
	args.push_back( newline );
	bb->getInstList().push_back( new CallInst( ThePrintf, args ) );
	break;
    }
    case IN_STR : 
    {
	if (echo) bb->setName("IN_STR");
	// Make room for the value result
	incr_stack_index(bb);
	GetElementPtrInst* gep_value = 
	    cast<GetElementPtrInst>(get_stack_pointer(bb));
	CastInst* caster = 
	    new CastInst( gep_value, PointerType::get( Type::SByteTy ) );

	// Make room for the count result
	incr_stack_index(bb);
	GetElementPtrInst* gep_count = 
	    cast<GetElementPtrInst>(get_stack_pointer(bb));

	// Call scanf(3)
	std::vector<Value*> args;
	args.push_back( InStrFormat );
	args.push_back( caster );
	CallInst* scanf = new CallInst( TheScanf, args );
	bb->getInstList().push_back( scanf );

	// Store the result
	bb->getInstList().push_back( new StoreInst( scanf, gep_count ) );
	break;
    }
    case IN_NUM : 
    {
	if (echo) bb->setName("IN_NUM");
	// Make room for the value result
	incr_stack_index(bb);
	GetElementPtrInst* gep_value = 
	    cast<GetElementPtrInst>(get_stack_pointer(bb));

	// Make room for the count result
	incr_stack_index(bb);
	GetElementPtrInst* gep_count = 
	    cast<GetElementPtrInst>(get_stack_pointer(bb));

	// Call scanf(3)
	std::vector<Value*> args;
	args.push_back( InStrFormat );
	args.push_back( gep_value );
	CallInst* scanf = new CallInst( TheScanf, args );
	bb->getInstList().push_back( scanf );

	// Store the result
	bb->getInstList().push_back( new StoreInst( scanf, gep_count ) );
	break;
    }
    case IN_CHAR :
    {
	if (echo) bb->setName("IN_CHAR");
	// Make room for the value result
	incr_stack_index(bb);
	GetElementPtrInst* gep_value = 
	    cast<GetElementPtrInst>(get_stack_pointer(bb));

	// Make room for the count result
	incr_stack_index(bb);
	GetElementPtrInst* gep_count = 
	    cast<GetElementPtrInst>(get_stack_pointer(bb));

	// Call scanf(3)
	std::vector<Value*> args;
	args.push_back( InChrFormat );
	args.push_back( gep_value );
	CallInst* scanf = new CallInst( TheScanf, args );
	bb->getInstList().push_back( scanf );

	// Store the result
	bb->getInstList().push_back( new StoreInst( scanf, gep_count ) );
	break;
    }
    case OUT_STR : 
    {
	if (echo) bb->setName("OUT_STR");
	LoadInst* op1 = cast<LoadInst>(stack_top(bb));

	// Get the address of the format string
	std::vector<Value*> indexVec;
	indexVec.push_back( Zero );
	indexVec.push_back( Zero );
	GetElementPtrInst* format_gep = 
	    new GetElementPtrInst( StrFormat, indexVec );
	bb->getInstList().push_back( format_gep );
	// Build function call arguments
	std::vector<Value*> args;
	args.push_back( format_gep );
	args.push_back( op1 );
	// Call printf
	bb->getInstList().push_back( new CallInst( ThePrintf, args ) );
	break;
    }
    case OUT_NUM : 
    {
	if (echo) bb->setName("OUT_NUM");
	// Pop the numeric operand off the stack
	LoadInst* op1 = cast<LoadInst>(stack_top(bb));

	// Get the address of the format string
	std::vector<Value*> indexVec;
	indexVec.push_back( Zero );
	indexVec.push_back( Zero );
	GetElementPtrInst* format_gep = 
	    new GetElementPtrInst( NumFormat, indexVec );
	bb->getInstList().push_back( format_gep );

	// Build function call arguments
	std::vector<Value*> args;
	args.push_back( format_gep );
	args.push_back( op1 );

	// Call printf
	bb->getInstList().push_back( new CallInst( ThePrintf, args ) );
	break;
    }
    case OUT_CHAR :
    {
	if (echo) bb->setName("OUT_CHAR");
	// Pop the character operand off the stack
	LoadInst* op1 = cast<LoadInst>(stack_top(bb));

	// Get the address of the format string
	std::vector<Value*> indexVec;
	indexVec.push_back( Zero );
	indexVec.push_back( Zero );
	GetElementPtrInst* format_gep = 
	    new GetElementPtrInst( ChrFormat, indexVec );
	bb->getInstList().push_back( format_gep );

	// Build function call arguments
	std::vector<Value*> args;
	args.push_back( format_gep );
	args.push_back( op1 );
	// Call printf
	bb->getInstList().push_back( new CallInst( ThePrintf, args ) );
	break;
    }
    default :
    {
	ThrowException(std::string("Compiler Error: Unhandled token #"));
    }
    }

    // Return the basic block
    return bb;
}
