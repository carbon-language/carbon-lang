//===-- StackerCompiler.h - Interface to the Stacker Compiler ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and donated to the LLVM research 
// group and is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file defines the various variables that are shared among the 
//  different components of the parser...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_STACKERCOMPILER_H
#define LLVM_STACKERCOMPILER_H

#include <llvm/Constants.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Function.h>
#include <llvm/Instruction.h>
#include <llvm/Module.h>
#include <llvm/Assembly/Parser.h>
#include <Support/StringExtras.h>

using namespace llvm;

// Global variables exported from the lexer...
extern std::FILE *Stackerin;
extern int Stackerlineno;
extern char* Stackertext;
extern int Stackerleng;

/// @brief This class provides the Compiler for the Stacker language. 
/// 
/// The main method to call is \c compile. The other methods are
/// all internal to the compiler and protected. In general the 
/// handle_* methods are called by the BISON generated parser
/// (see StackerParser.y). The methods returning Instruction* all
/// produce some snippet of code to manipulate the stack in some
/// way. These functions are just conveniences as they are used
/// often by the compiler.
class StackerCompiler
{
    /// @name Constructors and Operators
    /// @{
    public:
	/// Default Constructor
	StackerCompiler();

	/// Destructor
	~StackerCompiler();
    private:
	/// Do not copy StackerCompilers
	StackerCompiler(const StackerCompiler&);

	/// Do not copy StackerCompilers.
	StackerCompiler& operator=(const StackerCompiler& );

    /// @}
    /// @name High Level Interface
    /// @{
    public:
	/// @brief Compile a single file to LLVM bytecode.
	///
	/// To use the StackerCompiler, just create one on
	/// the stack and call this method.
	Module* compile( 
	    const std::string& filename, ///< File to compile
	    bool echo, ///< Causes compiler to echo output
	    size_t stack_size ); ///< Size of generated stack
    /// @}
    /// @name Accessors
    /// @{
    public:
	/// @brief Returns the name of the file being compiled.
	std::string& filename() { return CurFilename; }

    /// @}
    /// @name Parse Handling Methods
    /// @{
    private:
	/// Allow only the parser to access these methods. No
	/// one else should call them.
	friend int Stackerparse();

	/// @brief Handle the start of a module
	Module* handle_module_start();

	/// @brief Handle the end of a module
	/// @param mod The module we're defining.
	Module* handle_module_end( Module* mod );

	/// @brief Handle the start of a list of definitions
	Module* handle_definition_list_start( );

	/// @brief Handle the end of a list of definitions
	/// @param mod The module we're constructing
	/// @param definition A definition (function) to add to the module
	Module* handle_definition_list_end( Module* mod, Function* definition );

	/// @brief Handle creation of the MAIN definition
	/// @param func The function to be used as the MAIN definition
	Function* handle_main_definition( Function* func );

	/// @brief Handle a forward definition
	/// @param name The name of the definition being declared
	Function* handle_forward( char* name );

	/// @brief Handle a general definition
	/// @param name The name of the definition being defined
	/// @param func The Function definition.
	Function* handle_definition( char* name, Function* func );

	/// @brief Handle the start of a definition's word list
	Function* handle_word_list_start();

	/// @brief Handle the end of a definition's word list
	/// @param func The function to which the basic block is added
	/// @param next The block to add to the function
	Function* handle_word_list_end( Function* func, BasicBlock* next );

	/// @brief Handle an if statement, possibly without an else
	/// @brief ifTrue The block to execute if true
	/// @brief ifFalse The optional block to execute if false
	BasicBlock* handle_if( char* ifTrue, char* ifFalse = 0 );

	/// @brief Handle a while statement
	/// @brief todo The block to repeatedly execute
	BasicBlock* handle_while( char* todo );

	/// @brief Handle an identifier to call the identified definition
	/// @param name The name of the identifier to be called.
	BasicBlock* handle_identifier( char * name );

	/// @brief Handle the push of a string onto the stack
	/// @param value The string to be pushed.
	BasicBlock* handle_string( char * value );

	/// @brief Handle the push of an integer onto the stack.
	/// @param value The integer value to be pushed.
	BasicBlock* handle_integer( const int32_t value );

	/// @brief Handle one of the reserved words (given as a token)
	BasicBlock* handle_word( int tkn );

    /// @}
    /// @name Utility functions
    /// @{
    public:
        /// @brief Throws an exception to indicate an error
        /// @param message The message to be output
	/// @param line Override for the current line no
	static inline void ThrowException( const std::string &message, 
		int line = -1) 	    
	{
	  if (line == -1) line = Stackerlineno;
	  // TODO: column number in exception
	  throw ParseException(TheInstance->CurFilename, message, line);
	}
    private:
	/// @brief Generate code to increment the stack index
	Instruction* incr_stack_index( BasicBlock* bb, Value* );
	/// @brief Generate code to decrement the stack index.
	Instruction* decr_stack_index( BasicBlock* bb, Value* );
	/// @brief Generate code to dereference the top of stack.
	Instruction* get_stack_pointer( BasicBlock* bb, Value* );
	/// @brief Generate code to push any value onto the stack.
	Instruction* push_value( BasicBlock* bb, Value* value );
	/// @brief Generate code to push a constant integer onto the stack.
	Instruction* push_integer( BasicBlock* bb, int32_t value );
	/// @brief Generate code to pop an integer off the stack.
	Instruction* pop_integer( BasicBlock* bb );
	/// @brief Generate code to push a string pointer onto the stack.
	Instruction* push_string( BasicBlock* bb, const char* value );
	/// @brief Generate code to pop a string pointer off the stack.
	Instruction* pop_string( BasicBlock* bb );
	/// @brief Generate code to get the top stack element.
	Instruction* stack_top( BasicBlock* bb, Value* index );
	/// @brief Generate code to get the top stack element as a string.
	Instruction* stack_top_string( BasicBlock* bb, Value* index );
	/// @brief Generate code to replace the top element of the stack.
	Instruction* replace_top( BasicBlock* bb, Value* new_top, Value* index);

    /// @}
    /// @name Data Members (used during parsing)
    /// @{
    public:
        static StackerCompiler*	TheInstance;	///< The instance for the parser

    private:
	std::string 		CurFilename;	///< Current file name
	Module* 		TheModule; 	///< Module instance we'll build
        Function*		TheFunction;	///< Function we're building
	FunctionType* 		DefinitionType; ///< FT for Definitions
	GlobalVariable*		TheStack;	///< For referencing _stack_
	GlobalVariable*		TheIndex;	///< For referencing _index_
	Function*		TheScanf;	///< External input function
	Function*		ThePrintf;	///< External output function
	Function*		TheExit;	///< External exit function
	GlobalVariable*		StrFormat;	///< Format for strings
	GlobalVariable*		NumFormat;	///< Format for numbers
	GlobalVariable*		ChrFormat;	///< Format for chars
	GlobalVariable*		InStrFormat;	///< Format for input strings
	GlobalVariable*		InNumFormat;	///< Format for input numbers
	GlobalVariable*		InChrFormat;	///< Format for input chars
	ConstantInt*		Zero;		///< long constant 0
	ConstantInt*		One;		///< long constant 1
	ConstantInt*		Two;		///< long constant 2
	ConstantInt*		Three;		///< long constant 3
	ConstantInt*		Four;		///< long constant 4
	ConstantInt*		Five;		///< long constant 5
	ConstantInt*		IZero;		///< int constant 0
	ConstantInt*		IOne;		///< int constant 1
	ConstantInt*		ITwo;		///< int constant 2
	std::vector<Value*> 	no_arguments;	///< no arguments for Stacker
	bool 			echo;		///< Echo flag
	size_t			stack_size;	///< Size of stack to gen.
	ArrayType*		stack_type;	///< The type of the stack
    /// @}
};

#endif
