//===-- llvmAsmParser.y - Parser for llvm assembly files --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This file implements the bison parser for LLVM assembly languages files.
//
//===----------------------------------------------------------------------===//

%debug

%{
#include "StackerCompiler.h"
#include "llvm/SymbolTable.h"
#include "llvm/Module.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "Support/STLExtras.h"
#include "Support/DepthFirstIterator.h"
#include <list>
#include <utility>
#include <algorithm>

#define YYERROR_VERBOSE 1
#define SCI StackerCompiler::TheInstance

int yyerror(const char *ErrorMsg); // Forward declarations to prevent "implicit
int yylex();                       // declaration" of xxx warnings.
int yyparse();

%}

%union 
{
  llvm::Module*		ModuleVal;
  llvm::Function* 	FunctionVal;
  llvm::BasicBlock*	BasicBlockVal;
  uint32_t              IntegerVal;
  char*                 StringVal;
}

/* Typed Productions */
%type <ModuleVal>	Module DefinitionList
%type <FunctionVal>	Definition ForwardDef ColonDef MainDef
%type <FunctionVal>	WordList
%type <BasicBlockVal>	Word

/* Typed Tokens */
%token <IntegerVal>	INTEGER
%token <StringVal>	STRING IDENTIFIER

/* Terminal Tokens */
%token 			SEMI COLON FORWARD MAIN DUMP
%token  		TRUE FALSE LESS MORE LESS_EQUAL MORE_EQUAL NOT_EQUAL EQUAL
%token 			PLUS MINUS INCR DECR MULT DIV MODULUS NEGATE ABS MIN MAX STAR_SLASH 
%token 			AND OR XOR LSHIFT RSHIFT 
%token 			DROP DROP2 NIP NIP2 DUP DUP2 SWAP SWAP2	OVER OVER2 ROT ROT2 
%token			RROT RROT2 TUCK TUCK2 ROLL PICK SELECT
%token 			MALLOC FREE GET PUT
%token 			IF ELSE ENDIF WHILE END RECURSE RETURN EXIT
%token 			TAB SPACE CR IN_STR IN_NUM IN_CHAR OUT_STR OUT_NUM OUT_CHAR

/* Start Token */
%start Module

%%

/* A module is just a DefinitionList */
Module 		: 				{ SCI->handle_module_start( ); } 
	 	DefinitionList 			{ $$ = SCI->handle_module_end( $2 ); } ;

/* A Definitionlist is just a sequence of definitions */
DefinitionList	: DefinitionList Definition 	{ $$ = SCI->handle_definition_list_end( $1, $2 ); }
		| /* empty */ 			{ $$ = SCI->handle_definition_list_start(); } ;

/* A definition can be one of three flavors */
Definition 	: ForwardDef 			{ $$ = $1; }
	   	| ColonDef			{ $$ = $1; }
	   	| MainDef			{ $$ = $1; } ;

/* Forward definitions just introduce a name */
ForwardDef : FORWARD IDENTIFIER SEMI 		{ $$ = SCI->handle_forward( $2 ); } ;

/* The main definition has to generate additional code so we treat it specially */
MainDef : COLON MAIN WordList SEMI		{ $$ = SCI->handle_main_definition($3); } ;

/* Regular definitions have a name and a WordList */
ColonDef : COLON IDENTIFIER WordList SEMI 	{ $$ = SCI->handle_definition( $2, $3 ); } ;

/* A WordList is just a sequence of words */
WordList : WordList Word 			{ $$ = SCI->handle_word_list_end( $1, $2 ); } 
	 | /* empty */				{ $$ = SCI->handle_word_list_start(); } ;

/* A few "words" have a funky syntax */
/* FIXME: The body of compound words can currently only be function calls */
/* This is not acceptable, it should be a WordList, but that produces a Function */
/* Which is hard to merge into the function the compound statement is working on */
Word : IF IDENTIFIER ELSE IDENTIFIER ENDIF	{ $$ = SCI->handle_if( $2, $4 ); } 
     | IF IDENTIFIER ENDIF			{ $$ = SCI->handle_if( $2 ); } 
     | WHILE IDENTIFIER END			{ $$ = SCI->handle_while( $2 ); } ;

/* A few words are handled specially */
Word : IDENTIFIER 				{ $$ = SCI->handle_identifier( $1 ); } ;
Word : STRING 					{ $$ = SCI->handle_string( $1 ); } ;
Word : INTEGER 					{ $$ = SCI->handle_integer( $1 ); } ;

/* Everything else is a terminal symbol and goes to handle_word */
Word : TRUE					{ $$ = SCI->handle_word( TRUE ); } ;
Word : FALSE					{ $$ = SCI->handle_word( FALSE ); } ;
Word : LESS					{ $$ = SCI->handle_word( LESS ); } ;
Word : MORE					{ $$ = SCI->handle_word( MORE ); } ;
Word : LESS_EQUAL				{ $$ = SCI->handle_word( LESS_EQUAL ); } ;
Word : MORE_EQUAL				{ $$ = SCI->handle_word( MORE_EQUAL ); } ;
Word : NOT_EQUAL				{ $$ = SCI->handle_word( NOT_EQUAL ); } ;
Word : EQUAL					{ $$ = SCI->handle_word( EQUAL ); } ;
Word : PLUS					{ $$ = SCI->handle_word( PLUS ); } ;
Word : MINUS					{ $$ = SCI->handle_word( MINUS ); } ;
Word : INCR					{ $$ = SCI->handle_word( INCR ); } ;
Word : DECR					{ $$ = SCI->handle_word( DECR ); } ;
Word : MULT					{ $$ = SCI->handle_word( MULT ); } ;
Word : DIV					{ $$ = SCI->handle_word( DIV ); } ;
Word : MODULUS					{ $$ = SCI->handle_word( MODULUS ); } ;
Word : NEGATE					{ $$ = SCI->handle_word( NEGATE ); } ;
Word : ABS					{ $$ = SCI->handle_word( ABS ); } ;
Word : MIN					{ $$ = SCI->handle_word( MIN ); } ;
Word : MAX					{ $$ = SCI->handle_word( MAX ); } ;
Word : STAR_SLASH				{ $$ = SCI->handle_word( STAR_SLASH ); } ;
Word : AND					{ $$ = SCI->handle_word( AND ); } ;
Word : OR					{ $$ = SCI->handle_word( OR ); } ;
Word : XOR					{ $$ = SCI->handle_word( XOR ); } ;
Word : LSHIFT					{ $$ = SCI->handle_word( LSHIFT ); } ;
Word : RSHIFT					{ $$ = SCI->handle_word( RSHIFT ); } ;
Word : DROP					{ $$ = SCI->handle_word( DROP ); } ;
Word : DROP2					{ $$ = SCI->handle_word( DROP2 ); } ;
Word : NIP					{ $$ = SCI->handle_word( NIP ); } ;
Word : NIP2					{ $$ = SCI->handle_word( NIP2 ); } ;
Word : DUP					{ $$ = SCI->handle_word( DUP ); } ;
Word : DUP2					{ $$ = SCI->handle_word( DUP2 ); } ;
Word : SWAP					{ $$ = SCI->handle_word( SWAP ); } ;
Word : SWAP2					{ $$ = SCI->handle_word( SWAP2 ); } ;
Word : OVER					{ $$ = SCI->handle_word( OVER ); } ;
Word : OVER2					{ $$ = SCI->handle_word( OVER2 ); } ;
Word : ROT					{ $$ = SCI->handle_word( ROT ); } ;
Word : ROT2					{ $$ = SCI->handle_word( ROT2 ); } ;
Word : RROT					{ $$ = SCI->handle_word( RROT ); } ;
Word : RROT2					{ $$ = SCI->handle_word( RROT2 ); } ;
Word : TUCK					{ $$ = SCI->handle_word( TUCK ); } ;
Word : TUCK2					{ $$ = SCI->handle_word( TUCK2 ); } ;
Word : ROLL					{ $$ = SCI->handle_word( ROLL ); } ;
Word : PICK					{ $$ = SCI->handle_word( PICK ); } ;
Word : SELECT					{ $$ = SCI->handle_word( SELECT ); } ;
Word : MALLOC					{ $$ = SCI->handle_word( MALLOC ); } ;
Word : FREE					{ $$ = SCI->handle_word( FREE ); } ;
Word : GET					{ $$ = SCI->handle_word( GET ); } ;
Word : PUT					{ $$ = SCI->handle_word( PUT ); } ;
Word : RECURSE					{ $$ = SCI->handle_word( RECURSE ); } ;
Word : RETURN					{ $$ = SCI->handle_word( RETURN ); } ;
Word : EXIT					{ $$ = SCI->handle_word( EXIT ); } ;
Word : TAB					{ $$ = SCI->handle_word( TAB ); };
Word : SPACE					{ $$ = SCI->handle_word( SPACE ); } ;
Word : CR					{ $$ = SCI->handle_word( CR ); } ;
Word : IN_STR					{ $$ = SCI->handle_word( IN_STR ); } ;
Word : IN_NUM					{ $$ = SCI->handle_word( IN_NUM ); } ;
Word : IN_CHAR					{ $$ = SCI->handle_word( IN_CHAR ); } ;
Word : OUT_STR					{ $$ = SCI->handle_word( OUT_STR ); } ;
Word : OUT_NUM					{ $$ = SCI->handle_word( OUT_NUM ); } ;
Word : OUT_CHAR					{ $$ = SCI->handle_word( OUT_CHAR ); } ;
Word : DUMP					{ $$ = SCI->handle_word( DUMP ); } ;

%%

/* Handle messages a little more nicely than the default yyerror */
int yyerror(const char *ErrorMsg) {
  std::string where 
    = std::string((SCI->filename() == "-") ? std::string("<stdin>") : SCI->filename())
                  + ":" + utostr((unsigned) Stackerlineno ) + ": ";
  std::string errMsg = std::string(ErrorMsg) + "\n" + where + " while reading ";
  if (yychar == YYEMPTY)
    errMsg += "end-of-file.";
  else
    errMsg += "token: '" + std::string(Stackertext, Stackerleng) + "'";
  StackerCompiler::ThrowException(errMsg);
  return 0;
}
