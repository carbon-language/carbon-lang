/*===-- Lexer.l - Scanner for llvm assembly files ----------------*- C++ -*--=//
//
//  This file implements the flex scanner for LLVM assembly languages files.
//
//===------------------------------------------------------------------------=*/

%option prefix="llvmAsm"
%option yylineno
%option nostdinit
%option never-interactive
%option batch
%option noyywrap
%option nodefault
%option 8bit
%option outfile="Lexer.cpp"
%option ecs
%option noreject
%option noyymore

%{
#include "ParserInternals.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include <list>
#include "llvmAsmParser.h"

#define RET_TOK(type, Enum, sym) \
  llvmAsmlval.type = Instruction::Enum; return sym


// TODO: All of the static identifiers are figured out by the lexer, 
// these should be hashed.


// atoull - Convert an ascii string of decimal digits into the unsigned long
// long representation... this does not have to do input error checking, 
// because we know that the input will be matched by a suitable regex...
//
uint64_t atoull(const char *Buffer) {
  uint64_t Result = 0;
  for (; *Buffer; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes) {  // Uh, oh, overflow detected!!!
      ThrowException("constant bigger than 64 bits detected!");
    }
  }
  return Result;
}


#define YY_NEVER_INTERACTIVE 1
%}



/* Comments start with a ; and go till end of line */
Comment    ;.*

/* Variable(Def) identifiers start with a % sign */
VarID       %[a-zA-Z$._][a-zA-Z$._0-9]*

/* Label identifiers end with a colon */
Label       [a-zA-Z$._0-9]+:

/* Quoted names can contain any character except " and \ */
StringConstant \"[^\"]+\"


/* [PN]Integer: match positive and negative literal integer values that
 * are preceeded by a '%' character.  These represent unnamed variable slots.
 */
EPInteger     %[0-9]+
ENInteger    %-[0-9]+


/* E[PN]Integer: match positive and negative literal integer values */
PInteger   [0-9]+
NInteger  -[0-9]+

%%

{Comment}       { /* Ignore comments for now */ }

begin           { return BEGINTOK; }
end             { return END; }
true            { return TRUE;  }
false           { return FALSE; }
declare         { return DECLARE; }
implementation  { return IMPLEMENTATION; }

-               { cerr << "deprecated argument '-' used!\n"; return '-'; }
bb              { cerr << "deprecated type 'bb' used!\n"; llvmAsmlval.TypeVal = Type::LabelTy; return LABEL;}

void            { llvmAsmlval.TypeVal = Type::VoidTy  ; return VOID;   }
bool            { llvmAsmlval.TypeVal = Type::BoolTy  ; return BOOL;   }
sbyte           { llvmAsmlval.TypeVal = Type::SByteTy ; return SBYTE;  }
ubyte           { llvmAsmlval.TypeVal = Type::UByteTy ; return UBYTE;  }
short           { llvmAsmlval.TypeVal = Type::ShortTy ; return SHORT;  }
ushort          { llvmAsmlval.TypeVal = Type::UShortTy; return USHORT; }
int             { llvmAsmlval.TypeVal = Type::IntTy   ; return INT;    }
uint            { llvmAsmlval.TypeVal = Type::UIntTy  ; return UINT;   }
long            { llvmAsmlval.TypeVal = Type::LongTy  ; return LONG;   }
ulong           { llvmAsmlval.TypeVal = Type::ULongTy ; return ULONG;  }
float           { llvmAsmlval.TypeVal = Type::FloatTy ; return FLOAT;  }
double          { llvmAsmlval.TypeVal = Type::DoubleTy; return DOUBLE; }

type            { llvmAsmlval.TypeVal = Type::TypeTy  ; return TYPE;   }

label           { llvmAsmlval.TypeVal = Type::LabelTy ; return LABEL;  }

neg             { RET_TOK(UnaryOpVal, Neg, NEG); }
not             { RET_TOK(UnaryOpVal, Not, NOT); }

phi             { return PHI; }
call            { return CALL; }
add             { RET_TOK(BinaryOpVal, Add, ADD); }
sub             { RET_TOK(BinaryOpVal, Sub, SUB); }
mul             { RET_TOK(BinaryOpVal, Mul, MUL); }
div             { RET_TOK(BinaryOpVal, Div, DIV); }
rem             { RET_TOK(BinaryOpVal, Rem, REM); }
setne           { RET_TOK(BinaryOpVal, SetNE, SETNE); }
seteq           { RET_TOK(BinaryOpVal, SetEQ, SETEQ); }
setlt           { RET_TOK(BinaryOpVal, SetLT, SETLT); }
setgt           { RET_TOK(BinaryOpVal, SetGT, SETGT); }
setle           { RET_TOK(BinaryOpVal, SetLE, SETLE); }
setge           { RET_TOK(BinaryOpVal, SetGE, SETGE); }

ret             { RET_TOK(TermOpVal, Ret, RET); }
br              { RET_TOK(TermOpVal, Br, BR); }
switch          { RET_TOK(TermOpVal, Switch, SWITCH); }


malloc          { RET_TOK(MemOpVal, Malloc, MALLOC); }
alloca          { RET_TOK(MemOpVal, Alloca, ALLOCA); }
free            { RET_TOK(MemOpVal, Free, FREE); }
load            { RET_TOK(MemOpVal, Load, LOAD); }
store           { RET_TOK(MemOpVal, Store, STORE); }
getfield        { RET_TOK(MemOpVal, GetField, GETFIELD); }
putfield        { RET_TOK(MemOpVal, PutField, PUTFIELD); }


{VarID}         { llvmAsmlval.StrVal = strdup(yytext+1); return VAR_ID; }
{Label}         { 
                  yytext[strlen(yytext)-1] = 0;  // nuke colon
		  llvmAsmlval.StrVal = strdup(yytext); 
		  return LABELSTR; 
                }

{StringConstant} { 
                  yytext[strlen(yytext)-1] = 0;           // nuke end quote
		  llvmAsmlval.StrVal = strdup(yytext+1);  // Nuke start quote 
		  return STRINGCONSTANT;
                }


{PInteger}      { llvmAsmlval.UInt64Val = atoull(yytext); return EUINT64VAL; }
{NInteger}      { 
                  uint64_t Val = atoull(yytext+1);
		  // +1:  we have bigger negative range
		  if (Val > (uint64_t)INT64_MAX+1)
		    ThrowException("Constant too large for signed 64 bits!");
                  llvmAsmlval.SInt64Val = -Val; 
		  return ESINT64VAL; 
                }


{EPInteger}     { llvmAsmlval.UIntVal = atoull(yytext+1); return UINTVAL; }
{ENInteger}     {
                  uint64_t Val = atoull(yytext+2);
		  // +1:  we have bigger negative range
		  if (Val > (uint64_t)INT32_MAX+1)
		    ThrowException("Constant too large for signed 32 bits!");
                  llvmAsmlval.SIntVal = -Val;
		  return SINTVAL;
                }


[ \t\n]         { /* Ignore whitespace */ }
.               { /*printf("'%s'", yytext);*/ return yytext[0]; }

%%
