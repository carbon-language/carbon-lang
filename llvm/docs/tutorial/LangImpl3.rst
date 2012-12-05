========================================
Kaleidoscope: Code generation to LLVM IR
========================================

.. contents::
   :local:

Written by `Chris Lattner <mailto:sabre@nondot.org>`_

Chapter 3 Introduction
======================

Welcome to Chapter 3 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. This chapter shows you how to transform
the `Abstract Syntax Tree <LangImpl2.html>`_, built in Chapter 2, into
LLVM IR. This will teach you a little bit about how LLVM does things, as
well as demonstrate how easy it is to use. It's much more work to build
a lexer and parser than it is to generate LLVM IR code. :)

**Please note**: the code in this chapter and later require LLVM 2.2 or
later. LLVM 2.1 and before will not work with it. Also note that you
need to use a version of this tutorial that matches your LLVM release:
If you are using an official LLVM release, use the version of the
documentation included with your release or on the `llvm.org releases
page <http://llvm.org/releases/>`_.

Code Generation Setup
=====================

In order to generate LLVM IR, we want some simple setup to get started.
First we define virtual code generation (codegen) methods in each AST
class:

.. code-block:: c++

    /// ExprAST - Base class for all expression nodes.
    class ExprAST {
    public:
      virtual ~ExprAST() {}
      virtual Value *Codegen() = 0;
    };

    /// NumberExprAST - Expression class for numeric literals like "1.0".
    class NumberExprAST : public ExprAST {
      double Val;
    public:
      NumberExprAST(double val) : Val(val) {}
      virtual Value *Codegen();
    };
    ...

The Codegen() method says to emit IR for that AST node along with all
the things it depends on, and they all return an LLVM Value object.
"Value" is the class used to represent a "`Static Single Assignment
(SSA) <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
register" or "SSA value" in LLVM. The most distinct aspect of SSA values
is that their value is computed as the related instruction executes, and
it does not get a new value until (and if) the instruction re-executes.
In other words, there is no way to "change" an SSA value. For more
information, please read up on `Static Single
Assignment <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
- the concepts are really quite natural once you grok them.

Note that instead of adding virtual methods to the ExprAST class
hierarchy, it could also make sense to use a `visitor
pattern <http://en.wikipedia.org/wiki/Visitor_pattern>`_ or some other
way to model this. Again, this tutorial won't dwell on good software
engineering practices: for our purposes, adding a virtual method is
simplest.

The second thing we want is an "Error" method like we used for the
parser, which will be used to report errors found during code generation
(for example, use of an undeclared parameter):

.. code-block:: c++

    Value *ErrorV(const char *Str) { Error(Str); return 0; }

    static Module *TheModule;
    static IRBuilder<> Builder(getGlobalContext());
    static std::map<std::string, Value*> NamedValues;

The static variables will be used during code generation. ``TheModule``
is the LLVM construct that contains all of the functions and global
variables in a chunk of code. In many ways, it is the top-level
structure that the LLVM IR uses to contain code.

The ``Builder`` object is a helper object that makes it easy to generate
LLVM instructions. Instances of the
```IRBuilder`` <http://llvm.org/doxygen/IRBuilder_8h-source.html>`_
class template keep track of the current place to insert instructions
and has methods to create new instructions.

The ``NamedValues`` map keeps track of which values are defined in the
current scope and what their LLVM representation is. (In other words, it
is a symbol table for the code). In this form of Kaleidoscope, the only
things that can be referenced are function parameters. As such, function
parameters will be in this map when generating code for their function
body.

With these basics in place, we can start talking about how to generate
code for each expression. Note that this assumes that the ``Builder``
has been set up to generate code *into* something. For now, we'll assume
that this has already been done, and we'll just use it to emit code.

Expression Code Generation
==========================

Generating LLVM code for expression nodes is very straightforward: less
than 45 lines of commented code for all four of our expression nodes.
First we'll do numeric literals:

.. code-block:: c++

    Value *NumberExprAST::Codegen() {
      return ConstantFP::get(getGlobalContext(), APFloat(Val));
    }

In the LLVM IR, numeric constants are represented with the
``ConstantFP`` class, which holds the numeric value in an ``APFloat``
internally (``APFloat`` has the capability of holding floating point
constants of Arbitrary Precision). This code basically just creates
and returns a ``ConstantFP``. Note that in the LLVM IR that constants
are all uniqued together and shared. For this reason, the API uses the
"foo::get(...)" idiom instead of "new foo(..)" or "foo::Create(..)".

.. code-block:: c++

    Value *VariableExprAST::Codegen() {
      // Look this variable up in the function.
      Value *V = NamedValues[Name];
      return V ? V : ErrorV("Unknown variable name");
    }

References to variables are also quite simple using LLVM. In the simple
version of Kaleidoscope, we assume that the variable has already been
emitted somewhere and its value is available. In practice, the only
values that can be in the ``NamedValues`` map are function arguments.
This code simply checks to see that the specified name is in the map (if
not, an unknown variable is being referenced) and returns the value for
it. In future chapters, we'll add support for `loop induction
variables <LangImpl5.html#for>`_ in the symbol table, and for `local
variables <LangImpl7.html#localvars>`_.

.. code-block:: c++

    Value *BinaryExprAST::Codegen() {
      Value *L = LHS->Codegen();
      Value *R = RHS->Codegen();
      if (L == 0 || R == 0) return 0;

      switch (Op) {
      case '+': return Builder.CreateFAdd(L, R, "addtmp");
      case '-': return Builder.CreateFSub(L, R, "subtmp");
      case '*': return Builder.CreateFMul(L, R, "multmp");
      case '<':
        L = Builder.CreateFCmpULT(L, R, "cmptmp");
        // Convert bool 0/1 to double 0.0 or 1.0
        return Builder.CreateUIToFP(L, Type::getDoubleTy(getGlobalContext()),
                                    "booltmp");
      default: return ErrorV("invalid binary operator");
      }
    }

Binary operators start to get more interesting. The basic idea here is
that we recursively emit code for the left-hand side of the expression,
then the right-hand side, then we compute the result of the binary
expression. In this code, we do a simple switch on the opcode to create
the right LLVM instruction.

In the example above, the LLVM builder class is starting to show its
value. IRBuilder knows where to insert the newly created instruction,
all you have to do is specify what instruction to create (e.g. with
``CreateFAdd``), which operands to use (``L`` and ``R`` here) and
optionally provide a name for the generated instruction.

One nice thing about LLVM is that the name is just a hint. For instance,
if the code above emits multiple "addtmp" variables, LLVM will
automatically provide each one with an increasing, unique numeric
suffix. Local value names for instructions are purely optional, but it
makes it much easier to read the IR dumps.

`LLVM instructions <../LangRef.html#instref>`_ are constrained by strict
rules: for example, the Left and Right operators of an `add
instruction <../LangRef.html#i_add>`_ must have the same type, and the
result type of the add must match the operand types. Because all values
in Kaleidoscope are doubles, this makes for very simple code for add,
sub and mul.

On the other hand, LLVM specifies that the `fcmp
instruction <../LangRef.html#i_fcmp>`_ always returns an 'i1' value (a
one bit integer). The problem with this is that Kaleidoscope wants the
value to be a 0.0 or 1.0 value. In order to get these semantics, we
combine the fcmp instruction with a `uitofp
instruction <../LangRef.html#i_uitofp>`_. This instruction converts its
input integer into a floating point value by treating the input as an
unsigned value. In contrast, if we used the `sitofp
instruction <../LangRef.html#i_sitofp>`_, the Kaleidoscope '<' operator
would return 0.0 and -1.0, depending on the input value.

.. code-block:: c++

    Value *CallExprAST::Codegen() {
      // Look up the name in the global module table.
      Function *CalleeF = TheModule->getFunction(Callee);
      if (CalleeF == 0)
        return ErrorV("Unknown function referenced");

      // If argument mismatch error.
      if (CalleeF->arg_size() != Args.size())
        return ErrorV("Incorrect # arguments passed");

      std::vector<Value*> ArgsV;
      for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        ArgsV.push_back(Args[i]->Codegen());
        if (ArgsV.back() == 0) return 0;
      }

      return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
    }

Code generation for function calls is quite straightforward with LLVM.
The code above initially does a function name lookup in the LLVM
Module's symbol table. Recall that the LLVM Module is the container that
holds all of the functions we are JIT'ing. By giving each function the
same name as what the user specifies, we can use the LLVM symbol table
to resolve function names for us.

Once we have the function to call, we recursively codegen each argument
that is to be passed in, and create an LLVM `call
instruction <../LangRef.html#i_call>`_. Note that LLVM uses the native C
calling conventions by default, allowing these calls to also call into
standard library functions like "sin" and "cos", with no additional
effort.

This wraps up our handling of the four basic expressions that we have so
far in Kaleidoscope. Feel free to go in and add some more. For example,
by browsing the `LLVM language reference <../LangRef.html>`_ you'll find
several other interesting instructions that are really easy to plug into
our basic framework.

Function Code Generation
========================

Code generation for prototypes and functions must handle a number of
details, which make their code less beautiful than expression code
generation, but allows us to illustrate some important points. First,
lets talk about code generation for prototypes: they are used both for
function bodies and external function declarations. The code starts
with:

.. code-block:: c++

    Function *PrototypeAST::Codegen() {
      // Make the function type:  double(double,double) etc.
      std::vector<Type*> Doubles(Args.size(),
                                 Type::getDoubleTy(getGlobalContext()));
      FunctionType *FT = FunctionType::get(Type::getDoubleTy(getGlobalContext()),
                                           Doubles, false);

      Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule);

This code packs a lot of power into a few lines. Note first that this
function returns a "Function\*" instead of a "Value\*". Because a
"prototype" really talks about the external interface for a function
(not the value computed by an expression), it makes sense for it to
return the LLVM Function it corresponds to when codegen'd.

The call to ``FunctionType::get`` creates the ``FunctionType`` that
should be used for a given Prototype. Since all function arguments in
Kaleidoscope are of type double, the first line creates a vector of "N"
LLVM double types. It then uses the ``Functiontype::get`` method to
create a function type that takes "N" doubles as arguments, returns one
double as a result, and that is not vararg (the false parameter
indicates this). Note that Types in LLVM are uniqued just like Constants
are, so you don't "new" a type, you "get" it.

The final line above actually creates the function that the prototype
will correspond to. This indicates the type, linkage and name to use, as
well as which module to insert into. "`external
linkage <../LangRef.html#linkage>`_" means that the function may be
defined outside the current module and/or that it is callable by
functions outside the module. The Name passed in is the name the user
specified: since "``TheModule``" is specified, this name is registered
in "``TheModule``"s symbol table, which is used by the function call
code above.

.. code-block:: c++

      // If F conflicted, there was already something named 'Name'.  If it has a
      // body, don't allow redefinition or reextern.
      if (F->getName() != Name) {
        // Delete the one we just made and get the existing one.
        F->eraseFromParent();
        F = TheModule->getFunction(Name);

The Module symbol table works just like the Function symbol table when
it comes to name conflicts: if a new function is created with a name
that was previously added to the symbol table, the new function will get
implicitly renamed when added to the Module. The code above exploits
this fact to determine if there was a previous definition of this
function.

In Kaleidoscope, I choose to allow redefinitions of functions in two
cases: first, we want to allow 'extern'ing a function more than once, as
long as the prototypes for the externs match (since all arguments have
the same type, we just have to check that the number of arguments
match). Second, we want to allow 'extern'ing a function and then
defining a body for it. This is useful when defining mutually recursive
functions.

In order to implement this, the code above first checks to see if there
is a collision on the name of the function. If so, it deletes the
function we just created (by calling ``eraseFromParent``) and then
calling ``getFunction`` to get the existing function with the specified
name. Note that many APIs in LLVM have "erase" forms and "remove" forms.
The "remove" form unlinks the object from its parent (e.g. a Function
from a Module) and returns it. The "erase" form unlinks the object and
then deletes it.

.. code-block:: c++

        // If F already has a body, reject this.
        if (!F->empty()) {
          ErrorF("redefinition of function");
          return 0;
        }

        // If F took a different number of args, reject.
        if (F->arg_size() != Args.size()) {
          ErrorF("redefinition of function with different # args");
          return 0;
        }
      }

In order to verify the logic above, we first check to see if the
pre-existing function is "empty". In this case, empty means that it has
no basic blocks in it, which means it has no body. If it has no body, it
is a forward declaration. Since we don't allow anything after a full
definition of the function, the code rejects this case. If the previous
reference to a function was an 'extern', we simply verify that the
number of arguments for that definition and this one match up. If not,
we emit an error.

.. code-block:: c++

      // Set names for all arguments.
      unsigned Idx = 0;
      for (Function::arg_iterator AI = F->arg_begin(); Idx != Args.size();
           ++AI, ++Idx) {
        AI->setName(Args[Idx]);

        // Add arguments to variable symbol table.
        NamedValues[Args[Idx]] = AI;
      }
      return F;
    }

The last bit of code for prototypes loops over all of the arguments in
the function, setting the name of the LLVM Argument objects to match,
and registering the arguments in the ``NamedValues`` map for future use
by the ``VariableExprAST`` AST node. Once this is set up, it returns the
Function object to the caller. Note that we don't check for conflicting
argument names here (e.g. "extern foo(a b a)"). Doing so would be very
straight-forward with the mechanics we have already used above.

.. code-block:: c++

    Function *FunctionAST::Codegen() {
      NamedValues.clear();

      Function *TheFunction = Proto->Codegen();
      if (TheFunction == 0)
        return 0;

Code generation for function definitions starts out simply enough: we
just codegen the prototype (Proto) and verify that it is ok. We then
clear out the ``NamedValues`` map to make sure that there isn't anything
in it from the last function we compiled. Code generation of the
prototype ensures that there is an LLVM Function object that is ready to
go for us.

.. code-block:: c++

      // Create a new basic block to start insertion into.
      BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
      Builder.SetInsertPoint(BB);

      if (Value *RetVal = Body->Codegen()) {

Now we get to the point where the ``Builder`` is set up. The first line
creates a new `basic block <http://en.wikipedia.org/wiki/Basic_block>`_
(named "entry"), which is inserted into ``TheFunction``. The second line
then tells the builder that new instructions should be inserted into the
end of the new basic block. Basic blocks in LLVM are an important part
of functions that define the `Control Flow
Graph <http://en.wikipedia.org/wiki/Control_flow_graph>`_. Since we
don't have any control flow, our functions will only contain one block
at this point. We'll fix this in `Chapter 5 <LangImpl5.html>`_ :).

.. code-block:: c++

      if (Value *RetVal = Body->Codegen()) {
        // Finish off the function.
        Builder.CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        return TheFunction;
      }

Once the insertion point is set up, we call the ``CodeGen()`` method for
the root expression of the function. If no error happens, this emits
code to compute the expression into the entry block and returns the
value that was computed. Assuming no error, we then create an LLVM `ret
instruction <../LangRef.html#i_ret>`_, which completes the function.
Once the function is built, we call ``verifyFunction``, which is
provided by LLVM. This function does a variety of consistency checks on
the generated code, to determine if our compiler is doing everything
right. Using this is important: it can catch a lot of bugs. Once the
function is finished and validated, we return it.

.. code-block:: c++

      // Error reading body, remove function.
      TheFunction->eraseFromParent();
      return 0;
    }

The only piece left here is handling of the error case. For simplicity,
we handle this by merely deleting the function we produced with the
``eraseFromParent`` method. This allows the user to redefine a function
that they incorrectly typed in before: if we didn't delete it, it would
live in the symbol table, with a body, preventing future redefinition.

This code does have a bug, though. Since the ``PrototypeAST::Codegen``
can return a previously defined forward declaration, our code can
actually delete a forward declaration. There are a number of ways to fix
this bug, see what you can come up with! Here is a testcase:

::

    extern foo(a b);     # ok, defines foo.
    def foo(a b) c;      # error, 'c' is invalid.
    def bar() foo(1, 2); # error, unknown function "foo"

Driver Changes and Closing Thoughts
===================================

For now, code generation to LLVM doesn't really get us much, except that
we can look at the pretty IR calls. The sample code inserts calls to
Codegen into the "``HandleDefinition``", "``HandleExtern``" etc
functions, and then dumps out the LLVM IR. This gives a nice way to look
at the LLVM IR for simple functions. For example:

::

    ready> 4+5;
    Read top-level expression:
    define double @0() {
    entry:
      ret double 9.000000e+00
    }

Note how the parser turns the top-level expression into anonymous
functions for us. This will be handy when we add `JIT
support <LangImpl4.html#jit>`_ in the next chapter. Also note that the
code is very literally transcribed, no optimizations are being performed
except simple constant folding done by IRBuilder. We will `add
optimizations <LangImpl4.html#trivialconstfold>`_ explicitly in the next
chapter.

::

    ready> def foo(a b) a*a + 2*a*b + b*b;
    Read function definition:
    define double @foo(double %a, double %b) {
    entry:
      %multmp = fmul double %a, %a
      %multmp1 = fmul double 2.000000e+00, %a
      %multmp2 = fmul double %multmp1, %b
      %addtmp = fadd double %multmp, %multmp2
      %multmp3 = fmul double %b, %b
      %addtmp4 = fadd double %addtmp, %multmp3
      ret double %addtmp4
    }

This shows some simple arithmetic. Notice the striking similarity to the
LLVM builder calls that we use to create the instructions.

::

    ready> def bar(a) foo(a, 4.0) + bar(31337);
    Read function definition:
    define double @bar(double %a) {
    entry:
      %calltmp = call double @foo(double %a, double 4.000000e+00)
      %calltmp1 = call double @bar(double 3.133700e+04)
      %addtmp = fadd double %calltmp, %calltmp1
      ret double %addtmp
    }

This shows some function calls. Note that this function will take a long
time to execute if you call it. In the future we'll add conditional
control flow to actually make recursion useful :).

::

    ready> extern cos(x);
    Read extern:
    declare double @cos(double)

    ready> cos(1.234);
    Read top-level expression:
    define double @1() {
    entry:
      %calltmp = call double @cos(double 1.234000e+00)
      ret double %calltmp
    }

This shows an extern for the libm "cos" function, and a call to it.

.. TODO:: Abandon Pygments' horrible `llvm` lexer. It just totally gives up
   on highlighting this due to the first line.

::

    ready> ^D
    ; ModuleID = 'my cool jit'

    define double @0() {
    entry:
      %addtmp = fadd double 4.000000e+00, 5.000000e+00
      ret double %addtmp
    }

    define double @foo(double %a, double %b) {
    entry:
      %multmp = fmul double %a, %a
      %multmp1 = fmul double 2.000000e+00, %a
      %multmp2 = fmul double %multmp1, %b
      %addtmp = fadd double %multmp, %multmp2
      %multmp3 = fmul double %b, %b
      %addtmp4 = fadd double %addtmp, %multmp3
      ret double %addtmp4
    }

    define double @bar(double %a) {
    entry:
      %calltmp = call double @foo(double %a, double 4.000000e+00)
      %calltmp1 = call double @bar(double 3.133700e+04)
      %addtmp = fadd double %calltmp, %calltmp1
      ret double %addtmp
    }

    declare double @cos(double)

    define double @1() {
    entry:
      %calltmp = call double @cos(double 1.234000e+00)
      ret double %calltmp
    }

When you quit the current demo, it dumps out the IR for the entire
module generated. Here you can see the big picture with all the
functions referencing each other.

This wraps up the third chapter of the Kaleidoscope tutorial. Up next,
we'll describe how to `add JIT codegen and optimizer
support <LangImpl4.html>`_ to this so we can actually start running
code!

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the LLVM code generator. Because this uses the LLVM libraries, we need
to link them in. To do this, we use the
`llvm-config <http://llvm.org/cmds/llvm-config.html>`_ tool to inform
our makefile/command line about which options to use:

.. code-block:: bash

    # Compile
    clang++ -g -O3 toy.cpp `llvm-config --cppflags --ldflags --libs core` -o toy
    # Run
    ./toy

Here is the code:

.. code-block:: c++

    // To build this:
    // See example below.

    #include "llvm/DerivedTypes.h"
    #include "llvm/IRBuilder.h"
    #include "llvm/LLVMContext.h"
    #include "llvm/Module.h"
    #include "llvm/Analysis/Verifier.h"
    #include <cstdio>
    #include <string>
    #include <map>
    #include <vector>
    using namespace llvm;

    //===----------------------------------------------------------------------===//
    // Lexer
    //===----------------------------------------------------------------------===//

    // The lexer returns tokens [0-255] if it is an unknown character, otherwise one
    // of these for known things.
    enum Token {
      tok_eof = -1,

      // commands
      tok_def = -2, tok_extern = -3,

      // primary
      tok_identifier = -4, tok_number = -5
    };

    static std::string IdentifierStr;  // Filled in if tok_identifier
    static double NumVal;              // Filled in if tok_number

    /// gettok - Return the next token from standard input.
    static int gettok() {
      static int LastChar = ' ';

      // Skip any whitespace.
      while (isspace(LastChar))
        LastChar = getchar();

      if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
        IdentifierStr = LastChar;
        while (isalnum((LastChar = getchar())))
          IdentifierStr += LastChar;

        if (IdentifierStr == "def") return tok_def;
        if (IdentifierStr == "extern") return tok_extern;
        return tok_identifier;
      }

      if (isdigit(LastChar) || LastChar == '.') {   // Number: [0-9.]+
        std::string NumStr;
        do {
          NumStr += LastChar;
          LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), 0);
        return tok_number;
      }

      if (LastChar == '#') {
        // Comment until end of line.
        do LastChar = getchar();
        while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if (LastChar != EOF)
          return gettok();
      }

      // Check for end of file.  Don't eat the EOF.
      if (LastChar == EOF)
        return tok_eof;

      // Otherwise, just return the character as its ascii value.
      int ThisChar = LastChar;
      LastChar = getchar();
      return ThisChar;
    }

    //===----------------------------------------------------------------------===//
    // Abstract Syntax Tree (aka Parse Tree)
    //===----------------------------------------------------------------------===//

    /// ExprAST - Base class for all expression nodes.
    class ExprAST {
    public:
      virtual ~ExprAST() {}
      virtual Value *Codegen() = 0;
    };

    /// NumberExprAST - Expression class for numeric literals like "1.0".
    class NumberExprAST : public ExprAST {
      double Val;
    public:
      NumberExprAST(double val) : Val(val) {}
      virtual Value *Codegen();
    };

    /// VariableExprAST - Expression class for referencing a variable, like "a".
    class VariableExprAST : public ExprAST {
      std::string Name;
    public:
      VariableExprAST(const std::string &name) : Name(name) {}
      virtual Value *Codegen();
    };

    /// BinaryExprAST - Expression class for a binary operator.
    class BinaryExprAST : public ExprAST {
      char Op;
      ExprAST *LHS, *RHS;
    public:
      BinaryExprAST(char op, ExprAST *lhs, ExprAST *rhs)
        : Op(op), LHS(lhs), RHS(rhs) {}
      virtual Value *Codegen();
    };

    /// CallExprAST - Expression class for function calls.
    class CallExprAST : public ExprAST {
      std::string Callee;
      std::vector<ExprAST*> Args;
    public:
      CallExprAST(const std::string &callee, std::vector<ExprAST*> &args)
        : Callee(callee), Args(args) {}
      virtual Value *Codegen();
    };

    /// PrototypeAST - This class represents the "prototype" for a function,
    /// which captures its name, and its argument names (thus implicitly the number
    /// of arguments the function takes).
    class PrototypeAST {
      std::string Name;
      std::vector<std::string> Args;
    public:
      PrototypeAST(const std::string &name, const std::vector<std::string> &args)
        : Name(name), Args(args) {}

      Function *Codegen();
    };

    /// FunctionAST - This class represents a function definition itself.
    class FunctionAST {
      PrototypeAST *Proto;
      ExprAST *Body;
    public:
      FunctionAST(PrototypeAST *proto, ExprAST *body)
        : Proto(proto), Body(body) {}

      Function *Codegen();
    };

    //===----------------------------------------------------------------------===//
    // Parser
    //===----------------------------------------------------------------------===//

    /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
    /// token the parser is looking at.  getNextToken reads another token from the
    /// lexer and updates CurTok with its results.
    static int CurTok;
    static int getNextToken() {
      return CurTok = gettok();
    }

    /// BinopPrecedence - This holds the precedence for each binary operator that is
    /// defined.
    static std::map<char, int> BinopPrecedence;

    /// GetTokPrecedence - Get the precedence of the pending binary operator token.
    static int GetTokPrecedence() {
      if (!isascii(CurTok))
        return -1;

      // Make sure it's a declared binop.
      int TokPrec = BinopPrecedence[CurTok];
      if (TokPrec <= 0) return -1;
      return TokPrec;
    }

    /// Error* - These are little helper functions for error handling.
    ExprAST *Error(const char *Str) { fprintf(stderr, "Error: %s\n", Str);return 0;}
    PrototypeAST *ErrorP(const char *Str) { Error(Str); return 0; }
    FunctionAST *ErrorF(const char *Str) { Error(Str); return 0; }

    static ExprAST *ParseExpression();

    /// identifierexpr
    ///   ::= identifier
    ///   ::= identifier '(' expression* ')'
    static ExprAST *ParseIdentifierExpr() {
      std::string IdName = IdentifierStr;

      getNextToken();  // eat identifier.

      if (CurTok != '(') // Simple variable ref.
        return new VariableExprAST(IdName);

      // Call.
      getNextToken();  // eat (
      std::vector<ExprAST*> Args;
      if (CurTok != ')') {
        while (1) {
          ExprAST *Arg = ParseExpression();
          if (!Arg) return 0;
          Args.push_back(Arg);

          if (CurTok == ')') break;

          if (CurTok != ',')
            return Error("Expected ')' or ',' in argument list");
          getNextToken();
        }
      }

      // Eat the ')'.
      getNextToken();

      return new CallExprAST(IdName, Args);
    }

    /// numberexpr ::= number
    static ExprAST *ParseNumberExpr() {
      ExprAST *Result = new NumberExprAST(NumVal);
      getNextToken(); // consume the number
      return Result;
    }

    /// parenexpr ::= '(' expression ')'
    static ExprAST *ParseParenExpr() {
      getNextToken();  // eat (.
      ExprAST *V = ParseExpression();
      if (!V) return 0;

      if (CurTok != ')')
        return Error("expected ')'");
      getNextToken();  // eat ).
      return V;
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    static ExprAST *ParsePrimary() {
      switch (CurTok) {
      default: return Error("unknown token when expecting an expression");
      case tok_identifier: return ParseIdentifierExpr();
      case tok_number:     return ParseNumberExpr();
      case '(':            return ParseParenExpr();
      }
    }

    /// binoprhs
    ///   ::= ('+' primary)*
    static ExprAST *ParseBinOpRHS(int ExprPrec, ExprAST *LHS) {
      // If this is a binop, find its precedence.
      while (1) {
        int TokPrec = GetTokPrecedence();

        // If this is a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (TokPrec < ExprPrec)
          return LHS;

        // Okay, we know this is a binop.
        int BinOp = CurTok;
        getNextToken();  // eat binop

        // Parse the primary expression after the binary operator.
        ExprAST *RHS = ParsePrimary();
        if (!RHS) return 0;

        // If BinOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as its LHS.
        int NextPrec = GetTokPrecedence();
        if (TokPrec < NextPrec) {
          RHS = ParseBinOpRHS(TokPrec+1, RHS);
          if (RHS == 0) return 0;
        }

        // Merge LHS/RHS.
        LHS = new BinaryExprAST(BinOp, LHS, RHS);
      }
    }

    /// expression
    ///   ::= primary binoprhs
    ///
    static ExprAST *ParseExpression() {
      ExprAST *LHS = ParsePrimary();
      if (!LHS) return 0;

      return ParseBinOpRHS(0, LHS);
    }

    /// prototype
    ///   ::= id '(' id* ')'
    static PrototypeAST *ParsePrototype() {
      if (CurTok != tok_identifier)
        return ErrorP("Expected function name in prototype");

      std::string FnName = IdentifierStr;
      getNextToken();

      if (CurTok != '(')
        return ErrorP("Expected '(' in prototype");

      std::vector<std::string> ArgNames;
      while (getNextToken() == tok_identifier)
        ArgNames.push_back(IdentifierStr);
      if (CurTok != ')')
        return ErrorP("Expected ')' in prototype");

      // success.
      getNextToken();  // eat ')'.

      return new PrototypeAST(FnName, ArgNames);
    }

    /// definition ::= 'def' prototype expression
    static FunctionAST *ParseDefinition() {
      getNextToken();  // eat def.
      PrototypeAST *Proto = ParsePrototype();
      if (Proto == 0) return 0;

      if (ExprAST *E = ParseExpression())
        return new FunctionAST(Proto, E);
      return 0;
    }

    /// toplevelexpr ::= expression
    static FunctionAST *ParseTopLevelExpr() {
      if (ExprAST *E = ParseExpression()) {
        // Make an anonymous proto.
        PrototypeAST *Proto = new PrototypeAST("", std::vector<std::string>());
        return new FunctionAST(Proto, E);
      }
      return 0;
    }

    /// external ::= 'extern' prototype
    static PrototypeAST *ParseExtern() {
      getNextToken();  // eat extern.
      return ParsePrototype();
    }

    //===----------------------------------------------------------------------===//
    // Code Generation
    //===----------------------------------------------------------------------===//

    static Module *TheModule;
    static IRBuilder<> Builder(getGlobalContext());
    static std::map<std::string, Value*> NamedValues;

    Value *ErrorV(const char *Str) { Error(Str); return 0; }

    Value *NumberExprAST::Codegen() {
      return ConstantFP::get(getGlobalContext(), APFloat(Val));
    }

    Value *VariableExprAST::Codegen() {
      // Look this variable up in the function.
      Value *V = NamedValues[Name];
      return V ? V : ErrorV("Unknown variable name");
    }

    Value *BinaryExprAST::Codegen() {
      Value *L = LHS->Codegen();
      Value *R = RHS->Codegen();
      if (L == 0 || R == 0) return 0;

      switch (Op) {
      case '+': return Builder.CreateFAdd(L, R, "addtmp");
      case '-': return Builder.CreateFSub(L, R, "subtmp");
      case '*': return Builder.CreateFMul(L, R, "multmp");
      case '<':
        L = Builder.CreateFCmpULT(L, R, "cmptmp");
        // Convert bool 0/1 to double 0.0 or 1.0
        return Builder.CreateUIToFP(L, Type::getDoubleTy(getGlobalContext()),
                                    "booltmp");
      default: return ErrorV("invalid binary operator");
      }
    }

    Value *CallExprAST::Codegen() {
      // Look up the name in the global module table.
      Function *CalleeF = TheModule->getFunction(Callee);
      if (CalleeF == 0)
        return ErrorV("Unknown function referenced");

      // If argument mismatch error.
      if (CalleeF->arg_size() != Args.size())
        return ErrorV("Incorrect # arguments passed");

      std::vector<Value*> ArgsV;
      for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        ArgsV.push_back(Args[i]->Codegen());
        if (ArgsV.back() == 0) return 0;
      }

      return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
    }

    Function *PrototypeAST::Codegen() {
      // Make the function type:  double(double,double) etc.
      std::vector<Type*> Doubles(Args.size(),
                                 Type::getDoubleTy(getGlobalContext()));
      FunctionType *FT = FunctionType::get(Type::getDoubleTy(getGlobalContext()),
                                           Doubles, false);

      Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule);

      // If F conflicted, there was already something named 'Name'.  If it has a
      // body, don't allow redefinition or reextern.
      if (F->getName() != Name) {
        // Delete the one we just made and get the existing one.
        F->eraseFromParent();
        F = TheModule->getFunction(Name);

        // If F already has a body, reject this.
        if (!F->empty()) {
          ErrorF("redefinition of function");
          return 0;
        }

        // If F took a different number of args, reject.
        if (F->arg_size() != Args.size()) {
          ErrorF("redefinition of function with different # args");
          return 0;
        }
      }

      // Set names for all arguments.
      unsigned Idx = 0;
      for (Function::arg_iterator AI = F->arg_begin(); Idx != Args.size();
           ++AI, ++Idx) {
        AI->setName(Args[Idx]);

        // Add arguments to variable symbol table.
        NamedValues[Args[Idx]] = AI;
      }

      return F;
    }

    Function *FunctionAST::Codegen() {
      NamedValues.clear();

      Function *TheFunction = Proto->Codegen();
      if (TheFunction == 0)
        return 0;

      // Create a new basic block to start insertion into.
      BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
      Builder.SetInsertPoint(BB);

      if (Value *RetVal = Body->Codegen()) {
        // Finish off the function.
        Builder.CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        return TheFunction;
      }

      // Error reading body, remove function.
      TheFunction->eraseFromParent();
      return 0;
    }

    //===----------------------------------------------------------------------===//
    // Top-Level parsing and JIT Driver
    //===----------------------------------------------------------------------===//

    static void HandleDefinition() {
      if (FunctionAST *F = ParseDefinition()) {
        if (Function *LF = F->Codegen()) {
          fprintf(stderr, "Read function definition:");
          LF->dump();
        }
      } else {
        // Skip token for error recovery.
        getNextToken();
      }
    }

    static void HandleExtern() {
      if (PrototypeAST *P = ParseExtern()) {
        if (Function *F = P->Codegen()) {
          fprintf(stderr, "Read extern: ");
          F->dump();
        }
      } else {
        // Skip token for error recovery.
        getNextToken();
      }
    }

    static void HandleTopLevelExpression() {
      // Evaluate a top-level expression into an anonymous function.
      if (FunctionAST *F = ParseTopLevelExpr()) {
        if (Function *LF = F->Codegen()) {
          fprintf(stderr, "Read top-level expression:");
          LF->dump();
        }
      } else {
        // Skip token for error recovery.
        getNextToken();
      }
    }

    /// top ::= definition | external | expression | ';'
    static void MainLoop() {
      while (1) {
        fprintf(stderr, "ready> ");
        switch (CurTok) {
        case tok_eof:    return;
        case ';':        getNextToken(); break;  // ignore top-level semicolons.
        case tok_def:    HandleDefinition(); break;
        case tok_extern: HandleExtern(); break;
        default:         HandleTopLevelExpression(); break;
        }
      }
    }

    //===----------------------------------------------------------------------===//
    // "Library" functions that can be "extern'd" from user code.
    //===----------------------------------------------------------------------===//

    /// putchard - putchar that takes a double and returns 0.
    extern "C"
    double putchard(double X) {
      putchar((char)X);
      return 0;
    }

    //===----------------------------------------------------------------------===//
    // Main driver code.
    //===----------------------------------------------------------------------===//

    int main() {
      LLVMContext &Context = getGlobalContext();

      // Install standard binary operators.
      // 1 is lowest precedence.
      BinopPrecedence['<'] = 10;
      BinopPrecedence['+'] = 20;
      BinopPrecedence['-'] = 20;
      BinopPrecedence['*'] = 40;  // highest.

      // Prime the first token.
      fprintf(stderr, "ready> ");
      getNextToken();

      // Make the module, which holds all the code.
      TheModule = new Module("my cool jit", Context);

      // Run the main "interpreter loop" now.
      MainLoop();

      // Print out all of the generated code.
      TheModule->dump();

      return 0;
    }

`Next: Adding JIT and Optimizer Support <LangImpl4.html>`_

