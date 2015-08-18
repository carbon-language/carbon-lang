========================================
Kaleidoscope: Code generation to LLVM IR
========================================

.. contents::
   :local:

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
      NumberExprAST(double Val) : Val(Val) {}
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
`IRBuilder <http://llvm.org/doxygen/IRBuilder_8h-source.html>`_
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
    clang++ -g -O3 toy.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o toy
    # Run
    ./toy

Here is the code:

.. literalinclude:: ../../examples/Kaleidoscope/Chapter3/toy.cpp
   :language: c++

`Next: Adding JIT and Optimizer Support <LangImpl4.html>`_

