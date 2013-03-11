=======================================================
Kaleidoscope: Extending the Language: Mutable Variables
=======================================================

.. contents::
   :local:

Chapter 7 Introduction
======================

Welcome to Chapter 7 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. In chapters 1 through 6, we've built a
very respectable, albeit simple, `functional programming
language <http://en.wikipedia.org/wiki/Functional_programming>`_. In our
journey, we learned some parsing techniques, how to build and represent
an AST, how to build LLVM IR, and how to optimize the resultant code as
well as JIT compile it.

While Kaleidoscope is interesting as a functional language, the fact
that it is functional makes it "too easy" to generate LLVM IR for it. In
particular, a functional language makes it very easy to build LLVM IR
directly in `SSA
form <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.
Since LLVM requires that the input code be in SSA form, this is a very
nice property and it is often unclear to newcomers how to generate code
for an imperative language with mutable variables.

The short (and happy) summary of this chapter is that there is no need
for your front-end to build SSA form: LLVM provides highly tuned and
well tested support for this, though the way it works is a bit
unexpected for some.

Why is this a hard problem?
===========================

To understand why mutable variables cause complexities in SSA
construction, consider this extremely simple C example:

.. code-block:: c

    int G, H;
    int test(_Bool Condition) {
      int X;
      if (Condition)
        X = G;
      else
        X = H;
      return X;
    }

In this case, we have the variable "X", whose value depends on the path
executed in the program. Because there are two different possible values
for X before the return instruction, a PHI node is inserted to merge the
two values. The LLVM IR that we want for this example looks like this:

.. code-block:: llvm

    @G = weak global i32 0   ; type of @G is i32*
    @H = weak global i32 0   ; type of @H is i32*

    define i32 @test(i1 %Condition) {
    entry:
      br i1 %Condition, label %cond_true, label %cond_false

    cond_true:
      %X.0 = load i32* @G
      br label %cond_next

    cond_false:
      %X.1 = load i32* @H
      br label %cond_next

    cond_next:
      %X.2 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
      ret i32 %X.2
    }

In this example, the loads from the G and H global variables are
explicit in the LLVM IR, and they live in the then/else branches of the
if statement (cond\_true/cond\_false). In order to merge the incoming
values, the X.2 phi node in the cond\_next block selects the right value
to use based on where control flow is coming from: if control flow comes
from the cond\_false block, X.2 gets the value of X.1. Alternatively, if
control flow comes from cond\_true, it gets the value of X.0. The intent
of this chapter is not to explain the details of SSA form. For more
information, see one of the many `online
references <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_.

The question for this article is "who places the phi nodes when lowering
assignments to mutable variables?". The issue here is that LLVM
*requires* that its IR be in SSA form: there is no "non-ssa" mode for
it. However, SSA construction requires non-trivial algorithms and data
structures, so it is inconvenient and wasteful for every front-end to
have to reproduce this logic.

Memory in LLVM
==============

The 'trick' here is that while LLVM does require all register values to
be in SSA form, it does not require (or permit) memory objects to be in
SSA form. In the example above, note that the loads from G and H are
direct accesses to G and H: they are not renamed or versioned. This
differs from some other compiler systems, which do try to version memory
objects. In LLVM, instead of encoding dataflow analysis of memory into
the LLVM IR, it is handled with `Analysis
Passes <../WritingAnLLVMPass.html>`_ which are computed on demand.

With this in mind, the high-level idea is that we want to make a stack
variable (which lives in memory, because it is on the stack) for each
mutable object in a function. To take advantage of this trick, we need
to talk about how LLVM represents stack variables.

In LLVM, all memory accesses are explicit with load/store instructions,
and it is carefully designed not to have (or need) an "address-of"
operator. Notice how the type of the @G/@H global variables is actually
"i32\*" even though the variable is defined as "i32". What this means is
that @G defines *space* for an i32 in the global data area, but its
*name* actually refers to the address for that space. Stack variables
work the same way, except that instead of being declared with global
variable definitions, they are declared with the `LLVM alloca
instruction <../LangRef.html#i_alloca>`_:

.. code-block:: llvm

    define i32 @example() {
    entry:
      %X = alloca i32           ; type of %X is i32*.
      ...
      %tmp = load i32* %X       ; load the stack value %X from the stack.
      %tmp2 = add i32 %tmp, 1   ; increment it
      store i32 %tmp2, i32* %X  ; store it back
      ...

This code shows an example of how you can declare and manipulate a stack
variable in the LLVM IR. Stack memory allocated with the alloca
instruction is fully general: you can pass the address of the stack slot
to functions, you can store it in other variables, etc. In our example
above, we could rewrite the example to use the alloca technique to avoid
using a PHI node:

.. code-block:: llvm

    @G = weak global i32 0   ; type of @G is i32*
    @H = weak global i32 0   ; type of @H is i32*

    define i32 @test(i1 %Condition) {
    entry:
      %X = alloca i32           ; type of %X is i32*.
      br i1 %Condition, label %cond_true, label %cond_false

    cond_true:
      %X.0 = load i32* @G
      store i32 %X.0, i32* %X   ; Update X
      br label %cond_next

    cond_false:
      %X.1 = load i32* @H
      store i32 %X.1, i32* %X   ; Update X
      br label %cond_next

    cond_next:
      %X.2 = load i32* %X       ; Read X
      ret i32 %X.2
    }

With this, we have discovered a way to handle arbitrary mutable
variables without the need to create Phi nodes at all:

#. Each mutable variable becomes a stack allocation.
#. Each read of the variable becomes a load from the stack.
#. Each update of the variable becomes a store to the stack.
#. Taking the address of a variable just uses the stack address
   directly.

While this solution has solved our immediate problem, it introduced
another one: we have now apparently introduced a lot of stack traffic
for very simple and common operations, a major performance problem.
Fortunately for us, the LLVM optimizer has a highly-tuned optimization
pass named "mem2reg" that handles this case, promoting allocas like this
into SSA registers, inserting Phi nodes as appropriate. If you run this
example through the pass, for example, you'll get:

.. code-block:: bash

    $ llvm-as < example.ll | opt -mem2reg | llvm-dis
    @G = weak global i32 0
    @H = weak global i32 0

    define i32 @test(i1 %Condition) {
    entry:
      br i1 %Condition, label %cond_true, label %cond_false

    cond_true:
      %X.0 = load i32* @G
      br label %cond_next

    cond_false:
      %X.1 = load i32* @H
      br label %cond_next

    cond_next:
      %X.01 = phi i32 [ %X.1, %cond_false ], [ %X.0, %cond_true ]
      ret i32 %X.01
    }

The mem2reg pass implements the standard "iterated dominance frontier"
algorithm for constructing SSA form and has a number of optimizations
that speed up (very common) degenerate cases. The mem2reg optimization
pass is the answer to dealing with mutable variables, and we highly
recommend that you depend on it. Note that mem2reg only works on
variables in certain circumstances:

#. mem2reg is alloca-driven: it looks for allocas and if it can handle
   them, it promotes them. It does not apply to global variables or heap
   allocations.
#. mem2reg only looks for alloca instructions in the entry block of the
   function. Being in the entry block guarantees that the alloca is only
   executed once, which makes analysis simpler.
#. mem2reg only promotes allocas whose uses are direct loads and stores.
   If the address of the stack object is passed to a function, or if any
   funny pointer arithmetic is involved, the alloca will not be
   promoted.
#. mem2reg only works on allocas of `first
   class <../LangRef.html#t_classifications>`_ values (such as pointers,
   scalars and vectors), and only if the array size of the allocation is
   1 (or missing in the .ll file). mem2reg is not capable of promoting
   structs or arrays to registers. Note that the "scalarrepl" pass is
   more powerful and can promote structs, "unions", and arrays in many
   cases.

All of these properties are easy to satisfy for most imperative
languages, and we'll illustrate it below with Kaleidoscope. The final
question you may be asking is: should I bother with this nonsense for my
front-end? Wouldn't it be better if I just did SSA construction
directly, avoiding use of the mem2reg optimization pass? In short, we
strongly recommend that you use this technique for building SSA form,
unless there is an extremely good reason not to. Using this technique
is:

-  Proven and well tested: llvm-gcc and clang both use this technique
   for local mutable variables. As such, the most common clients of LLVM
   are using this to handle a bulk of their variables. You can be sure
   that bugs are found fast and fixed early.
-  Extremely Fast: mem2reg has a number of special cases that make it
   fast in common cases as well as fully general. For example, it has
   fast-paths for variables that are only used in a single block,
   variables that only have one assignment point, good heuristics to
   avoid insertion of unneeded phi nodes, etc.
-  Needed for debug info generation: `Debug information in
   LLVM <../SourceLevelDebugging.html>`_ relies on having the address of
   the variable exposed so that debug info can be attached to it. This
   technique dovetails very naturally with this style of debug info.

If nothing else, this makes it much easier to get your front-end up and
running, and is very simple to implement. Lets extend Kaleidoscope with
mutable variables now!

Mutable Variables in Kaleidoscope
=================================

Now that we know the sort of problem we want to tackle, lets see what
this looks like in the context of our little Kaleidoscope language.
We're going to add two features:

#. The ability to mutate variables with the '=' operator.
#. The ability to define new variables.

While the first item is really what this is about, we only have
variables for incoming arguments as well as for induction variables, and
redefining those only goes so far :). Also, the ability to define new
variables is a useful thing regardless of whether you will be mutating
them. Here's a motivating example that shows how we could use these:

::

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;

    # Recursive fib, we could do this before.
    def fib(x)
      if (x < 3) then
        1
      else
        fib(x-1)+fib(x-2);

    # Iterative fib.
    def fibi(x)
      var a = 1, b = 1, c in
      (for i = 3, i < x in
         c = a + b :
         a = b :
         b = c) :
      b;

    # Call it.
    fibi(10);

In order to mutate variables, we have to change our existing variables
to use the "alloca trick". Once we have that, we'll add our new
operator, then extend Kaleidoscope to support new variable definitions.

Adjusting Existing Variables for Mutation
=========================================

The symbol table in Kaleidoscope is managed at code generation time by
the '``NamedValues``' map. This map currently keeps track of the LLVM
"Value\*" that holds the double value for the named variable. In order
to support mutation, we need to change this slightly, so that it
``NamedValues`` holds the *memory location* of the variable in question.
Note that this change is a refactoring: it changes the structure of the
code, but does not (by itself) change the behavior of the compiler. All
of these changes are isolated in the Kaleidoscope code generator.

At this point in Kaleidoscope's development, it only supports variables
for two things: incoming arguments to functions and the induction
variable of 'for' loops. For consistency, we'll allow mutation of these
variables in addition to other user-defined variables. This means that
these will both need memory locations.

To start our transformation of Kaleidoscope, we'll change the
NamedValues map so that it maps to AllocaInst\* instead of Value\*. Once
we do this, the C++ compiler will tell us what parts of the code we need
to update:

.. code-block:: c++

    static std::map<std::string, AllocaInst*> NamedValues;

Also, since we will need to create these alloca's, we'll use a helper
function that ensures that the allocas are created in the entry block of
the function:

.. code-block:: c++

    /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
    /// the function.  This is used for mutable variables etc.
    static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                              const std::string &VarName) {
      IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                     TheFunction->getEntryBlock().begin());
      return TmpB.CreateAlloca(Type::getDoubleTy(getGlobalContext()), 0,
                               VarName.c_str());
    }

This funny looking code creates an IRBuilder object that is pointing at
the first instruction (.begin()) of the entry block. It then creates an
alloca with the expected name and returns it. Because all values in
Kaleidoscope are doubles, there is no need to pass in a type to use.

With this in place, the first functionality change we want to make is to
variable references. In our new scheme, variables live on the stack, so
code generating a reference to them actually needs to produce a load
from the stack slot:

.. code-block:: c++

    Value *VariableExprAST::Codegen() {
      // Look this variable up in the function.
      Value *V = NamedValues[Name];
      if (V == 0) return ErrorV("Unknown variable name");

      // Load the value.
      return Builder.CreateLoad(V, Name.c_str());
    }

As you can see, this is pretty straightforward. Now we need to update
the things that define the variables to set up the alloca. We'll start
with ``ForExprAST::Codegen`` (see the `full code listing <#code>`_ for
the unabridged code):

.. code-block:: c++

      Function *TheFunction = Builder.GetInsertBlock()->getParent();

      // Create an alloca for the variable in the entry block.
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

        // Emit the start code first, without 'variable' in scope.
      Value *StartVal = Start->Codegen();
      if (StartVal == 0) return 0;

      // Store the value into the alloca.
      Builder.CreateStore(StartVal, Alloca);
      ...

      // Compute the end condition.
      Value *EndCond = End->Codegen();
      if (EndCond == 0) return EndCond;

      // Reload, increment, and restore the alloca.  This handles the case where
      // the body of the loop mutates the variable.
      Value *CurVar = Builder.CreateLoad(Alloca);
      Value *NextVar = Builder.CreateFAdd(CurVar, StepVal, "nextvar");
      Builder.CreateStore(NextVar, Alloca);
      ...

This code is virtually identical to the code `before we allowed mutable
variables <LangImpl5.html#forcodegen>`_. The big difference is that we
no longer have to construct a PHI node, and we use load/store to access
the variable as needed.

To support mutable argument variables, we need to also make allocas for
them. The code for this is also pretty simple:

.. code-block:: c++

    /// CreateArgumentAllocas - Create an alloca for each argument and register the
    /// argument in the symbol table so that references to it will succeed.
    void PrototypeAST::CreateArgumentAllocas(Function *F) {
      Function::arg_iterator AI = F->arg_begin();
      for (unsigned Idx = 0, e = Args.size(); Idx != e; ++Idx, ++AI) {
        // Create an alloca for this variable.
        AllocaInst *Alloca = CreateEntryBlockAlloca(F, Args[Idx]);

        // Store the initial value into the alloca.
        Builder.CreateStore(AI, Alloca);

        // Add arguments to variable symbol table.
        NamedValues[Args[Idx]] = Alloca;
      }
    }

For each argument, we make an alloca, store the input value to the
function into the alloca, and register the alloca as the memory location
for the argument. This method gets invoked by ``FunctionAST::Codegen``
right after it sets up the entry block for the function.

The final missing piece is adding the mem2reg pass, which allows us to
get good codegen once again:

.. code-block:: c++

        // Set up the optimizer pipeline.  Start with registering info about how the
        // target lays out data structures.
        OurFPM.add(new DataLayout(*TheExecutionEngine->getDataLayout()));
        // Promote allocas to registers.
        OurFPM.add(createPromoteMemoryToRegisterPass());
        // Do simple "peephole" optimizations and bit-twiddling optzns.
        OurFPM.add(createInstructionCombiningPass());
        // Reassociate expressions.
        OurFPM.add(createReassociatePass());

It is interesting to see what the code looks like before and after the
mem2reg optimization runs. For example, this is the before/after code
for our recursive fib function. Before the optimization:

.. code-block:: llvm

    define double @fib(double %x) {
    entry:
      %x1 = alloca double
      store double %x, double* %x1
      %x2 = load double* %x1
      %cmptmp = fcmp ult double %x2, 3.000000e+00
      %booltmp = uitofp i1 %cmptmp to double
      %ifcond = fcmp one double %booltmp, 0.000000e+00
      br i1 %ifcond, label %then, label %else

    then:       ; preds = %entry
      br label %ifcont

    else:       ; preds = %entry
      %x3 = load double* %x1
      %subtmp = fsub double %x3, 1.000000e+00
      %calltmp = call double @fib(double %subtmp)
      %x4 = load double* %x1
      %subtmp5 = fsub double %x4, 2.000000e+00
      %calltmp6 = call double @fib(double %subtmp5)
      %addtmp = fadd double %calltmp, %calltmp6
      br label %ifcont

    ifcont:     ; preds = %else, %then
      %iftmp = phi double [ 1.000000e+00, %then ], [ %addtmp, %else ]
      ret double %iftmp
    }

Here there is only one variable (x, the input argument) but you can
still see the extremely simple-minded code generation strategy we are
using. In the entry block, an alloca is created, and the initial input
value is stored into it. Each reference to the variable does a reload
from the stack. Also, note that we didn't modify the if/then/else
expression, so it still inserts a PHI node. While we could make an
alloca for it, it is actually easier to create a PHI node for it, so we
still just make the PHI.

Here is the code after the mem2reg pass runs:

.. code-block:: llvm

    define double @fib(double %x) {
    entry:
      %cmptmp = fcmp ult double %x, 3.000000e+00
      %booltmp = uitofp i1 %cmptmp to double
      %ifcond = fcmp one double %booltmp, 0.000000e+00
      br i1 %ifcond, label %then, label %else

    then:
      br label %ifcont

    else:
      %subtmp = fsub double %x, 1.000000e+00
      %calltmp = call double @fib(double %subtmp)
      %subtmp5 = fsub double %x, 2.000000e+00
      %calltmp6 = call double @fib(double %subtmp5)
      %addtmp = fadd double %calltmp, %calltmp6
      br label %ifcont

    ifcont:     ; preds = %else, %then
      %iftmp = phi double [ 1.000000e+00, %then ], [ %addtmp, %else ]
      ret double %iftmp
    }

This is a trivial case for mem2reg, since there are no redefinitions of
the variable. The point of showing this is to calm your tension about
inserting such blatent inefficiencies :).

After the rest of the optimizers run, we get:

.. code-block:: llvm

    define double @fib(double %x) {
    entry:
      %cmptmp = fcmp ult double %x, 3.000000e+00
      %booltmp = uitofp i1 %cmptmp to double
      %ifcond = fcmp ueq double %booltmp, 0.000000e+00
      br i1 %ifcond, label %else, label %ifcont

    else:
      %subtmp = fsub double %x, 1.000000e+00
      %calltmp = call double @fib(double %subtmp)
      %subtmp5 = fsub double %x, 2.000000e+00
      %calltmp6 = call double @fib(double %subtmp5)
      %addtmp = fadd double %calltmp, %calltmp6
      ret double %addtmp

    ifcont:
      ret double 1.000000e+00
    }

Here we see that the simplifycfg pass decided to clone the return
instruction into the end of the 'else' block. This allowed it to
eliminate some branches and the PHI node.

Now that all symbol table references are updated to use stack variables,
we'll add the assignment operator.

New Assignment Operator
=======================

With our current framework, adding a new assignment operator is really
simple. We will parse it just like any other binary operator, but handle
it internally (instead of allowing the user to define it). The first
step is to set a precedence:

.. code-block:: c++

     int main() {
       // Install standard binary operators.
       // 1 is lowest precedence.
       BinopPrecedence['='] = 2;
       BinopPrecedence['<'] = 10;
       BinopPrecedence['+'] = 20;
       BinopPrecedence['-'] = 20;

Now that the parser knows the precedence of the binary operator, it
takes care of all the parsing and AST generation. We just need to
implement codegen for the assignment operator. This looks like:

.. code-block:: c++

    Value *BinaryExprAST::Codegen() {
      // Special case '=' because we don't want to emit the LHS as an expression.
      if (Op == '=') {
        // Assignment requires the LHS to be an identifier.
        VariableExprAST *LHSE = dynamic_cast<VariableExprAST*>(LHS);
        if (!LHSE)
          return ErrorV("destination of '=' must be a variable");

Unlike the rest of the binary operators, our assignment operator doesn't
follow the "emit LHS, emit RHS, do computation" model. As such, it is
handled as a special case before the other binary operators are handled.
The other strange thing is that it requires the LHS to be a variable. It
is invalid to have "(x+1) = expr" - only things like "x = expr" are
allowed.

.. code-block:: c++

        // Codegen the RHS.
        Value *Val = RHS->Codegen();
        if (Val == 0) return 0;

        // Look up the name.
        Value *Variable = NamedValues[LHSE->getName()];
        if (Variable == 0) return ErrorV("Unknown variable name");

        Builder.CreateStore(Val, Variable);
        return Val;
      }
      ...

Once we have the variable, codegen'ing the assignment is
straightforward: we emit the RHS of the assignment, create a store, and
return the computed value. Returning a value allows for chained
assignments like "X = (Y = Z)".

Now that we have an assignment operator, we can mutate loop variables
and arguments. For example, we can now run code like this:

::

    # Function to print a double.
    extern printd(x);

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;

    def test(x)
      printd(x) :
      x = 4 :
      printd(x);

    test(123);

When run, this example prints "123" and then "4", showing that we did
actually mutate the value! Okay, we have now officially implemented our
goal: getting this to work requires SSA construction in the general
case. However, to be really useful, we want the ability to define our
own local variables, lets add this next!

User-defined Local Variables
============================

Adding var/in is just like any other other extensions we made to
Kaleidoscope: we extend the lexer, the parser, the AST and the code
generator. The first step for adding our new 'var/in' construct is to
extend the lexer. As before, this is pretty trivial, the code looks like
this:

.. code-block:: c++

    enum Token {
      ...
      // var definition
      tok_var = -13
    ...
    }
    ...
    static int gettok() {
    ...
        if (IdentifierStr == "in") return tok_in;
        if (IdentifierStr == "binary") return tok_binary;
        if (IdentifierStr == "unary") return tok_unary;
        if (IdentifierStr == "var") return tok_var;
        return tok_identifier;
    ...

The next step is to define the AST node that we will construct. For
var/in, it looks like this:

.. code-block:: c++

    /// VarExprAST - Expression class for var/in
    class VarExprAST : public ExprAST {
      std::vector<std::pair<std::string, ExprAST*> > VarNames;
      ExprAST *Body;
    public:
      VarExprAST(const std::vector<std::pair<std::string, ExprAST*> > &varnames,
                 ExprAST *body)
      : VarNames(varnames), Body(body) {}

      virtual Value *Codegen();
    };

var/in allows a list of names to be defined all at once, and each name
can optionally have an initializer value. As such, we capture this
information in the VarNames vector. Also, var/in has a body, this body
is allowed to access the variables defined by the var/in.

With this in place, we can define the parser pieces. The first thing we
do is add it as a primary expression:

.. code-block:: c++

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= ifexpr
    ///   ::= forexpr
    ///   ::= varexpr
    static ExprAST *ParsePrimary() {
      switch (CurTok) {
      default: return Error("unknown token when expecting an expression");
      case tok_identifier: return ParseIdentifierExpr();
      case tok_number:     return ParseNumberExpr();
      case '(':            return ParseParenExpr();
      case tok_if:         return ParseIfExpr();
      case tok_for:        return ParseForExpr();
      case tok_var:        return ParseVarExpr();
      }
    }

Next we define ParseVarExpr:

.. code-block:: c++

    /// varexpr ::= 'var' identifier ('=' expression)?
    //                    (',' identifier ('=' expression)?)* 'in' expression
    static ExprAST *ParseVarExpr() {
      getNextToken();  // eat the var.

      std::vector<std::pair<std::string, ExprAST*> > VarNames;

      // At least one variable name is required.
      if (CurTok != tok_identifier)
        return Error("expected identifier after var");

The first part of this code parses the list of identifier/expr pairs
into the local ``VarNames`` vector.

.. code-block:: c++

      while (1) {
        std::string Name = IdentifierStr;
        getNextToken();  // eat identifier.

        // Read the optional initializer.
        ExprAST *Init = 0;
        if (CurTok == '=') {
          getNextToken(); // eat the '='.

          Init = ParseExpression();
          if (Init == 0) return 0;
        }

        VarNames.push_back(std::make_pair(Name, Init));

        // End of var list, exit loop.
        if (CurTok != ',') break;
        getNextToken(); // eat the ','.

        if (CurTok != tok_identifier)
          return Error("expected identifier list after var");
      }

Once all the variables are parsed, we then parse the body and create the
AST node:

.. code-block:: c++

      // At this point, we have to have 'in'.
      if (CurTok != tok_in)
        return Error("expected 'in' keyword after 'var'");
      getNextToken();  // eat 'in'.

      ExprAST *Body = ParseExpression();
      if (Body == 0) return 0;

      return new VarExprAST(VarNames, Body);
    }

Now that we can parse and represent the code, we need to support
emission of LLVM IR for it. This code starts out with:

.. code-block:: c++

    Value *VarExprAST::Codegen() {
      std::vector<AllocaInst *> OldBindings;

      Function *TheFunction = Builder.GetInsertBlock()->getParent();

      // Register all variables and emit their initializer.
      for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
        const std::string &VarName = VarNames[i].first;
        ExprAST *Init = VarNames[i].second;

Basically it loops over all the variables, installing them one at a
time. For each variable we put into the symbol table, we remember the
previous value that we replace in OldBindings.

.. code-block:: c++

        // Emit the initializer before adding the variable to scope, this prevents
        // the initializer from referencing the variable itself, and permits stuff
        // like this:
        //  var a = 1 in
        //    var a = a in ...   # refers to outer 'a'.
        Value *InitVal;
        if (Init) {
          InitVal = Init->Codegen();
          if (InitVal == 0) return 0;
        } else { // If not specified, use 0.0.
          InitVal = ConstantFP::get(getGlobalContext(), APFloat(0.0));
        }

        AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
        Builder.CreateStore(InitVal, Alloca);

        // Remember the old variable binding so that we can restore the binding when
        // we unrecurse.
        OldBindings.push_back(NamedValues[VarName]);

        // Remember this binding.
        NamedValues[VarName] = Alloca;
      }

There are more comments here than code. The basic idea is that we emit
the initializer, create the alloca, then update the symbol table to
point to it. Once all the variables are installed in the symbol table,
we evaluate the body of the var/in expression:

.. code-block:: c++

      // Codegen the body, now that all vars are in scope.
      Value *BodyVal = Body->Codegen();
      if (BodyVal == 0) return 0;

Finally, before returning, we restore the previous variable bindings:

.. code-block:: c++

      // Pop all our variables from scope.
      for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
        NamedValues[VarNames[i].first] = OldBindings[i];

      // Return the body computation.
      return BodyVal;
    }

The end result of all of this is that we get properly scoped variable
definitions, and we even (trivially) allow mutation of them :).

With this, we completed what we set out to do. Our nice iterative fib
example from the intro compiles and runs just fine. The mem2reg pass
optimizes all of our stack variables into SSA registers, inserting PHI
nodes where needed, and our front-end remains simple: no "iterated
dominance frontier" computation anywhere in sight.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
mutable variables and var/in support. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cppflags --ldflags --libs core jit native` -O3 -o toy
    # Run
    ./toy

Here is the code:

.. code-block:: c++

    #include "llvm/DerivedTypes.h"
    #include "llvm/ExecutionEngine/ExecutionEngine.h"
    #include "llvm/ExecutionEngine/JIT.h"
    #include "llvm/IRBuilder.h"
    #include "llvm/LLVMContext.h"
    #include "llvm/Module.h"
    #include "llvm/PassManager.h"
    #include "llvm/Analysis/Verifier.h"
    #include "llvm/Analysis/Passes.h"
    #include "llvm/DataLayout.h"
    #include "llvm/Transforms/Scalar.h"
    #include "llvm/Support/TargetSelect.h"
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
      tok_identifier = -4, tok_number = -5,

      // control
      tok_if = -6, tok_then = -7, tok_else = -8,
      tok_for = -9, tok_in = -10,

      // operators
      tok_binary = -11, tok_unary = -12,

      // var definition
      tok_var = -13
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
        if (IdentifierStr == "if") return tok_if;
        if (IdentifierStr == "then") return tok_then;
        if (IdentifierStr == "else") return tok_else;
        if (IdentifierStr == "for") return tok_for;
        if (IdentifierStr == "in") return tok_in;
        if (IdentifierStr == "binary") return tok_binary;
        if (IdentifierStr == "unary") return tok_unary;
        if (IdentifierStr == "var") return tok_var;
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
      const std::string &getName() const { return Name; }
      virtual Value *Codegen();
    };

    /// UnaryExprAST - Expression class for a unary operator.
    class UnaryExprAST : public ExprAST {
      char Opcode;
      ExprAST *Operand;
    public:
      UnaryExprAST(char opcode, ExprAST *operand)
        : Opcode(opcode), Operand(operand) {}
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

    /// IfExprAST - Expression class for if/then/else.
    class IfExprAST : public ExprAST {
      ExprAST *Cond, *Then, *Else;
    public:
      IfExprAST(ExprAST *cond, ExprAST *then, ExprAST *_else)
      : Cond(cond), Then(then), Else(_else) {}
      virtual Value *Codegen();
    };

    /// ForExprAST - Expression class for for/in.
    class ForExprAST : public ExprAST {
      std::string VarName;
      ExprAST *Start, *End, *Step, *Body;
    public:
      ForExprAST(const std::string &varname, ExprAST *start, ExprAST *end,
                 ExprAST *step, ExprAST *body)
        : VarName(varname), Start(start), End(end), Step(step), Body(body) {}
      virtual Value *Codegen();
    };

    /// VarExprAST - Expression class for var/in
    class VarExprAST : public ExprAST {
      std::vector<std::pair<std::string, ExprAST*> > VarNames;
      ExprAST *Body;
    public:
      VarExprAST(const std::vector<std::pair<std::string, ExprAST*> > &varnames,
                 ExprAST *body)
      : VarNames(varnames), Body(body) {}

      virtual Value *Codegen();
    };

    /// PrototypeAST - This class represents the "prototype" for a function,
    /// which captures its name, and its argument names (thus implicitly the number
    /// of arguments the function takes), as well as if it is an operator.
    class PrototypeAST {
      std::string Name;
      std::vector<std::string> Args;
      bool isOperator;
      unsigned Precedence;  // Precedence if a binary op.
    public:
      PrototypeAST(const std::string &name, const std::vector<std::string> &args,
                   bool isoperator = false, unsigned prec = 0)
      : Name(name), Args(args), isOperator(isoperator), Precedence(prec) {}

      bool isUnaryOp() const { return isOperator && Args.size() == 1; }
      bool isBinaryOp() const { return isOperator && Args.size() == 2; }

      char getOperatorName() const {
        assert(isUnaryOp() || isBinaryOp());
        return Name[Name.size()-1];
      }

      unsigned getBinaryPrecedence() const { return Precedence; }

      Function *Codegen();

      void CreateArgumentAllocas(Function *F);
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

    /// ifexpr ::= 'if' expression 'then' expression 'else' expression
    static ExprAST *ParseIfExpr() {
      getNextToken();  // eat the if.

      // condition.
      ExprAST *Cond = ParseExpression();
      if (!Cond) return 0;

      if (CurTok != tok_then)
        return Error("expected then");
      getNextToken();  // eat the then

      ExprAST *Then = ParseExpression();
      if (Then == 0) return 0;

      if (CurTok != tok_else)
        return Error("expected else");

      getNextToken();

      ExprAST *Else = ParseExpression();
      if (!Else) return 0;

      return new IfExprAST(Cond, Then, Else);
    }

    /// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
    static ExprAST *ParseForExpr() {
      getNextToken();  // eat the for.

      if (CurTok != tok_identifier)
        return Error("expected identifier after for");

      std::string IdName = IdentifierStr;
      getNextToken();  // eat identifier.

      if (CurTok != '=')
        return Error("expected '=' after for");
      getNextToken();  // eat '='.


      ExprAST *Start = ParseExpression();
      if (Start == 0) return 0;
      if (CurTok != ',')
        return Error("expected ',' after for start value");
      getNextToken();

      ExprAST *End = ParseExpression();
      if (End == 0) return 0;

      // The step value is optional.
      ExprAST *Step = 0;
      if (CurTok == ',') {
        getNextToken();
        Step = ParseExpression();
        if (Step == 0) return 0;
      }

      if (CurTok != tok_in)
        return Error("expected 'in' after for");
      getNextToken();  // eat 'in'.

      ExprAST *Body = ParseExpression();
      if (Body == 0) return 0;

      return new ForExprAST(IdName, Start, End, Step, Body);
    }

    /// varexpr ::= 'var' identifier ('=' expression)?
    //                    (',' identifier ('=' expression)?)* 'in' expression
    static ExprAST *ParseVarExpr() {
      getNextToken();  // eat the var.

      std::vector<std::pair<std::string, ExprAST*> > VarNames;

      // At least one variable name is required.
      if (CurTok != tok_identifier)
        return Error("expected identifier after var");

      while (1) {
        std::string Name = IdentifierStr;
        getNextToken();  // eat identifier.

        // Read the optional initializer.
        ExprAST *Init = 0;
        if (CurTok == '=') {
          getNextToken(); // eat the '='.

          Init = ParseExpression();
          if (Init == 0) return 0;
        }

        VarNames.push_back(std::make_pair(Name, Init));

        // End of var list, exit loop.
        if (CurTok != ',') break;
        getNextToken(); // eat the ','.

        if (CurTok != tok_identifier)
          return Error("expected identifier list after var");
      }

      // At this point, we have to have 'in'.
      if (CurTok != tok_in)
        return Error("expected 'in' keyword after 'var'");
      getNextToken();  // eat 'in'.

      ExprAST *Body = ParseExpression();
      if (Body == 0) return 0;

      return new VarExprAST(VarNames, Body);
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= ifexpr
    ///   ::= forexpr
    ///   ::= varexpr
    static ExprAST *ParsePrimary() {
      switch (CurTok) {
      default: return Error("unknown token when expecting an expression");
      case tok_identifier: return ParseIdentifierExpr();
      case tok_number:     return ParseNumberExpr();
      case '(':            return ParseParenExpr();
      case tok_if:         return ParseIfExpr();
      case tok_for:        return ParseForExpr();
      case tok_var:        return ParseVarExpr();
      }
    }

    /// unary
    ///   ::= primary
    ///   ::= '!' unary
    static ExprAST *ParseUnary() {
      // If the current token is not an operator, it must be a primary expr.
      if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
        return ParsePrimary();

      // If this is a unary operator, read it.
      int Opc = CurTok;
      getNextToken();
      if (ExprAST *Operand = ParseUnary())
        return new UnaryExprAST(Opc, Operand);
      return 0;
    }

    /// binoprhs
    ///   ::= ('+' unary)*
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

        // Parse the unary expression after the binary operator.
        ExprAST *RHS = ParseUnary();
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
    ///   ::= unary binoprhs
    ///
    static ExprAST *ParseExpression() {
      ExprAST *LHS = ParseUnary();
      if (!LHS) return 0;

      return ParseBinOpRHS(0, LHS);
    }

    /// prototype
    ///   ::= id '(' id* ')'
    ///   ::= binary LETTER number? (id, id)
    ///   ::= unary LETTER (id)
    static PrototypeAST *ParsePrototype() {
      std::string FnName;

      unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
      unsigned BinaryPrecedence = 30;

      switch (CurTok) {
      default:
        return ErrorP("Expected function name in prototype");
      case tok_identifier:
        FnName = IdentifierStr;
        Kind = 0;
        getNextToken();
        break;
      case tok_unary:
        getNextToken();
        if (!isascii(CurTok))
          return ErrorP("Expected unary operator");
        FnName = "unary";
        FnName += (char)CurTok;
        Kind = 1;
        getNextToken();
        break;
      case tok_binary:
        getNextToken();
        if (!isascii(CurTok))
          return ErrorP("Expected binary operator");
        FnName = "binary";
        FnName += (char)CurTok;
        Kind = 2;
        getNextToken();

        // Read the precedence if present.
        if (CurTok == tok_number) {
          if (NumVal < 1 || NumVal > 100)
            return ErrorP("Invalid precedecnce: must be 1..100");
          BinaryPrecedence = (unsigned)NumVal;
          getNextToken();
        }
        break;
      }

      if (CurTok != '(')
        return ErrorP("Expected '(' in prototype");

      std::vector<std::string> ArgNames;
      while (getNextToken() == tok_identifier)
        ArgNames.push_back(IdentifierStr);
      if (CurTok != ')')
        return ErrorP("Expected ')' in prototype");

      // success.
      getNextToken();  // eat ')'.

      // Verify right number of names for operator.
      if (Kind && ArgNames.size() != Kind)
        return ErrorP("Invalid number of operands for operator");

      return new PrototypeAST(FnName, ArgNames, Kind != 0, BinaryPrecedence);
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
    static std::map<std::string, AllocaInst*> NamedValues;
    static FunctionPassManager *TheFPM;

    Value *ErrorV(const char *Str) { Error(Str); return 0; }

    /// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
    /// the function.  This is used for mutable variables etc.
    static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                              const std::string &VarName) {
      IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                     TheFunction->getEntryBlock().begin());
      return TmpB.CreateAlloca(Type::getDoubleTy(getGlobalContext()), 0,
                               VarName.c_str());
    }

    Value *NumberExprAST::Codegen() {
      return ConstantFP::get(getGlobalContext(), APFloat(Val));
    }

    Value *VariableExprAST::Codegen() {
      // Look this variable up in the function.
      Value *V = NamedValues[Name];
      if (V == 0) return ErrorV("Unknown variable name");

      // Load the value.
      return Builder.CreateLoad(V, Name.c_str());
    }

    Value *UnaryExprAST::Codegen() {
      Value *OperandV = Operand->Codegen();
      if (OperandV == 0) return 0;

      Function *F = TheModule->getFunction(std::string("unary")+Opcode);
      if (F == 0)
        return ErrorV("Unknown unary operator");

      return Builder.CreateCall(F, OperandV, "unop");
    }

    Value *BinaryExprAST::Codegen() {
      // Special case '=' because we don't want to emit the LHS as an expression.
      if (Op == '=') {
        // Assignment requires the LHS to be an identifier.
        VariableExprAST *LHSE = dynamic_cast<VariableExprAST*>(LHS);
        if (!LHSE)
          return ErrorV("destination of '=' must be a variable");
        // Codegen the RHS.
        Value *Val = RHS->Codegen();
        if (Val == 0) return 0;

        // Look up the name.
        Value *Variable = NamedValues[LHSE->getName()];
        if (Variable == 0) return ErrorV("Unknown variable name");

        Builder.CreateStore(Val, Variable);
        return Val;
      }

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
      default: break;
      }

      // If it wasn't a builtin binary operator, it must be a user defined one. Emit
      // a call to it.
      Function *F = TheModule->getFunction(std::string("binary")+Op);
      assert(F && "binary operator not found!");

      Value *Ops[2] = { L, R };
      return Builder.CreateCall(F, Ops, "binop");
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

    Value *IfExprAST::Codegen() {
      Value *CondV = Cond->Codegen();
      if (CondV == 0) return 0;

      // Convert condition to a bool by comparing equal to 0.0.
      CondV = Builder.CreateFCmpONE(CondV,
                                  ConstantFP::get(getGlobalContext(), APFloat(0.0)),
                                    "ifcond");

      Function *TheFunction = Builder.GetInsertBlock()->getParent();

      // Create blocks for the then and else cases.  Insert the 'then' block at the
      // end of the function.
      BasicBlock *ThenBB = BasicBlock::Create(getGlobalContext(), "then", TheFunction);
      BasicBlock *ElseBB = BasicBlock::Create(getGlobalContext(), "else");
      BasicBlock *MergeBB = BasicBlock::Create(getGlobalContext(), "ifcont");

      Builder.CreateCondBr(CondV, ThenBB, ElseBB);

      // Emit then value.
      Builder.SetInsertPoint(ThenBB);

      Value *ThenV = Then->Codegen();
      if (ThenV == 0) return 0;

      Builder.CreateBr(MergeBB);
      // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
      ThenBB = Builder.GetInsertBlock();

      // Emit else block.
      TheFunction->getBasicBlockList().push_back(ElseBB);
      Builder.SetInsertPoint(ElseBB);

      Value *ElseV = Else->Codegen();
      if (ElseV == 0) return 0;

      Builder.CreateBr(MergeBB);
      // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
      ElseBB = Builder.GetInsertBlock();

      // Emit merge block.
      TheFunction->getBasicBlockList().push_back(MergeBB);
      Builder.SetInsertPoint(MergeBB);
      PHINode *PN = Builder.CreatePHI(Type::getDoubleTy(getGlobalContext()), 2,
                                      "iftmp");

      PN->addIncoming(ThenV, ThenBB);
      PN->addIncoming(ElseV, ElseBB);
      return PN;
    }

    Value *ForExprAST::Codegen() {
      // Output this as:
      //   var = alloca double
      //   ...
      //   start = startexpr
      //   store start -> var
      //   goto loop
      // loop:
      //   ...
      //   bodyexpr
      //   ...
      // loopend:
      //   step = stepexpr
      //   endcond = endexpr
      //
      //   curvar = load var
      //   nextvar = curvar + step
      //   store nextvar -> var
      //   br endcond, loop, endloop
      // outloop:

      Function *TheFunction = Builder.GetInsertBlock()->getParent();

      // Create an alloca for the variable in the entry block.
      AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

      // Emit the start code first, without 'variable' in scope.
      Value *StartVal = Start->Codegen();
      if (StartVal == 0) return 0;

      // Store the value into the alloca.
      Builder.CreateStore(StartVal, Alloca);

      // Make the new basic block for the loop header, inserting after current
      // block.
      BasicBlock *LoopBB = BasicBlock::Create(getGlobalContext(), "loop", TheFunction);

      // Insert an explicit fall through from the current block to the LoopBB.
      Builder.CreateBr(LoopBB);

      // Start insertion in LoopBB.
      Builder.SetInsertPoint(LoopBB);

      // Within the loop, the variable is defined equal to the PHI node.  If it
      // shadows an existing variable, we have to restore it, so save it now.
      AllocaInst *OldVal = NamedValues[VarName];
      NamedValues[VarName] = Alloca;

      // Emit the body of the loop.  This, like any other expr, can change the
      // current BB.  Note that we ignore the value computed by the body, but don't
      // allow an error.
      if (Body->Codegen() == 0)
        return 0;

      // Emit the step value.
      Value *StepVal;
      if (Step) {
        StepVal = Step->Codegen();
        if (StepVal == 0) return 0;
      } else {
        // If not specified, use 1.0.
        StepVal = ConstantFP::get(getGlobalContext(), APFloat(1.0));
      }

      // Compute the end condition.
      Value *EndCond = End->Codegen();
      if (EndCond == 0) return EndCond;

      // Reload, increment, and restore the alloca.  This handles the case where
      // the body of the loop mutates the variable.
      Value *CurVar = Builder.CreateLoad(Alloca, VarName.c_str());
      Value *NextVar = Builder.CreateFAdd(CurVar, StepVal, "nextvar");
      Builder.CreateStore(NextVar, Alloca);

      // Convert condition to a bool by comparing equal to 0.0.
      EndCond = Builder.CreateFCmpONE(EndCond,
                                  ConstantFP::get(getGlobalContext(), APFloat(0.0)),
                                      "loopcond");

      // Create the "after loop" block and insert it.
      BasicBlock *AfterBB = BasicBlock::Create(getGlobalContext(), "afterloop", TheFunction);

      // Insert the conditional branch into the end of LoopEndBB.
      Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

      // Any new code will be inserted in AfterBB.
      Builder.SetInsertPoint(AfterBB);

      // Restore the unshadowed variable.
      if (OldVal)
        NamedValues[VarName] = OldVal;
      else
        NamedValues.erase(VarName);


      // for expr always returns 0.0.
      return Constant::getNullValue(Type::getDoubleTy(getGlobalContext()));
    }

    Value *VarExprAST::Codegen() {
      std::vector<AllocaInst *> OldBindings;

      Function *TheFunction = Builder.GetInsertBlock()->getParent();

      // Register all variables and emit their initializer.
      for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
        const std::string &VarName = VarNames[i].first;
        ExprAST *Init = VarNames[i].second;

        // Emit the initializer before adding the variable to scope, this prevents
        // the initializer from referencing the variable itself, and permits stuff
        // like this:
        //  var a = 1 in
        //    var a = a in ...   # refers to outer 'a'.
        Value *InitVal;
        if (Init) {
          InitVal = Init->Codegen();
          if (InitVal == 0) return 0;
        } else { // If not specified, use 0.0.
          InitVal = ConstantFP::get(getGlobalContext(), APFloat(0.0));
        }

        AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
        Builder.CreateStore(InitVal, Alloca);

        // Remember the old variable binding so that we can restore the binding when
        // we unrecurse.
        OldBindings.push_back(NamedValues[VarName]);

        // Remember this binding.
        NamedValues[VarName] = Alloca;
      }

      // Codegen the body, now that all vars are in scope.
      Value *BodyVal = Body->Codegen();
      if (BodyVal == 0) return 0;

      // Pop all our variables from scope.
      for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
        NamedValues[VarNames[i].first] = OldBindings[i];

      // Return the body computation.
      return BodyVal;
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
           ++AI, ++Idx)
        AI->setName(Args[Idx]);

      return F;
    }

    /// CreateArgumentAllocas - Create an alloca for each argument and register the
    /// argument in the symbol table so that references to it will succeed.
    void PrototypeAST::CreateArgumentAllocas(Function *F) {
      Function::arg_iterator AI = F->arg_begin();
      for (unsigned Idx = 0, e = Args.size(); Idx != e; ++Idx, ++AI) {
        // Create an alloca for this variable.
        AllocaInst *Alloca = CreateEntryBlockAlloca(F, Args[Idx]);

        // Store the initial value into the alloca.
        Builder.CreateStore(AI, Alloca);

        // Add arguments to variable symbol table.
        NamedValues[Args[Idx]] = Alloca;
      }
    }

    Function *FunctionAST::Codegen() {
      NamedValues.clear();

      Function *TheFunction = Proto->Codegen();
      if (TheFunction == 0)
        return 0;

      // If this is an operator, install it.
      if (Proto->isBinaryOp())
        BinopPrecedence[Proto->getOperatorName()] = Proto->getBinaryPrecedence();

      // Create a new basic block to start insertion into.
      BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
      Builder.SetInsertPoint(BB);

      // Add all arguments to the symbol table and create their allocas.
      Proto->CreateArgumentAllocas(TheFunction);

      if (Value *RetVal = Body->Codegen()) {
        // Finish off the function.
        Builder.CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        // Optimize the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
      }

      // Error reading body, remove function.
      TheFunction->eraseFromParent();

      if (Proto->isBinaryOp())
        BinopPrecedence.erase(Proto->getOperatorName());
      return 0;
    }

    //===----------------------------------------------------------------------===//
    // Top-Level parsing and JIT Driver
    //===----------------------------------------------------------------------===//

    static ExecutionEngine *TheExecutionEngine;

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
          // JIT the function, returning a function pointer.
          void *FPtr = TheExecutionEngine->getPointerToFunction(LF);

          // Cast it to the right type (takes no arguments, returns a double) so we
          // can call it as a native function.
          double (*FP)() = (double (*)())(intptr_t)FPtr;
          fprintf(stderr, "Evaluated to %f\n", FP());
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

    /// printd - printf that takes a double prints it as "%f\n", returning 0.
    extern "C"
    double printd(double X) {
      printf("%f\n", X);
      return 0;
    }

    //===----------------------------------------------------------------------===//
    // Main driver code.
    //===----------------------------------------------------------------------===//

    int main() {
      InitializeNativeTarget();
      LLVMContext &Context = getGlobalContext();

      // Install standard binary operators.
      // 1 is lowest precedence.
      BinopPrecedence['='] = 2;
      BinopPrecedence['<'] = 10;
      BinopPrecedence['+'] = 20;
      BinopPrecedence['-'] = 20;
      BinopPrecedence['*'] = 40;  // highest.

      // Prime the first token.
      fprintf(stderr, "ready> ");
      getNextToken();

      // Make the module, which holds all the code.
      TheModule = new Module("my cool jit", Context);

      // Create the JIT.  This takes ownership of the module.
      std::string ErrStr;
      TheExecutionEngine = EngineBuilder(TheModule).setErrorStr(&ErrStr).create();
      if (!TheExecutionEngine) {
        fprintf(stderr, "Could not create ExecutionEngine: %s\n", ErrStr.c_str());
        exit(1);
      }

      FunctionPassManager OurFPM(TheModule);

      // Set up the optimizer pipeline.  Start with registering info about how the
      // target lays out data structures.
      OurFPM.add(new DataLayout(*TheExecutionEngine->getDataLayout()));
      // Provide basic AliasAnalysis support for GVN.
      OurFPM.add(createBasicAliasAnalysisPass());
      // Promote allocas to registers.
      OurFPM.add(createPromoteMemoryToRegisterPass());
      // Do simple "peephole" optimizations and bit-twiddling optzns.
      OurFPM.add(createInstructionCombiningPass());
      // Reassociate expressions.
      OurFPM.add(createReassociatePass());
      // Eliminate Common SubExpressions.
      OurFPM.add(createGVNPass());
      // Simplify the control flow graph (deleting unreachable blocks, etc).
      OurFPM.add(createCFGSimplificationPass());

      OurFPM.doInitialization();

      // Set the global so the code gen can use this.
      TheFPM = &OurFPM;

      // Run the main "interpreter loop" now.
      MainLoop();

      TheFPM = 0;

      // Print out all of the generated code.
      TheModule->dump();

      return 0;
    }

`Next: Conclusion and other useful LLVM tidbits <LangImpl8.html>`_

