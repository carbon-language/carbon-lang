============================================================
Kaleidoscope: Extending the Language: User-defined Operators
============================================================

.. contents::
   :local:

Chapter 6 Introduction
======================

Welcome to Chapter 6 of the "`Implementing a language with
LLVM <index.html>`_" tutorial. At this point in our tutorial, we now
have a fully functional language that is fairly minimal, but also
useful. There is still one big problem with it, however. Our language
doesn't have many useful operators (like division, logical negation, or
even any comparisons besides less-than).

This chapter of the tutorial takes a wild digression into adding
user-defined operators to the simple and beautiful Kaleidoscope
language. This digression now gives us a simple and ugly language in
some ways, but also a powerful one at the same time. One of the great
things about creating your own language is that you get to decide what
is good or bad. In this tutorial we'll assume that it is okay to use
this as a way to show some interesting parsing techniques.

At the end of this tutorial, we'll run through an example Kaleidoscope
application that `renders the Mandelbrot set <#example>`_. This gives an
example of what you can build with Kaleidoscope and its feature set.

User-defined Operators: the Idea
================================

The "operator overloading" that we will add to Kaleidoscope is more
general than languages like C++. In C++, you are only allowed to
redefine existing operators: you can't programatically change the
grammar, introduce new operators, change precedence levels, etc. In this
chapter, we will add this capability to Kaleidoscope, which will let the
user round out the set of operators that are supported.

The point of going into user-defined operators in a tutorial like this
is to show the power and flexibility of using a hand-written parser.
Thus far, the parser we have been implementing uses recursive descent
for most parts of the grammar and operator precedence parsing for the
expressions. See `Chapter 2 <LangImpl2.html>`_ for details. Without
using operator precedence parsing, it would be very difficult to allow
the programmer to introduce new operators into the grammar: the grammar
is dynamically extensible as the JIT runs.

The two specific features we'll add are programmable unary operators
(right now, Kaleidoscope has no unary operators at all) as well as
binary operators. An example of this is:

::

    # Logical unary not.
    def unary!(v)
      if v then
        0
      else
        1;

    # Define > with the same precedence as <.
    def binary> 10 (LHS RHS)
      RHS < LHS;

    # Binary "logical or", (note that it does not "short circuit")
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;

    # Define = with slightly lower precedence than relationals.
    def binary= 9 (LHS RHS)
      !(LHS < RHS | LHS > RHS);

Many languages aspire to being able to implement their standard runtime
library in the language itself. In Kaleidoscope, we can implement
significant parts of the language in the library!

We will break down implementation of these features into two parts:
implementing support for user-defined binary operators and adding unary
operators.

User-defined Binary Operators
=============================

Adding support for user-defined binary operators is pretty simple with
our current framework. We'll first add support for the unary/binary
keywords:

.. code-block:: c++

    enum Token {
      ...
      // operators
      tok_binary = -11, tok_unary = -12
    };
    ...
    static int gettok() {
    ...
        if (IdentifierStr == "for") return tok_for;
        if (IdentifierStr == "in") return tok_in;
        if (IdentifierStr == "binary") return tok_binary;
        if (IdentifierStr == "unary") return tok_unary;
        return tok_identifier;

This just adds lexer support for the unary and binary keywords, like we
did in `previous chapters <LangImpl5.html#iflexer>`_. One nice thing
about our current AST, is that we represent binary operators with full
generalisation by using their ASCII code as the opcode. For our extended
operators, we'll use this same representation, so we don't need any new
AST or parser support.

On the other hand, we have to be able to represent the definitions of
these new operators, in the "def binary\| 5" part of the function
definition. In our grammar so far, the "name" for the function
definition is parsed as the "prototype" production and into the
``PrototypeAST`` AST node. To represent our new user-defined operators
as prototypes, we have to extend the ``PrototypeAST`` AST node like
this:

.. code-block:: c++

    /// PrototypeAST - This class represents the "prototype" for a function,
    /// which captures its argument names as well as if it is an operator.
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
    };

Basically, in addition to knowing a name for the prototype, we now keep
track of whether it was an operator, and if it was, what precedence
level the operator is at. The precedence is only used for binary
operators (as you'll see below, it just doesn't apply for unary
operators). Now that we have a way to represent the prototype for a
user-defined operator, we need to parse it:

.. code-block:: c++

    /// prototype
    ///   ::= id '(' id* ')'
    ///   ::= binary LETTER number? (id, id)
    static PrototypeAST *ParsePrototype() {
      std::string FnName;

      unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
      unsigned BinaryPrecedence = 30;

      switch (CurTok) {
      default:
        return ErrorP("Expected function name in prototype");
      case tok_identifier:
        FnName = IdentifierStr;
        Kind = 0;
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

This is all fairly straightforward parsing code, and we have already
seen a lot of similar code in the past. One interesting part about the
code above is the couple lines that set up ``FnName`` for binary
operators. This builds names like "binary@" for a newly defined "@"
operator. This then takes advantage of the fact that symbol names in the
LLVM symbol table are allowed to have any character in them, including
embedded nul characters.

The next interesting thing to add, is codegen support for these binary
operators. Given our current structure, this is a simple addition of a
default case for our existing binary operator node:

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
      default: break;
      }

      // If it wasn't a builtin binary operator, it must be a user defined one. Emit
      // a call to it.
      Function *F = TheModule->getFunction(std::string("binary")+Op);
      assert(F && "binary operator not found!");

      Value *Ops[2] = { L, R };
      return Builder.CreateCall(F, Ops, "binop");
    }

As you can see above, the new code is actually really simple. It just
does a lookup for the appropriate operator in the symbol table and
generates a function call to it. Since user-defined operators are just
built as normal functions (because the "prototype" boils down to a
function with the right name) everything falls into place.

The final piece of code we are missing, is a bit of top-level magic:

.. code-block:: c++

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

      if (Value *RetVal = Body->Codegen()) {
        ...

Basically, before codegening a function, if it is a user-defined
operator, we register it in the precedence table. This allows the binary
operator parsing logic we already have in place to handle it. Since we
are working on a fully-general operator precedence parser, this is all
we need to do to "extend the grammar".

Now we have useful user-defined binary operators. This builds a lot on
the previous framework we built for other operators. Adding unary
operators is a bit more challenging, because we don't have any framework
for it yet - lets see what it takes.

User-defined Unary Operators
============================

Since we don't currently support unary operators in the Kaleidoscope
language, we'll need to add everything to support them. Above, we added
simple support for the 'unary' keyword to the lexer. In addition to
that, we need an AST node:

.. code-block:: c++

    /// UnaryExprAST - Expression class for a unary operator.
    class UnaryExprAST : public ExprAST {
      char Opcode;
      ExprAST *Operand;
    public:
      UnaryExprAST(char opcode, ExprAST *operand)
        : Opcode(opcode), Operand(operand) {}
      virtual Value *Codegen();
    };

This AST node is very simple and obvious by now. It directly mirrors the
binary operator AST node, except that it only has one child. With this,
we need to add the parsing logic. Parsing a unary operator is pretty
simple: we'll add a new function to do it:

.. code-block:: c++

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

The grammar we add is pretty straightforward here. If we see a unary
operator when parsing a primary operator, we eat the operator as a
prefix and parse the remaining piece as another unary operator. This
allows us to handle multiple unary operators (e.g. "!!x"). Note that
unary operators can't have ambiguous parses like binary operators can,
so there is no need for precedence information.

The problem with this function, is that we need to call ParseUnary from
somewhere. To do this, we change previous callers of ParsePrimary to
call ParseUnary instead:

.. code-block:: c++

    /// binoprhs
    ///   ::= ('+' unary)*
    static ExprAST *ParseBinOpRHS(int ExprPrec, ExprAST *LHS) {
      ...
        // Parse the unary expression after the binary operator.
        ExprAST *RHS = ParseUnary();
        if (!RHS) return 0;
      ...
    }
    /// expression
    ///   ::= unary binoprhs
    ///
    static ExprAST *ParseExpression() {
      ExprAST *LHS = ParseUnary();
      if (!LHS) return 0;

      return ParseBinOpRHS(0, LHS);
    }

With these two simple changes, we are now able to parse unary operators
and build the AST for them. Next up, we need to add parser support for
prototypes, to parse the unary operator prototype. We extend the binary
operator code above with:

.. code-block:: c++

    /// prototype
    ///   ::= id '(' id* ')'
    ///   ::= binary LETTER number? (id, id)
    ///   ::= unary LETTER (id)
    static PrototypeAST *ParsePrototype() {
      std::string FnName;

      unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
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
        ...

As with binary operators, we name unary operators with a name that
includes the operator character. This assists us at code generation
time. Speaking of, the final piece we need to add is codegen support for
unary operators. It looks like this:

.. code-block:: c++

    Value *UnaryExprAST::Codegen() {
      Value *OperandV = Operand->Codegen();
      if (OperandV == 0) return 0;

      Function *F = TheModule->getFunction(std::string("unary")+Opcode);
      if (F == 0)
        return ErrorV("Unknown unary operator");

      return Builder.CreateCall(F, OperandV, "unop");
    }

This code is similar to, but simpler than, the code for binary
operators. It is simpler primarily because it doesn't need to handle any
predefined operators.

Kicking the Tires
=================

It is somewhat hard to believe, but with a few simple extensions we've
covered in the last chapters, we have grown a real-ish language. With
this, we can do a lot of interesting things, including I/O, math, and a
bunch of other things. For example, we can now add a nice sequencing
operator (printd is defined to print out the specified value and a
newline):

::

    ready> extern printd(x);
    Read extern:
    declare double @printd(double)

    ready> def binary : 1 (x y) 0;  # Low-precedence operator that ignores operands.
    ..
    ready> printd(123) : printd(456) : printd(789);
    123.000000
    456.000000
    789.000000
    Evaluated to 0.000000

We can also define a bunch of other "primitive" operations, such as:

::

    # Logical unary not.
    def unary!(v)
      if v then
        0
      else
        1;

    # Unary negate.
    def unary-(v)
      0-v;

    # Define > with the same precedence as <.
    def binary> 10 (LHS RHS)
      RHS < LHS;

    # Binary logical or, which does not short circuit.
    def binary| 5 (LHS RHS)
      if LHS then
        1
      else if RHS then
        1
      else
        0;

    # Binary logical and, which does not short circuit.
    def binary& 6 (LHS RHS)
      if !LHS then
        0
      else
        !!RHS;

    # Define = with slightly lower precedence than relationals.
    def binary = 9 (LHS RHS)
      !(LHS < RHS | LHS > RHS);

    # Define ':' for sequencing: as a low-precedence operator that ignores operands
    # and just returns the RHS.
    def binary : 1 (x y) y;

Given the previous if/then/else support, we can also define interesting
functions for I/O. For example, the following prints out a character
whose "density" reflects the value passed in: the lower the value, the
denser the character:

::

    ready>

    extern putchard(char)
    def printdensity(d)
      if d > 8 then
        putchard(32)  # ' '
      else if d > 4 then
        putchard(46)  # '.'
      else if d > 2 then
        putchard(43)  # '+'
      else
        putchard(42); # '*'
    ...
    ready> printdensity(1): printdensity(2): printdensity(3):
           printdensity(4): printdensity(5): printdensity(9):
           putchard(10);
    **++.
    Evaluated to 0.000000

Based on these simple primitive operations, we can start to define more
interesting things. For example, here's a little function that solves
for the number of iterations it takes a function in the complex plane to
converge:

::

    # Determine whether the specific location diverges.
    # Solve for z = z^2 + c in the complex plane.
    def mandleconverger(real imag iters creal cimag)
      if iters > 255 | (real*real + imag*imag > 4) then
        iters
      else
        mandleconverger(real*real - imag*imag + creal,
                        2*real*imag + cimag,
                        iters+1, creal, cimag);

    # Return the number of iterations required for the iteration to escape
    def mandleconverge(real imag)
      mandleconverger(real, imag, 0, real, imag);

This "``z = z2 + c``" function is a beautiful little creature that is
the basis for computation of the `Mandelbrot
Set <http://en.wikipedia.org/wiki/Mandelbrot_set>`_. Our
``mandelconverge`` function returns the number of iterations that it
takes for a complex orbit to escape, saturating to 255. This is not a
very useful function by itself, but if you plot its value over a
two-dimensional plane, you can see the Mandelbrot set. Given that we are
limited to using putchard here, our amazing graphical output is limited,
but we can whip together something using the density plotter above:

::

    # Compute and plot the mandlebrot set with the specified 2 dimensional range
    # info.
    def mandelhelp(xmin xmax xstep   ymin ymax ystep)
      for y = ymin, y < ymax, ystep in (
        (for x = xmin, x < xmax, xstep in
           printdensity(mandleconverge(x,y)))
        : putchard(10)
      )

    # mandel - This is a convenient helper function for plotting the mandelbrot set
    # from the specified position with the specified Magnification.
    def mandel(realstart imagstart realmag imagmag)
      mandelhelp(realstart, realstart+realmag*78, realmag,
                 imagstart, imagstart+imagmag*40, imagmag);

Given this, we can try plotting out the mandlebrot set! Lets try it out:

::

    ready> mandel(-2.3, -1.3, 0.05, 0.07);
    *******************************+++++++++++*************************************
    *************************+++++++++++++++++++++++*******************************
    **********************+++++++++++++++++++++++++++++****************************
    *******************+++++++++++++++++++++.. ...++++++++*************************
    *****************++++++++++++++++++++++.... ...+++++++++***********************
    ***************+++++++++++++++++++++++.....   ...+++++++++*********************
    **************+++++++++++++++++++++++....     ....+++++++++********************
    *************++++++++++++++++++++++......      .....++++++++*******************
    ************+++++++++++++++++++++.......       .......+++++++******************
    ***********+++++++++++++++++++....                ... .+++++++*****************
    **********+++++++++++++++++.......                     .+++++++****************
    *********++++++++++++++...........                    ...+++++++***************
    ********++++++++++++............                      ...++++++++**************
    ********++++++++++... ..........                        .++++++++**************
    *******+++++++++.....                                   .+++++++++*************
    *******++++++++......                                  ..+++++++++*************
    *******++++++.......                                   ..+++++++++*************
    *******+++++......                                     ..+++++++++*************
    *******.... ....                                      ...+++++++++*************
    *******.... .                                         ...+++++++++*************
    *******+++++......                                    ...+++++++++*************
    *******++++++.......                                   ..+++++++++*************
    *******++++++++......                                   .+++++++++*************
    *******+++++++++.....                                  ..+++++++++*************
    ********++++++++++... ..........                        .++++++++**************
    ********++++++++++++............                      ...++++++++**************
    *********++++++++++++++..........                     ...+++++++***************
    **********++++++++++++++++........                     .+++++++****************
    **********++++++++++++++++++++....                ... ..+++++++****************
    ***********++++++++++++++++++++++.......       .......++++++++*****************
    ************+++++++++++++++++++++++......      ......++++++++******************
    **************+++++++++++++++++++++++....      ....++++++++********************
    ***************+++++++++++++++++++++++.....   ...+++++++++*********************
    *****************++++++++++++++++++++++....  ...++++++++***********************
    *******************+++++++++++++++++++++......++++++++*************************
    *********************++++++++++++++++++++++.++++++++***************************
    *************************+++++++++++++++++++++++*******************************
    ******************************+++++++++++++************************************
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************
    Evaluated to 0.000000
    ready> mandel(-2, -1, 0.02, 0.04);
    **************************+++++++++++++++++++++++++++++++++++++++++++++++++++++
    ***********************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    *********************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++.
    *******************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++...
    *****************+++++++++++++++++++++++++++++++++++++++++++++++++++++++++.....
    ***************++++++++++++++++++++++++++++++++++++++++++++++++++++++++........
    **************++++++++++++++++++++++++++++++++++++++++++++++++++++++...........
    ************+++++++++++++++++++++++++++++++++++++++++++++++++++++..............
    ***********++++++++++++++++++++++++++++++++++++++++++++++++++........        .
    **********++++++++++++++++++++++++++++++++++++++++++++++.............
    ********+++++++++++++++++++++++++++++++++++++++++++..................
    *******+++++++++++++++++++++++++++++++++++++++.......................
    ******+++++++++++++++++++++++++++++++++++...........................
    *****++++++++++++++++++++++++++++++++............................
    *****++++++++++++++++++++++++++++...............................
    ****++++++++++++++++++++++++++......   .........................
    ***++++++++++++++++++++++++.........     ......    ...........
    ***++++++++++++++++++++++............
    **+++++++++++++++++++++..............
    **+++++++++++++++++++................
    *++++++++++++++++++.................
    *++++++++++++++++............ ...
    *++++++++++++++..............
    *+++....++++................
    *..........  ...........
    *
    *..........  ...........
    *+++....++++................
    *++++++++++++++..............
    *++++++++++++++++............ ...
    *++++++++++++++++++.................
    **+++++++++++++++++++................
    **+++++++++++++++++++++..............
    ***++++++++++++++++++++++............
    ***++++++++++++++++++++++++.........     ......    ...........
    ****++++++++++++++++++++++++++......   .........................
    *****++++++++++++++++++++++++++++...............................
    *****++++++++++++++++++++++++++++++++............................
    ******+++++++++++++++++++++++++++++++++++...........................
    *******+++++++++++++++++++++++++++++++++++++++.......................
    ********+++++++++++++++++++++++++++++++++++++++++++..................
    Evaluated to 0.000000
    ready> mandel(-0.9, -1.4, 0.02, 0.03);
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************
    **********+++++++++++++++++++++************************************************
    *+++++++++++++++++++++++++++++++++++++++***************************************
    +++++++++++++++++++++++++++++++++++++++++++++**********************************
    ++++++++++++++++++++++++++++++++++++++++++++++++++*****************************
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++*************************
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++**********************
    +++++++++++++++++++++++++++++++++.........++++++++++++++++++*******************
    +++++++++++++++++++++++++++++++....   ......+++++++++++++++++++****************
    +++++++++++++++++++++++++++++.......  ........+++++++++++++++++++**************
    ++++++++++++++++++++++++++++........   ........++++++++++++++++++++************
    +++++++++++++++++++++++++++.........     ..  ...+++++++++++++++++++++**********
    ++++++++++++++++++++++++++...........        ....++++++++++++++++++++++********
    ++++++++++++++++++++++++.............       .......++++++++++++++++++++++******
    +++++++++++++++++++++++.............        ........+++++++++++++++++++++++****
    ++++++++++++++++++++++...........           ..........++++++++++++++++++++++***
    ++++++++++++++++++++...........                .........++++++++++++++++++++++*
    ++++++++++++++++++............                  ...........++++++++++++++++++++
    ++++++++++++++++...............                 .............++++++++++++++++++
    ++++++++++++++.................                 ...............++++++++++++++++
    ++++++++++++..................                  .................++++++++++++++
    +++++++++..................                      .................+++++++++++++
    ++++++........        .                               .........  ..++++++++++++
    ++............                                         ......    ....++++++++++
    ..............                                                    ...++++++++++
    ..............                                                    ....+++++++++
    ..............                                                    .....++++++++
    .............                                                    ......++++++++
    ...........                                                     .......++++++++
    .........                                                       ........+++++++
    .........                                                       ........+++++++
    .........                                                           ....+++++++
    ........                                                             ...+++++++
    .......                                                              ...+++++++
                                                                        ....+++++++
                                                                       .....+++++++
                                                                        ....+++++++
                                                                        ....+++++++
                                                                        ....+++++++
    Evaluated to 0.000000
    ready> ^D

At this point, you may be starting to realize that Kaleidoscope is a
real and powerful language. It may not be self-similar :), but it can be
used to plot things that are!

With this, we conclude the "adding user-defined operators" chapter of
the tutorial. We have successfully augmented our language, adding the
ability to extend the language in the library, and we have shown how
this can be used to build a simple but interesting end-user application
in Kaleidoscope. At this point, Kaleidoscope can build a variety of
applications that are functional and can call functions with
side-effects, but it can't actually define and mutate a variable itself.

Strikingly, variable mutation is an important feature of some languages,
and it is not at all obvious how to `add support for mutable
variables <LangImpl7.html>`_ without having to add an "SSA construction"
phase to your front-end. In the next chapter, we will describe how you
can add variable mutation without building SSA in your front-end.

Full Code Listing
=================

Here is the complete code listing for our running example, enhanced with
the if/then/else and for expressions.. To build this example, use:

.. code-block:: bash

    # Compile
    clang++ -g toy.cpp `llvm-config --cppflags --ldflags --libs core jit native` -O3 -o toy
    # Run
    ./toy

On some platforms, you will need to specify -rdynamic or
-Wl,--export-dynamic when linking. This ensures that symbols defined in
the main executable are exported to the dynamic linker and so are
available for symbol resolution at run time. This is not needed if you
compile your support code into a shared library, although doing that
will cause problems on Windows.

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
      tok_binary = -11, tok_unary = -12
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

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= ifexpr
    ///   ::= forexpr
    static ExprAST *ParsePrimary() {
      switch (CurTok) {
      default: return Error("unknown token when expecting an expression");
      case tok_identifier: return ParseIdentifierExpr();
      case tok_number:     return ParseNumberExpr();
      case '(':            return ParseParenExpr();
      case tok_if:         return ParseIfExpr();
      case tok_for:        return ParseForExpr();
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
    static std::map<std::string, Value*> NamedValues;
    static FunctionPassManager *TheFPM;

    Value *ErrorV(const char *Str) { Error(Str); return 0; }

    Value *NumberExprAST::Codegen() {
      return ConstantFP::get(getGlobalContext(), APFloat(Val));
    }

    Value *VariableExprAST::Codegen() {
      // Look this variable up in the function.
      Value *V = NamedValues[Name];
      return V ? V : ErrorV("Unknown variable name");
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
      //   ...
      //   start = startexpr
      //   goto loop
      // loop:
      //   variable = phi [start, loopheader], [nextvariable, loopend]
      //   ...
      //   bodyexpr
      //   ...
      // loopend:
      //   step = stepexpr
      //   nextvariable = variable + step
      //   endcond = endexpr
      //   br endcond, loop, endloop
      // outloop:

      // Emit the start code first, without 'variable' in scope.
      Value *StartVal = Start->Codegen();
      if (StartVal == 0) return 0;

      // Make the new basic block for the loop header, inserting after current
      // block.
      Function *TheFunction = Builder.GetInsertBlock()->getParent();
      BasicBlock *PreheaderBB = Builder.GetInsertBlock();
      BasicBlock *LoopBB = BasicBlock::Create(getGlobalContext(), "loop", TheFunction);

      // Insert an explicit fall through from the current block to the LoopBB.
      Builder.CreateBr(LoopBB);

      // Start insertion in LoopBB.
      Builder.SetInsertPoint(LoopBB);

      // Start the PHI node with an entry for Start.
      PHINode *Variable = Builder.CreatePHI(Type::getDoubleTy(getGlobalContext()), 2, VarName.c_str());
      Variable->addIncoming(StartVal, PreheaderBB);

      // Within the loop, the variable is defined equal to the PHI node.  If it
      // shadows an existing variable, we have to restore it, so save it now.
      Value *OldVal = NamedValues[VarName];
      NamedValues[VarName] = Variable;

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

      Value *NextVar = Builder.CreateFAdd(Variable, StepVal, "nextvar");

      // Compute the end condition.
      Value *EndCond = End->Codegen();
      if (EndCond == 0) return EndCond;

      // Convert condition to a bool by comparing equal to 0.0.
      EndCond = Builder.CreateFCmpONE(EndCond,
                                  ConstantFP::get(getGlobalContext(), APFloat(0.0)),
                                      "loopcond");

      // Create the "after loop" block and insert it.
      BasicBlock *LoopEndBB = Builder.GetInsertBlock();
      BasicBlock *AfterBB = BasicBlock::Create(getGlobalContext(), "afterloop", TheFunction);

      // Insert the conditional branch into the end of LoopEndBB.
      Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

      // Any new code will be inserted in AfterBB.
      Builder.SetInsertPoint(AfterBB);

      // Add a new entry to the PHI node for the backedge.
      Variable->addIncoming(NextVar, LoopEndBB);

      // Restore the unshadowed variable.
      if (OldVal)
        NamedValues[VarName] = OldVal;
      else
        NamedValues.erase(VarName);


      // for expr always returns 0.0.
      return Constant::getNullValue(Type::getDoubleTy(getGlobalContext()));
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

      // If this is an operator, install it.
      if (Proto->isBinaryOp())
        BinopPrecedence[Proto->getOperatorName()] = Proto->getBinaryPrecedence();

      // Create a new basic block to start insertion into.
      BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
      Builder.SetInsertPoint(BB);

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

`Next: Extending the language: mutable variables / SSA
construction <LangImpl7.html>`_

