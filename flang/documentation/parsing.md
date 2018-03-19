The F18 Parser
==============
This program source code implements a parser for the Fortran programming
language.

The draft ISO standard for Fortran 2018 dated July 2017 was used as the
primary definition of the language.
The parser also accepts many features from previous versions of the standard
that are no longer part of the Fortran 2018 language.

It also accepts many features that have never been part of any version
of the standard Fortran language but have been supported by previous
implementations and are known or suspected to remain in use.
As a general principle, we want to recognize and implement any
such feature so long as it does not conflict with requirements of the
current standard for Fortran.

The parser is implemented in standard ISO C++ and requires the 2017
edition of the language and library.  The parser constitutes a reentrant
library with no mutable or constructed static data.
Best modern C++ programming practices are observed to ensure that
the ownership of dynamic memory is clear, that value rather than
object semantics are defined for the data structures, that
most functions are free from invisible side effects, and that the
strictest available type checking is enforced by the C++ compiler
when the Fortran parser is built.
Class inheritance is rare and dynamic polymorphism is avoided in
favor of modern discriminated unions.
To the furthest reasonable extent, the parser has been implemented
in a declarative fashion that corresponds closely to the
text of the Fortran language standard.

The several major modules of the Fortran parser are composed into a
top-level Parsing class, by means of which one may drive the parsing of
a source file and receive its parse tree and error messages.
The interface of the Parsing class corresponds to the two major passes
of the parser, which are described below.

Prescanning and Preprocessing
-----------------------------
The first pass is performed by an instance of the Prescanner class,
with help from an instance of Preprocessor.

The prescanner generates the "cooked character stream", in which
* line ends have been normalized
* all INCLUDE files have been expanded
* all Fortran continuation lines have been collapsed
* all comments and insignificant spaces have been removed
* fixed form right margins have been clipped
* extra blank card columns have been inserted into character literals
and Hollerith constants
* preprocessing directives have been implemented
* preprocessing macro invocations have been expanded
* legacy *D* lines in fixed form source have been omitted or included

Lines in the cooked character stream can be of arbitrary length.
The purpose of the cooked character stream is to enable the implementation
of a parser whose sole concern is the recognition of the Fortran language
from productions that closely correspond to the grammar that is presented
in the Fortran standard.

The preprocessing phase interacts with the prescanner by means of
token sequences.
These token sequences, which partition input lines into (pointers to)
contiguous blocks of characters, are the only place in this Fortran parser
in which we have a reified tokenization of the program source.
The prescanner builds token sequences out of source lines and supplies them
to the preprocessor, which interprets directives and expands macro
invocations.
The token sequences returned by the preprocessor are marshaled to
form the cooked character stream that is the output of the prescanner.
The preprocessor and prescanner can both instantiate new temporary
instances of the Prescanner class to locate, open, and process any
include files.

The tight interaction and mutual designs of the prescanner and preprocessor
enable a principled implementation of preprocessing for the Fortran
language that implements a reasonable facsimile of the C language
preprocessor that is fully aware of Fortran's source forms, line
continuation mechanisms, case insensitivity, &c.

The content of the cooked character stream is available and is useful
for debugging, being as it is a value being forwarded from the first major
pass to the second.

Source Provenance
-----------------
The prescanner constructs a chronicle of every file that
is read by the parser, viz. the original source file and all that it
directly or indirectly includes.  One copy of the content of each of
these files is mapped or read into the address space of the parser.
Memory mapping is used initially, but files with DOS line breaks or
a missing terminal newline are immediately normalized in a buffer
when necessary.

The virtual input stream, which marshals every appearance of every file
and every expansion of every macro invocation, is not materialized
as an actual stream of bytes.
There is, however, a mapping from each byte position in this virtual
input stream back to whence it came (maintained by an instance
of the AllSources class).
Offsets into this virtual input stream constitute values of the
Provenance class.
Provenance values, and contiguous ranges thereof, are used to describe
and delimit source positions for messaging.

Further, every byte in the cooked character stream supplied by the
prescanner to the parser can be inexpensively mapped to its provenance.
Simple `const char *` pointers to characters in the cooked character stream,
or to contiguous ranges thereof, are used as source position indicators
within the parser and in the parse tree.

Messages
--------
Message texts, and snprintf-like formatting strings for constructing
messages, are instantiated in the various components of the parser
with C++ user defined character literals tagged with `_en_US`
(signifying the dialect of English used in the United States) so that
they may be easily identified, localized, and mapped.
As described above, messages are associated with source code positions
by means of provenance values.

The Parse Tree
--------------
Each of the many numbered requirement productions in the standard Fortran
language grammar, as well as the productions implied by legacy extensions
and preserved obsolescent features, maps to a distinct class in the
parse tree so as to maximize the efficacy of static type checking
by the C++ compiler.

A transcription of the Fortran grammar appears, with production requirement
numbers, in the commentary before these class definitions, so that one
may easily refer to the standard (or to the parse tree definitions while
reading that document).

There are three paradigms that collectively implement most of the
parse tree classes:
* wrappers, in which a single data member `v` has been encapsulated
  in a new type
* tuples, in which several values of arbitrary type have been
  encapsulated in a single data member `t` whose type is an instance
  of `std::tuple<>`
* discriminated unions, in which one value whose type is a dynamic selection
  from a set of distinct types is saved in a data member `u` whose type
  is an instance of `std::variant<>`

The use of these patterns is a design convenience, and exceptions to them
are common where it makes better sense to do so.

Parse tree entities should be viewed as values, not objects.
They are assembled with C++ move semantics during parse tree construction.
Their default and copy constructors are deliberately deleted in their
declarations.

There is a general purpose library by means of which parse trees may be
traversed.

Parsing
-------
This compiler attempts to recognize the entire cooked character stream
(see above) as a Fortran program, and records the reductions made
during a successful recognition as a parse tree value.
Its grammar is that of a whole source file, not just of its possible
statements, and it has no global data structures that track the subprogram
hierarchy or the structure of nested block constructs.
It performs (essentially) no semantic analysis along the way, deferring
that work to the next pass of the compiler.

The resulting parse tree contains necessarily ambiguous parses that
cannot be resolved without recourse to a symbol table.
Most notably, leading assignments to array elements can be misrecognized
as statement function definitions, and array element references can be
misrecognized as function calls.
The semantic analysis phase of the compiler performs local rewrites
of the parse tree once it can be disambiguated.

Formally speaking, this parser uses recursive descent with localized
backtracking as its basic architecture.
It is not generated as a table or code from a specification of the
grammar; rather, it _is_ the grammar, as declaratively respecified
using a small collection of basic token recognition objects and a library of
"parser combinator" template functions that compose them to form more
complicated recognizers and their correspondences to the construction
of parse tree values.

Unparsing
---------
Parse trees can be converted back into free form Fortran source code.
This formatter is not really a classical "pretty printer", but is
more of a data structure dump whose output is suitable for compilation
by another compiler.
It can also be useful for testing the parser, as a reparse of an
unparsed parse tree should be identical to the original.
