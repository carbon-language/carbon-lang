# Syntax of Basic Carbon Features

The purpose of this proposal is to establish some basic syntactic
elements of the Carbon language and make sure that the grammar is
unambiguous and can be parsed by an LALR parser such as `yacc` or
`bison`.  The grammar presented here has indeed been checked by
`bison`. The language features in this basic grammar include control
flow via `if` and `while`, functions, simple structures, choice, and
pattern matching. The main syntactic categories are `declaration`,
`statement`, and `expression`. Establishing these syntactic categories
should help the other proposals choose syntax that is compatible with
the rest of the language.

The grammar is based on the Carbon language overview and on the
proposals for pattern matching, structs, tuples, and
metaprogramming. There may be places that this grammar does not
accurately capture what was intended in those proposal, which should
trigger some useful discussion and revisions.

We begin with a summary of the three main syntactic categories:

* `declaration` includes function, structure, and choice definitions.

* `statement` includes variable definitions, assignment, blocks, `if`,
    `while`, `match`, `break`, `continue`, and `return`.
    
* `expression` plays several roles. In an initial attempt these roles
    were separate, with three different syntactic categories, but that
    led to ambiguities in the grammar. Folding them into one category
    resolved the ambiguities. The intent is that the three roles
    are teased apart during type checking.

    1. The `expression` category plays the usual role of expressions
       that produce a value, such as integer literals and arithmetic
       expression.

    2. To fascilitate metaprogramming and reflection, `expression`
       also includes type expressions, include literals such as `Int`
       and `Bool` and constructors for function types, etc.

    3. `expression` is also used for patterns, for example, in the
        `case` of a `match`, also for describing the parameters of
        a function, and on the left-hand side of variable definitions.


## Expressions

The following grammar defines the concrete syntax for
expressions. Below we comment on a few unusual aspects of the grammar.

    expression:
      identifier
    | expression '.' identifier
    | expression '[' expression ']'
    | expression ':' identifier
    | integer_literal
    | "true"
    | "false"
    | '(' field_list ')'
    | expression "==" expression
    | expression '+' expression
    | expression '-' expression
    | expression "&&" expression
    | expression "||" expression
    | '!' expression
    | '-' expression
    | expression '(' field_list ')'
    | "Int"
    | "Bool"
    | "Type"
    | "auto"
    | "fn" expression "->" expression
    ;
    field_list:
      /* empty */
    | field
    | field ',' field_list
    ;
    field:
      expression
    | identifier '=' expression
    ;

The grammar rule

    expression:  expression ':' identifier

is for pattern variables. For example, in a variable definition such as

    var Int: x = 0;
    
the `Int: x` is parsed with the grammar rule for pattern variables.
In the above grammar rule, the `expression` to the left of the `:`
must evaluate to a type at compile time.

The grammar rule

    expression:  '(' field_list ')'
    
is primarily for constructing a tuple, but it is also used for
creating tuple types and tuple patterns, depending on the context in
which the expression occurs.

## Statements

The following grammar defines the concrete syntax for statements.

    statement:
      "var" expression '=' expression ';'
    | expression '=' expression ';'
    | expression ';'
    | "if" '(' expression ')' statement "else" statement
    | "while" '(' expression ')' statement
    | "break" ';'
    | "continue" ';'
    | "return" expression ';'
    | '{' statement_list '}'
    | "match" '(' expression ')' '{' clause_list '}'
    ;
    statement_list:
      statement
    | statement statement_list
    ;
    clause_list:
      /* empty */
    | clause clause_list
    ;
    clause:
      "case" expression "=>" statement
    | "default" "=>" statement 
    ;

In the grammar rule for the variable definition statement

    statement:  "var" expression '=' expression ';'
    
the left-hand-side `expression` is used as a pattern, so it would
typically evaluate to a variable pattern or some other kind of value
(such as a tuple) that contains variable patterns.

Likewise, in the rule for `case`

    clause:  "case" expression "=>" statement

the `expression` is used as a pattern.


# Declarations

The following grammar defines the syntax for declarations.

    declaration:
      "fn" identifier expression "->" expression '{' statement_list '}'
    | "struct" identifier '{' member_list '}'
    | "choice" identifier '{' alternative_list '}'
    ;
    member:
      "var" expression ':' identifier ';'
    | "method" identifier expression "->" expression '{' statement_list '}'
    ;
    member_list:
      /* empty */
    | member member_list
    ;
    alternative:
      "alt" identifier expression ';'
    ;
    alternative_list:
      /* empty */
    | alternative alternative_list
    ;
    declaration_list:
      /* empty */
    | declaration declaration_list
    ;

In the grammar fule for function definitions

    declaration:  "fn" identifier expression "->" expression '{' statement_list '}'

the `expression` before the `->` is used as a pattern to describe the
parameters of the function whereas the `expression` after the `->`
must evaluate to a type at compile-time, to express the return type of
the function. The story is the same for method definitions.

In the rule for field declarations

    member:  "var" expression ':' identifier ';'

the `expression` must evaluate to a type at compile time.
The same is true for the `expression` in the grammar
rule for an alternative:

    alternative:  "alt" identifier expression ';'
