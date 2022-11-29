# Issue 2251, many violations of style rule that `case` blocks have braces 
Issue 2251 is the starting point to create a list of connected issues.
## PDF Rendering
Categorizing issue 2251 is done using Google Docs.
It's easy to quickly make topics into bold font style
and rearrange comments in the Google Docs unordered list.
_____
third_party/issue_manage/issue_2251.pdf

## Categorized Topics Meant for Resolution
The PDF file attempted to organize the topics within
the issue 2251.
Now that there's some categorization done
from the issue outward,
the topics on issue have been sufficiently identified.
## Carbon Docs Temporary Clone 
Now it will help to temporarily make 
a tree that starts from the perspective of
the Carbon docs.
Each categorized topic on issue can then
be adjusted to fit inside the
Carbon docs appropriately.
This Carbon docs tree should be written in
markdown here.
The pdf file then becomes deletable.
## Categorized Topics Before Cloning Carbon Docs Tree
- style rule
  - braces
    - conditional
      - if
    - switch
      - case blocks
        - variable scope
          - -Wimplicit-fallthrough
    - loop statements
    - casing
      - being followed
    - comments on closing namespaces
      - clang-format checks
  - we deviate
    - explorer
      - fix
    - not followed
      - accumulate more
      - cost
      - not retain style rules
  - add automated checks
    - clang
  - remove the style rule
  - don't automatically check
    - The code inconsistencies in explorer concerning case may be
irrelevant
  - leads decision
  - code structure
  - pull requests
    - frequent contributors
    - style compliance
    - review process
    - contributor gap
      - unequal treatment
      - new contributors
      - global consistency
    - Modern and evolving
      - easy to learn
- match
  - syntax
  - braces
  - fallthrough
    - explicit
- goto
- design
  - Expressions
    - interpreted
      - partial precedence ordering
        - Expression components
        - most developers understand
        - parentheses
          - fallthrough
- C++ code mirror what we expect Carbon code to look like
  - can't change fallthrough behavior
    - break will still be there
    - example by zygoloid
       - Recursion
       - references
  - braces have less utility and are more just repetitive when there's no variable
being scoped
- Carbon without C++ experience
  - Lists
____
Modern and evolving
Solid language foundations that are easy to learn, especially if you have used C++
## Carbon Docs Design Table of Contents
-   [Introduction](#introduction) 
    -   [This document is provisional](#this-document-is-provisional) 
    -   [Tour of the basics](#tour-of-the-basics) 
-   [Code and comments](#code-and-comments) 
-   [Build modes](#build-modes) 
-   [Types are values](#types-are-values) 
-   [Primitive types](#primitive-types) 
    -   [`bool`](#bool) 
    -   [Integer types](#integer-types) 
        -   [Integer literals](#integer-literals) 
    -   [Floating-point types](#floating-point-types) 
        -   [Floating-point literals](#floating-point-literals) 
    -   [String types](#string-types) 
        -   [String literals](#string-literals) 
 -   [Value categories and value phases](#value-categories-and-value-phases) 
 -   [Composite types](#composite-types) 
     -   [Tuples](#tuples) 
     -   [Struct types](#struct-types) 
     -   [Pointer types](#pointer-types) 
     -   [Arrays and slices](#arrays-and-slices) 
 -   [Expressions](#expressions) 
 -   [Declarations, Definitions, and Scopes](#declarations-definitions-and-scopes) 
 -   [Patterns](#patterns) 
     -   [Binding patterns](#binding-patterns) 
     -   [Destructuring patterns](#destructuring-patterns) 
     -   [Refutable patterns](#refutable-patterns) 
 -   [Name-binding declarations](#name-binding-declarations) 
     -   [Constant `let` declarations](#constant-let-declarations) 
     -   [Variable `var` declarations](#variable-var-declarations) 
     -   [`auto`](#auto) 
 -   [Functions](#functions) 
     -   [Parameters](#parameters) 
     -   [`auto` return type](#auto-return-type) 
     -   [Blocks and statements](#blocks-and-statements) 
     -   [Assignment statements](#assignment-statements) 
     -   [Control flow](#control-flow) 
         -   [`if` and `else`](#if-and-else) 
         -   [Loops](#loops) 
             -   [`while`](#while) 
             -   [`for`](#for) 
             -   [`break`](#break) 
             -   [`continue`](#continue) 
         -   [`return`](#return) 
             -   [`returned var`](#returned-var) 
         -   [`match`](#match) 
 -   [User-defined types](#user-defined-types) 
     -   [Classes](#classes) 
         -   [Assignment](#assignment) 
         -   [Class functions and factory functions](#class-functions-and-factory-functions) 
         -   [Methods](#methods) 
         -   [Inheritance](#inheritance) 
         -   [Access control](#access-control) 
         -   [Destructors](#destructors) 
         -   [`const`](#const) 
         -   [Unformed state](#unformed-state) 
         -   [Move](#move) 
         -   [Mixins](#mixins) 
     -   [Choice types](#choice-types) 
 -   [Names](#names) 
     -   [Files, libraries, packages](#files-libraries-packages) 
     -   [Package declaration](#package-declaration) 
     -   [Imports](#imports) 
     -   [Name visibility](#name-visibility) 
     -   [Package scope](#package-scope) 
     -   [Namespaces](#namespaces) 
     -   [Naming conventions](#naming-conventions) 
     -   [Aliases](#aliases) 
     -   [Name lookup](#name-lookup) 
         -   [Name lookup for common types](#name-lookup-for-common-types) 
 -   [Generics](#generics) 
     -   [Checked and template parameters](#checked-and-template-parameters) 
     -   [Interfaces and implementations](#interfaces-and-implementations) 
     -   [Combining constraints](#combining-constraints) 
     -   [Associated types](#associated-types) 
     -   [Generic entities](#generic-entities) 
         -   [Generic Classes](#generic-classes) 
         -   [Generic choice types](#generic-choice-types) 
         -   [Generic interfaces](#generic-interfaces) 
         -   [Generic implementations](#generic-implementations) 
     -   [Other features](#other-features) 
     -   [Generic type equality and `observe` declarations](#generic-type-equality-and-observe-declarations) 
     -   [Operator overloading](#operator-overloading) 
         -   [Common type](#common-type) 
 -   [Bidirectional interoperability with C and C++](#bidirectional-interoperability-with-c-and-c) 
     -   [Goals](#goals) 
     -   [Non-goals](#non-goals) 
     -   [Importing and `#include`](#importing-and-include) 
     -   [ABI and dynamic linking](#abi-and-dynamic-linking) 
     -   [Operator overloading](#operator-overloading-1) 
     -   [Templates](#templates) 
     -   [Standard types](#standard-types) 
     -   [Inheritance](#inheritance-1) 
     -   [Enums](#enums) 
 -   [Unfinished tales](#unfinished-tales) 
     -   [Safety](#safety) 
     -   [Lifetime and move semantics](#lifetime-and-move-semantics) 
     -   [Metaprogramming](#metaprogramming) 
     -   [Pattern matching as function overload resolution](#pattern-matching-as-function-overload-resolution) 
     -   [Error handling](#error-handling) 
     -   [Execution abstractions](#execution-abstractions) 
         -   [Abstract machine and execution model](#abstract-machine-and-execution-model) 
         -   [Lambdas](#lambdas) 
         -   [Co-routines](#co-routines) 
         -   [Concurrency](#concurrency)


