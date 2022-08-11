" Part of the Carbon Language project, under the Apache License v2.0 with LLVM
" Exceptions. See /LICENSE for license information.
" SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if exists("b:current_syntax")
  finish
endif

syn match carbonIdentifier '[a-zA-Z0-9_]\+' contained
syn match carbonNominalType '[a-zA-Z0-9_]\+' contained
syn match carbonComment "//.*$" contains=carbonTodo,carbonPreprocess
syn keyword carbonTodo TODO contained
syn keyword carbonPreprocess RUN AUTOUPDATE CHECK contained

" carbon primitive types and literals
syn keyword carbonBooleanType bool
syn match carbonIntType 'i\d\+'
syn match carbonUnsignedIntType 'u\d\+'
syn match carbonFloatType 'f\d\+'
syn keyword carbonStringType String
syn keyword carbonBoolean true false
syn match carbonNumber '\<[0-9][_0-9]*\(\.[_0-9]\+\(e[-+]\?[1-9][0-9]*\)\?\)\?\>'
syn match carbonHexLiteral '\<0x[_0-9A-F]\+\(\.[_0-9A-F]\+\(p[+-]\?[1-9][0-9]*\)\?\)\?\>'
syn match carbonBinLiteral '\<0b[_01]\+\>'
syn region carbonStringLiteral start=+"+ end=+"+ skip=+\\"+
syn region carbonBlockStringLiteral start=+"""+ end=+"""+ skip=+\\"""+

" carbon declaration introducers
syn keyword carbonNamespaceDeclaration namespace
syn keyword carbonVariableDeclaration var nextgroup=carbonIdentifier skipwhite
syn keyword carbonVariableDeclarationMod returned
syn keyword carbonConstantDeclaration let nextgroup=carbonIdentifier skipwhite
syn keyword carbonFunctionDeclaration fn
syn keyword carbonClassDeclaration class  nextgroup=carbonNominalType skipwhite
syn keyword carbonClassDeclarationMod base abstract final
syn keyword carbonClassMethodDeclaration fn destructor
syn keyword carbonClassMethodDeclarationMod private virtual abstract protected impl
syn keyword carbonAliasDeclaration alias nextgroup=carbonNominalType skipwhite
syn keyword carbonInterfaceDeclaration interface nextgroup=carbonNominalType skipwhite
syn keyword carbonChoiceDeclaration choice nextgroup=carbonNominalType skipwhite
syn keyword carbonPackageDeclaration package nextgroup=carbonIdentifier skipwhite
syn keyword carbonLibraryDeclaration library nextgroup=carbonStringLiteral skipwhite

" carbon control flow
syn keyword carbonConditional if then else
syn keyword carbonLoop while for in
syn keyword carbonSwitch match case default
syn keyword carbonControlFlowStatement break continue return

" carbon operators
syn keyword carbonLogicalOperator and or not

" handle any other keywords
syn keyword carbonKeywordExtends extends nextgroup=carbonNominalType skipwhite
syn keyword carbonKeywordSelf Self
syn keyword carbonKeywordAs as
syn keyword carbonKeywordTemplate template
syn keyword carbonKeywordExternal external
syn keyword carbonKeywordForAll forall
syn keyword carbonKeywordAPI api
syn keyword carbonKeywordImport import nextgroup=carbonIdentifier skipwhite

hi def link carbonIdentifier Identifier
hi def link carbonNominalType Type
hi def link carbonComment Comment
hi def link carbonTodo Todo
hi def link carbonPreprocess PreProc
hi def link carbonBooleanType carbonType
hi def link carbonIntType carbonType
hi def link carbonUnsignedIntType carbonType
hi def link carbonFloatType carbonType
hi def link carbonStringType carbonType
hi def link carbonType Type
hi def link carbonBoolean Boolean
hi def link carbonHexLiteral carbonNumber
hi def link carbonBinLiteral carbonNumber
hi def link carbonNumber Number
hi def link carbonStringLiteral carbonString
hi def link carbonBlockStringLiteral carbonString
hi def link carbonString String
hi def link carbonNamespaceDeclaration carbonDeclaration
hi def link carbonVariableDeclaration carbonDeclaration
hi def link carbonVariableDeclarationMod carbonDeclaration
hi def link carbonConstantDeclaration carbonDeclaration
hi def link carbonFunctionDeclaration carbonDeclaration
hi def link carbonClassDeclaration carbonDeclaration
hi def link carbonClassDeclarationMod carbonDeclaration
hi def link carbonClassMethodDeclaration carbonDeclaration
hi def link carbonClassMethodDeclarationMod carbonDeclaration
hi def link carbonAliasDeclaration carbonDeclaration
hi def link carbonInterfaceDeclaration carbonDeclaration
hi def link carbonChoiceDeclaration carbonDeclaration
hi def link carbonPackageDeclaration Include
hi def link carbonLibraryDeclaration Include
hi def link carbonDeclaration Structure
hi def link carbonKeywordExtends carbonKeyword
hi def link carbonKeywordSelf carbonKeyword
hi def link carbonKeywordAs carbonKeyword
hi def link carbonKeywordTemplate carbonKeyword
hi def link carbonKeywordExternal carbonKeyword
hi def link carbonKeywordForAll carbonKeyword
hi def link carbonKeywordAPI Structure
hi def link carbonKeywordImport Include
hi def link carbonKeyword Keyword
hi def link carbonConditional Conditional
hi def link carbonLoop Repeat
hi def link carbonSwitch Repeat
hi def link carbonControlFlowStatement Statement
hi def link carbonLogicalOperator carbonOperator
hi def link carbonOperator Operator

let b:current_syntax = "carbon"
