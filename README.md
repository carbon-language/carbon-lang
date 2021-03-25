# Barcon

Swift implementation of executable semantics of
[Carbon](https://carbon-language/carbon-lang).

## Preparation

1. Have Swift installed and in your path.
2. `git submodule update --init`

## To Build

    make build
    
Note: the first build [may fail](https://github.com/roop/citron/issues/3)
reporting conflicts in the grammar.  Just build again if this happens.

## To Test

    make test

- Note: the first test [may fail](https://github.com/roop/citron/issues/3)
reporting conflicts in the grammar.  Just test again if this happens.

## To work on the project in Xcode

    make build || ls Sources/Syntax/Parser.swift
    open Package.swift
